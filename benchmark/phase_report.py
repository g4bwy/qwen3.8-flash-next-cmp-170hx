#!/usr/bin/env python3
"""phase_report: attribute decode-step time from torch profiler traces.

Run against the logs/prof output of a phase_probe cell (server booted with
PROF=1). Reads the per-rank chrome traces; keeps real decode steps (worker
spans execute_context_0(...)_generation_1(...); the 3:1 interspersed
generation_0(0) spans are the drafter's draft waves, same iteration).
Event dur/ts units are MICROSECONDS.

Outputs, per real decode step:
1. cadence, and GPU busy vs gap per rank (gap here is small by construction:
   PP idle shows up as NCCL SPIN kernels, which are counted busy)
2. kernel category rollup PER RANK, NCCL spin split out - the PP step is the
   serial sum of compute across ranks; spin time is bubble, not cost
3. duration histogram per rank: where the launch-tax vs byte-time boundary sits
4. top kernels per rank

usage: phase_report.py [--dir logs/prof] [--top 8]
stdlib only.
"""
import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from glob import glob

STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
SPAN_CATS = {"cpu_op", "user_annotation"}
WAITER_US = 2000.0  # an nccl kernel longer than this is a wait, not a transfer

CATEGORIES = [
    ("moe", r"topkGating|moe_align|count_and_sort|expert|moe_fused|process_input.*Quantiz"),
    ("qsa-attn", r"qsa|persistent_topk"),
    ("gdn/state", r"gdn|causal_conv|hc_|delta|chunk|scan|post_update"),
    ("humming", r"humming"),
    ("gemm-bf16", r"gemm|nvjet|cutlass|Matmul|matmul"),
    ("marlin-int8", r"[Mm]arlin"),
    ("sampler", r"softmax|multinomial|argmax|rejection|penalt|renorm|sample_"),
    ("copy/memset", r"[Mm]emcpy|[Mm]emset|cast|[Cc]opy"),
    ("reduce/norm", r"[Rr]educe|norm|silu|act_|rope|Rotary"),
    ("elementwise", r"elementwise|vectorized|unrolled|fill|CatArray|where"),
]


def med(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_rank(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    steps, gpu, drafts = [], [], 0
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        name, cat, ts = e.get("name", ""), e.get("cat", ""), e.get("ts", 0.0)
        if cat in GPU_CATS:
            gpu.append((ts, ts + e["dur"], name))
        elif cat in SPAN_CATS and not cat.startswith("gpu"):
            m = STEP_RE.match(name)
            if m:
                ctx_r, _ct, gen_r, _gt = (int(x) for x in m.groups())
                if ctx_r == 0 and gen_r == 1:
                    steps.append(ts)
                elif ctx_r == 0 and gen_r == 0:
                    drafts += 1
    gpu.sort()
    steps.sort()
    return steps, drafts, gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/prof")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    files = sorted(glob(a.dir + "/*rank*.pt.trace.json*"))
    if not files:
        print(f"no trace files in {a.dir} - run phase_probe after a PROF=1 boot")
        return 1
    data = {}
    for path in files:
        m = re.search(r"rank[_-]?(\d+)", path)
        rank = int(m.group(1)) if m else len(data)
        data[rank] = load_rank(path)

    n_ref = min(len(data[r][0]) for r in data) or 1
    print(f"{len(files)} rank traces; real decode steps per rank: "
          + ", ".join(f"r{r}={len(data[r][0])}" for r in sorted(data))
          + f" (draft-wave spans excluded: "
          + ", ".join(f"r{r}={data[r][1]}" for r in sorted(data)) + ")")
    print(f"budgets below divide by the per-rank real step count; "
          f"n_ref={n_ref}\n")

    print("== cadence per rank: period | GPU busy (incl spin) | true gap | waiters ==")
    print("rank |steps |period_ms |busy_ms |gap_ms |spin_ms |waiter_kernels/step")
    for r in sorted(data):
        steps, _d, gpu = data[r]
        if len(steps) < 3:
            continue
        n = len(steps)
        periods = [steps[i + 1] - steps[i] for i in range(n - 1)]
        edges = steps[1:] + [steps[-1] + med(periods)]
        busy, spin, waiters = [], 0, 0
        for s, e in zip(steps, edges):
            win = sorted((max(gs, s), min(ge, e)) for gs, ge, _n in gpu if ge > s and gs < e)
            total, cs, ce = 0.0, None, None
            for a0, a1 in win:
                if ce is None:
                    cs, ce = a0, a1
                elif a0 > ce:
                    total += ce - cs
                    cs, ce = a0, a1
                else:
                    ce = max(ce, a1)
            if ce is not None:
                total += ce - cs
            busy.append(total)
        spin = sum(ge - gs for gs, ge, nm in gpu if "nccl" in nm and ge - gs > WAITER_US)
        waiters = sum(1 for _s, _e, nm in gpu if "nccl" in nm and _e - _s > WAITER_US) / n
        b_ms = med(busy) / 1000
        p_ms = med(periods) / 1000
        nccl_all = sum(ge - gs for gs, ge, nm in gpu if "nccl" in nm) / n / 1000
        print(f"{r:>4} |{n:>5} |{p_ms:>10.2f} |{b_ms:>8.2f} |{p_ms - b_ms:>7.2f} "
              f"|{nccl_all:>8.2f} |{waiters:>5.1f} (wait-time kernels >2ms)")

    # per-rank category rollup on compute kernels; nccl handled separately
    cat_ms = {r: defaultdict(float) for r in data}
    ker_us = {r: defaultdict(float) for r in data}
    ker_ct = {r: defaultdict(int) for r in data}
    hist = {}
    for r in sorted(data):
        steps, _d, gpu = data[r]
        n = max(len(steps), 1)
        durs = []
        for gs, ge, nm in gpu:
            d = ge - gs
            if "nccl" in nm:
                cat_ms[r]["nccl-spin"] += d / n / 1000
                continue
            durs.append(d)
            for cat, pat in CATEGORIES:
                if re.search(pat, nm):
                    cat_ms[r][cat] += d / n / 1000
                    break
            else:
                cat_ms[r]["other"] += d / n / 1000
            ker_us[r][nm[:60]] += d / n / 1000
            ker_ct[r][nm[:60]] += 1 / n
        buckets = [(0, 2), (2, 5), (5, 20), (20, 100), (100, 1e12)]
        cnt = [sum(1 for x in durs if lo <= x < hi) / n for lo, hi in buckets]
        ms5 = sum(x for x in durs if x < 5) / n / 1000
        ms100 = sum(x for x in durs if x >= 100) / n / 1000
        hist[r] = (len(durs) / n, cnt, ms5, ms100, med(durs))

    names = ["nccl-spin"] + [c for c, _ in CATEGORIES] + ["other"]
    print("\n== kernel time per real step, ms per rank "
          "(nccl-spin = GPU occupied waiting, not data movement; "
          "PP step ~ sum of compute columns) ==")
    print("category".ljust(13) + "".join(f"{'r'+str(r):>8}" for r in sorted(data)) + f"{'tot':>8}")
    for c in names:
        row, s = c.ljust(13), 0.0
        for r in sorted(data):
            v = cat_ms[r].get(c, 0.0)
            s += v
            row += f"{v:>8.2f}"
        print(row + f"{s:>8.2f}")

    print("\n== compute kernel duration histogram (nccl excluded) ==")
    print("rank | kt/s | <2us |2-5us |5-20 |20-100 |>100 | ms<5us | ms>100us | med_us")
    for r in sorted(hist):
        kt, cnt, ms5, ms100, m = hist[r]
        print(f"{r:>4} |{kt:>5.0f} |{cnt[0]:>5.0f} |{cnt[1]:>5.0f} |{cnt[2]:>5.0f} "
              f"|{cnt[3]:>6.0f} |{cnt[4]:>5.0f} |{ms5:>7.2f} |{ms100:>9.2f} |{m:>6.2f}")

    for r in sorted(data):
        top = sorted(ker_us[r].items(), key=lambda kv: -kv[1])[:a.top]
        if not top:
            continue
        print(f"\n== top {a.top} kernels, rank {r} (us per real step) ==")
        for nm, us in top:
            print(f"  {us*1000:>7.0f}  x{ker_ct[r][nm]:>5.1f}  {nm}")

    print("\nreading guide: (1) wall time per step ~= cadence column; the compute-"
          "column sums\napproximate it because PP stages are serial for one stream. "
          "(2) spin/bubble is\nstructural for single-stream PP and shrinks as "
          "concurrency fills the pipeline; do not\noptimize it directly. (3) the "
          "actionable numbers are the compute columns and the\nhistogram: byte-side "
          "categories (gemm/humming/marlin) vs latency/state-side\n(qsa/gdn/moe-"
          "align/copy/norm/other).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
