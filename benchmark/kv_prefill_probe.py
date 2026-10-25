#!/usr/bin/env python3
"""kv_prefill_probe: profile one cold 1M prefill and dump a per-kernel inventory.

Made for one question, where the fp8 KV tax actually goes. The tax is measured and
prefill-only, about 9.1 s on a cold 1M TTFT and nothing at decode, but the recorded
explanation, that it lives in the attention read path, was inference from a
single-sided capture and a kernel that reads fp8 faster than bf16 has since
disproved it. The only way to settle it is a two-sided capture, one boot per cache
dtype, diffed by kernel.

Run once per boot, then compare the two inventories:

    PROF=1 ./serve-4x-1m.sh --kv-cache-dtype auto
    python3 benchmark/kv_prefill_probe.py --arm bf16
    PROF=1 ./serve-4x-1m.sh
    python3 benchmark/kv_prefill_probe.py --arm fp8
    python3 benchmark/kv_prefill_probe.py --compare \
        logs/kvab/inventory-bf16-s130100.json logs/kvab/inventory-fp8-s130100.json

The engine must be booted with PROF=1. Without it the profiler config has no
max_iterations and a start call opens an unbounded window that writes until the box
runs out of memory, so this tool refuses rather than risk it. Inventories land under
logs/, which is local scratch and never committed.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")

# Ordered first-match classifier. Same intent as phase_report.py, narrower scope:
# this only needs to separate the paths a cache dtype can plausibly move.
CATEGORIES = [
    ("qsa-read", ("_qsa_sparse_paged_gqa_splitk", "_qsa_merge_splitk")),
    ("qsa-index", ("_qsa_mqa_paged", "persistent_topk", "cooperative_topk")),
    (
        "qsa-prepare",
        (
            "qsa_prepare",
            "_store_qsa_rows",
            "_compress_qsa_groups",
            "gemma_rmsnorm",
            "mrope",
            "qwen_gdn",
        ),
    ),
    ("kv-write", ("reshape_and_cache", "cast_to_fp8", "triton_reshape")),
    ("nccl", ("nccl", "record_param_comms")),
    ("gdn-state", ("gdn", "chunk_scaled_dot", "fused_recurrent", "causal_conv")),
    ("gemm", ("gemm", "nvjet", "cutlass", "humming", "Matmul", "matmul")),
    ("copy/memset", ("memcpy", "memset", "copy_device_to_device", "CatArray")),
    ("elementwise", ("elementwise", "unrolled", "vectorized")),
]
# The QSA indexer and rope work moves into these names when the fused prepare is
# unavailable, which is one of the things being tested. Count them as index work
# only when they are not already claimed above.
AMBIGUOUS = ("rmsnorm", "norm", "rope", "index")


def classify(name):
    low = name.lower()
    for cat, keys in CATEGORIES:
        if any(k.lower() in low for k in keys):
            return cat
    if any(k in low for k in AMBIGUOUS):
        return "qsa-prepare"
    return "other"


def get(url, timeout=10):
    return urllib.request.urlopen(url, timeout=timeout).read()


def post(url):
    req = urllib.request.Request(url, data=b"", method="POST")
    return urllib.request.urlopen(req, timeout=60).read()


def metrics(url):
    out = {}
    for line in get(url, timeout=5).decode().splitlines():
        for key in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
        ):
            if line.startswith(key + " ") or line.startswith(key + "{"):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def check_boot(arm, boot_log):
    """Refuse to measure the wrong thing. The dtype is not exposed by any endpoint."""
    if not boot_log or not os.path.exists(boot_log):
        print(
            f"WARNING: no boot log at {boot_log!r}, cannot verify cache dtype. "
            "Trust the arm label or point --boot-log at the live one."
        )
        return
    with open(boot_log, errors="ignore") as handle:
        text = handle.read()
    if "torch_profiler_dir" not in text or "max_iterations" not in text:
        raise SystemExit(
            "this boot has no bounded profiler config. Reboot with PROF=1. A "
            "config without max_iterations opens an unbounded capture and the "
            "writer swaps the machine."
        )
    want = "'kv_cache_dtype': 'fp8'" if arm == "fp8" else None
    if want and want not in text:
        raise SystemExit(f"--arm {arm} but the boot log does not show fp8 KV, aborting")
    if arm == "bf16" and "'kv_cache_dtype': 'fp8'" in text:
        raise SystemExit("--arm bf16 but the boot log shows fp8 KV, aborting")
    print(f"boot log confirms arm {arm}: profiler configured, dtype consistent")


def newest_traces(trace_dir, since):
    """One file per rank, newest only. A directory of stale captures is a trap."""
    by_rank = {}
    for entry in os.listdir(trace_dir):
        if "rank" not in entry or not entry.endswith((".json", ".json.gz")):
            continue
        path = os.path.join(trace_dir, entry)
        if os.path.getmtime(path) < since:
            continue
        rank = entry.split("rank")[-1].split(".")[0]
        if rank not in by_rank or os.path.getmtime(path) > os.path.getmtime(
            by_rank[rank]
        ):
            by_rank[rank] = path
    return by_rank


def inventory(paths):
    per_kernel = defaultdict(lambda: [0.0, 0])
    for rank, path in sorted(paths.items()):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", errors="ignore") as handle:
            data = json.load(handle)
        events = data.get("traceEvents", data if isinstance(data, list) else [])
        kept = 0
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") not in (
                "kernel",
                "gpu_memcpy",
                "gpu_memset",
            ):
                continue
            name = ev.get("name", "?")
            dur = float(ev.get("dur", 0.0) or 0.0)
            slot = per_kernel[name]
            slot[0] += dur
            slot[1] += 1
            kept += 1
        print(f"  rank {rank}: {kept} device events from {os.path.basename(path)}")
        del data, events
    return {k: {"us": round(v[0], 1), "n": v[1]} for k, v in per_kernel.items()}


def run_cell(args):
    meturl = args.base_url.rstrip("/") + "/metrics"
    base = args.base_url.rstrip("/")
    m0 = metrics(meturl)
    if m0.get("vllm:num_requests_running", 0) or m0.get("vllm:num_requests_waiting", 0):
        raise SystemExit(
            "requests are in flight, this tool must own the lane while it profiles"
        )
    print(
        f"idle confirmed, prompt_tokens_total at start "
        f"{m0['vllm:prompt_tokens_total']:.0f}"
    )

    cmd = [
        sys.executable,
        BENCH,
        "--base-url",
        args.base_url,
        "--ctx",
        str(args.ctx),
        "--fills",
        "100",
        "--concurrency",
        "1",
        "--decode",
        "1",
        "--salt",
        str(args.salt),
        "--label",
        f"kvab {args.arm} salt {args.salt}",
    ]
    started = time.time()
    proc = subprocess.Popen(cmd)

    # Arm on a clock, not on the prompt counter. vllm:prompt_tokens_total moves
    # when a request completes, so watching it can only report the prefill after
    # it is over, which is how a first version of this tool profiled an idle
    # engine and got nothing. This cell is one request, so its prefill runs from
    # about a second after launch until roughly ttft_s.
    print(
        f"cell launched, arming in {args.arm_after_s:.0f}s "
        f"of a ~{args.ttft_s:.0f}s prefill"
    )
    while time.time() - started < args.arm_after_s:
        if proc.poll() is not None:
            raise SystemExit("the cell exited before the trigger, nothing profiled")
        time.sleep(0.5)
    tok_a = metrics(meturl)["vllm:prompt_tokens_total"]
    post(base + "/start_profile")
    print(
        f"profiler armed at {(tok_a - m0['vllm:prompt_tokens_total']) / 1e3:.0f}k "
        f"prompt tokens, holding {args.window_s:.0f}s"
    )
    time.sleep(args.window_s)
    post(base + "/stop_profile")
    time.sleep(3.0)
    tok_b = metrics(meturl)["vllm:prompt_tokens_total"]
    print("profiler stopped, waiting for the cell and the trace files")
    proc.wait()

    deadline = time.time() + 240
    while time.time() < deadline:
        traces = newest_traces(args.trace_dir, started)
        if traces and all(
            time.time() - os.path.getmtime(p) > 5 for p in traces.values()
        ):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"no fresh traces in {args.trace_dir} within 240s")

    print(f"parsing {len(traces)} rank traces")
    inv = inventory(traces)
    total_us = sum(v["us"] for v in inv.values())
    moved = max(tok_b - tok_a, 1.0)
    cats = defaultdict(float)
    for name, v in inv.items():
        cats[classify(name)] += v["us"]
    out = {
        "arm": args.arm,
        "salt": args.salt,
        "ctx": args.ctx,
        "window_s": args.window_s,
        "prompt_tokens_in_window": int(moved),
        "device_us_in_window": round(total_us, 1),
        "us_per_1k_tokens": {
            k: round(v["us"] / moved * 1000, 3) for k, v in inv.items()
        },
        "kernels": inv,
        "categories_us": {
            k: round(v, 1) for k, v in sorted(cats.items(), key=lambda x: -x[1])
        },
        "traces": {r: os.path.basename(p) for r, p in traces.items()},
    }
    if total_us < 1e6:
        raise SystemExit(
            f"the window captured only {total_us / 1e3:.1f} ms of device time, it "
            "missed the prefill. Raise --arm-after-s and check the cell is running."
        )
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"inventory-{args.arm}-s{args.salt}.json")
    with open(dest, "w") as handle:
        json.dump(out, handle, indent=1)
    print(
        f"\nwindow: {moved / 1e3:.0f}k prompt tokens, "
        f"{total_us / 1e6:.2f} s of device time"
    )
    print("category shares of device time in window:")
    for cat, us in out["categories_us"].items():
        print(f"  {cat:<14} {us / 1e6:8.3f} s  {100 * us / total_us:5.1f}%")
    print("\ntop kernels per 1k prompt tokens, us:")
    ranked = sorted(out["us_per_1k_tokens"].items(), key=lambda x: -x[1])[:12]
    for name, per_k in ranked:
        print(f"  {per_k:8.2f}  {name[:74]}")
    print(f"\nwrote {dest}")


def compare(paths):
    tables = []
    for p in paths:
        with open(p) as handle:
            tables.append(json.load(handle))
    names = sorted(
        {n for t in tables for n in t["us_per_1k_tokens"]},
        key=lambda n: -max(abs(t["us_per_1k_tokens"].get(n, 0.0)) for t in tables),
    )
    head = (
        "kernel".ljust(58) + "".join(f"{t['arm']:>12}" for t in tables) + "     delta"
    )
    print(f"us per 1k prompt tokens, {' vs '.join(t['arm'] for t in tables)}")
    print(head)
    for n in names[:30]:
        vals = [t["us_per_1k_tokens"].get(n, 0.0) for t in tables]
        row = n[:57].ljust(58) + "".join(f"{v:>12.2f}" for v in vals)
        if len(vals) == 2:
            row += f"{vals[1] - vals[0]:>+11.2f}"
        print(row)
    print("\ncategory totals, us per 1k prompt tokens")
    cats = sorted({c for t in tables for c in t["categories_us"]})
    print(
        "category".ljust(16) + "".join(f"{t['arm']:>12}" for t in tables) + "     delta"
    )
    for c in sorted(
        cats, key=lambda c: -sum(t["categories_us"].get(c, 0) for t in tables)
    ):
        vals = [
            t["categories_us"].get(c, 0.0) / t["prompt_tokens_in_window"] * 1000
            for t in tables
        ]
        row = c.ljust(16) + "".join(f"{v:>12.2f}" for v in vals)
        if len(vals) == 2:
            row += f"{vals[1] - vals[0]:>+11.2f}"
        print(row)
    parts = []
    for t in tables:
        per_k = sum(t["categories_us"].values()) / t["prompt_tokens_in_window"] * 1000
        win = t["prompt_tokens_in_window"] / 1e3
        parts.append(f"{t['arm']} {per_k:.1f} (window {win:.0f}k)")
    print("\ntotals, us per 1k tokens: " + ", ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://fender.lan:8000")
    ap.add_argument("--arm", choices=["bf16", "fp8"], default="fp8")
    ap.add_argument("--ctx", type=int, default=1000000)
    ap.add_argument(
        "--salt",
        type=int,
        default=130100,
        help="use the same salt on both arms, it selects the prompt",
    )
    ap.add_argument(
        "--arm-after-s",
        type=float,
        default=25.0,
        help="seconds of prefill to let elapse before arming",
    )
    ap.add_argument(
        "--ttft-s",
        type=float,
        default=85.0,
        help="expected cold TTFT, about 76 on bf16 and 85 on fp8, hint only",
    )
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--boot-log", default=os.path.join(HERE, "..", "boot-4x-1m.log"))
    ap.add_argument("--trace-dir", default=os.path.join(HERE, "..", "logs", "prof"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "logs", "kvab"))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)
    check_boot(args.arm, args.boot_log)
    return run_cell(args)


if __name__ == "__main__":
    main()
