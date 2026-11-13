#!/usr/bin/env python3
# Context-fill + concurrency benchmark for the Qwen3.8-Flash-Next PP3 servers.
#
# Each (fill %, concurrency N) cell builds N fresh prompts of that token
# depth (unique salt per stream, so every stream pays a full prefill), fires
# them together, and reports serving metrics for the cell:
#
#   TTFT mean/p95 s  first token per stream, measured from batch start;
#                    queueing shows up here first
#   wall s           batch start to last completion
#   in tok/s         total prompt tokens / wall      (aggregate prefill load)
#   out tok/s        total completion tokens / wall  (aggregate throughput)
#   win avg tok/s/req
#                    (completion-1)/(finish-first) per stream, mean: the
#                    window the user actually experiences, STARVED while
#                    other streams still prefill (chunked prefill paces
#                    decode to ~1 step/s). Not batched-decode capacity.
#   sust agg tok/s   peak sustained aggregate generation rate sampled from
#                    vllm:generation_tokens_total at 0.5 s (mean of the
#                    fastest 3 samples). Needs a decode long enough to form
#                    an all-decoding phase: use --decode >= 512, the default
#                    1024. Under a short decode this column only catches
#                    transient overlap peaks.
#   acc len          1 + accepted/drafts diffed from /metrics over the whole
#                    cell. Full-window and legacy: since vLLM base 8407bc0bb6
#                    the counters also include drafts emitted during the
#                    prefill-phase steps, which dilutes this toward the prefill
#                    ratio. For cross-base comparisons read acc win below or
#                    run accept_window.py.
#   preempt          vLLM preemptions during the cell; nonzero means the cell
#                    exceeded the KV pool and its numbers are not throughput
#   match            percent of sentence lines in the last 40 lines of each
#                    stream that reached the repeat loop (thinking blocks
#                    and the max_tokens-cut tail line are excluded); "-"
#                    means no stream looped inside the budget. Under 98,
#                    read it with the acc columns: deep greedy loops drift on
#                    some 1M salts with healthy acceptance (content lottery),
#                    while a values corruption drops acc too and repeats
#                    on every salt of the boot
#   acc win          1 + accepted/drafts over the all-decoding window only,
#                    diffed from the 0.5 s counter samples; "-" when the cell
#                    is too short or the window holds no sample pair
#   bg               drafts the idle probes found around the cell (before plus
#                    after it); nonzero means other traffic polluted acc and
#                    the rate columns
#
# Under each row prints "tel=<gen>x<lanes>/<peak W>": the lowest PCIe link state
# and the highest power draw any GPU reached while that cell ran. Idle 170HX
# links renegotiate down, so the sampler runs in a separate process during the
# cell only; it is fail-soft and never changes a measured number.
#
# /metrics is global: run the box idle or the acceptance column is polluted.
# The KV pool caps concurrency (262k pool measured 4.39 single contexts);
# watch the preempt column before believing a big wall s.
#
#   ./benchmark/bench.py --base-url http://<host>:8000 --ctx 262144 \
#       --fills 25,50,75,100 --concurrency 1,2,4
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from mtp_ab_probe import REPEAT_SENTENCE, build_prompt, calibrate, metrics, tail_for


def stream_run(base, model, prompt, max_tokens, timeout):
    """One streaming greedy completion; absolute start/first/finish times, the
    usage block, and the completion text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    first = None
    usage = {}
    pieces = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            text = choices[0].get("text") if choices else ""
            if text:
                if first is None:
                    first = time.time()
                pieces.append(text)
    end = time.time()
    return {
        "start": start,
        "first": first or end,
        "end": end,
        "usage": usage,
        "text": "".join(pieces),
    }


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def gen_sampler(base, t0, stop, samples, timeout, tag=None, counter_samples=None):
    """Aggregate generation tok/s samples every 0.5 s until stopped.

    counter_samples, when passed, collects (absolute time, drafts, accepted)
    so the caller can diff the drafter counters over an inner window instead of
    the whole cell. The times are absolute because the window bounds are read
    off the stream events, not off t0.
    """
    prev_t = time.time()
    prev = metrics(base, timeout)
    beat = prev_t
    while not stop.wait(0.5):
        t = time.time()
        now = metrics(base, timeout)
        rate = (now["gen_tokens"] - prev["gen_tokens"]) / (t - prev_t)
        samples.append((t - t0, rate))
        if counter_samples is not None:
            counter_samples.append((t, now["drafts"], now["accepted"]))
        prev_t, prev = t, now
        if tag and t - beat >= 30.0:
            print(
                f"  ... {tag}: {(t - t0) / 60:.1f} min, now {rate:,.0f} tok/s agg",
                flush=True,
            )
            beat = t


def idle_probe(base, timeout, seconds=2.0):
    """Background counter drift over a quiet stretch: (drafts, gen_tokens)."""
    a = metrics(base, timeout)
    time.sleep(seconds)
    b = metrics(base, timeout)
    return b["drafts"] - a["drafts"], b["gen_tokens"] - a["gen_tokens"]


def tail_match(texts, window=40, min_hits=5):
    """Share of REPEAT_SENTENCE lines in the tail of any stream that looped.

    This checkpoint is a reasoner. /v1/completions streams its planning
    block (which may quote the sentence on its own line) before the loop,
    and max_tokens cuts the final line mid-word, so line-position heuristics
    break. Judge the last `window` lines of each stream instead, dropping
    the cut tail line, and only for streams that show the loop (>= min_hits
    exact lines somewhere; a healthy loop runs to the end). A stream still
    thinking at its token budget is skipped: inconclusive content behavior,
    not corruption. None when no stream qualified.
    """
    judged = []
    for text in texts:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if sum(1 for ln in lines if ln == REPEAT_SENTENCE) < min_hits:
            continue
        judged.extend((lines[:-1] if len(lines) > 1 else lines)[-window:])
    if not judged:
        return None
    return sum(1 for ln in judged if ln == REPEAT_SENTENCE) / len(judged)


def completion_hash(texts):
    """Identity of the exact values this cell streamed back."""
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()


def window_acc(counter_samples, t_start, t_end, min_span=10.0):
    """Acceptance length over the samples inside [t_start, t_end].

    Entries are (absolute time, drafts, accepted). None when the window holds
    no sample pair, when the pair spans under min_span seconds, or when it saw
    no drafts: a thin slice of the counters is noise.
    """
    inside = [s for s in counter_samples if t_start <= s[0] <= t_end]
    if len(inside) < 2:
        return None
    lo, hi = inside[0], inside[-1]
    if hi[0] - lo[0] < min_span or hi[1] - lo[1] <= 0:
        return None
    return 1.0 + (hi[2] - lo[2]) / (hi[1] - lo[1])


def fmt_pct(v):
    return "-" if v is None else f"{v * 100:.0f}"


def fmt2(v):
    return "-" if v is None else f"{v:.2f}"


# --- In-cell environment telemetry (cmp.wiki Benchmark_protocol rev 1456) ---
# An idle 170HX link renegotiates down to Gen1, so link state only means
# anything when sampled while the cell runs. The sampler is a separate process
# appending to a temp CSV: a stalled probe then costs the measured window
# nothing. Everything here is fail-soft, telemetry never fails a cell.
TEL_INTERVAL = 2.0
TEL_KEYS = ("cap", "draw", "gen", "lanes")
TEL_INTS = ("gen", "lanes")
TEL_GEN_BY_GTS = ((2.5, 1), (5.0, 2), (8.0, 3), (16.0, 4), (32.0, 5), (64.0, 6))


def _tel_sh(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip()[:100] or f"rc {out}")
    return out.stdout


def _tel_num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _tel_gen(gts):
    if gts is None:
        return None
    return min(TEL_GEN_BY_GTS, key=lambda g: abs(g[0] - gts))[1]


def _tel_sysfs_bdf(nv_bdf):
    """'00000000:01:00.0' (nvidia-smi) to '0000:01:00.0' (lspci, sysfs)."""
    parts = nv_bdf.strip().lower().split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        parts[0] = parts[0][-4:]
    return ":".join(parts)


def _tel_query(*fields):
    """nvidia-smi index -> [field values], for one or more fields per call."""
    text = _tel_sh(
        [
            "nvidia-smi",
            "--query-gpu=index," + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    out = {}
    for line in text.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) == len(fields) + 1:
            out[cols[0]] = cols[1:]
    return out


def _tel_link(bdf):
    """(gen, lanes) from the LnkSta line; sysfs exposes the same registers
    when pciutils is not installed."""
    try:
        sta = re.search(r"LnkSta:[^\n]*", _tel_sh(["lspci", "-vv", "-s", bdf]))
        if sta:
            speed = re.search(r"Speed\s+([\d.]+)", sta.group(0))
            width = re.search(r"Width\s+x(\d+)", sta.group(0))
            if speed and width:
                return _tel_gen(_tel_num(speed.group(1))), int(width.group(1))
    except Exception:
        pass
    node = f"/sys/bus/pci/devices/{bdf}/"
    with open(node + "current_link_speed") as fh:
        gen = _tel_gen(_tel_num(fh.read().split()[0]))
    with open(node + "current_link_width") as fh:
        return gen, int(fh.read().strip())


def telemetry_loop(path):
    """Child entry point: one CSV line per GPU per 2 s until the parent stops it."""
    try:
        head = _tel_query("pci.bus_id", "power.limit", "power.draw")
    except Exception:
        return
    bdfs = {k: _tel_sysfs_bdf(v[0]) for k, v in head.items()}
    with open(path, "w") as fh:
        while True:
            try:
                power = _tel_query("power.limit", "power.draw")
            except Exception:
                power = {}
            # One stamp per pass, so the parent counts passes, not GPU rows.
            stamp = time.time()
            for idx, bdf in sorted(bdfs.items()):
                cols = power.get(idx) or ["", ""]
                try:
                    gen, lanes = _tel_link(bdf)
                except Exception:
                    gen = lanes = None
                vals = (_tel_num(cols[0]), _tel_num(cols[1]), gen, lanes)
                body = ",".join("" if v is None else str(v) for v in vals)
                fh.write(f"{stamp:.1f},{idx},{body}\n")
                fh.flush()
            time.sleep(TEL_INTERVAL)


def telemetry_start():
    try:
        fd, path = tempfile.mkstemp(prefix="bench-tel-", suffix=".csv")
        os.close(fd)
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--tel-loop", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return None, None, f"spawn: {exc}"
    return proc, path, None


def telemetry_stop(proc, path):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(2)
    reason, lines = None, []
    try:
        with open(path) as fh:
            lines = [ln.strip().split(",") for ln in fh if ln.strip()]
    except (OSError, TypeError) as exc:
        reason = f"read: {exc}"
    with contextlib.suppress(OSError):
        os.unlink(path)
    rows = [ln for ln in lines if len(ln) == 6]
    if not rows:
        return {"unavailable": reason or "no samples (nvidia-smi unreachable?)"}
    gpus, passes = {}, set()
    for ts, idx, limit, draw, gen, lanes in rows:
        passes.add(ts)
        acc = gpus.setdefault(idx, {})
        for key, raw in zip(TEL_KEYS, (limit, draw, gen, lanes), strict=True):
            num = _tel_num(raw)
            if num is None:
                continue
            acc.setdefault(key, []).append(int(num) if key in TEL_INTS else num)
    order = sorted(gpus, key=lambda i: _tel_num(i) or 0)

    def _pick(idx, key, reduce):
        vals = gpus[idx].get(key) or []
        return reduce(vals) if vals else None

    out = {
        "power_cap_w": [_pick(i, "cap", max) for i in order],
        "power_peak_w": [_pick(i, "draw", max) for i in order],
        "power_last_w": [_pick(i, "draw", lambda v: v[-1]) for i in order],
        "pcie_speed_min_gen_seen": [_pick(i, "gen", min) for i in order],
        "pcie_width_min_lanes_seen": [_pick(i, "lanes", min) for i in order],
        "samples": len(passes),
    }
    if any(v is None for v in out["power_peak_w"] + out["pcie_width_min_lanes_seen"]):
        out["note"] = "power or link probe failed on some GPU"
    return out


def fmt_tel(tel):
    if not tel or "unavailable" in tel:
        return "n/a"
    gens = [v for v in tel["pcie_speed_min_gen_seen"] if v]
    lanes = [v for v in tel["pcie_width_min_lanes_seen"] if v]
    peaks = [v for v in tel["power_peak_w"] if v]
    link = f"{int(min(gens))}x{int(min(lanes))}" if gens and lanes else "n/a"
    return f"{link}/{max(peaks):.0f}W" if peaks else f"{link}/n/a"


def run_cell(base, model, depth, n, cpt, salt, decode, timeout, tag=None):
    prompts = [
        build_prompt(depth, cpt, salt + i, tail_for("repeat", salt + i), task="repeat")
        for i in range(n)
    ]
    bg_drafts_pre, bg_gen_pre = idle_probe(base, timeout)
    before = metrics(base, timeout)
    t0 = time.time()
    stop = threading.Event()
    samples = []
    counter_samples = []
    tel_proc, tel_path, tel_err = telemetry_start()
    sampler = threading.Thread(
        target=gen_sampler,
        args=(base, t0, stop, samples, timeout, tag, counter_samples),
        daemon=True,
    )
    sampler.start()
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(
            pool.map(lambda p: stream_run(base, model, p, decode, timeout), prompts)
        )
    stop.set()
    sampler.join()
    telemetry = telemetry_stop(tel_proc, tel_path)
    if tel_err:
        telemetry = {"unavailable": tel_err}
    # vLLM flushes step stats on a ~5 s interval; reading the counters
    # 2 s after the last chunk misses the cell's own tail, which then
    # shows up as background drafts in the post-probe (observed 2026-10-22).
    time.sleep(6.0)
    after = metrics(base, timeout)
    bg_drafts_post, bg_gen_post = idle_probe(base, timeout)

    t0 = min(r["start"] for r in runs)
    wall = max(r["end"] for r in runs) - t0
    rates = sorted((v for _, v in samples), reverse=True)
    sustained = sum(rates[:3]) / min(3, len(rates)) if rates else float("nan")
    ttfts = [r["first"] - t0 for r in runs]
    ptoks = sum(r["usage"].get("prompt_tokens", depth) for r in runs)
    ctoks = sum(max(1, r["usage"].get("completion_tokens", decode)) for r in runs)
    steady = [
        (ct - 1) / (r["end"] - r["first"])
        for r in runs
        for ct in [max(1, r["usage"].get("completion_tokens", decode))]
        if r["end"] > r["first"] and ct > 1
    ]
    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted"] - before["accepted"]
    texts = [r.get("text", "") for r in runs]
    # All streams decode together between the last first token and the first
    # finish; the counters diffed over that slice exclude the prefill drafts.
    decode_start = max(r["first"] for r in runs)
    decode_end = min(r["end"] for r in runs)
    return {
        "concurrency": n,
        "prompt_tokens_each": round(ptoks / n),
        "ttft_mean_s": sum(ttfts) / len(ttfts),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_max_s": max(ttfts),
        "wall_s": wall,
        "in_tps": ptoks / wall,
        "out_tps": ctoks / wall,
        "win_avg_tps_per_req": sum(steady) / len(steady) if steady else float("nan"),
        "sustained_agg_tps": sustained,
        "samples": [[round(t, 2), round(v, 2)] for t, v in samples],
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else float("nan"),
        "acc_window_len": window_acc(counter_samples, decode_start, decode_end),
        "tail_match_frac": tail_match(texts),
        "completion_sha256": completion_hash(texts),
        "per_stream_sha256": [
            hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts
        ],
        "bg_drafts_pre": bg_drafts_pre,
        "bg_drafts_post": bg_drafts_post,
        "bg_gen_pre": bg_gen_pre,
        "bg_gen_post": bg_gen_post,
        "preemptions": after["preemptions"] - before["preemptions"],
        "telemetry": telemetry,
        "total_prompt_tokens": ptoks,
        "total_completion_tokens": ctoks,
        "ttfts_s": ttfts,
    }


def dump_json(args, label, rows):
    if not args.json:
        return
    tmp = args.json + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(
            {"label": label, "ctx": args.ctx, "decode": args.decode, "rows": rows},
            fh,
            indent=2,
        )
    os.replace(tmp, args.json)


def main():
    # Sampler child mode, ahead of argparse: a cell run needs --ctx, while the
    # sampler never touches the server and must not be bound by its CLI.
    if "--tel-loop" in sys.argv:
        telemetry_loop(sys.argv[sys.argv.index("--tel-loop") + 1])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--ctx",
        type=int,
        required=True,
        help="server max_model_len, for computing fill percent",
    )
    ap.add_argument(
        "--fills",
        default="25,50,75,100",
        help="comma-separated percent-of-context points",
    )
    ap.add_argument(
        "--concurrency",
        default="1",
        help="comma-separated stream counts per fill point",
    )
    ap.add_argument(
        "--decode",
        type=int,
        default=1024,
        help="tokens per stream; short windows under ~512 measure drafter "
        "warmup, not steady decode",
    )
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--salt",
        type=int,
        default=None,
        help="prompt series offset; default is time-derived so runs "
        "never reuse prefixes cached by earlier runs",
    )
    ap.add_argument("--json", default=None, help="also dump results to this file")
    ap.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to wait after every cell so thermal and pool state settle",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="fire one throwaway request before the grid, to clear the "
        "first-MTP-step JIT stall and the cold allocator state",
    )
    args = ap.parse_args()

    models_url = args.base_url.rstrip("/") + "/v1/models"
    listed = json.loads(urllib.request.urlopen(models_url, timeout=30).read())
    model = args.model or listed["data"][0]["id"]
    cpt = calibrate(args.base_url, model, 1, args.timeout)
    label = args.label or f"ctx {args.ctx:,}"
    fills = [float(p) for p in args.fills.split(",")]
    concs = [int(c) for c in args.concurrency.split(",")]

    print(f"\n### {label}  (decode {args.decode} per stream, greedy, unique prompts)\n")
    print(
        "| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s "
        "| in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s "
        "| acc len | preempt | match | acc win | bg |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:"
        "|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    salt = args.salt if args.salt is not None else int(time.time()) // 10 * 10
    if args.warmup:
        # salt-1, not salt: cell 1 stream 0 uses seed salt, so seeding the
        # warmup the same way would leave its 8k prefix in the cache and hand
        # cell 1 a warm start.
        warm = build_prompt(
            8192, cpt, salt - 1, tail_for("repeat", salt - 1), task="repeat"
        )
        stream_run(args.base_url, model, warm, 64, args.timeout)
        time.sleep(3)
        print("warmup done", flush=True)
    for pct in fills:
        for n in concs:
            # 100% fill = largest prompt with room for decode inside --ctx;
            # the 1% shave covers calibration drift. Oversized = HTTP 400.
            depth = int((args.ctx - args.decode) * pct / 100.0 * 0.99)
            tag = f"fill {pct:g}% conc {n} ({depth:,} tok each)"
            print(f"| starting {tag} at {time.strftime('%H:%M:%S')}", flush=True)
            row = run_cell(
                args.base_url, model, depth, n, cpt, salt, args.decode,
                args.timeout, tag=tag,
            )
            row["fill_pct"] = pct
            salt += 100
            rows.append(row)
            print(
                f"| {pct:g} | {n} | {row['prompt_tokens_each']:,} "
                f"| {row['ttft_mean_s']:.1f} | {row['ttft_p95_s']:.1f} "
                f"| {row['wall_s']:.1f} | {row['in_tps']:,.0f} | {row['out_tps']:,.0f} "
                f"| {row['win_avg_tps_per_req']:.1f} | {row['sustained_agg_tps']:.0f} "
                f"| {row['mean_acceptance_length']:.2f} "
                f"| {row['preemptions']:.0f} | {fmt_pct(row['tail_match_frac'])} "
                f"| {fmt2(row['acc_window_len'])} "
                f"| {row['bg_drafts_pre'] + row['bg_drafts_post']:.0f} |",
                flush=True,
            )
            print(f"  tel={fmt_tel(row['telemetry'])}", flush=True)
            if row["bg_drafts_pre"] or row["bg_drafts_post"]:
                print(
                    f"  WARNING: background traffic during cell ({tag}): "
                    f"pre {row['bg_drafts_pre']:.0f} post "
                    f"{row['bg_drafts_post']:.0f} drafts; acceptance and rate "
                    "columns are suspect",
                    flush=True,
                )
            frac = row["tail_match_frac"]
            if frac is not None and frac < 0.98:
                print(
                    f"  WARNING: cell output diverges from the repeat-task tail "
                    f"({frac * 100:.0f}% lines match, acc win "
                    f"{fmt2(row['acc_window_len'])}, acc full "
                    f"{row['mean_acceptance_length']:.2f}); read together: "
                    "healthy acc = deep-loop content drift, collapsed acc on "
                    "every salt = suspect the values",
                    flush=True,
                )
            dump_json(args, label, rows)
            if args.cooldown:
                time.sleep(args.cooldown)



if __name__ == "__main__":
    main()
