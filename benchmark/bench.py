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
#   acc len          1 + accepted/drafts diffed from /metrics over the cell
#   preempt          vLLM preemptions during the cell; nonzero means the cell
#                    exceeded the KV pool and its numbers are not throughput
#
# /metrics is global: run the box idle or the acceptance column is polluted.
# The KV pool caps concurrency (262k pool measured 4.39 single contexts);
# watch the preempt column before believing a big wall s.
#
#   ./benchmark/bench.py --base-url http://<host>:8000 --ctx 262144 \
#       --fills 25,50,75,100 --concurrency 1,2,4
import argparse
import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from mtp_ab_probe import build_prompt, calibrate, metrics, tail_for


def stream_run(base, model, prompt, max_tokens, timeout):
    """One streaming greedy completion; absolute start/first/finish times."""
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
            if first is None and choices and choices[0].get("text"):
                first = time.time()
    end = time.time()
    return {"start": start, "first": first or end, "end": end, "usage": usage}


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def gen_sampler(base, t0, stop, samples, timeout):
    """Aggregate generation tok/s samples every 0.5 s until stopped."""
    prev_t = time.time()
    prev_v = metrics(base, timeout)["gen_tokens"]
    while not stop.wait(0.5):
        t = time.time()
        v = metrics(base, timeout)["gen_tokens"]
        samples.append((t - t0, (v - prev_v) / (t - prev_t)))
        prev_t, prev_v = t, v


def run_cell(base, model, depth, n, cpt, salt, decode, timeout):
    prompts = [
        build_prompt(depth, cpt, salt + i, tail_for("repeat", salt + i), task="repeat")
        for i in range(n)
    ]
    before = metrics(base, timeout)
    t0 = time.time()
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(
        target=gen_sampler, args=(base, t0, stop, samples, timeout), daemon=True
    )
    sampler.start()
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(
            pool.map(lambda p: stream_run(base, model, p, decode, timeout), prompts)
        )
    stop.set()
    sampler.join(5)
    time.sleep(2.0)
    after = metrics(base, timeout)

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
        "mean_acceptance_length": 1.0 + accepted / drafts if drafts else float("nan"),
        "preemptions": after["preemptions"] - before["preemptions"],
        "total_prompt_tokens": ptoks,
        "total_completion_tokens": ctoks,
        "ttfts_s": ttfts,
    }


def main():
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
    ap.add_argument("--json", default=None, help="also dump results to this file")
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
        "| acc len | preempt |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    salt = 1000
    for pct in fills:
        for n in concs:
            # 100% fill = largest prompt with room for decode inside --ctx;
            # the 1% shave covers calibration drift. Oversized = HTTP 400.
            depth = int((args.ctx - args.decode) * pct / 100.0 * 0.99)
            row = run_cell(
                args.base_url, model, depth, n, cpt, salt, args.decode, args.timeout
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
                f"| {row['preemptions']:.0f} |"
            )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {"label": label, "ctx": args.ctx, "decode": args.decode, "rows": rows},
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
