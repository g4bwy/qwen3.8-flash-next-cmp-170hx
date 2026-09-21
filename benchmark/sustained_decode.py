#!/usr/bin/env python3
"""Sustained concurrent-decode capacity, isolated from prefill effects.

Short prompts, long decode, N streams fired together, and
vllm:generation_tokens_total sampled at 0.5 s. The plateau is the true
batched-decode rate at batch N (aggregate and per stream); bench.py's
window-average numbers include the chunked-prefill starvation phase and
read far lower with big prompts.

    ./benchmark/sustained_decode.py --base-url http://<host>:8000
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from bench import stream_run  # noqa: E402
from mtp_ab_probe import build_prompt, calibrate, get_text, tail_for  # noqa: E402


def gen_total(base, timeout):
    for line in get_text(base.rstrip("/") + "/metrics", timeout).splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            return float(line.rsplit(" ", 1)[1])
    raise RuntimeError("no vllm:generation_tokens_total in /metrics")


def run(base, model, n, depth, decode, timeout, cpt):
    prompts = [
        build_prompt(depth, cpt, 900 + n * 50 + i, tail_for("repeat", 1), task="repeat")
        for i in range(n)
    ]
    t0 = time.time()
    stop = threading.Event()
    samples = []

    def sampler():
        prev_t, prev_v = time.time(), gen_total(base, timeout)
        while not stop.wait(0.5):
            t, v = time.time(), gen_total(base, timeout)
            samples.append((t - t0, (v - prev_v) / (t - prev_t)))
            prev_t, prev_v = t, v

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(
            pool.map(lambda p: stream_run(base, model, p, decode, timeout), prompts)
        )
    stop.set()
    th.join(3)
    firsts = sorted(r["first"] - t0 for r in runs)
    rates = sorted((v for _, v in samples), reverse=True)
    plateau = sum(rates[:4]) / min(4, len(rates)) if rates else float("nan")
    print(
        f"| {n} | {max(r['end'] for r in runs) - t0:.1f} "
        f"| {', '.join(f'{x:.1f}' for x in firsts)} "
        f"| {plateau:,.0f} | {plateau / n:,.0f} |"
    )
    return {"n": n, "plateau_agg_tps": plateau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrencies", default="1,2,4")
    ap.add_argument("--depth", type=int, default=8000, help="prompt tokens")
    ap.add_argument("--decode", type=int, default=1024, help="tokens per stream")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    models_url = args.base_url.rstrip("/") + "/v1/models"
    listed = json.loads(urllib.request.urlopen(models_url, timeout=30).read())
    model = args.model or listed["data"][0]["id"]
    cpt = calibrate(args.base_url, model, 7, args.timeout)
    print(f"\n### sustained decode: depth {args.depth}, decode {args.decode}\n")
    print("| streams | wall s | first tokens s | agg tok/s | per-stream tok/s |")
    print("|---:|---:|---|---:|---:|")
    out = []
    for n in (int(c) for c in args.concurrencies.split(",")):
        out.append(
            run(args.base_url, model, n, args.depth, args.decode, args.timeout, cpt)
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {"depth": args.depth, "decode": args.decode, "runs": out}, fh, indent=2
            )


if __name__ == "__main__":
    main()
