#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed-workload decode rate probe. Successor to the /tmp/ab3.py harness.

Prompts are built with mtp_ab_probe's corpus so workloads are reproducible
from the repo alone. Reports wall seconds for ignore_eos generations at a
pinned length, single stream then N-stream, best-of-reps aggregation.
Run on an idle server; idle 3-5 minutes after the N-stream cells before
trusting any single-stream number from the same boot.

  ./decode_probe.py --depth 16384 --reps 3
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mtp_ab_probe import build_prompt, tail_for  # noqa: E402


def post(base, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        json.load(r)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next-fp8")
    ap.add_argument("--depth", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--concurrencies", default="1,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    prompt = build_prompt(
        args.depth,
        1.4,
        args.seed,
        tail_for("copy", args.seed),
        task="copy",
    )

    for conc_s in args.concurrencies.split(","):
        conc = int(conc_s)
        walls = []
        for _ in range(args.reps):
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [
                    ex.submit(post, base, args.model, prompt, args.gen, args.timeout)
                    for _ in range(conc)
                ]
                walls.append(max(f.result() for f in futs))
        agg = [conc * args.gen / w for w in walls]
        print(
            f"{conc} stream{'s' if conc > 1 else ' '}: gen={conc * args.gen} "
            f"median wall={statistics.median(walls):5.2f}s "
            f"best agg={max(agg):6.1f} tok/s  per-stream="
            f"{max(agg) / conc:5.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
