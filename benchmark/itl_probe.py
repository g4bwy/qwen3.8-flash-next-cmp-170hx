#!/usr/bin/env python3
"""Is decode starved because it is slow, or because it is skipped?

Fires N concurrent near-full-context requests, records every streamed token's
arrival time, then classifies each inter-token gap by whether some other request
was still prefilling at that moment. Uniform large gaps mean mixed steps are
genuinely expensive (slow but scheduled). A bimodal distribution, long silence
then a burst, means streams are being held out of steps entirely (deferred).

  ../..../.venv/bin/python itl_probe.py [--base-url http://fender.lan:8000]
                                        [--streams 3] [--depth 995000]
                                        [--decode 2048] [--salt 106000]
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from mtp_ab_probe import (  # noqa: E402
    build_prompt,
    calibrate,
    get_text,
    metrics,
    tail_for,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default=None)
ap.add_argument("--streams", type=int, default=3)
ap.add_argument("--depth", type=int, default=995000)
ap.add_argument("--decode", type=int, default=2048)
ap.add_argument("--salt", type=int, default=106000)
ap.add_argument("--timeout", type=float, default=900.0)
a = ap.parse_args()

base = a.base_url.rstrip("/")
model = (
    a.model
    or json.loads(urllib.request.urlopen(base + "/v1/models", timeout=15).read())[
        "data"
    ][0]["id"]
)
cpt = calibrate(base, model, 1, a.timeout)
prompts = [
    build_prompt(
        a.depth, cpt, a.salt + i, tail_for("repeat", a.salt + i), task="repeat"
    )
    for i in range(a.streams)
]
print(
    f"model={model} chars/token={cpt:.2f} streams={a.streams} "
    f"depth={a.depth} decode={a.decode}",
    flush=True,
)


def prompt_total(url, timeout):
    total = 0.0
    for line in get_text(url.rstrip("/") + "/metrics", timeout).splitlines():
        if line.startswith("vllm:prompt_tokens_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


before = metrics(base, a.timeout)
before_prompt = prompt_total(base, a.timeout)
lock = threading.Lock()
runs = {}
t0 = time.time()


def fire(i, prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": a.decode,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    stamps, start = [], time.time()
    with urllib.request.urlopen(req, timeout=a.timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or ():
                if choice.get("text"):
                    stamps.append(time.time() - start)
    with lock:
        runs[i] = stamps


with ThreadPoolExecutor(max_workers=a.streams) as pool:
    for i, p in enumerate(prompts):
        pool.submit(fire, i, p)
wall = time.time() - t0
after = metrics(base, a.timeout)

firsts = {i: (min(s) if s else None) for i, s in runs.items()}
print(f"\nwall {wall:.1f} s; per-stream tokens and first-token time")
for i in sorted(runs):
    print(f"  stream {i}: {len(runs[i]):>5} tokens, first at {firsts[i]:.1f} s")

starved, content = [], []
for i, stamps in runs.items():
    others = [firsts[j] for j in runs if j != i and firsts[j] is not None]
    if not others:
        continue
    for k in range(1, len(stamps)):
        gap_ms = (stamps[k] - stamps[k - 1]) * 1000
        (starved if min(others) > stamps[k - 1] else content).append(gap_ms)


def report(tag, gaps):
    if not gaps:
        print(f"{tag}: no samples")
        return
    gaps = sorted(gaps)
    q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
    med = statistics.median(gaps)
    # bimodality: fraction of gaps under 0.5x median vs over 2x median
    lo = sum(1 for g in gaps if g < 0.5 * med) / len(gaps)
    hi = sum(1 for g in gaps if g > 2 * med) / len(gaps)
    print(
        f"{tag:<26} n={len(gaps):>6} mean {statistics.mean(gaps):7.1f} ms  "
        f"p50 {med:7.1f}  p95 {q(0.95):7.1f}  max {gaps[-1]:8.1f}  "
        f"<0.5xp50 {lo:5.1%}  >2xp50 {hi:5.1%}"
    )


print("\ninter-token gap, milliseconds")
report("while another prefill ran", starved)
report("all prefills done", content)
dg = after.get("gen_tokens", 0) - before.get("gen_tokens", 0)
dp = prompt_total(base, a.timeout) - before_prompt
print(
    f"\nengine deltas over the cell: generation {dg:.0f} tokens (expected "
    f"{a.streams * a.decode}), prompt {dp / 1e6:.2f} M (expected "
    f"{a.streams * a.depth / 1e6:.2f} M)"
)
dpre = after.get("preemptions", 0) - before.get("preemptions", 0)
print(f"preemptions delta {dpre:.0f}")
