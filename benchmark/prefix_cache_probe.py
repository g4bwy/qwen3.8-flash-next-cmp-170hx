#!/usr/bin/env python3
"""Measure how much of a prompt vLLM reuses instead of re-prefilling.

Send a prompt, then send the same prompt with a few characters appended, and
read the server's own counters to see how many prompt tokens came from the
prefix cache. A Mamba-hybrid model in `align` cache mode reuses only at
retained checkpoints, so the appended case is the number that matters for
multi-turn sessions.

Counters used: `vllm:prompt_tokens_total` and `vllm:prompt_tokens_cached_total`.
The probe needs no restart and writes nothing. It does not clear the cache, so
run each prompt shape once, or accept that a first send can already hit.

Usage:
    ./prefix_cache_probe.py                      # against fender.lan:8000
    ./prefix_cache_probe.py --lengths 16000,64000
    ./prefix_cache_probe.py --base-url http://host:8000 --model NAME
"""

import argparse
import json
import re
import time
import urllib.request

PROMPT_TOKENS = "vllm:prompt_tokens_total"
CACHED_TOKENS = "vllm:prompt_tokens_cached_total"

# ~12 tokens per repetition, so the multiplier tracks the token target.
UNIT = " the quick brown fox jumps over a lazy dog while counting pebbles\n"


def get(url, data=None):
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return resp.read().decode()


def counters(base):
    text = get(base + "/metrics")
    out = {}
    for key in (PROMPT_TOKENS, CACHED_TOKENS):
        m = re.search(re.escape(key) + r"\{[^}]*\} ([0-9.e+]+)", text)
        out[key] = float(m.group(1)) if m else 0.0
    return out


def n_tokens(base, model, text):
    body = json.dumps({"model": model, "prompt": text}).encode()
    return len(json.loads(get(base + "/tokenize", body))["tokens"])


def build(base, model, target):
    """Return text that tokenizes to close to `target` tokens."""
    text = UNIT * max(1, target // 12)
    for _ in range(12):
        n = n_tokens(base, model, text)
        if n >= target:
            break
        text += UNIT * max(1, (target - n) // 12)
    return text[: int(len(text) * min(1.0, target / max(n, 1)))]


def request(base, model, prompt):
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    ).encode()
    before = counters(base)
    t0 = time.time()
    reply = json.loads(get(base + "/v1/completions", body))
    wall = time.time() - t0
    after = counters(base)
    dp = after[PROMPT_TOKENS] - before[PROMPT_TOKENS]
    dc = after[CACHED_TOKENS] - before[CACHED_TOKENS]
    return {
        "prompt": reply["usage"]["prompt_tokens"],
        "wall": wall,
        "cached": dc,
        "recomputed": dp - dc,
    }


def show(label, r):
    hit = 100.0 * r["cached"] / max(1.0, r["cached"] + r["recomputed"])
    print(
        f"  {label:26s} prompt={r['prompt']:7d} wall={r['wall']:6.2f}s "
        f"cached={r['cached']:7.0f} recomputed={r['recomputed']:7.0f} hit={hit:5.1f}%"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://fender.lan:8000")
    ap.add_argument("--model", default="qwen3.8-flash-next-fp8")
    ap.add_argument(
        "--lengths",
        default="16000,64000",
        help="comma-separated prompt token targets",
    )
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    health = get(base + "/health") or "ok"
    print(f"server {base} reachable ({len(health)}B health body)")
    for target in (int(x) for x in args.lengths.split(",")):
        text = build(base, args.model, target)
        print(f"prompt target {target} tokens")
        show("first send of this prompt", request(base, args.model, text))
        show("identical prompt again", request(base, args.model, text))
        show("appended two characters", request(base, args.model, text + " hi"))
        print(
            "  read: 'recomputed' on the appended row is what your session pays "
            "per turn"
        )
        print(
            "  note: the cache is not cleared between runs, so a 'first send' can "
            "still hit"
        )


if __name__ == "__main__":
    main()
