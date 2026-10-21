#!/usr/bin/env python3
"""Realistic-traffic acceptance probe.

The bench.py corpora are repeat/copy tasks whose acceptance pins near the
drafter ceiling; live sessions accept far less (mean length ~2.9 on this
deployment vs 4.9 synthetic). This tool measures per-position draft
acceptance on four NON-repetitive content types so depth decisions are
made against data that resembles real requests.

For each corpus type it streams one completion, brackets the decode
phase with spec-decode counter snapshots (global and per-position), and
reports acceptance length and per-position survival. Prompts are
deterministic from a seed; each run appends a random tail line so
prefix cache hits do not skew consecutive trials of the same type.

Usage: ./accept_realistic.py [--base-url URL] [--tokens 512] [--depth 40000]
"""

import argparse
import json
import random
import time
import urllib.request

D = "vllm:spec_decode_num_drafts_total"
A = "vllm:spec_decode_num_accepted_tokens_total"
PP = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def counters(base):
    out = {D: 0.0, A: 0.0}
    with urllib.request.urlopen(base + "/metrics", timeout=15) as r:
        for line in r:
            line = line.decode()
            if line.startswith("#"):
                continue
            if PP in line:
                pos = line.split('position="')[1][0]
                out[f"pos{pos}"] = float(line.split()[-1])
            elif line.split("{")[0].strip() in (D, A):
                out[line.split("{")[0].strip()] = float(line.split()[-1])
    return out


WORDS = [
    "river",
    "model",
    "window",
    "garden",
    "signal",
    "theory",
    "kitchen",
    "fiber",
    "import",
    "clause",
    "volume",
    "harbor",
    "frame",
    "token",
    "sample",
    "letter",
    "branch",
    "custom",
    "router",
    "device",
    "policy",
    "buffer",
    "market",
    "anchor",
    "series",
    "record",
    "table",
    "column",
    "region",
    "cache",
]
IDENTS = ("svc_alpha", "ingest_v2", "quotaCtl", "frameBuf", "llmRouter", "evictQ")


def corpus(kind, target_chars, rng):
    parts = []
    n = 0
    if kind == "prose":
        while n < target_chars:
            s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))
            p = f"{s.capitalize()}. "
            parts.append(p)
            n += len(p)
    elif kind == "code":
        while n < target_chars:
            fn = rng.choice(IDENTS) + str(rng.randint(1, 99))
            block = (
                f"def {fn}(items, limit={rng.randint(1, 64)}):\n"
                f"    acc = {rng.randint(0, 9)}\n"
                "    for i, x in enumerate(items):\n"
                f"        if x % {rng.randint(2, 9)} == {rng.randint(0, 3)}"
                " and i < limit:\n"
                "            acc += hash(str(x)) % 97\n"
                "        else:\n"
                "            acc -= 1\n"
                "    return acc\n\n"
            )
            parts.append(block)
            n += len(block)
    elif kind == "json":
        while n < target_chars:
            rec = {
                rng.choice(IDENTS): rng.randint(0, 10**6),
                "tag": rng.choice(WORDS),
                "ok": rng.random() > 0.5,
                "vals": [rng.randint(-50, 50) for _ in range(rng.randint(2, 6))],
            }
            line = json.dumps(rec, separators=(",", ":")) + "\n"
            parts.append(line)
            n += len(line)
    else:  # mixed: table + prose + a question
        while n < target_chars:
            row = "| " + " | ".join(str(rng.randint(0, 999)) for _ in range(4)) + " |\n"
            s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 9)))
            parts.append(row)
            parts.append(f"{s.capitalize()}.\n")
            n += len(row) + len(s) + 3
    return "".join(parts)[:target_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://fender.lan:8000")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--depth", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=10) as r:
        model = json.load(r)["data"][0]["id"]

    print(f"{'kind':8} {'acc_len':>8}  pos0   pos1   pos2   pos3  gen_tps  secs")
    for kind in ("prose", "code", "json", "mixed"):
        rng = random.Random(f"{args.seed}-{kind}")
        text = corpus(kind, int(args.depth * 3.6), rng)
        prompt = (
            f"Context follows. Answer at the end with a short summary of "
            f"item {rng.randint(1000, 9999)}.\n---BEGIN---\n{text}\n---END---\n"
            "Summary:"
        )
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": args.tokens,
                "temperature": 0,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            args.base_url + "/v1/completions",
            data=body,
            headers={"content-type": "application/json"},
        )
        pre = counters(args.base_url)
        t0 = time.time()
        first = None
        snap = {}
        chars = 0
        with urllib.request.urlopen(req, timeout=1200) as resp:
            for raw in resp:
                if not raw.startswith(b"data:") or b"[DONE]" in raw:
                    if b"[DONE]" in raw:
                        break
                    continue
                if not snap:
                    try:
                        ev = json.loads(raw[5:])
                        txt = ev["choices"][0].get("text") or ""
                    except Exception:
                        continue
                    chars += len(txt)
                    if chars > 400:
                        snap = counters(args.base_url)
                        first = time.time() - t0
        post = counters(args.base_url)
        snap = snap or pre
        d = post[D] - snap[D]
        a = post[A] - snap[A]
        if d <= 0:
            print(f"{kind:8} window empty (graph replay?) - raise --tokens")
            continue
        pos = [(post.get(f"pos{i}", 0) - snap.get(f"pos{i}", 0)) / d for i in range(4)]
        gen = time.time() - t0 - (first or 0)
        ps = "  ".join(f"{p:.3f}" for p in pos)
        print(
            f"{kind:8} {1 + a / d:8.3f}  {ps}  "
            f"{(args.tokens / gen if gen > 0 else 0):7.1f}  {gen:5.1f}"
        )


if __name__ == "__main__":
    main()
