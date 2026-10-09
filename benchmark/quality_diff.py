#!/usr/bin/env python3
"""Diff two logprob_capture.py fingerprints.

usage: quality_diff.py A.jsonl B.jsonl [--label-a NAME --label-b NAME]
Reports: top-1 flip rate, logprob delta distribution, greedy tail
exact-match and common-prefix statistics.
"""
import argparse
import json


def load(path):
    lp, gr = {}, {}
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        if r["kind"] == "logprobs":
            lp[r["win"]] = r["top1"]
        else:
            gr[r["prompt_i"]] = r["text"]
    return lp, gr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    lp_a, gr_a = load(a.a)
    lp_b, gr_b = load(a.b)
    wins = sorted(set(lp_a) & set(lp_b))
    n = flips = 0
    deltas = []
    big = 0
    for w in wins:
        for ta, tb in zip(lp_a[w], lp_b[w]):
            if not ta or not tb:
                continue
            n += 1
            flips += ta[0] != tb[0]
            d = abs(ta[1] - tb[1])
            deltas.append(d)
            big += d > 0.5
    deltas.sort()
    q = lambda p: deltas[int(p * (len(deltas) - 1))] if deltas else 0
    print(f"windows: {len(wins)}  positions: {n:,}")
    print(f"top-1 flips: {flips:,} ({100 * flips / max(1, n):.3f} %)")
    print(f"top-1 logprob |delta|: mean {sum(deltas) / max(1, len(deltas)):.4f} "
          f"p50 {q(.5):.4f} p95 {q(.95):.4f} p99 {q(.99):.4f} max {deltas[-1] if deltas else 0:.4f}")
    print(f"positions moved > 0.5 nats: {big:,} ({100 * big / max(1, n):.3f} %)")

    keys = sorted(set(gr_a) & set(gr_b))
    same = sum(gr_a[k] == gr_b[k] for k in keys)
    import os.path
    cp = [len(os.path.commonprefix([gr_a[k], gr_b[k]])) for k in keys]
    ln = [max(len(gr_a[k]), len(gr_b[k])) for k in keys]
    first_diff = sum(1 for c, l in zip(cp, ln) if c < l)
    print(f"greedy tails: {len(keys)}, exact-equal: {same} "
          f"({100 * same / max(1, len(keys)):.1f} %), diverged: {first_diff}")
    if cp:
        print(f"common prefix chars: mean {sum(cp) / len(cp):.0f} of mean len "
              f"{sum(ln) / len(ln):.0f}")


if __name__ == "__main__":
    main()
