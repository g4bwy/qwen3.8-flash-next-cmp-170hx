#!/usr/bin/env python3
"""Diff two logprob_capture.py fingerprints.

usage: quality_diff.py A.jsonl B.jsonl [--label-a NAME --label-b NAME]
       [--max-mean-kl X --max-p99-kl X --min-top1-pct X]
Reports: top-1 flip rate, logprob delta distribution, union KL over the
captured top-k sets (k>1 captures only), greedy tail exact-match and
common-prefix statistics, per-depth breakdown for long-context rows.

Two uses. Same boot captured twice measures the noise floor; a candidate
boot versus a stored reference measures the format or kernel change. When
threshold flags are given the exit code is 1 on any violation, so the diff
can gate a certification run.

KL note: vLLM returns only the top-k logprobs per position. Both sides are
renormalized over the union of their sets, and a token missing from one side
gets that side's worst captured logprob minus one nat. The proxy is fixed,
so read the number against the noise-floor diff, not as an absolute KL.
"""
import argparse
import gzip
import json
import math
import os


def load(path):
    meta = {}
    lp, gr = {}, {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "meta":
                meta = r
            elif kind == "logprobs" and "error" not in r:
                lp[r["win"]] = r
            elif kind == "greedy" and "error" not in r:
                gr[r["prompt_i"]] = r["text"]
    return meta, lp, gr


def rank1(entries):
    """entries: list of [tok, logprob, rank] -> rank-1 pair or None."""
    if not entries:
        return None
    return tuple(min(entries, key=lambda e: e[2])[:2])


def top1_of(row, win):
    """rank-1 (tok, logprob) pairs per position from either row format."""
    seq = row.get("top1") or row.get("top") or []
    out = []
    for e in seq:
        if e is None:
            out.append(None)
        elif row.get("top1") is not None:
            out.append(tuple(e))
        else:
            out.append(rank1(e))
    return out


def union_kl(ea, eb, missing_nat=1.0):
    """KL(P_a || P_b) over the union of two top-k sets, both renormalized."""
    pa = {t: lp for t, lp, _ in ea}
    pb = {t: lp for t, lp, _ in eb}
    if not pa or not pb:
        return None
    union = set(pa) | set(pb)
    fa = min(pa.values()) - missing_nat
    fb = min(pb.values()) - missing_nat
    va = [math.exp(pa.get(t, fa)) for t in union]
    vb = [math.exp(pb.get(t, fb)) for t in union]
    sa, sb = sum(va), sum(vb)
    if sa <= 0 or sb <= 0:
        return None
    kl = 0.0
    for x, y in zip(va, vb):
        p, q = x / sa, y / sb
        if p > 0.0:
            kl += p * math.log(p / q)
    return kl


def dist(vals):
    if not vals:
        return "n/a"
    vals = sorted(vals)
    q = lambda p: vals[int(p * (len(vals) - 1))]  # noqa: E731
    return (f"mean {sum(vals) / len(vals):.5f} p50 {q(.5):.5f} "
            f"p95 {q(.95):.5f} p99 {q(.99):.5f} max {vals[-1]:.5f}")


def report(lp_a, lp_b, k_mode):
    """Aggregate per-position comparisons over the shared windows."""
    n = flips = miss = 0
    deltas, kls = [], []
    for w in sorted(set(lp_a) & set(lp_b), key=str):
        ra, rb = lp_a[w], lp_b[w]
        if k_mode:
            seqa, seqb = ra.get("top") or [], rb.get("top") or []
            for ea, eb in zip(seqa, seqb):
                if not ea or not eb:
                    continue
                n += 1
                t1a, t1b = rank1(ea), rank1(eb)
                flips += t1a[0] != t1b[0]
                deltas.append(abs(t1a[1] - t1b[1]))
                b1 = {t for t, _, _ in eb}
                miss += t1a[0] not in b1
                kl = union_kl(ea, eb)
                if kl is not None:
                    kls.append(kl)
        else:
            for ta, tb in zip(top1_of(ra, w), top1_of(rb, w)):
                if not ta or not tb:
                    continue
                n += 1
                flips += ta[0] != tb[0]
                deltas.append(abs(ta[1] - tb[1]))
    return n, flips, deltas, kls, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--max-mean-kl", type=float, default=None)
    ap.add_argument("--max-p99-kl", type=float, default=None)
    ap.add_argument("--min-top1-pct", type=float, default=None)
    ap.add_argument("--require-greedy-equal", action="store_true",
                    help="fail unless every greedy tail matches exactly "
                         "(determinism runs between identical configs)")
    a = ap.parse_args()

    meta_a, lp_a, gr_a = load(a.a)
    meta_b, lp_b, gr_b = load(a.b)
    k_mode = bool(lp_a and lp_b
                  and "top" in next(iter(lp_a.values()))
                  and "top" in next(iter(lp_b.values())))
    for tag, m in (("A", meta_a), ("B", meta_b)):
        lab = m.get("label") or a.__dict__.get(f"label_{tag.lower()}")
        print(f"{tag}: k={m.get('k', 1)} depths={m.get('depths', [])} "
              f"label={lab!r} ts={m.get('ts', '?')} model={m.get('model', '?')}")
    if meta_a.get("seed") != meta_b.get("seed") or \
            meta_a.get("depths") != meta_b.get("depths") or \
            meta_a.get("k") != meta_b.get("k") or \
            meta_a.get("lp_stride") != meta_b.get("lp_stride"):
        print("WARNING: corpus/depth settings differ between captures, "
              "only shared windows are compared")
    cs_a, cs_b = meta_a.get("corpus_sha256"), meta_b.get("corpus_sha256")
    if cs_a and cs_b and cs_a != cs_b:
        print("INCOMPARABLE: the two captures used different corpus files, "
              "so their windows hold different text and the flip rate "
              "measures content mismatch, not value drift. Re-capture the "
              "reference against the current corpus before reading this "
              "number.")

    n, flips, deltas, kls, miss = report(lp_a, lp_b, k_mode)
    print(f"positions: {n:,}  top-1 flips: {flips:,} "
          f"({100 * flips / max(1, n):.3f} %)")
    print(f"top-1 logprob |delta|: {dist(deltas)}")
    if k_mode:
        p99 = sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else None
        print(f"union KL(P_a||P_b): {dist(kls)}")
        if p99 is not None:
            print(f"positions where A's top-1 is absent from B: {miss:,}")
    else:
        print("(k=1 captures: no KL, rerun capture with --k 8)")

    keys = sorted(set(gr_a) & set(gr_b), key=str)
    same = sum(gr_a[k] == gr_b[k] for k in keys)
    cp = [len(os.path.commonprefix([gr_a[k], gr_b[k]])) for k in keys]
    ln = [max(len(gr_a[k]), len(gr_b[k])) for k in keys]
    diverged = sum(1 for c, x in zip(cp, ln) if c < x)
    print(f"greedy tails: {len(keys)}, exact-equal: {same} "
          f"({100 * same / max(1, len(keys)):.1f} %), diverged: {diverged}")
    if cp:
        print(f"common prefix chars: mean {sum(cp) / len(cp):.0f} of mean len "
              f"{sum(ln) / len(ln):.0f}")

    if k_mode and any("depth" in lp_a[w] for w in lp_a):
        print("per-depth KL:")
        by_depth = {}
        for w, ra in lp_a.items():
            d = ra.get("depth")
            if d and w in lp_b:
                seqb = lp_b[w].get("top") or []
                vals = [union_kl(ea, eb) for ea, eb in zip(ra["top"], seqb)
                        if ea and eb]
                by_depth.setdefault(d, []).extend(
                    v for v in vals if v is not None)
        for d in sorted(by_depth):
            print(f"  {d:>7} tok: {dist(by_depth[d])}")

    fails = []
    top1_pct = 100.0 * (n - flips) / max(1, n)
    if a.min_top1_pct is not None and top1_pct < a.min_top1_pct:
        fails.append(f"top-1 match {top1_pct:.3f} % < {a.min_top1_pct}")
    mean_kl = sum(kls) / len(kls) if kls else None
    if (a.max_mean_kl is not None and k_mode
            and (mean_kl is None or mean_kl > a.max_mean_kl)):
        fails.append(f"mean KL {mean_kl} > {a.max_mean_kl}")
    if a.max_p99_kl is not None and k_mode:
        p99 = sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else None
        if p99 is None or p99 > a.max_p99_kl:
            fails.append(f"p99 KL {p99} > {a.max_p99_kl}")
    if a.require_greedy_equal and same != len(keys):
        fails.append(f"greedy tails differ: {len(keys) - same}/{len(keys)}")
    if fails:
        print("VERDICT: FAIL")
        for f in fails:
            print(f"  - {f}")
        raise SystemExit(1)
    print("VERDICT: PASS" if (a.max_mean_kl or a.min_top1_pct
                              or a.require_greedy_equal) else "VERDICT: ok")


if __name__ == "__main__":
    main()
