#!/usr/bin/env python3
"""Decode-window acceptance bracket for certification.

From base 8407bc0bb6-era onward (and since #62129 landed), the spec-decode
counters also include drafts emitted during prefill-phase steps, so
full-window acceptance reads dilute toward the prefill ratio. This tool
streams a fixed prompt, snapshots the counters after a token threshold
(inside the decode phase), snapshots again at stream end, and reports
1 + accepted/drafts for that inner window. Background drafts are checked
before and after (2 s idle probes).

Usage: ./accept_window.py --base-url http://host:8000 [--salt 7770]
                          [--depth 988986] [--tokens 1024] [--trials 3]
"""

import argparse
import importlib.util
import json
import threading
import time
import urllib.request

HERE = __import__("pathlib").Path(__file__).parent
spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

D = "vllm:spec_decode_num_drafts_total"
A = "vllm:spec_decode_num_accepted_tokens_total"


def counters(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r:
            line = line.decode()
            if line.startswith("#"):
                continue
            k = line.split("{")[0].strip()
            if k in (D, A):
                out[k] = float(line.split()[1])
    return out


def idle_drafts(base):
    a = counters(base)
    time.sleep(2)
    b = counters(base)
    return b[D] - a[D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://fender.lan:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--salt", type=int, default=7770)
    ap.add_argument("--depth", type=int, default=int((1000000 - 1024) * 0.99))
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--mark-chars", type=int, default=500)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    model = args.model
    if model is None:
        with urllib.request.urlopen(args.base_url + "/v1/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]

    cpt = bench.calibrate(args.base_url, model, 1, 120)
    prompt = bench.build_prompt(
        args.depth, cpt, args.salt, bench.tail_for("repeat", args.salt), task="repeat"
    )

    rows = []
    for trial in range(1, args.trials + 1):
        bg_pre = idle_drafts(args.base_url)
        t0 = time.time()
        tokcount = 0
        snap = {}
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
        with urllib.request.urlopen(req, timeout=1200) as resp:
            for raw in resp:
                if not raw.startswith(b"data:"):
                    continue
                if b"[DONE]" in raw:
                    break
                if not snap:
                    try:
                        ev = json.loads(raw[5:])
                        tokcount += len(ev["choices"][0].get("text") or "")
                    except Exception:
                        continue
                    if tokcount > args.mark_chars:
                        snap = counters(args.base_url)
        post = counters(args.base_url)
        bg_post = idle_drafts(args.base_url)
        if not snap or post[D] == snap[D]:
            print(f"trial{trial}: sampling raced the stream (snap missing); retry")
            continue
        d = post[D] - snap[D]
        a = post[A] - snap[A]
        acc = 1 + a / d
        rows.append(acc)
        print(
            f"trial{trial}: t+{snap and time.time() - t0:.1f}s window drafts {d:.0f} "
            f"accepted {a:.0f} -> decode-window acceptance {acc:.3f} "
            f"| bg pre/post {bg_pre:.0f}/{bg_post:.0f}"
        )
    if rows:
        print(f"MEAN {sum(rows) / len(rows):.3f}  n={len(rows)}")


if __name__ == "__main__":
    main()
