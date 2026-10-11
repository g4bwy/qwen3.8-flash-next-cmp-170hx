#!/usr/bin/env python3
"""Acceptance probe on real maki-corpus text, at real client sampling.

Prompts are concatenations of corpus records (the model's own past
outputs), so draft acceptance is measured on the distribution the draft
vocabulary was built from, not synthetic repetition. Streams launch
staggered to create genuine joining windows: the phase where patch 8's
small head was measured to lose acceptance. Reports mean acceptance
length from server counters plus per-position accepted shares.

usage: corpus_accept_probe.py --depth-tokens 45000 --window 4096
       --conc 4 --stagger 1.5 [--out file.json]
"""
import argparse
import json
import random
import re
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def metrics():
    txt = urllib.request.urlopen(BASE.rstrip("/") + "/metrics", timeout=30).read().decode()
    d = {}
    for line in txt.splitlines():
        m = re.match(r"vllm:spec_decode_num_drafts_total\S* (\S+)", line)
        if m:
            d["drafts"] = float(m.group(1))
        m = re.match(r"vllm:spec_decode_num_accepted_tokens_total\S* (\S+)", line)
        if m:
            d["accepted"] = float(m.group(1))
        m = re.match(
            r'vllm:spec_decode_num_accepted_tokens_per_pos_total{'
            r'.*?position="(\d+)"} (\S+)', line)
        if m:
            d[f"pos{m.group(1)}"] = float(m.group(2))
    return d


def stream(prompt, window, stop_ev, temperature=None):
    payload = {
        "model": "qwen3.8-flash-next-fp8",
        "prompt": prompt,
        "max_tokens": window,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if stop_ev.is_set():
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            if first is None and ch.get("text"):
                first = time.time() - t0
            usage = obj.get("usage")
            if usage:
                ntok = usage.get("completion_tokens", ntok)
    return {"start": t0, "ttft": first, "end": time.time(), "toks": ntok}


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--corpus", default="draft_corpus.jsonl")
    ap.add_argument("--depth-tokens", type=int, default=45000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--stagger", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit to use the server generation-config defaults")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    BASE = a.base_url

    chars = a.depth_tokens * 4
    rng = random.Random(a.seed)
    pool = []
    with open(a.corpus, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("output") or ""
            if len(t) > 500:
                pool.append(t)
    prompts = []
    for _ in range(a.conc):
        parts, total = [], 0
        while total < chars:
            t = rng.choice(pool)
            parts.append(t)
            total += len(t)
        prompts.append("\n\n".join(parts)[:chars])

    before = metrics()
    stop = threading.Event()
    runs = []
    threads = []

    def run(i):
        r = stream(prompts[i], a.window, stop, a.temperature)
        r["i"] = i
        runs.append(r)

    for i in range(a.conc):
        th = threading.Thread(target=run, args=(i,), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(a.stagger)
    for th in threads:
        th.join()
    time.sleep(2)
    after = metrics()

    dd = after["drafts"] - before["drafts"]
    da = after["accepted"] - before["accepted"]
    pos = {p: after.get(f"pos{p}", 0) - before.get(f"pos{p}", 0)
           for p in range(3)}
    span = max(r["end"] for r in runs) - min(r["start"] for r in runs)
    out = {
        "label": a.label, "seed": a.seed, "conc": a.conc,
        "temperature": a.temperature,
        "depth_tokens_est": a.depth_tokens, "window": a.window,
        "stagger": a.stagger, "span_s": round(span, 1),
        "drafts": dd, "mean_acceptance_length": round(1 + da / dd, 3) if dd else None,
        "pos_shares": {p: round(v / da, 3) for p, v in pos.items()} if da else None,
        "per_stream": [
            {"i": r["i"], "ttft": round(r["ttft"], 1) if r["ttft"] else None,
             "toks": r["toks"],
             "rate": round(r["toks"] / (r["end"] - (r["start"] + (r["ttft"] or 0))), 1)
             if r["toks"] else None}
            for r in sorted(runs, key=lambda x: x["i"])],
    }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
