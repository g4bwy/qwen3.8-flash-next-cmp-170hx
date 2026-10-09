#!/usr/bin/env python3
"""Capture a quantization-quality fingerprint from a running server.

Fixed, seeded sample of corpus records. Two probes per run:
  logprobs  prompt_logprobs=1 over one 4k-char window; stores per position
            the rank-1 token id and logprob. Two runs compare by top-1
            flip rate and logprob drift.
  greedy    temperature-0 continuations; compare by exact text agreement.

usage: logprob_capture.py --base-url URL --corpus draft_corpus.jsonl
       --out capture.jsonl [--windows 256 --tail-prompts 64 --tail-tokens 64]
"""
import argparse
import json
import random
import urllib.request


def post(base, path, payload):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--tail-prompts", type=int, default=64)
    ap.add_argument("--tail-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=816)
    a = ap.parse_args()

    model = a.model or json.loads(
        urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models",
                               timeout=30).read())["data"][0]["id"]

    rng = random.Random(a.seed)
    with open(a.corpus, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    texts = []
    for i in rng.sample(range(len(lines)), (a.windows + a.tail_prompts) * 4):
        try:
            rec = json.loads(lines[i])
        except json.JSONDecodeError:
            continue
        t = rec.get("output") or rec.get("text") or ""
        if 800 <= len(t):
            texts.append(t[:4000])
        if len(texts) >= a.windows + a.tail_prompts:
            break
    print(f"sampled {len(texts)} records", flush=True)

    with open(a.out, "w") as out:
        for wi in range(a.windows):
            try:
                r = post(a.base_url, "/v1/completions", {
                    "model": model, "prompt": texts[wi], "max_tokens": 1,
                    "temperature": 0, "stream": False,
                    "prompt_logprobs": 1,
                })
            except Exception as exc:  # noqa: BLE001
                out.write(json.dumps({"kind": "logprobs", "win": wi,
                                      "error": str(exc)[:200]}) + "\n")
                continue
            pl = r["choices"][0].get("prompt_logprobs") or []
            row = {"kind": "logprobs", "win": wi, "n": len(pl), "top1": []}
            for entry in pl:
                if not entry:
                    row["top1"].append(None)
                    continue
                best = min(entry.items(), key=lambda kv: kv[1]["rank"])
                row["top1"].append([best[0], round(best[1]["logprob"], 4)])
            out.write(json.dumps(row) + "\n")
            if (wi + 1) % 32 == 0:
                print(f"logprobs {wi + 1}/{a.windows}", flush=True)

        for ti in range(a.tail_prompts):
            idx = a.windows + ti
            if idx >= len(texts):
                break
            try:
                r = post(a.base_url, "/v1/completions", {
                    "model": model, "prompt": texts[idx],
                    "max_tokens": a.tail_tokens, "temperature": 0,
                    "stream": False, "echo": False,
                })
            except Exception as exc:  # noqa: BLE001
                out.write(json.dumps({"kind": "greedy", "prompt_i": ti,
                                      "error": str(exc)[:200]}) + "\n")
                continue
            ch = r["choices"][0]
            out.write(json.dumps({"kind": "greedy", "prompt_i": ti,
                                  "text": ch["text"]}) + "\n")
            if (ti + 1) % 16 == 0:
                print(f"greedy {ti + 1}/{a.tail_prompts}", flush=True)
    print("capture written:", a.out)


if __name__ == "__main__":
    main()
