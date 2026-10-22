#!/usr/bin/env python3
"""Capture a quantization-quality fingerprint from a running server.

Fixed, seeded sample of corpus records. Two probes per run:
  logprobs  prompt_logprobs over one 4k-char window; stores per position the
            rank-1 token id and logprob (k=1, old format) or the whole top-k
            set with ranks (k>1). Runs compare by top-1 flip rate, logprob
            drift, and union KL (quality_diff.py).
  greedy    temperature-0 continuations; compare by exact text agreement.

--depths adds a long-context axis: synthetic prompts built by mtp_ab_probe's
seeded filler at the named token counts, logprobs and a greedy tail each.
That is where a KV format change shows damage the short windows cannot see.
Depth rows make the engine produce top-k at every prompt position, the
heaviest probe this tool runs; on a full pool start at 8200 and step up.

Store captures per boot, always with a matching repeat run, so the diff can
be read against the same-config noise floor.

usage: logprob_capture.py --base-url URL --corpus ../draft_corpus.jsonl
       --out capture.jsonl [--label bf16kv-ref --k 8 --windows 256
       --tail-prompts 64 --tail-tokens 64 --depths 8200,16400,49152
       --depth-tokens 64 --lp-stride 1 --seed 816]
"""
import argparse
import hashlib
import json
import random
import time
import urllib.request


def post(base, path, payload):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def capture_logprobs(base, model, text, k, stride):
    """One teacher-forced pass; returns a compact per-position top-k row."""
    r = post(base, "/v1/completions", {
        "model": model, "prompt": text, "max_tokens": 1,
        "temperature": 0, "stream": False,
        "prompt_logprobs": k,
    })
    pl = r["choices"][0].get("prompt_logprobs") or []
    kept = []
    for i, entry in enumerate(pl):
        if i % stride:
            continue
        if not entry:
            kept.append(None)
            continue
        if k == 1:  # old format: rank-1 pair only
            best = min(entry.items(), key=lambda kv: kv[1]["rank"])
            kept.append([best[0], round(best[1]["logprob"], 4)])
        else:
            kept.append([[tid, round(v["logprob"], 4), v["rank"]]
                         for tid, v in entry.items()])
    return r["usage"]["prompt_tokens"], kept


def capture_greedy(base, model, text, tokens):
    r = post(base, "/v1/completions", {
        "model": model, "prompt": text, "max_tokens": tokens,
        "temperature": 0, "stream": False, "echo": False,
    })
    return r["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="",
                    help="free text describing this boot, stored in the header")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--tail-prompts", type=int, default=64)
    ap.add_argument("--tail-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=1,
                    help="top-k per position; 1 keeps the old compact format")
    ap.add_argument("--depths", default="",
                    help="comma list of synthetic prompt token counts, "
                         "e.g. 8200,16400,49152")
    ap.add_argument("--depth-tokens", type=int, default=64,
                    help="greedy continuation length per depth prompt")
    ap.add_argument("--depth-seed", type=int, default=4242)
    ap.add_argument("--lp-stride", type=int, default=1,
                    help="keep every Nth position for logprobs storage")
    ap.add_argument("--seed", type=int, default=816)
    a = ap.parse_args()

    model = a.model or json.loads(
        urllib.request.urlopen(a.base_url.rstrip("/") + "/v1/models",
                               timeout=30).read())["data"][0]["id"]
    depths = [int(x) for x in a.depths.split(",") if x.strip()]

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
        if len(t) >= 800:
            texts.append(t[:4000])
        if len(texts) >= a.windows + a.tail_prompts:
            break
    print(f"sampled {len(texts)} records", flush=True)

    cptr = None
    if depths:
        from mtp_ab_probe import build_prompt, calibrate, tail_for
        cptr = calibrate(a.base_url.rstrip("/"), model, a.depth_seed, 60)
        print(f"chars/token: {cptr:.2f}", flush=True)

    with open(a.out, "w") as out:
        out.write(json.dumps({
            "kind": "meta", "tool": "logprob_capture/2", "label": a.label,
            "base": a.base_url, "model": model, "k": a.k,
            "windows": a.windows, "tail_prompts": a.tail_prompts,
            "tail_tokens": a.tail_tokens, "depths": depths,
            "depth_tokens": a.depth_tokens, "depth_seed": a.depth_seed,
            "lp_stride": a.lp_stride, "seed": a.seed,
            "chars_per_token": cptr, "ts": time.strftime("%F %T"),
            "corpus": a.corpus,
            "corpus_sha256": hashlib.sha256(
                open(a.corpus, "rb").read()
            ).hexdigest(),
        }) + "\n")

        for wi in range(min(a.windows, len(texts))):
            try:
                ntok, kept = capture_logprobs(
                    a.base_url, model, texts[wi], a.k, a.lp_stride)
            except Exception as exc:  # noqa: BLE001
                out.write(json.dumps({"kind": "logprobs", "win": wi,
                                      "error": str(exc)[:200]}) + "\n")
                continue
            row = {"kind": "logprobs", "win": wi, "n": ntok,
                   "top1" if a.k == 1 else "top": kept}
            out.write(json.dumps(row) + "\n")
            if (wi + 1) % 32 == 0:
                print(f"logprobs {wi + 1}/{a.windows}", flush=True)

        for ti in range(a.tail_prompts):
            idx = a.windows + ti
            if idx >= len(texts):
                break
            try:
                text = capture_greedy(a.base_url, model, texts[idx],
                                      a.tail_tokens)
            except Exception as exc:  # noqa: BLE001
                out.write(json.dumps({"kind": "greedy", "prompt_i": ti,
                                      "error": str(exc)[:200]}) + "\n")
                continue
            out.write(json.dumps({"kind": "greedy", "prompt_i": ti,
                                  "text": text}) + "\n")
            if (ti + 1) % 16 == 0:
                print(f"greedy {ti + 1}/{a.tail_prompts}", flush=True)

        for di, depth in enumerate(depths):
            tag = f"d{depth}#{di}"
            text = build_prompt(depth, cptr, a.depth_seed + di,
                                tail_for("repeat", a.depth_seed + di))
            try:
                ntok, kept = capture_logprobs(
                    a.base_url, model, text, a.k, a.lp_stride)
                text_tail = capture_greedy(a.base_url, model, text,
                                           a.depth_tokens)
            except Exception as exc:  # noqa: BLE001
                out.write(json.dumps({"kind": "logprobs", "win": tag,
                                      "depth": depth,
                                      "error": str(exc)[:200]}) + "\n")
                continue
            out.write(json.dumps({"kind": "logprobs", "win": tag, "n": ntok,
                                  "depth": depth,
                                  "top1" if a.k == 1 else "top": kept}) + "\n")
            out.write(json.dumps({"kind": "greedy", "prompt_i": tag,
                                  "depth": depth, "text": text_tail}) + "\n")
            print(f"depth {depth}: {ntok} prompt tokens", flush=True)

    print("capture written:", a.out)


if __name__ == "__main__":
    main()
