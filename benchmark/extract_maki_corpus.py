#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract assistant-side text from maki session logs into a draft-vocab corpus.

maki (https://github.com/tontinton/maki) is a terminal coding agent; when it
runs on the model being served, its stored sessions are that model's own
generated text, which is exactly what a draft vocabulary must be counted over.
Users of other agent harnesses: port this script to wherever yours keeps
transcripts, or feed build_draft_vocab.py any JSONL with the assistant text
under "output".

Feed the result to build_draft_vocab.py. The corpus is this model's own
generated text (content, thinking, and serialized tool inputs), filtered to
sessions whose header model matches --model-match, so the draft vocabulary is
counted over what the checkpoint actually emits on your real workload.

  ./extract_maki_corpus.py --out draft_corpus.jsonl
  ./build_draft_vocab.py --model-path <ckpt> --corpus draft_corpus.jsonl

Everything runs locally; the JSONL is what you copy to the serve box.
"""

import argparse
import glob
import json
import os
import sys


def iter_parts(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        try:
            model = json.loads(header).get("model", "")
        except json.JSONDecodeError:
            return None
        if not model:
            return None
        yield_from = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("t") != "msg":
                continue
            msg = obj.get("d") or {}
            if msg.get("role") != "assistant":
                continue
            for part in msg.get("content") or []:
                ptype = part.get("type")
                if ptype == "text":
                    yield_from.append(part.get("text") or "")
                elif ptype == "thinking":
                    yield_from.append(part.get("thinking") or "")
                elif ptype == "tool_use":
                    yield_from.append(json.dumps(part.get("input"),
                                                 ensure_ascii=False))
        return (model, yield_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir",
                    default=os.path.expanduser("~/.local/state/maki/sessions"))
    ap.add_argument("--model-match", default="qwen3.8-flash-next")
    ap.add_argument("--out", default="draft_corpus.jsonl")
    ap.add_argument("--max-mb", type=float, default=60.0,
                    help="cap assistant text written (~4 MB per 1M tokens)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sessions_dir, "*.jsonl")),
                   key=os.path.getmtime, reverse=True)
    written = 0
    chars = 0
    used_files = 0
    budget = args.max_mb * 1_000_000
    with open(args.out, "w", encoding="utf-8") as out:
        for path in files:
            if chars >= budget:
                break
            res = iter_parts(path)
            if not res:
                continue
            model, parts = res
            if args.model_match not in model:
                continue
            used_files += 1
            for text in parts:
                if not text:
                    continue
                out.write(json.dumps({"output": text},
                                     ensure_ascii=False) + "\n")
                written += 1
                chars += len(text)
                if chars >= budget:
                    break
    print(f"sessions used : {used_files}")
    print(f"parts written : {written}")
    print(f"chars         : {chars:,} (~{chars / 4e6:.1f}M tokens)")
    print(f"wrote         : {args.out}")


if __name__ == "__main__":
    sys.exit(main())
