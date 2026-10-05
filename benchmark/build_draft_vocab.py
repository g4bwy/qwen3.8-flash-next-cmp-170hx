#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the MTP draft-vocabulary artifacts for Qwen4Exp.

Creates the two files patch 0008 (patches/0008-mtp-draft-vocab-head.patch)
switches the drafter onto:

  mtp_draft_vocab_ids.pt       int64 tensor of selected target-vocab ids
  mtp_draft_lm_head.safetensors  mtp.draft_lm_head.weight, those rows of the
                               checkpoint's lm_head, plus a new entry in
                               model.safetensors.index.json

The id list MUST be counted from this model's own outputs. A frequency list
from generic web text covers ~92% of generations and silently costs ~10%
throughput, because a token outside the list can never be proposed. Feed
corpus files of assistant text (plain text, or JSONL with the text under
"output", "content", or "text"). Check the printed coverage before booting:
aim for 97%+ on traffic like yours.

Install order: apply patch 0008 first, then run this script, or neither.
Files present without the patch make draft weight loading fail on the extra
tensor. Rollback: delete the two files and the index entry, or rerun with
--out-dir /dev/null-equivalent scratch.

  ./build_draft_vocab.py --corpus my_outputs.jsonl more.txt --size 40000
"""

import argparse
import collections
import json
import os
import sys


def read_corpus_texts(paths):
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield line
                        continue
                    for key in ("output", "text", "content"):
                        val = obj.get(key)
                        if isinstance(val, str) and val:
                            yield val
                            break
        else:
            with open(p, encoding="utf-8") as f:
                yield f.read()


def find_tensor(files, name):
    from safetensors import safe_open

    for fn in files:
        with safe_open(fn, framework="pt") as f:
            if name in f.keys():
                return fn, name
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=40000)
    ap.add_argument("--head-tensor", default=None,
                    help="checkpoint name of the lm_head; default: probe")
    ap.add_argument("--out-dir", default=None, help="default: --model-path")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or args.model_path
    ids_path = os.path.join(out_dir, "mtp_draft_vocab_ids.pt")
    head_path = os.path.join(out_dir, "mtp_draft_lm_head.safetensors")
    for p in (ids_path, head_path):
        if os.path.exists(p) and not args.force:
            sys.exit(f"{p} exists; rerun with --force to overwrite")

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    counts = collections.Counter()
    n_tokens = 0
    for text in read_corpus_texts(args.corpus):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        counts.update(ids)
        n_tokens += len(ids)
    if not counts:
        sys.exit("corpus tokenized to zero tokens")
    if args.size >= len(counts):
        print(f"warning: corpus uses only {len(counts)} distinct ids; "
              f"capping draft vocab at that")
    top = [tid for tid, _ in counts.most_common(args.size)]
    covered = sum(counts[t] for t in top) / n_tokens
    specials = {tok.pad_token_id, tok.unk_token_id}
    top = [t for t in top if t not in specials]
    top_sorted = sorted(top)
    print(f"tokens counted   : {n_tokens:,}")
    print(f"draft vocab size : {len(top_sorted)}")
    print(f"coverage         : {100 * covered:.2f}% of corpus continuations")

    index = os.path.join(args.model_path, "model.safetensors.index.json")
    if not os.path.exists(index):
        sys.exit("expected model.safetensors.index.json beside the checkpoint")
    wmap = json.load(open(index))["weight_map"]
    candidates = [args.head_tensor] if args.head_tensor else [
        "lm_head.weight",
        "model.lm_head.weight",
        "model.language_model.lm_head.weight",
        "mtp.shared_head.head.weight",
    ]
    src_name = None
    for c in candidates:
        if c in wmap:
            src_name = c
            break
    if src_name is None:
        sys.exit(f"no lm_head tensor found; tried {candidates}")
    src_file = os.path.join(args.model_path, wmap[src_name])
    print(f"head source      : {src_name} in {os.path.basename(src_file)}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src_file, framework="pt") as f:
        head = f.get_tensor(src_name)
    rows = head.index_select(0, torch.tensor(top_sorted, dtype=torch.long))
    print(f"head rows        : {tuple(head.shape)} -> {tuple(rows.shape)} "
          f"({rows.dtype})")
    save_file({"mtp.draft_lm_head.weight": rows.contiguous().clone()},
              head_path)
    torch.save(torch.tensor(top_sorted, dtype=torch.long), ids_path)

    new_index = json.load(open(index))
    new_index["weight_map"]["mtp.draft_lm_head.weight"] = os.path.basename(
        head_path)
    total = sum(os.path.getsize(os.path.join(args.model_path, fn))
                for fn in set(new_index["weight_map"].values()))
    new_index["metadata"] = {**new_index.get("metadata", {}),
                             "total_size": total}
    tmp = index + ".tmp"
    json.dump(new_index, open(tmp, "w"), indent=1)
    os.replace(tmp, index)
    print(f"wrote            : {head_path}")
    print(f"wrote            : {ids_path}")
    print(f"updated          : {index}")
    print("boot order: apply patch 0008 FIRST; without it this checkpoint "
          "directory fails draft weight loading.")


if __name__ == "__main__":
    sys.exit(main())
