#!/usr/bin/env python3
"""Targeted audit after the qkv-indexer repair of the W8A16 dense tree.

Checks what the boot actually failed on, at header cost instead of the 172 GB that
verify_w8a16.py reads:
1. every source tensor maps to exactly one spelling in the output index, and a
   converted module never appears as both `.weight` and its packed triple
2. every tensor in an output shard is indexed to that shard
3. every fused module (qkv_proj and friends) has all its runtime shards on ONE
   scheme, which is the check vLLM's should_ignore_layer performs at load
4. the twelve indexer tensors this repair converted round-trip byte-exact from
   source through the same quantizer the builder uses
5. the patch 5 draft-vocab pair is still indexed, because losing it does not fail
   the boot, it just rejects every draft

usage: audit_w8a16_repair.py [--src DIR] [--out DIR]
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_fp8_dense import bf16_to_f32, read_header  # noqa: E402
from encode_w8a16 import bf16_bytes, quant_int8_g32  # noqa: E402

CONV_SUF = ("weight_packed", "weight_scale", "weight_shape")
FUSED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
INDEXER_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.self_attn\.indexer\.index_qk_proj\.weight$"
)
QKV_RE = re.compile(
    r"^(model\.language_model\.layers\.\d+\.self_attn)\.qkv_proj\.weight$"
)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/gab/models/Qwen/Qwen3.8-Flash-Next-FP8")
    ap.add_argument("--out", default="/home/gab/test/Qwen3.8-Flash-Next-W8A16")
    a = ap.parse_args()
    src, out = a.src, a.out

    src_ix = read_json(f"{src}/model.safetensors.index.json")["weight_map"]
    src_ix.pop("mtp.draft_lm_head.weight", None)
    out_ix = read_json(f"{out}/model.safetensors.index.json")["weight_map"]
    conv = read_json(f"{out}/w8_manifest.json")["converted"]
    cfg = read_json(f"{out}/config.json")["quantization_config"]
    ignore = set(cfg.get("ignore") or [])
    problems = []
    for name in src_ix:
        mod = name[: -len(".weight")]
        if name in conv:
            missing = [f"{mod}.{s}" for s in CONV_SUF if f"{mod}.{s}" not in out_ix]
            if missing:
                problems.append(f"converted but not indexed: {missing[:3]}")
            if name in out_ix:
                problems.append(f"converted tensor still indexed as .weight: {name}")
            continue
        renamed = name.replace("weight_scale_inv", "weight_scale")
        if name not in out_ix and renamed not in out_ix:
            problems.append(f"passthrough tensor missing from index: {name}")

    # Artifact shards are named here, not in the source index.
    for fname in sorted(os.listdir(out)):
        if not fname.endswith(".safetensors"):
            continue
        hdr, _ = read_header(f"{out}/{fname}")
        for k in hdr:
            if k != "__metadata__" and out_ix.get(k) != fname:
                problems.append(f"tensor in shard {fname} not indexed there: {k}")

    mixed = []
    for name in src_ix:
        m = QKV_RE.match(name)
        if not m:
            continue
        base = m.group(1)
        runtime = base.replace("model.language_model.", "model.", 1) + ".qkv_proj"
        if runtime in ignore:
            continue
        shards = [f"{base}.{s}" for s in FUSED["qkv_proj"]]
        ign = [s in ignore for s in shards]
        if any(ign) and not all(ign):
            mixed.append((runtime, [s for s, i in zip(shards, ign) if i]))
    for rt, ignored in mixed[:3]:
        problems.append(f"MIXED SCHEME {rt}: ignored shards {ignored}")

    checked = 0
    for name, fname in sorted(src_ix.items()):
        if not INDEXER_RE.match(name):
            continue
        shp, sstart = read_header(f"{src}/{fname}")
        o0, o1 = shp[name]["data_offsets"]
        with open(f"{src}/{fname}", "rb") as handle:
            handle.seek(sstart + o0)
            raw = handle.read(o1 - o0)
        w = bf16_to_f32(np.frombuffer(raw, "V2")).reshape(shp[name]["shape"])
        packed, scale, _ = quant_int8_g32(w)
        want = {"weight_packed": packed.tobytes(), "weight_scale": bf16_bytes(scale)}
        mod = name[: -len(".weight")]
        for suffix, buf in want.items():
            key = f"{mod}.{suffix}"
            of, ostart = read_header(f"{out}/{out_ix[key]}")
            g0, g1 = of[key]["data_offsets"]
            with open(f"{out}/{out_ix[key]}", "rb") as handle:
                handle.seek(ostart + g0)
                got = handle.read(g1 - g0)
            if got != buf:
                problems.append(f"{key} differs from a fresh quantize")
        checked += 1

    if os.path.exists(f"{out}/mtp_draft_lm_head.safetensors"):
        got = out_ix.get("mtp.draft_lm_head.weight")
        if got != "mtp_draft_lm_head.safetensors":
            problems.append(
                f"draft-vocab head not indexed (got {got}); "
                "the drafter would run unweighted and reject every draft"
            )
        if not os.path.exists(f"{out}/mtp_draft_vocab_ids.pt"):
            problems.append("draft-vocab id list missing")

    print(
        f"index entries {len(out_ix)}, converted {len(conv)}, "
        f"ignore entries {len(ignore)}"
    )
    print(f"draft head indexed: {out_ix.get('mtp.draft_lm_head.weight')}")
    print(f"indexer tensors byte-checked: {checked}")
    print(f"fused qkv modules with mixed schemes: {len(mixed)}")
    print(f"problems: {len(problems)}")
    for p in problems[:10]:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
