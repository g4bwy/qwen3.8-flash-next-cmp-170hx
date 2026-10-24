#!/usr/bin/env python3
"""W8A16-dense checkpoint encoder (v2, int8).

Source: official Qwen3.8-Flash-Next-FP8. Converts the 228 main-body dense
bf16 modules to int8 symmetric group-32, pack-quantized uint8b128 layout
(weight_packed int32 [N, K/4], weight_scale bf16 [N, K/32], weight_shape
int64 [2] = (K, N)). Experts, PLE and held modules pass through byte-exact.
Output config: compressed-tensors mixed groups (see ct_config.py).

usage: encode_w8a16.py SRC OUT [--files a,b,c] [--assemble-only]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_config import CONVERT_RE, build_ct_config  # noqa: E402
from encode_fp8_dense import bf16_to_f32, read_header  # noqa: E402


def out_name(name: str) -> str:
    """compressed-tensors reads block fp8 scales as weight_scale, the
    official checkpoint spells it weight_scale_inv: rename at emit time,
    bytes are untouched."""
    if name.endswith(".weight_scale_inv"):
        return name[: -len("_inv")]
    return name

G = 32
SKIP = {"model.safetensors.index.json.orig", "mtp_draft_lm_head.safetensors",
        "mtp_draft_vocab_ids.pt"}


def quant_int8_g32(w: np.ndarray):
    o, i = w.shape
    assert i % G == 0 and i % 4 == 0
    blocks = w.reshape(o, i // G, G)
    amax = np.abs(blocks).max(axis=2)
    s = np.where(amax == 0, 1.0, amax / 127.0)
    q = np.clip(np.round(blocks / s[:, :, None]), -127, 127)
    scale = s.astype(np.float32)
    scale = (view_bf16_round(scale))
    deq = q * scale[:, :, None]
    err = float(((w - deq.reshape(o, i)) ** 2).mean() ** 0.5
                / ((w ** 2).mean() ** 0.5 or 1e-12))
    u8 = np.ascontiguousarray((q + 128).astype(np.uint8).reshape(o, i // 4, 4))
    packed = u8.view(np.int32).reshape(o, i // 4)
    return packed, scale, err


def view_bf16_round(x: np.ndarray) -> np.ndarray:
    from encode_fp8_dense import f32_to_bf16_bytes, bf16_to_f32 as B
    return B(f32_to_bf16_bytes(x.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--files", default="")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--merge-prior", action="store_true",
                    help="with --assemble-only after a partial --files repair, keep "
                         "the conversions recorded by the previous full build")
    a = ap.parse_args()

    sid = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"]
    sid.pop("mtp.draft_lm_head.weight", None)
    by_file = {}
    for k, f in sid.items():
        by_file.setdefault(f, []).append(k)
    os.makedirs(a.out, exist_ok=True)

    if a.assemble_only:
        assemble(a, sid, merge_prior=a.merge_prior)
        return

    man = {"converted": {}, "passthrough": 0}
    files = a.files.split(",") if a.files else sorted(by_file)
    for fname in files:
        names = by_file.get(fname, [])
        if not names:
            print("skip", fname)
            continue
        hdr, dstart = read_header(f"{a.src}/{fname}")
        with open(f"{a.src}/{fname}", "rb") as fh:
            blob = fh.read()
        entries, blobs, size = [], [], 0

        def add(name, dtype, shape, buf):
            entries.append((name, {"dtype": dtype, "shape": list(shape),
                                   "nbytes": len(buf)}))
            blobs.append(buf)

        for name in names:
            m = hdr[name]
            s0, s1 = m["data_offsets"]
            raw = blob[dstart + s0: dstart + s1]
            if CONVERT_RE.match(name):
                w = bf16_to_f32(np.frombuffer(raw, "V2")).reshape(m["shape"])
                packed, scale, err = quant_int8_g32(w)
                mod = name[: -len(".weight")]
                add(name.replace("weight", "weight_packed"), "I32",
                    packed.shape, packed.tobytes())
                add(mod + ".weight_scale", "BF16", scale.shape,
                    bf16_bytes(scale))
                add(mod + ".weight_shape", "I64", (2,),
                    np.array([w.shape[1], w.shape[0]], dtype=np.int64).tobytes())
                man["converted"][name] = {
                    "shape": m["shape"], "rel_rms": round(err, 5),
                    "src_sha16": hashlib.sha256(raw).hexdigest()[:16]}
            else:
                add(out_name(name), m["dtype"], m["shape"], raw)
                man["passthrough"] += 1
        with open(f"{a.out}/{fname}", "wb") as ofh:
            hdr_json, off = {}, 0
            for n2, meta in entries:
                hdr_json[n2] = {"dtype": meta["dtype"], "shape": meta["shape"],
                                "data_offsets": [off, off + meta["nbytes"]]}
                off += meta["nbytes"]
            hb = json.dumps(hdr_json, separators=(",", ":")).encode()
            pad = (-len(hb)) % 8
            ofh.write(np.uint64(len(hb) + pad).tobytes() + hb + b" " * pad)
            for b_ in blobs:
                ofh.write(b_)
        print("done", fname, flush=True)
    json.dump(man, open(f"{a.out}/manifest.{files[0]}.json", "w"))
    print(json.dumps({"converted": len(man["converted"]),
                      "passthrough": man["passthrough"]}))


def bf16_bytes(x: np.ndarray) -> bytes:
    from encode_fp8_dense import f32_to_bf16_bytes
    return f32_to_bf16_bytes(x.astype(np.float32)).tobytes()


def assemble(a, sid, merge_prior=False):
    """Write config.json, the shard index and w8_manifest.json from the run manifests.

    Per-run manifests are consumed and deleted here, so a repair pass over a few
    shards would otherwise forget every module a previous pass converted: the
    index would re-declare 228 packed tensors as plain `.weight` and the ignore
    list would grow back. `merge_prior` seeds from the last w8_manifest.json.
    """
    conv = {}
    pt = 0
    if merge_prior:
        prior_path = f"{a.out}/w8_manifest.json"
        if not os.path.exists(prior_path):
            raise SystemExit("--merge-prior needs the existing w8_manifest.json")
        with open(prior_path) as handle:
            prior = json.load(handle)
        conv.update(prior.get("converted", {}))
        pt += prior.get("passthrough", 0)
        print(f"seeded {len(conv)} converted modules from the prior manifest")
    for f in sorted(os.listdir(a.out)):
        if f.startswith("manifest."):
            d = json.load(open(f"{a.out}/{f}"))
            conv.update(d["converted"])
            pt += d["passthrough"]
            os.remove(f"{a.out}/{f}")
    out_index = {}
    from encode_w8a16 import out_name
    for name, fname in sid.items():
        if name in conv:
            mod = name[: -len(".weight")]
            for suf in ("weight_packed", "weight_scale", "weight_shape"):
                out_index[f"{mod}.{suf}"] = fname
        else:
            out_index[out_name(name)] = fname
    # A model dir can hold artifacts that are not derived from the source tree,
    # notably the patch 5 draft-vocab head. Index whatever the output shards carry
    # that the source never named, otherwise the drafter allocates a head, loads no
    # weights into it, prints "scores a 40000-token draft vocabulary", and every
    # draft is rejected (acceptance exactly 1.000) on an otherwise clean boot.
    known = set(out_index)
    src_files = set(sid.values())
    added = 0
    for f in sorted(os.listdir(a.out)):
        if not f.endswith(".safetensors") or f in src_files:
            continue
        hdr, _ = read_header(f"{a.out}/{f}")
        for k in hdr:
            if k != "__metadata__" and k not in known:
                out_index[k] = f
                known.add(k)
                added += 1
    if added:
        print(f"carried {added} artifact tensor(s) into the index")
    official = json.load(open(f"{a.src}/config.json"))
    mods = official["quantization_config"]["modules_to_not_convert"]
    conv_mods = {k[: -len(".weight")] for k in conv}
    held = [m for m in mods if m not in conv_mods]
    cfg = json.load(open(f"{a.src}/config.json"))
    cfg["quantization_config"] = build_ct_config(held)
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": 0}, "weight_map": out_index},
              open(f"{a.out}/model.safetensors.index.json", "w"), indent=2)
    mean = float(np.mean([v["rel_rms"] for v in conv.values()]))
    json.dump({"converted": conv, "passthrough": pt, "mean_rel_rms": mean},
              open(f"{a.out}/w8_manifest.json", "w"), indent=1)
    for f_ in sorted(os.listdir(a.src)):
        p = f"{a.src}/{f_}"
        if (not os.path.isfile(p) or f_.endswith(".safetensors")
                or f_ in SKIP or f_ in ("config.json",
                                        "model.safetensors.index.json")):
            continue
        shutil.copy2(p, f"{a.out}/{f_}")
    print(f"assembled: {len(out_index)} entries, {len(conv)} converted "
          f"(mean rel_rms {mean:.5f}), {pt} passthrough, held {len(held)}")


if __name__ == "__main__":
    main()
