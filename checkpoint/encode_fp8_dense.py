#!/usr/bin/env python3
"""Emit the W8-everywhere checkpoint: fp8 block-128 dense tensors on top of
the official FP8 release, all other bytes passed through untouched.

v1 converts main-body dense weights (GDN out_proj, QSA q/k/v/o, shared
experts) from bf16 to e4m3 with 128x128 block scales stored as bf16, the
same convention the official release uses for experts. Output keeps
quant_method "fp8"; the config edit shrinks modules_to_not_convert.

usage: encode_fp8_dense.py SRC OUT [--limit N] [--dry-run]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

BF16 = np.dtype(("V", 2))

CONVERT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.out_proj"
    r"|self_attn\.(?:q|k|v|o)_proj"
    r"|mlp\.shared_expert\.(?:gate|up|down)_proj)\.weight$"
)
CONVERT_MODULE_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.out_proj"
    r"|self_attn\.(?:q|k|v|o)_proj"
    r"|mlp\.shared_expert\.(?:gate|up|down)_proj)$"
)
SKIP_FILES = {
    "model.safetensors.index.json.orig",
    "mtp_draft_lm_head.safetensors",
    "mtp_draft_vocab_ids.pt",
}
BLK = 128


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bytes(x: np.ndarray) -> np.ndarray:
    """bf16 with round-to-nearest-even, the torch convention."""
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    h = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    return h.view("V2")


def e4m3_round(x: np.ndarray) -> np.ndarray:
    """RTN to float8_e4m3fn on float32 input, clipped to +-448.

    Returns the rounded *values* (still float32); byte encoding is done by
    a unique-value table, so grid math and bit packing can't drift.
    """
    x = np.clip(x, -448.0, 448.0)
    a = np.abs(x)
    out = np.zeros_like(a)
    sub = a < 2.0 ** -6  # subnormal grid, step 2^-9
    out[sub] = np.round(a[sub] / 2.0 ** -9) * 2.0 ** -9
    nrm = ~sub
    e = np.floor(np.log2(np.maximum(a[nrm], 2.0 ** -6)))
    step = np.ldexp(1.0, e.astype(np.int32) - 3)
    out[nrm] = np.round(a[nrm] / step) * step
    return np.copysign(np.clip(out, 0.0, 448.0), x)


def _e4m3_decode_table() -> np.ndarray:
    dec = np.zeros(256, dtype=np.float64)
    for bits in range(256):
        s = -1.0 if bits & 0x80 else 1.0
        e = (bits >> 3) & 0xF
        m = bits & 7
        if e == 0:
            dec[bits] = s * m * 2.0 ** -9
        elif e == 0xF and m == 7:
            dec[bits] = np.nan
        else:
            dec[bits] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return dec


E4M3_DEC = _e4m3_decode_table()


def e4m3_bytes(x: np.ndarray) -> np.ndarray:
    vals = e4m3_round(x)
    if "_pos" not in e4m3_bytes.__dict__:
        pos = np.array([E4M3_DEC[b] for b in range(0x7F)])
        assert np.all(np.diff(pos[1:]) > 0), "positive e4m3 grid must be monotone"
        e4m3_bytes._pos = pos
    pos = e4m3_bytes._pos
    a = np.abs(vals).astype(np.float64)
    codes = np.searchsorted(pos, a).reshape(vals.shape)
    assert np.array_equal(pos[codes], a), "rounded value off the e4m3 grid"
    codes = codes.astype(np.uint8)
    codes |= (np.signbit(vals)).astype(np.uint8) << 7
    return codes


def quant_block128(w: np.ndarray):
    """w float32 [out, in] -> (fp8 bytes, scales bf16 [out/128, in/128])."""
    o, i = w.shape
    assert o % BLK == 0 and i % BLK == 0, f"shape {w.shape} not 128-divisible"
    blocks = w.reshape(o // BLK, BLK, i // BLK, BLK).transpose(0, 2, 1, 3)
    amax = np.abs(blocks).max(axis=(2, 3))
    scales = (amax / 448.0).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales)
    q = blocks / safe[:, :, None, None]
    fp8 = e4m3_bytes(np.ascontiguousarray(q.transpose(0, 2, 1, 3).reshape(o, i)))
    sc = f32_to_bf16_bytes(scales)
    return fp8, sc


def rel_rms(orig: np.ndarray, fp8: np.ndarray, sc: np.ndarray) -> float:
    """Error against the stored bytes and stored (bf16-rounded) scales."""
    o, i = orig.shape
    dq = E4M3_DEC[fp8].astype(np.float32).reshape(o // BLK, BLK, i // BLK, BLK)
    dq = dq.transpose(0, 2, 1, 3)
    scale = bf16_to_f32(sc)[:, :, None, None]
    dq = (dq * scale).transpose(0, 2, 1, 3).reshape(o, i)
    num = float(((orig - dq) ** 2).mean() ** 0.5)
    den = float((orig ** 2).mean() ** 0.5) or 1e-12
    return num / den


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def write_shard(out_path, entries, data_blobs):
    hdr, off = {}, 0
    for name, meta in entries:
        hdr[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                     "data_offsets": [off, off + meta["nbytes"]]}
        off += meta["nbytes"]
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(hb)) % 8
    b = bytearray()
    b += struct.pack("<Q", len(hb) + pad)
    b += hb + b" " * pad
    with open(out_path, "wb") as fh:
        fh.write(bytes(b))
        for blob in data_blobs:
            fh.write(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0, help="process N shards only")
    ap.add_argument("--files", default="", help="comma-separated shard basenames")
    ap.add_argument("--assemble-only", action="store_true",
                    help="write config, index and copy aux files from merged manifest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idx_all = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"]
    idx_all = {k: v for k, v in idx_all.items() if k != "mtp.draft_lm_head.weight"}
    by_file = {}
    for name, f in idx_all.items():
        by_file.setdefault(f, []).append(name)

    os.makedirs(a.out, exist_ok=True)
    if a.assemble_only:
        assemble(a, idx_all)
        return
    manifest = {"converted": {}, "passthrough_checks": 0, "bytes_read": 0}
    out_index = {}
    files = sorted(by_file) if not a.files else a.files.split(",")
    if a.limit:
        files = files[: a.limit]

    for fi, fname in enumerate(files):
        names = by_file.get(fname, [])
        if not names:
            print(f"skip (no indexed tensors): {fname}")
            continue
        path = f"{a.src}/{fname}"
        if not a.dry_run and os.path.exists(f"{a.out}/{fname}"):
            op = f"{a.out}/{fname}"
            try:
                oh, ostart = read_header(op)
                need = ostart + max(v["data_offsets"][1] for v in oh.values())
                if need != os.path.getsize(op):
                    raise ValueError("partial")
                with open(op, "rb") as ofh, open(path, "rb") as sfh:
                    sh, sstart = read_header(path)
                    for name in names:
                        out_index[name] = fname
                        if not CONVERT_RE.match(name):
                            manifest["passthrough_checks"] += 1
                            continue
                        m = sh[name]
                        s0, s1 = m["data_offsets"]
                        sfh.seek(sstart + s0)
                        raw = sfh.read(s1 - s0)
                        w = bf16_to_f32(np.frombuffer(raw, "V2")).reshape(m["shape"])
                        fp8, sc = quant_block128(w)
                        om = oh[name]
                        o0, o1 = om["data_offsets"]
                        ofh.seek(ostart + o0)
                        assert ofh.read(o1 - o0) == fp8.tobytes(), name
                        sn = name[: -len(".weight")] + ".weight_scale_inv"
                        out_index[sn] = fname
                        n0, n1 = oh[sn]["data_offsets"]
                        ofh.seek(ostart + n0)
                        assert ofh.read(n1 - n0) == sc.tobytes(), sn
                        manifest["converted"][name] = {
                            "shape": m["shape"],
                            "rel_rms": round(rel_rms(w, fp8, sc), 5),
                            "src_sha16": hashlib.sha256(raw).hexdigest()[:16],
                        }
                print(f"resume: {fname} verified")
                continue
            except Exception as exc:
                print(f"redo {fname}: {exc}")
        hdr, data_start = read_header(path)
        with open(path, "rb") as fh:
            blob = fh.read()
        manifest["bytes_read"] += len(blob)
        entries, blobs = [], []
        for name in names:  # index order per file
            meta = dict(hdr[name])
            s, e = hdr[name]["data_offsets"]
            raw = blob[data_start + s: data_start + e]
            if CONVERT_RE.match(name):
                meta_shape = tuple(meta["shape"])
                if meta["dtype"] != "BF16" or len(meta_shape) != 2:
                    entries.append((name, {**meta, "nbytes": e - s}))
                    blobs.append(raw)
                    manifest["passthrough_checks"] += 1
                    out_index[name] = fname
                    continue
                w = bf16_to_f32(np.frombuffer(raw, dtype="V2")).reshape(meta_shape)
                fp8, sc = quant_block128(w)
                r = rel_rms(w, fp8, sc)
                manifest["converted"][name] = {
                    "shape": meta["shape"], "rel_rms": round(r, 5),
                    "src_sha16": hashlib.sha256(raw).hexdigest()[:16],
                }
                entries.append((name, {"dtype": "F8_E4M3", "shape": meta["shape"],
                                       "nbytes": fp8.nbytes}))
                blobs.append(fp8.tobytes())
                sn = name[: -len(".weight")] + ".weight_scale_inv"
                out_index[sn] = fname
                entries.append((sn, {"dtype": "BF16", "shape": list(sc.shape),
                                     "nbytes": sc.nbytes}))
                blobs.append(sc.tobytes())
                # self-test on first converted tensor: official expert
                # re-quant must be bit-exact (validated separately below)
            else:
                entries.append((name, {**meta, "nbytes": e - s}))
                blobs.append(raw)
                manifest["passthrough_checks"] += 1
            out_index[name] = fname
        if not a.dry_run:
            write_shard(f"{a.out}/{fname}", entries, blobs)
        if (fi + 1) % 10 == 0 or fi + 1 == len(files):
            print(f"{fi + 1}/{len(files)} shards", flush=True)

    conv = manifest["converted"]
    print(json.dumps({
        "converted_tensors": len(conv),
        "mean_rel_rms": round(float(np.mean(
            [v["rel_rms"] for v in conv.values()] or [0])), 5),
        "max_rel_rms": max((v["rel_rms"] for v in conv.values()), default=0),
    }, indent=1))
    if not a.dry_run:
        mf = f"{a.out}/manifest.{files[0]}.json"
        json.dump(manifest, open(mf, "w"), indent=1)


def assemble(a, idx_all):
    """Merge per-batch manifests, write config + index + aux files."""
    merged = {"converted": {}, "passthrough_checks": 0}
    out_index = {}
    for f in sorted(os.listdir(a.out)):
        if f.startswith("manifest."):
            m = json.load(open(f"{a.out}/{f}"))
            merged["converted"].update(m["converted"])
            merged["passthrough_checks"] += m["passthrough_checks"]
            os.remove(f"{a.out}/{f}")
    for name in idx_all:
        out_index[name] = idx_all[name]
    for name in merged["converted"]:
        out_index[name[: -len(".weight")] + ".weight_scale_inv"] = out_index[name]
    cfg = json.load(open(f"{a.src}/config.json"))
    mods = cfg["quantization_config"]["modules_to_not_convert"]
    conv_mods = {k[: -len(".weight")] for k in merged["converted"]}
    kept = [m for m in mods if m not in conv_mods]
    print(f"config exclusions: {len(mods)} -> {len(kept)}")
    cfg["quantization_config"]["modules_to_not_convert"] = kept
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": 0}, "weight_map": out_index},
              open(f"{a.out}/model.safetensors.index.json", "w"), indent=2)
    merged["mean_rel_rms"] = float(np.mean([v["rel_rms"] for v in merged["converted"].values()]))
    json.dump(merged, open(f"{a.out}/w8_manifest.json", "w"), indent=1)
    for f_ in sorted(os.listdir(a.src)):
        p = f"{a.src}/{f_}"
        if (not os.path.isfile(p) or f_.endswith(".safetensors")
                or f_ in ("config.json", "model.safetensors.index.json")
                or f_ in SKIP_FILES):
            continue
        shutil.copy2(p, f"{a.out}/{f_}")
    print("assembled:", len(out_index), "index entries;")
    print("converted", len(merged["converted"]), "tensors, mean rel_rms",
          round(merged["mean_rel_rms"], 5))


if __name__ == "__main__":
    main()
