#!/usr/bin/env python3
"""Dense bf16 weight bytes per pipeline rank, read from safetensors headers only.

The decode step's achieved bandwidth is these bytes divided by the gemm-bf16 time in a step-anatomy
trace (benchmark/phase_report.py). Kernel names in a captured decode trace carry no shapes, so the
checkpoint is the only source that says how many bytes each rank moves.

No GPU and no tensor data is read: each shard's JSON header is enough.

  .venv/bin/python benchmark/dense_weight_bytes.py /path/to/checkpoint \
      --partition 13,12,12,11 --gemm-ms 2.26,2.03,2.03,3.84
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import struct

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
               "I32": 4, "I64": 8, "F64": 8, "BOOL": 1, "F8_E8M0": 1}
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)")


def shard_tensors(ckpt):
    """{name: (dtype, shape)} over every shard. Reads the header of each file, nothing else."""
    paths = sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors")))
    if not paths:
        raise SystemExit(f"no model-*.safetensors under {ckpt}")
    out = {}
    for path in paths:
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, meta in header.items():
            if name != "__metadata__":
                out[name] = (meta["dtype"], tuple(meta["shape"]))
    return out


def is_dense_gemm_weight(dtype, shape, rest):
    """A 2-D BF16 matrix a dense bf16 GEMM reads. Routed experts, the PLE n-gram table, the token
    embedding and the vision tower are excluded: the first two are not bf16 GEMMs, the embedding is
    gathered by row, and the last never runs on the text path."""
    return (dtype == "BF16" and len(shape) == 2 and rest.endswith(".weight")
            and ".experts." not in rest and "ngram_embedding" not in rest
            and "embed_tokens" not in rest and not rest.startswith("model.visual."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--partition", default="13,12,12,11", help="VLLM_PP_LAYER_PARTITION")
    ap.add_argument("--gemm-ms", help="gemm-bf16 ms per real step per rank, from a phase report")
    args = ap.parse_args()

    part = [int(x) for x in args.partition.split(",")]
    gemm = [float(x) for x in args.gemm_ms.split(",")] if args.gemm_ms else None
    tensors = shard_tensors(args.checkpoint)
    nbytes = lambda dt, shape: math.prod(shape) * DTYPE_BYTES.get(dt, 0)

    per_layer = collections.defaultdict(int)
    kind, draft, head, top = {}, 0, 0, collections.defaultdict(int)
    for name, (dt, shape) in tensors.items():
        rest = name
        m = LAYER.match(name)
        if m:
            rest = m.group(2)
        if not is_dense_gemm_weight(dt, shape, rest):
            continue
        size = nbytes(dt, shape)
        if name.startswith("mtp."):
            draft += size                       # the drafter lives on the last rank
        elif m:
            idx = int(m.group(1))
            per_layer[idx] += size
            if "linear_attn" in rest:
                kind[idx] = "gdn"
            elif "self_attn" in rest:
                kind[idx] = "qsa"
        elif name.endswith("lm_head.weight"):
            head += size
        elif "hyper_connection_mixer" in name:
            top[name] += size               # applied once, on the first rank

    layers = sum(per_layer.values())
    print(f"{len(tensors)} tensors, {len(per_layer)} decoder layers "
          f"({collections.Counter(kind.values()).most_common()})")
    for label, idxs in (("gdn", [i for i, k in kind.items() if k == "gdn"]),
                        ("qsa", [i for i, k in kind.items() if k == "qsa"])):
        if idxs:
            print(f"  {label} mean dense {sum(per_layer[i] for i in idxs)/len(idxs)/2**20:.1f} MiB/layer")
    print(f"  lm_head {head/2**20:.1f} MiB, drafter {draft/2**20:.1f} MiB, "
          f"top-level mixer {sum(top.values())/2**20:.1f} MiB")
    print(f"  total dense per pass {(layers+head+draft)/2**30:.2f} GiB")

    ranks, start = [], 0
    for r, count in enumerate(part):
        size = sum(per_layer[i] for i in range(start, start + count))
        start += count
        if r == 0:
            size += sum(top.values())
        if r == len(part) - 1:
            size += head + draft
        ranks.append(size)
    print(f"\nbytes read per decode step, partition {part}")
    for r, size in enumerate(ranks):
        line = f"  rank {r}: {size/2**20:8.1f} MiB"
        if gemm and r < len(gemm):
            line += f"  / {gemm[r]:5.2f} ms = {size/(gemm[r]*1e-3)/1e9:5.0f} GB/s"
        print(line)
    if gemm:
        total, spent = sum(ranks), sum(gemm[:len(ranks)])
        print(f"  sum  : {total/2**20:8.1f} MiB  / {spent:5.2f} ms = {total/(spent*1e-3)/1e9:.0f} GB/s")
    print("\nThe drafter replays four times per step and the trace categories do not separate it,")
    print("so the last rank's rate carries more modelling error than the middle ranks'.")


if __name__ == "__main__":
    main()
