#!/usr/bin/env python3
"""Measure humming fp8 GEMM decode shapes against tuning_config overrides.

Why: this checkpoint quantizes routed experts (fp8, block 128) and PLE; every
other linear is bf16 and runs through cuBLAS F.linear, which is also why the
upstream cute-dsl decode plans are bf16-only and inert on sm80. This script
finds the real linear shapes in a checkpoint, times the production path per
dtype (F.linear for bf16, humming heuristic config for fp8) against candidates
with CUDA events, and reports achieved GB/s so a starved launch is visible
directly. CUPTI and kineto produced no GPU events on the unlocked-driver serve box, so timing is
torch.cuda.Event with an L2 flush between reps. Single GPU per process. Run on
an idle box: the 200 W cap makes any concurrent work a confounder.

  # step 1: shapes only, seconds
  python humming_decode_bench.py --model-path /ssd/ai/models/Qwen/Qwen3.8-Flash-Next-FP8 --dry-run

  # step 2: time the default config for every unique dense shape
  python humming_decode_bench.py --model-path ... --ms 4,16,33,128 --out bench-default.json

  # step 3: sweep candidates on the shapes step 2 flagged
  python humming_decode_bench.py --model-path ... --ms 4,16 --shapes-file picked.json \
      --sweep --out bench-sweep.json
"""

import argparse
import collections
import glob
import json
import os
import statistics
import sys

DEFAULT_MODEL = "/ssd/ai/models/Qwen/Qwen3.8-Flash-Next-FP8"
# Decoder families we care about. Patterns match the tensor base name, which
# has no .weight suffix (e.g. model.layers.3.mlp.gate).
SKIP_SUBSTR = ("mlp.gate", "router", "embed_tokens", "lm_head", "norm",
               "A_log", "dt_bias")


def parse_safetensors_header(path):
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def discover_shapes(model_path):
    """Return {(N, K, scale_kind, dtype): {"count": layers, "names": [..]}}."""
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
        files = [os.path.join(model_path, f) for f in files]
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    weights = {}
    scales = {}
    dtype_hist = collections.Counter()
    total_tensors = 0

    def norm_dtype(s):
        s = s.upper()
        return {"BF16": "BFLOAT16", "F16": "FLOAT16", "F32": "FLOAT32",
                "F8_E4M3": "FLOAT8_E4M3FN", "F8_E4M3FN": "FLOAT8_E4M3FN",
                "E4M3FN": "FLOAT8_E4M3FN"}.get(s, s)

    for fn in files:
        hdr = parse_safetensors_header(fn)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            total_tensors += 1
            shape = meta["shape"]
            dtype = norm_dtype(meta["dtype"])
            dtype_hist[dtype] += 1
            base = None
            if name.endswith(".weight"):
                base = name[: -len(".weight")]
            elif name.endswith(".weight_packed"):
                base = name[: -len(".weight_packed")]
            if base is not None:
                weights[base] = (shape, dtype)
            for suf in (".weight_scale_inv", ".weight_scale"):
                if name.endswith(suf):
                    scales[name[: -len(suf)]] = shape
                    break

    out = {}
    if not weights and total_tensors:
        print(f"DEBUG: {total_tensors} tensors across {len(files)} files, "
              f"no *.weight matched; dtype histogram: {dict(dtype_hist)}; "
              f"first scales: {list(scales)[:3]}")
    if os.environ.get("BENCH_VERBOSE") or len(weights) < 50:
        print(f"DEBUG: {total_tensors} tensors, {len(weights)} weight-like, "
              f"{len(scales)} scale-like; dtypes {dict(dtype_hist)}")
        import itertools as _it

        for nm in _it.islice(weights, 12):
            print("  W:", nm, weights[nm])
        for nm in _it.islice(scales, 6):
            print("  S:", nm, scales[nm])
    for base, (shape, dtype) in weights.items():
        if any(s in base for s in SKIP_SUBSTR) or len(shape) != 2:
            continue
        if dtype not in ("BFLOAT16", "FLOAT8_E4M3FN", "FLOAT8_E4M3", "FLOAT16"):
            continue
        n, k = shape
        scale = scales.get(base)
        if scale is None:
            kind = "none"  # unquantized; recorded, not benched
        elif dtype not in ("FLOAT8_E4M3FN", "FLOAT8_E4M3"):
            kind = "none"
        else:
            import math

            sn, sk = (scale + [1, 1])[:2]
            if (sn, sk) == (1, 1):
                kind = "tensor"
            elif sn == 1:
                kind = f"group:{k // max(sk, 1)}"
            elif sk == 1:
                kind = "channel" if sn == n else f"block:{sn and n // sn},{k // sk}"
            else:
                kind = f"block:{n // sn if sn else 0},{k // sk if sk else 0}"
        key = (n, k, kind, "fp8" if dtype.startswith("FLOAT8") else "bf16")
        e = out.setdefault(key, {"count": 0, "names": []})
        e["count"] += 1
        e["names"].append(base)
    return out


def weight_schema_config(kind):
    base = {"quant_method": "humming", "dtype": "float8e4m3"}
    if kind == "tensor":
        base["weight_scale_type"] = "tensor"
    elif kind == "channel":
        base["weight_scale_type"] = "channel"
    elif kind.startswith("group:"):
        base["weight_scale_type"] = "group"
        base["group_size"] = int(kind.split(":")[1])
    elif kind.startswith("block:"):
        gn, gk = (int(x) for x in kind.split(":")[1].split(","))
        base["weight_scale_type"] = "block"
        base["weight_scale_group_size_n"] = gn
        base["weight_scale_group_size"] = gk
    else:
        raise ValueError(f"unhandled scale kind {kind}")
    return base


def build_layer(n, k, kind, m, device):
    """Mirror the production fp8 flow with wheel-only APIs:
    schema -> prepare_layer_config (pad 256/128) -> transform_humming_tensors
    on the packed int32 weight view, exactly what vLLM's convert + prepare
    produce before humming_forward."""
    import torch
    from humming.schema.humming import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors

    ws = HummingWeightSchema.from_config(weight_schema_config(kind))
    lc = prepare_layer_config(
        shape_n=n, shape_k=k, weight_schema=ws,
        input_schema=HummingInputSchema(),
        pad_n_to_multiple=256, pad_k_to_multiple=128,
        has_bias=False, torch_dtype=torch.bfloat16)
    w = ((torch.randn(n, k, device=device, dtype=torch.float32) * 0.2)
         .clamp(-2, 2).to(torch.float8_e4m3fn))
    sn, sk = scale_shape(n, k, kind)
    s = torch.rand(sn, sk, device=device, dtype=torch.float32) + 0.5
    tensors = transform_humming_tensors(
        lc, {"weight": w.view(n, -1).view(torch.int32), "weight_scale": s})
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    return tensors, lc, x


def scale_shape(n, k, kind):
    import math

    if kind == "tensor":
        return 1, 1
    if kind == "channel":
        return n, 1
    if kind.startswith("group:"):
        return 1, k // int(kind.split(":")[1])
    gn, gk = (int(x) for x in kind.split(":")[1].split(","))
    return math.ceil(n / gn), math.ceil(k / gk)


def default_tuning(layer_config, m, compute_config):
    from vllm.utils.humming import get_heuristics_config

    return get_heuristics_config(
        layer_config=layer_config, shape_m=m,
        use_f16_accum=False, use_batch_invariant=False, gemm_type="dense")


def candidates(base):
    import copy

    out = [("default", None)]
    def clone(**kw):
        c = copy.deepcopy(base)
        c.update(kw)
        return c

    bm, bn, bk = base["block_shape"]
    out.append(("bn64", clone(block_shape=(bm, 64, bk))))
    if bn >= 128:
        out.append(("bn32", clone(block_shape=(bm, 32, bk))))
    out.append(("bk256", clone(block_shape=(bm, bn, min(bk * 2, 256)))))
    out.append(("ctas2", clone(num_ctas_per_sm=2)))
    out.append(("stages4", clone(num_stages=4)))
    if base.get("use_stream_k", True):
        out.append(("no_streamk", clone(use_stream_k=False)))
    else:
        out.append(("streamk", clone(use_stream_k=True)))
    out.append(("splits2", clone(num_write_splits=2, use_stream_k=True)))
    bm2 = 32
    wm, wn, wk = base["warp_shape"]
    out.append(("splits2_m32", clone(
        block_shape=(bm2, bn, bk), warp_shape=(bm2, wn, wk),
        num_write_splits=2, use_stream_k=True)))
    return out


def _clock_spin(device, ms=400.0):
    """Idle cards sit at base clocks and idle memory clocks; short kernels
    then measure sag, not the kernel. The decode shapes here are bandwidth-
    bound, so spin DRAM traffic (large fills), not just matmuls."""
    import time
    import torch

    big = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    t0 = time.monotonic()
    while (time.monotonic() - t0) * 1000.0 < ms:
        for _ in range(30):
            big.fill_(1)
        torch.cuda.synchronize()


def time_fn(fn, reps, flush_buf):
    import torch

    fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        flush_buf.zero_()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def run_case_bf16(n, k, m, args, device):
    import torch

    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    out_buf = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _clock_spin(device)
    cases = [
        ("F.linear", lambda: torch.nn.functional.linear(x, w)),
        ("linear_out", lambda: torch.nn.functional.linear(x, w, out=out_buf)),
        ("matmul_t", lambda: x @ w.t()),
    ]
    rows = []
    for name, fn in cases:
        try:
            fn()
            torch.cuda.synchronize()
            us = time_fn(fn, args.reps, flush)
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "error": str(e)[:120]})
            continue
        wbytes = 2 * (n * k + m * k + m * n)
        rows.append({"name": name, "us": round(us, 1),
                     "GB/s": round(wbytes / (us * 1e-6) / 1e9, 1)})
    return {"n": n, "k": k, "kind": "none", "m": m, "heuristic": None,
            "rows": rows}


def run_case(n, k, kind, m, args, device):
    import torch
    from vllm.utils.humming import humming_forward

    tensors, layer_config, x = build_layer(n, k, kind, m, device)
    compute_config = json.dumps(
        {"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"})
    locks = torch.zeros(1024, dtype=torch.int32, device=device)
    base = default_tuning(layer_config, m, compute_config)

    def call(tuning):
        return lambda: humming_forward(
            layer_config, inputs=x, weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"), locks=locks,
            compute_config=compute_config,
            tuning_config=None if tuning is None else json.dumps(
                {k2: v for k2, v in tuning.items()
                 if k2 not in ("num_sms",)}))

    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    _clock_spin(device)
    rows = []
    for name, tuning in (candidates(base) if args.sweep else [("default", None)]):
        try:
            y = call(tuning)()
            torch.cuda.synchronize()
            us = time_fn(call(tuning), args.reps, flush)
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "error": str(e)[:120]})
            continue
        out_rows = {"name": name, "us": round(us, 1)}
        if tuning is not None:
            out_rows["block"] = list(tuning.get("block_shape", []))
        rows.append(out_rows)
    wbytes = n * k + (n * k / 16384) * 4  # fp8 weights + fp32 128x128 block scales
    for r in rows:
        if "us" in r:
            r["GB/s"] = round(wbytes / (r["us"] * 1e-6) / 1e9, 1)
    return {"n": n, "k": k, "kind": kind, "m": m, "heuristic": base.get("block_shape"),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--ms", default="4,16,33")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--shapes-file", help="json list of [N,K,kind] to bench")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="skip weights under MB (1 covers fp8 experts)")
    ap.add_argument("--dtypes", default="bf16,fp8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    shapes = discover_shapes(args.model_path)
    want = set(args.dtypes.split(","))
    uniq = collections.defaultdict(int)
    for (n, k, kind, dt), info in shapes.items():
        if dt not in want:
            continue
        if dt == "fp8" and kind == "none":
            continue
        if n * k * (1 if dt == "fp8" else 2) / 1e6 < args.min_mb:
            continue
        uniq[(n, k, kind, dt)] += info["count"]

    print(f"{len(shapes)} matched tensors, {len(uniq)} unique shapes "
          f">= {args.min_mb} MB, dtypes {sorted(want)} "
          f"(count = layers sharing it):")
    for (n, k, kind, dt), cnt in sorted(uniq.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        print(f"  N={n:7d} K={k:6d} {kind:12s} {dt:4s} x{cnt}")
    if args.dry_run:
        return

    if args.shapes_file:
        picked = [tuple(x) for x in json.load(open(args.shapes_file))]
    else:
        picked = sorted(uniq.keys())

    import torch

    device = torch.device(args.device)
    results = []
    for picked_shape in picked:
        n, k, kind, dt = (list(picked_shape) + ["none", "fp8"])[:4]
        for m in (int(x) for x in args.ms.split(",")):
            r = (run_case_bf16(n, k, m, args, device) if dt == "bf16"
                 else run_case(n, k, kind, m, args, device))
            r["dt"] = dt
            results.append(r)
            hs = " ".join(f"{row['name']}={row.get('us', row.get('error'))}"
                          for row in r["rows"])
            print(f"N={n} K={k} M={m} {dt} :: {hs}", flush=True)
    if args.out:
        json.dump({"model": args.model_path, "results": results},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
