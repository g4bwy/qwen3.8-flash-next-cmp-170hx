#!/usr/bin/env python
"""Prefill-shape read A/B for the QSA sparse kernel: is int8 read 2x bf16?

The int8 lane booted at 135 s cold 1M TTFT against bf16 76 s at the SAME read
tile config (both take the is_fp8=False default table, BLOCK_N=32 warps=1 on
sm_80; only fp8 gets the 128-wide sm_80 override). Reads dominate prefill
attention (O(tokens x TOPK)) while the cache write is O(tokens), so the read
kernel is the first thing to price. This times the shipped
qsa_sparse_paged_attention at a mid-ramp prefill step for bf16 and int8.

Both dtypes run back-to-back on one card, so whatever neighbor load exists is
common-mode and the RATIO is the signal even if absolutes are inflated.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/wt-int8kv \
        /home/gab/test/vllm/.venv/bin/python \
        benchmark/qsa_read_prefill_ab.py --rows 2048 --depth 500000
"""
import argparse
import statistics
import torch
import triton

import vllm  # noqa: E402
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops  # noqa: E402

HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
PAGE_SIZE = 64
TOPK = 2048


def build(depth, rows, seed, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages = (depth + PAGE_SIZE - 1) // PAGE_SIZE
    dev = "cuda"
    idx = torch.stack(
        [
            torch.randperm(pages * PAGE_SIZE, generator=g)[:TOPK].sort().values
            for _ in range(rows)
        ]
    ).clamp(max=depth - 1).to(dtype=torch.int32, device=dev)
    count = torch.full((rows, 1), TOPK, dtype=torch.int32, device=dev)
    li = torch.cat([idx, count], dim=1).contiguous()

    nb = pages  # dense physical pages so every gathered byte is real, not page 0
    base = torch.randn((nb, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), generator=g)
    k_cache = base.to(dtype=dtype, device=dev).contiguous()
    v_cache = base.to(dtype=dtype, device=dev).contiguous()
    if dtype == torch.int8:
        # Realistic int8 codes and a fp32 per-(token,head) scale in [1e-3, 1e-2].
        k_cache = (base * 100).clamp(-127, 127).to(torch.int8).to(dev).contiguous()
        v_cache = (base * 100).clamp(-127, 127).to(torch.int8).to(dev).contiguous()
        ks = torch.full((nb, PAGE_SIZE, NUM_KV_HEADS), 0.005,
                        dtype=torch.float32, device=dev)
        vs = ks.clone()
    else:
        ks = vs = None
    bt = torch.arange(pages, dtype=torch.int32, device=dev).unsqueeze(0).contiguous()
    q = torch.randn((rows, NUM_Q_HEADS, HEAD_DIM), generator=g).to(
        dtype=torch.bfloat16, device=dev
    )
    t2r = torch.zeros((rows,), dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    gate = torch.ones_like(q)
    return (q, k_cache, v_cache, li, bt, t2r, out, gate, ks, vs)


def timed(args_tuple, dtype, rows, depth):
    q, k, v, li, bt, t2r, out, gate, ks, vs = args_tuple
    mode = KVQuantMode.INT8_PER_TOKEN_HEAD if dtype == torch.int8 else KVQuantMode.NONE
    fn = lambda: qsa_ops.qsa_sparse_paged_attention(  # noqa: E731
        q, k, v, li, bt, t2r, True, out,
        k_scale_cache=ks, v_scale_cache=vs, output_gate=gate, kv_quant_mode=mode,
    )
    fn()
    torch.cuda.synchronize()
    reps = 30
    best = 1e9
    for _ in range(4):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / reps)
    rows_, kvh, cols = q.shape[0], k.shape[2], li.shape[1] - 1
    b, w, t, sp = qsa_ops._select_config(rows_, kvh, True, cols,
                                         is_fp8=False)
    gsize = triton.cdiv(NUM_Q_HEADS, NUM_KV_HEADS)
    print(
        f"  {str(dtype):7s} BLOCK_N={b:3d} warps={w} tiles={t} splits={sp} "
        f"grid=({rows_},{kvh},{sp})  {best*1000:8.1f} us"
    )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=500000)
    a = ap.parse_args()
    print(f"vllm {vllm.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"rows={a.rows} depth={a.depth:,}  (PREFILL config)")
    free, total = torch.cuda.mem_get_info()
    print(f"free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB")
    t_bf = timed(build(a.depth, a.rows, 7, torch.bfloat16), torch.bfloat16, a.rows, a.depth)
    t_i8 = timed(build(a.depth, a.rows, 7, torch.int8), torch.int8, a.rows, a.depth)
    print(f"\nint8 / bf16 read ratio: {t_i8 / t_bf:.2f}x")


if __name__ == "__main__":
    main()
