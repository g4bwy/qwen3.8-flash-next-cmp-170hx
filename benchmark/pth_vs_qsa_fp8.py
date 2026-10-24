#!/usr/bin/env python
"""Equal-bytes A/B: the served fp8 QSA kernel against the pth int8 op.

Why this exists. The first A/B compared our kernel on a bf16 cache with their int8
cache, so arm A moved twice the bytes and won on size rather than on skill. That was
the wrong baseline: `serve-4x-1m.sh` passes `--kv-cache-dtype fp8` and the served tree
supports it, "fp8/fp8_e4m3: e4m3 bytes in a uint8 cache ... dequantized on load inside
the QSA Triton kernel" (vllm/models/qwen4_exp/nvidia/qsa.py:150-157 on the 13-patch
tip). One byte per element on both sides, so this run isolates kernel quality.

It must run against the SERVED tree, not /home/gab/test/vllm, whose QSA kernel is an
older bf16-only variant that asserts `k_cache.dtype == torch.bfloat16`. The assert
below catches that mistake.

    CUDA_VISIBLE_DEVICES=2 \
        PYTHONPATH=/home/gab/test/wt-p13:/home/gab/test/pth-build/python \
        /home/gab/test/vllm/.venv/bin/python benchmark/pth_vs_qsa_fp8.py --force
"""

import argparse
import functools
import inspect
import json
import math
import os
import statistics
import time

import torch

SO = "/home/gab/test/pth-build/python/pth_flash_attn/_C.abi3.so"
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
TOPK = 2048
PAGESIZE = 64  # --block-size 1616 splits to this for QSA on the 1M lane


def gather_ceiling(rows: int, row_bytes: int, reps: int) -> float:
    """GB/s this card delivers for THIS access pattern: random row gather.

    A linear read is the wrong ceiling for sparse attention, which pulls TOPK
    scattered rows of row_bytes each. uint8 sum() is also the wrong probe: it is
    an integer reduction, not a bandwidth path, and it reads 88 GB/s on a card
    that does 895 in production. So gather bf16 rows and reduce them, which costs
    the same DRAM traffic per byte and one cheap fp add.
    """
    n_rows = max(rows * 8, 1 << 16)
    elems = row_bytes // 2
    pool = torch.randn((n_rows, elems), dtype=torch.bfloat16, device="cuda")
    idx = torch.randint(0, n_rows, (rows,), device="cuda", dtype=torch.int64)
    for _ in range(5):
        pool.index_select(0, idx).sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        pool.index_select(0, idx).sum()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    gbps = rows * row_bytes / dt / 1e9
    del pool, idx
    torch.cuda.empty_cache()
    return gbps


def timeit(fn, reps):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps // 5):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / (reps / 5) * 1e6)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200000)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--splits", default="1,4,8,16,32,64")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    params = inspect.signature(qsa_ops.qsa_sparse_paged_attention).parameters
    if "output_gate" not in params:
        raise SystemExit(
            "this is the old bf16-only QSA kernel. Point PYTHONPATH at the served "
            "tree (/home/gab/test/wt-p13 or wt-rebase-1023), not /home/gab/test/vllm."
        )
    print(f"served kernel: {qsa_ops.__file__}")
    assert os.path.exists(SO)
    torch.ops.load_library(SO)

    device = "cuda"
    rows, depth = args.rows, args.depth
    g = torch.Generator().manual_seed(7)
    sel = torch.stack(
        [torch.randperm(depth, generator=g)[:TOPK].sort().values for _ in range(rows)]
    )
    pages = (depth + PAGESIZE - 1) // PAGESIZE
    touched = torch.unique(sel // PAGESIZE)
    nb = int(touched.numel())
    phys = torch.zeros(pages, dtype=torch.int32)
    phys[touched] = torch.arange(nb, dtype=torch.int32)

    # sm_80 Triton cannot compile the fp8e4nv type, so the served path stores e4m3
    # bits in a uint8 cache and dequantizes on load. The served tensor is a VIEW of
    # the unified [num_blocks, num_kv_heads, block_size, 2*head_dim] buffer, split
    # the way nvidia/qsa.py:145 splits it, so one token's K row and V row are
    # adjacent 256 B runs and its two KV heads sit 1616*512 bytes apart. A plain
    # contiguous [nb, block, heads, dim] cache puts the heads adjacent instead, which
    # the DRAM sees as one 512 B burst rather than two 256 B bursts and flatters our
    # kernel. Build the real strides.
    src = torch.randn((pages, NUM_KV_HEADS, PAGESIZE, 2 * HEAD_DIM), generator=g).to(
        torch.float8_e4m3fn
    )
    unified = src.view(torch.uint8).to(device)
    kc, vc = unified.transpose(1, 2).split(HEAD_DIM, dim=-1)
    print(
        f"arm A cache shape {tuple(kc.shape)} strides {kc.stride()} "
        f"({unified.numel() / 2**20:.0f} MiB for both tensors)"
    )

    count = torch.full((rows, 1), TOPK, dtype=torch.int32)
    li = torch.cat([sel.to(torch.int32), count], dim=1).contiguous().to(device)
    bt = phys[None].expand(rows, pages).contiguous().to(device)
    t2r = torch.zeros((rows,), dtype=torch.int32, device=device)
    q = (
        torch.randn((rows, NUM_Q_HEADS, HEAD_DIM), generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    gate = torch.ones_like(q)
    run_a = functools.partial(
        qsa_ops.qsa_sparse_paged_attention,
        q,
        kc,
        vc,
        li,
        bt,
        t2r,
        False,
        None,
        1.0,
        1.0,
        output_gate=gate,
    )
    orig = qsa_ops._select_config
    bn, wp, tiles, sp = orig(rows, NUM_KV_HEADS, False, TOPK, True)

    # Arm B: the same token set, int8 with per-token-head scales, BS=1 page list.
    uniq = torch.unique(sel)
    n = int(uniq.numel())
    pk = (uniq // PAGESIZE).long()
    off = (uniq % PAGESIZE).long()
    kx = kc[pk, off].float()
    vx = vc[pk, off].float()

    def quant(x):
        s = torch.clamp(x.abs().amax(-1) / 127.0, min=1e-6)
        qq = x * (1.0 / s).unsqueeze(-1)
        return torch.where(qq >= 0, qq + 0.5, qq - 0.5).clamp(-128, 127).to(
            torch.int8
        ), s

    k8, ks = quant(kx)
    v8, vs = quant(vx)
    slot_of = torch.zeros(int(uniq.max()) + 1, dtype=torch.int32)
    slot_of[uniq] = torch.arange(n, dtype=torch.int32)
    b = {
        "q": q,
        "out": torch.empty_like(q),
        "k": k8.view(n, 1, NUM_KV_HEADS, HEAD_DIM).contiguous().to(device),
        "v": v8.view(n, 1, NUM_KV_HEADS, HEAD_DIM).contiguous().to(device),
        "ks": ks.view(n, 1, NUM_KV_HEADS).to(device).float(),
        "vs": vs.view(n, 1, NUM_KV_HEADS).to(device).float(),
        "bt": slot_of[sel].contiguous().to(device),
        "sl": torch.full((rows,), TOPK, dtype=torch.int32, device=device),
        "qs": torch.arange(rows + 1, dtype=torch.int32, device=device),
    }
    dummy = torch.empty(1, device=device)

    def run_b(num_splits):
        ws_o = torch.empty(
            (max(num_splits, 1), rows, NUM_Q_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
        ws_l = torch.empty(
            (max(num_splits, 1), rows, NUM_Q_HEADS), dtype=torch.float32, device=device
        )

        def call():
            torch.ops.pth_flash_attn.pth_flash_attn_fwd(
                b["out"],
                b["q"],
                b["k"],
                b["v"],
                b["ks"],
                b["vs"],
                b["bt"],
                b["sl"],
                b["qs"],
                1,
                1,
                num_splits,
                1.0 / math.sqrt(HEAD_DIM),
                ws_o if num_splits > 1 else dummy,
                ws_l if num_splits > 1 else dummy,
                0,
                True,
            )

        return call

    mb = TOPK * NUM_KV_HEADS * HEAD_DIM * 2 * rows / 2**20
    print(
        f"both arms read {mb:.1f} MiB per layer-step ({rows} rows x {TOPK} tokens, "
        f"1 byte per element). arm A shipped splits={sp} BLOCK_N={bn} warps={wp}"
    )
    # One row is one token's K+V for one KV head: Hkv * D bytes per tensor.
    row_bytes = NUM_KV_HEADS * HEAD_DIM * 2
    calib0 = gather_ceiling(TOPK * rows, row_bytes, 30)
    print(
        f"calibration start: a random gather of {TOPK * rows} rows x "
        f"{row_bytes} B runs at {calib0:.0f} GB/s"
    )

    print("\npaired blocks, matched splits=64 (us per layer-step)")
    ratios = []
    for block in range(4):
        a = timeit(run_a, max(20, args.reps // 3))
        bb = timeit(run_b(sp), max(20, args.reps // 3))
        ratios.append(a / bb)
        print(f"  block {block}: A {a:7.1f}  B {bb:7.1f}  A/B {a / bb:5.2f}")

    print("\nsplit sweeps")
    A, B = {}, {}
    for s in [int(x) for x in args.splits.split(",") if x.strip()]:
        A[s] = timeit(run_a, max(20, args.reps // 3)) if s == sp else None
        B[s] = timeit(run_b(s), max(20, args.reps // 3))
        print(f"  B splits={s:3d}: {B[s]:7.1f} us")
    best_b = min(B, key=B.get)
    a_ref = timeit(run_a, args.reps)
    calib1 = gather_ceiling(TOPK * rows, row_bytes, 30)
    print(
        f"\nA shipped {a_ref:.1f} us   B best (splits={best_b}) {B[best_b]:.1f} us   "
        f"A/B {a_ref / B[best_b]:.2f}"
    )
    moved = TOPK * rows * row_bytes
    a_gbps = moved / (a_ref * 1e-6) / 1e9
    b_gbps = moved / (B[best_b] * 1e-6) / 1e9
    print(
        f"calibration end: {calib1:.0f} GB/s. Arm A reached {a_gbps:.0f} GB/s "
        f"({100 * a_gbps / calib1:.0f}% of it), arm B {b_gbps:.0f} GB/s "
        f"({100 * b_gbps / calib1:.0f}%)."
    )
    print(
        json.dumps(
            {
                "ceiling_start": round(calib0),
                "ceiling_end": round(calib1),
                "paired_ratio_mean": round(statistics.mean(ratios), 3),
                "paired_ratio_spread": round(max(ratios) - min(ratios), 3),
                "arm_a_us": round(a_ref, 1),
                "arm_b_us": round(B[best_b], 1),
                "step_share_pct": round(
                    100 * 3 * 4 * (a_ref - B[best_b]) / 1000 / 22.79, 2
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
