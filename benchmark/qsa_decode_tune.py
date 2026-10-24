#!/usr/bin/env python
"""Sweep the served QSA decode kernel's tile shape against its gather ceiling.

Why. The equal-bytes A/B (results.md, 2026-10-24) put our Triton sparse attention at
95 GB/s where a random gather of the same rows reaches 338 GB/s on this card, and the
split count barely moved it, 96.6 us at 16 against 99.6 at 64. That says the limit is
structural rather than a knob we forgot to turn. The candidate causes are the 32-token
tile against a 512 byte row, and the uint8 e4m3 byte decode inside the inner loop. This
sweeps the first.

Run it against the SERVED tree. /home/gab/test/vllm carries an older bf16-only variant.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/wt-p13 \
        /home/gab/test/vllm/.venv/bin/python benchmark/qsa_decode_tune.py
"""

import argparse
import functools
import inspect
import math
import statistics
import time

import torch

HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
TOPK = 2051  # indexer_budget + compress_ratio - 1
PAGESIZE = 1616  # --block-size 1616 on the 1M lane
WM_WARPS = (1, 2, 4, 8)


def gather_ceiling(rows: int, row_bytes: int, reps: int) -> float:
    n_rows = max(rows * 8, 1 << 16)
    pool = torch.randn((n_rows, row_bytes // 2), dtype=torch.bfloat16, device="cuda")
    idx = torch.randint(0, n_rows, (rows,), device="cuda", dtype=torch.int64)
    for _ in range(5):
        pool.index_select(0, idx).sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        pool.index_select(0, idx).sum()
    torch.cuda.synchronize()
    gbps = rows * row_bytes / ((time.perf_counter() - t0) / reps) / 1e9
    del pool, idx
    torch.cuda.empty_cache()
    return gbps


def build(rows: int, depth: int):
    device = "cuda"
    g = torch.Generator().manual_seed(7)
    sel = torch.stack(
        [torch.randperm(depth, generator=g)[:TOPK].sort().values for _ in range(rows)]
    )
    pages = (depth + PAGESIZE - 1) // PAGESIZE
    touched = torch.unique(sel // PAGESIZE)
    nb = int(touched.numel())
    phys = torch.zeros(pages, dtype=torch.int32)
    phys[touched] = torch.arange(nb, dtype=torch.int32)

    raw = torch.randn((nb * PAGESIZE, NUM_KV_HEADS, HEAD_DIM), generator=g)
    kc = (
        raw.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .view(nb, PAGESIZE, NUM_KV_HEADS, HEAD_DIM)
        .contiguous()
        .to(device)
    )
    vc = (
        (raw * 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .view(nb, PAGESIZE, NUM_KV_HEADS, HEAD_DIM)
        .contiguous()
        .to(device)
    )
    count = torch.full((rows, 1), TOPK, dtype=torch.int32)
    li = torch.cat([sel.to(torch.int32), count], dim=1).contiguous().to(device)
    q = (
        torch.randn((rows, NUM_Q_HEADS, HEAD_DIM), generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    return (
        q,
        kc,
        vc,
        li,
        phys[None].expand(rows, pages).contiguous().to(device),
        torch.zeros((rows,), dtype=torch.int32, device=device),
        torch.ones_like(q),
    )


class StagesProxy:
    """Wrap the Triton kernel and override num_stages at the launch site.

    The served kernel hardcodes num_stages=2 when it launches the split-K kernel, and
    the inner loop has a dependent chain, index load, then page lookup, then the K
    load that needs that address. Two stages is thin cover for a chain like that, so
    this is the cheapest possible test of whether we are latency-bound.
    """

    def __init__(self, inner, stages):
        self.inner = inner
        self.stages = stages

    def __getitem__(self, grid):
        launcher = self.inner[grid]

        def call(*a, **k):
            k["num_stages"] = self.stages
            return launcher(*a, **k)

        return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200000)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--block-n", default="16,32,64,128,256")
    ap.add_argument("--splits", default="4,8,16,32,64,128")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", default="", help="comma list, e.g. 1,2,3,4,6")
    args = ap.parse_args()

    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if (
        "output_gate"
        not in inspect.signature(qsa_ops.qsa_sparse_paged_attention).parameters
    ):
        raise SystemExit(
            "old bf16-only kernel, point PYTHONPATH at the served tree "
            "(/home/gab/test/wt-p13)"
        )
    print(f"served kernel: {qsa_ops.__file__}")

    q, kc, vc, li, bt, t2r, gate = build(args.rows, args.depth)
    run = functools.partial(
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
    moved = TOPK * NUM_KV_HEADS * HEAD_DIM * 2 * args.rows
    ceil0 = gather_ceiling(TOPK * args.rows, NUM_KV_HEADS * HEAD_DIM * 2, 30)
    print(
        f"{args.rows} rows x {TOPK} tokens, {moved / 2**20:.1f} MiB per layer-step, "
        f"gather ceiling {ceil0:.0f} GB/s"
    )

    def timeit(fn, reps):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(4):
            t0 = time.perf_counter()
            for _ in range(reps // 4):
                fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) / (reps / 4) * 1e6)
        return statistics.median(out)

    base = qsa_ops._select_config
    # is_fp8=True is what the served path passes, and on sm_80 it overrides the
    # GB300 table with BLOCK_N=128, warps=4, so ask the real question.
    bn0, wp0, tl0, sp0 = base(args.rows, NUM_KV_HEADS, False, TOPK, True)
    print(f"shipped: BLOCK_N={bn0} warps={wp0} tiles={tl0} splits={sp0}")
    print("\nBLOCK_N x splits, us per layer-step (0.00 = rejected)")
    splits = [int(x) for x in args.splits.split(",")]
    print("      " + "".join(f"{s:>8d}" for s in splits))
    best = (math.inf, None)
    for bn in [int(x) for x in args.block_n.split(",")]:
        row = f"bn={bn:<4d}"
        for s in splits:
            tiles = math.ceil(TOPK / bn)

            def make_cfg(block_n, num_tiles, num_splits, num_warps):
                def cfg(*_a):
                    return block_n, num_warps, num_tiles, num_splits

                return cfg

            qsa_ops._select_config = make_cfg(bn, tiles, min(s, tiles), args.warps)
            try:
                us = timeit(run, max(24, args.reps // 3))
                row += f"{us:>8.1f}"
                if us < best[0]:
                    best = (us, (bn, s))
            except Exception as e:  # a shape the kernel refuses is a real result
                row += f"{'--':>8}"
                if bn == bn0 and s == sp0:
                    print(f"   shipped shape failed: {type(e).__name__}: {e}")
        print(row)
    qsa_ops._select_config = base

    if args.stages:
        real_kernel = qsa_ops._qsa_sparse_paged_gqa_splitk_kernel
        print("\nnum_stages at the shipped BLOCK_N/splits, us per layer-step")
        for st in [int(x) for x in args.stages.split(",") if x.strip()]:
            qsa_ops._qsa_sparse_paged_gqa_splitk_kernel = StagesProxy(real_kernel, st)
            try:
                us_st = timeit(run, args.reps)
                print(f"  stages={st:2d}: {us_st:7.1f}")
            except Exception as e:
                print(f"  stages={st:2d}: rejected {type(e).__name__}")
        qsa_ops._qsa_sparse_paged_gqa_splitk_kernel = real_kernel

    us, (bn, s) = best
    gbps = moved / (us * 1e-6) / 1e9
    shipped_us = timeit(run, args.reps)
    print(
        f"\nbest ({bn}, splits {s}): {us:.1f} us = {gbps:.0f} GB/s "
        f"({100 * gbps / ceil0:.0f}% of ceiling)"
    )
    print(
        f"shipped again: {shipped_us:.1f} us -> the sweep buys "
        f"{100 * (shipped_us - us) / shipped_us:.0f}% of this kernel's time"
    )
    print(
        f"on the step: 3 layers x 4 serial ranks saves "
        f"{3 * 4 * (shipped_us - us) / 1000:.3f} ms of 22.79 ms = "
        f"{100 * 3 * 4 * (shipped_us - us) / 1000 / 22.79:.2f}%"
    )
    print(
        f"sanity: softmax scale 1/sqrt(D) = {HEAD_DIM**-0.5}, head_dim is a "
        f"power of two = {not (HEAD_DIM & (HEAD_DIM - 1))}"
    )


if __name__ == "__main__":
    main()
