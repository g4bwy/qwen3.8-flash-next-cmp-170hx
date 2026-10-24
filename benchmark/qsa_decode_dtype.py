#!/usr/bin/env python
"""Which costs more in QSA decode: the bytes, or decoding them?

Step 1 of the 3.8% problem. The served fp8 Triton kernel reaches 100 GB/s where a
random 512 B gather on this card reaches about 350 GB/s, and their int8 kernel reaches
320 GB/s at the same shape (results.md, 2026-10-24). No tile shape buys anything, so the
cost is inside the kernel. Two suspects: the e4m3 byte decode in the inner loop, or the
address math of a per-token gather.

This separates them with one ratio. Run the SAME kernel on the SAME selection with the
same strides and page size, once with a bf16 cache and once with the uint8 e4m3 cache.
bf16 moves exactly twice the bytes.

    time(bf16) / time(fp8) == 2.0   both are bandwidth-bound, the decode is free,
                                   so chase the gather instead
    time(bf16) / time(fp8) <  2.0   the fp8 path pays a per-element penalty on top of
                                   its byte advantage, so chase the decode

Arms run interleaved so any neighbour on this shared card hits both equally.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/wt-p13 \
        /home/gab/test/vllm/.venv/bin/python benchmark/qsa_decode_dtype.py
"""

import argparse
import functools
import inspect
import json
import statistics
import time

import torch

HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
TOPK = 2051  # indexer_budget + compress_ratio - 1
PAGESIZE = 1616  # --block-size 1616 on the 1M lane


def gather_ceiling(runs: int, run_bytes: int, reps: int) -> float:
    """GB/s for `runs` random reads of `run_bytes` each.

    The run length matters, so pass the real contiguous unit. In the unified
    [num_blocks, num_kv_heads, block_size, 2*head_dim] layout one (token, head)
    holds K and V back to back, so the unit is 2*head_dim elements: 512 B with an
    fp8 cache and 1024 B with bf16. The two KV heads are far apart and count as two
    runs.
    """
    n_rows = max(runs * 4, 1 << 15)
    pool_elems = n_rows * (run_bytes // 2)
    pool = torch.randn(pool_elems, dtype=torch.bfloat16, device="cuda")
    idx = torch.randint(0, n_rows, (runs,), device="cuda", dtype=torch.int64)
    view = pool.view(n_rows, run_bytes // 2)
    for _ in range(3):
        view.index_select(0, idx).sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        view.index_select(0, idx).sum()
    torch.cuda.synchronize()
    gbps = runs * run_bytes / ((time.perf_counter() - t0) / reps) / 1e9
    del pool, idx, view
    torch.cuda.empty_cache()
    return gbps


def timeit(fn, reps):
    for _ in range(12):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200000)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--reps", type=int, default=120)
    args = ap.parse_args()

    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    if (
        "output_gate"
        not in inspect.signature(qsa_ops.qsa_sparse_paged_attention).parameters
    ):
        raise SystemExit("stale tree, point PYTHONPATH at /home/gab/test/wt-p13")

    device = "cuda"
    g = torch.Generator().manual_seed(7)
    rows, depth = args.rows, args.depth
    sel = torch.stack(
        [torch.randperm(depth, generator=g)[:TOPK].sort().values for _ in range(rows)]
    )
    pages = (depth + PAGESIZE - 1) // PAGESIZE
    touched = torch.unique(sel // PAGESIZE)
    nb = int(touched.numel())
    phys = torch.zeros(pages, dtype=torch.int32)
    phys[touched] = torch.arange(nb, dtype=torch.int32)
    print(
        f"{rows} rows x {TOPK} tokens over a {depth} prefix, page {PAGESIZE}, "
        f"{nb} of {pages} pages resident"
    )

    src = torch.randn((pages, NUM_KV_HEADS, PAGESIZE, 2 * HEAD_DIM), generator=g)
    # Both arms read the same unified byte order, so the only difference is the
    # element width and whether the kernel must decode.
    u8 = src.to(torch.float8_e4m3fn).view(torch.uint8).to(device)
    bf = src.to(torch.bfloat16).to(device)
    kc8, vc8 = u8.transpose(1, 2).split(HEAD_DIM, dim=-1)
    kcb, vcb = bf.transpose(1, 2).split(HEAD_DIM, dim=-1)
    print(
        f"fp8  strides {kc8.stride()}  {u8.numel() / 2**20:.0f} MiB resident\n"
        f"bf16 strides {kcb.stride()}  {bf.numel() * 2 / 2**20:.0f} MiB resident"
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
    common = (li, bt, t2r, False)
    run8 = functools.partial(
        qsa_ops.qsa_sparse_paged_attention,
        q,
        kc8,
        vc8,
        *common,
        None,
        1.0,
        1.0,
        output_gate=gate,
    )
    runb = functools.partial(
        qsa_ops.qsa_sparse_paged_attention,
        q,
        kcb,
        vcb,
        *common,
        None,
        None,
        None,
        output_gate=gate,
    )

    moved = TOPK * rows * NUM_KV_HEADS * 2 * HEAD_DIM  # elements per layer-step
    mib8 = moved * 1 / 2**20
    mibb = moved * 2 / 2**20
    print(f"fp8 moves {mib8:.1f} MiB per layer-step, bf16 moves {mibb:.1f} MiB")

    print("\ninterleaved blocks (us per layer-step)")
    ratios = []
    for block in range(4):
        t8 = timeit(run8, max(24, args.reps // 3))
        tb = timeit(runb, max(24, args.reps // 3))
        ratios.append(tb / t8)
        print(
            f"  block {block}: fp8 {t8:7.1f}  bf16 {tb:7.1f}  bf16/fp8 {tb / t8:5.2f}"
        )

    t8 = timeit(run8, args.reps)
    tb = timeit(runb, args.reps)
    runs = TOPK * rows * NUM_KV_HEADS
    c8 = gather_ceiling(runs, 2 * HEAD_DIM * 1, 25)
    cb = gather_ceiling(runs, 2 * HEAD_DIM * 2, 25)
    e8 = moved / (t8 * 1e-6) / 1e9
    eb = moved * 2 / (tb * 1e-6) / 1e9
    ratio = tb / t8
    print(
        f"\nfp8  {t8:.1f} us, {e8:.0f} GB/s, ceiling {c8:.0f} GB/s "
        f"-> {100 * e8 / c8:.0f}% of it"
    )
    print(
        f"bf16 {tb:.1f} us, {eb:.0f} GB/s, ceiling {cb:.0f} GB/s "
        f"-> {100 * eb / cb:.0f}% of it"
    )
    verdict = (
        "bandwidth-bound: both arms sit at the same fraction of their own ceiling, "
        "so the e4m3 decode is free and the gather addressing is the suspect"
        if abs(ratio - 2.0) < 0.15
        else (
            "the fp8 path pays a per-element penalty: it reads half the bytes yet "
            "takes more than half the bf16 time, so decode is the suspect"
            if ratio < 2.0
            else "bf16 is slower than 2x, which should not happen: the fp8 decode "
            "is costing more than the bytes it saves"
        )
    )
    print(
        json.dumps(
            {
                "bf16_over_fp8": round(ratio, 3),
                "paired_mean": round(statistics.mean(ratios), 3),
                "paired_spread": round(max(ratios) - min(ratios), 3),
            },
            indent=2,
        )
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
