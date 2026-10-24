#!/usr/bin/env python
"""Measure the QSA sparse attention read floor on one card.

The question this answers: how much of the decode step is KV bytes that an
int8 cache would halve. Only 12 of the 48 layers are full attention, so the
whole sparse read may be too small a slice to be worth a cache format change.

Runs one kernel, a few hundred times, on one device. No engine, no server. The
kernel is Triton, so the stale August .so files in this tree do not matter, but the
Python must be current: use PYTHONPATH=/home/gab/test/vllm.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/vllm \
        /home/gab/test/vllm/.venv/bin/python \
        benchmark/qsa_byte_floor.py --depth 200000 --tokens 5

The preflight refuses to time on a contended card. A microbenchmark beside live
serving traffic measures the neighbor, not the kernel, and `nice` cannot help
because the contention is DRAM bandwidth and SM slots.
"""

import argparse
import functools
import json
import os
import statistics
import subprocess
import time

import torch

import vllm  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

qsa_sparse_paged_attention = qsa_ops.qsa_sparse_paged_attention

HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
PAGE_SIZE = 64
TOPK = 2048
FULL_ATTN_LAYERS = 12  # layer_types has 12 full_attention, 36 linear_attention


def preflight(max_util: int, min_free_mib: int, force: bool) -> None:
    """Refuse to time a kernel beside live traffic.

    Reads utilization and free memory for the device this process will actually
    land on, which is the first id listed in CUDA_VISIBLE_DEVICES.
    """
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.strip().splitlines():
        idx, util, used, total = [p.strip() for p in line.split(",")]
        if idx != visible:
            continue
        free = int(total) - int(used)
        report = {
            "device": idx,
            "util_pct": int(util),
            "free_mib": free,
            "needs_free_mib": min_free_mib,
            "max_util_pct": max_util,
        }
        print("preflight " + json.dumps(report))
        if int(util) > max_util or free < min_free_mib:
            if force:
                print("WARNING: contended device, these numbers are not kernel costs")
                return
            raise SystemExit(
                f"device {idx} is busy ({util}% util, {free} MiB free). Every "
                "card on this box serves live traffic, so a timing taken now "
                "measures the neighbor. Stop the serving stack first, or pass "
                "--force and label the result as contended."
            )
        return
    print(f"preflight: device {visible} not reported by nvidia-smi, proceeding")


def build(depth: int, rows: int, seed: int):
    """Selected tokens spread uniformly over a `depth` token logical prefix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages = (depth + PAGE_SIZE - 1) // PAGE_SIZE
    device = "cuda"

    # Each row selects TOPK distinct token positions from its own prefix.
    idx = torch.stack(
        [
            torch.randperm(pages * PAGE_SIZE, generator=g)[:TOPK].sort().values
            for _ in range(rows)
        ]
    ).clamp(max=depth - 1)
    idx = idx.to(dtype=torch.int32, device=device)
    count = torch.full((rows, 1), TOPK, dtype=torch.int32, device=device)
    logical_indices = torch.cat([idx, count], dim=1).contiguous()

    # Physical pages: touch only the pages the selection can reach.
    touched = torch.unique(idx // PAGE_SIZE).to(torch.int64)
    nb = int(touched.numel())
    k_cache = torch.randn((nb, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), generator=g)
    v_cache = torch.randn((nb, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), generator=g)
    k_cache = k_cache.to(dtype=torch.bfloat16, device=device).contiguous()
    v_cache = v_cache.to(dtype=torch.bfloat16, device=device).contiguous()

    # Page table maps logical page -> physical page, dense over the prefix so
    # untouched pages land on page 0.
    bt_cpu = torch.zeros((1, pages), dtype=torch.int32)
    bt_cpu[0, touched.cpu()] = torch.arange(nb, dtype=torch.int32)
    block_table = bt_cpu.to(device).contiguous()

    q = torch.randn((rows, NUM_Q_HEADS, HEAD_DIM), generator=g).to(
        dtype=torch.bfloat16, device=device
    )
    token_to_req = torch.zeros((rows,), dtype=torch.int32, device=device)
    return q, k_cache, v_cache, logical_indices, block_table, token_to_req, nb


def stream_ceiling(nbytes: int, reps: int) -> float:
    """Achieved read bandwidth on this card right now, for this footprint.

    Their bench/sol_copy.cu does this per access pattern. Ours is a reduce over a
    contiguous buffer, which is the same one-read-per-byte shape as the KV scan.
    """
    n = nbytes // 2  # bf16
    buf = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        buf.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        buf.sum()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return nbytes / dt / 1e9


def timeit(fn, reps: int, warmup: int = 20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps // 5):
            fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) / (reps / 5) * 1000)
    return statistics.median(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200000, help="logical prefix tokens")
    ap.add_argument("--tokens", type=int, default=5, help="query rows per step")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--step-ms", type=float, default=22.79, help="measured decode step")
    ap.add_argument("--ceiling-gbps", type=float, default=1593.0)
    ap.add_argument("--max-util", type=int, default=5, help="device util %% to allow")
    ap.add_argument("--min-free-mib", type=int, default=2048)
    ap.add_argument("--force", action="store_true", help="time anyway, label it")
    ap.add_argument(
        "--splits",
        default="",
        help="comma list of forced num_splits, e.g. 8,16,32,64. Empty uses the "
        "shipped _select_config, which is tuned on GB300 and asks for 64-way "
        "split-K at our decode shape",
    )
    args = ap.parse_args()

    preflight(args.max_util, args.min_free_mib, args.force)

    q, k_cache, v_cache, li, bt, t2r, nb = build(args.depth, args.tokens, seed=7)
    print(f"vllm {vllm.__version__} from {vllm.__file__}")
    cache_mib = 2 * k_cache.numel() * 2 / 2**20
    print(
        f"device {torch.cuda.get_device_name(0)}  pages touched {nb}  "
        f"cache {cache_mib:.0f} MiB across both tensors"
    )

    fn = lambda: qsa_sparse_paged_attention(  # noqa: E731
        q, k_cache, v_cache, li, bt, t2r, False
    )

    original = qsa_ops._select_config
    rows, kv_heads, cols = q.shape[0], k_cache.shape[2], li.shape[1] - 1
    b, w, t, s = original(rows, kv_heads, False, cols)
    print(
        f"shipped decode config: BLOCK_N={b} warps={w} tiles={t} splits={s} "
        f"-> grid ({rows}, {kv_heads}, {s}) = {rows * kv_heads * s} CTAs on "
        f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs"
    )
    total = TOPK * NUM_KV_HEADS * HEAD_DIM * 2 * 2 * args.tokens  # K+V, bf16
    kv_mib = total / 2**20
    partial_mib = s * rows * q.shape[1] * q.shape[2] * 4 / 2**20
    print(
        f"split-K partial workspace: {partial_mib:.1f} MiB fp32 written then read "
        f"back, {2 * partial_mib:.1f} MiB of traffic, against {kv_mib:.1f} MiB of KV. "
        f"GA100 L2 is 40 MiB, so these partials may never reach DRAM."
    )

    per_call = timeit(fn, args.reps)

    if args.splits:

        def patched(num_rows, num_kv_heads, use_prefill, num_columns, forced):
            return b, w, t, min(forced, t)

        for want in [int(x) for x in args.splits.split(",") if x.strip()]:
            qsa_ops._select_config = functools.partial(patched, forced=want)
            ms = timeit(fn, args.reps)
            print(f"  splits={want:3d}: {ms * 1000:7.1f} us")
        qsa_ops._select_config = original
        per_call = timeit(fn, args.reps)
        print(f"  shipped    : {per_call * 1000:7.1f} us")

    measured = stream_ceiling(total, max(20, args.reps // 3))
    print(
        f"calibration: a pure {kv_mib:.1f} MiB streaming read here runs at "
        f"{measured:.0f} GB/s (stored die ceiling {args.ceiling_gbps:.0f} GB/s)"
    )
    gbps = total / (per_call * 1e-3) / 1e9
    print(
        f"one layer-step: {per_call * 1000:.1f} us for {kv_mib:.2f} MiB of selected "
        f"KV = {gbps:.0f} GB/s ({100 * gbps / measured:.0f}% of this card's own "
        f"ceiling, {100 * gbps / args.ceiling_gbps:.0f}% of the stored one)"
        f"{100 * gbps / args.ceiling_gbps:.0f}% of the stored one)"
    )

    per_rank = FULL_ATTN_LAYERS / 4
    rank_ms = per_call * per_rank
    print(
        f"{per_rank:.0f} full-attn layers per PP rank -> {rank_ms * 1000:.0f} us of KV "
        f"read per rank-step"
    )
    wall_us = rank_ms * 1000 * 4  # PP stages are serial for one stream
    print(
        f"all 4 ranks serial: {wall_us:.0f} us of a {args.step_ms:.2f} ms step = "
        f"{100 * wall_us / 1000 / args.step_ms:.1f}%"
    )
    print(
        f"int8 KV halves those bytes, so the whole prize is "
        f"{50 * wall_us / 1000 / args.step_ms:.2f}% of the decode step"
    )
    print(
        "and at the current 4.96x pool it also doubles resident capacity: "
        "4,961,194 -> 9,922,388 tokens"
    )


if __name__ == "__main__":
    main()
