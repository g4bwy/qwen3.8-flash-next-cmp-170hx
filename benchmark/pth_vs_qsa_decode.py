#!/usr/bin/env python
"""Paired A/B: our Triton sparse attention against the pth int8 op, same selection.

Both arms attend the SAME 2048 selected tokens per query row over the same logical
prefix, at the real serving shape (D=256, 24 query heads, 2 KV heads, 5 rows per
decode step from MTP-4). Arm A reads bf16, arm B reads int8, so arm B moves half the
bytes by construction. The output of interest is the ratio, not the absolute
microseconds.

Why paired. Every card on this box serves live traffic, so an absolute timing taken
beside it measures the neighbour. Interleaving A and B inside one process means both
arms see the same contention, which makes the ratio far more robust than either
number. Still label any contended run as contended.

Their op indexes the block table by request, not by query row (docs/api.md:48, and
j = block_table[r][i] with r = req_of(token) at csrc/pth_attn/cache_geometry.cuh:10).
Our selector picks a different set per row, and at decode a step has 5 rows in one
request, so the 5 rows are passed as 5 pseudo-requests with query_start_loc
[0,1,2,3,4,5]. Each then carries its own per-token page list at BS=1.

    CUDA_VISIBLE_DEVICES=2 \
        PYTHONPATH=/home/gab/test/vllm:/home/gab/test/pth-build/python \
        /home/gab/test/vllm/.venv/bin/python benchmark/pth_vs_qsa_decode.py --force
"""

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import time

import torch

SO = "/home/gab/test/pth-build/python/pth_flash_attn/_C.abi3.so"
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
TOPK = 2048
PAGESIZE_OURS = 64  # what the manager gives QSA on the 1M lane


def preflight(max_util: int, force: bool) -> bool:
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    # PP stages burst, so one sample can read 0% beside a loaded card. Take the
    # worst of several.
    utils, frees = [], []
    for _ in range(6):
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        util, used, total = [int(p.strip()) for p in out.strip().split(",")]
        utils.append(util)
        frees.append(int(total) - int(used))
        time.sleep(0.25)
    util, free = max(utils), min(frees)
    contended = util > max_util
    print(
        f"preflight device {visible}: peak util {util}% (samples {utils}), "
        f"min free {free} MiB"
    )
    if contended and not force:
        raise SystemExit(
            f"device {visible} peaked at {util}%. Pass --force to run a paired "
            "comparison anyway; do not quote absolute times from it."
        )
    return contended


def make_selection(rows: int, depth: int, seed: int):
    """Per-row sets of TOPK token positions drawn from a `depth` token prefix."""
    g = torch.Generator().manual_seed(seed)
    per_row = [
        torch.randperm(depth, generator=g)[:TOPK].sort().values for _ in range(rows)
    ]
    return torch.stack(per_row)  # [rows, TOPK] logical token ids


def build_arm_a(sel: torch.Tensor, depth: int):
    """Our kernel: bf16 paged cache, per-request page table, packed index buffer."""
    device = "cuda"
    pages = (depth + PAGESIZE_OURS - 1) // PAGESIZE_OURS
    touched = torch.unique(sel // PAGESIZE_OURS)
    nb = int(touched.numel())
    phys = torch.full((pages,), -1, dtype=torch.int32)
    phys[touched] = torch.arange(nb, dtype=torch.int32)

    kv = torch.randn(
        (nb * PAGESIZE_OURS, NUM_KV_HEADS, HEAD_DIM),
        generator=torch.Generator().manual_seed(3),
    )
    k_cache = kv.to(torch.bfloat16).view(nb, PAGESIZE_OURS, NUM_KV_HEADS, HEAD_DIM)
    v_cache = (
        (kv * 0.5).to(torch.bfloat16).view(nb, PAGESIZE_OURS, NUM_KV_HEADS, HEAD_DIM)
    )
    count = torch.full((sel.shape[0], 1), TOPK, dtype=torch.int32)
    logical_indices = torch.cat([sel.to(torch.int32), count], dim=1).contiguous()
    # One request holding all rows, which is what decode really looks like.
    block_table = phys.clamp(min=0)[None].expand(sel.shape[0], pages).contiguous()
    token_to_req = torch.zeros((sel.shape[0],), dtype=torch.int32)
    q = torch.randn(
        (sel.shape[0], NUM_Q_HEADS, HEAD_DIM),
        generator=torch.Generator().manual_seed(5),
    ).to(torch.bfloat16)
    to_gpu = lambda t: t.to(device)  # noqa: E731
    return (
        to_gpu(q),
        to_gpu(k_cache.contiguous()),
        to_gpu(v_cache.contiguous()),
        to_gpu(logical_indices),
        to_gpu(block_table),
        to_gpu(token_to_req),
        kv,
        phys,
    )


def build_arm_b(sel: torch.Tensor, kv, phys):
    """Their kernel at BS=1: int8 per-token pages, block table is a token list."""
    device = "cuda"
    # Distinct logical tokens across all rows become physical int8 rows.
    uniq = torch.unique(sel)
    n = int(uniq.numel())
    # A logical token lives in phys[token // PAGESIZE] at offset token % PAGESIZE.
    slot = (
        phys[(uniq // PAGESIZE_OURS).long()].long() * PAGESIZE_OURS
        + (uniq % PAGESIZE_OURS).long()
    )
    k_u = kv[slot].float()
    v_u = (kv[slot] * 0.5).float()

    def quant(x):
        s = torch.clamp(x.abs().amax(-1) / 127.0, min=1e-6)
        qq = x * (1.0 / s).unsqueeze(-1)
        qq = torch.where(qq >= 0, qq + 0.5, qq - 0.5).clamp(-128.0, 127.0)
        return qq.to(torch.int8), s

    k8, ks = quant(k_u)
    v8, vs = quant(v_u)
    # Their validator wants scales as [NB, BS, Hkv], rank 3, matching the cache.
    ks = ks.view(n, 1, NUM_KV_HEADS)
    vs = vs.view(n, 1, NUM_KV_HEADS)
    slot_of = torch.full((int(uniq.max()) + 1,), 0, dtype=torch.int32)
    slot_of[uniq] = torch.arange(n, dtype=torch.int32)
    rows = sel.shape[0]
    block_table = slot_of[sel].contiguous().to(torch.int32)  # [rows, TOPK]
    seq_lens = torch.full((rows,), TOPK, dtype=torch.int32)
    query_start_loc = torch.arange(rows + 1, dtype=torch.int32)
    q = torch.randn(
        (rows, NUM_Q_HEADS, HEAD_DIM), generator=torch.Generator().manual_seed(5)
    ).to(torch.bfloat16)
    made = {
        "q": q.to(device),
        "out": torch.empty_like(q).to(device),
        "k": k8.view(n, 1, NUM_KV_HEADS, HEAD_DIM).contiguous().to(device),
        "v": v8.view(n, 1, NUM_KV_HEADS, HEAD_DIM).contiguous().to(device),
        "ks": ks.to(device).float(),
        "vs": vs.to(device).float(),
        "bt": block_table.to(device),
        "sl": seq_lens.to(device),
        "qs": query_start_loc.to(device),
    }
    return made, 2 * n * NUM_KV_HEADS * HEAD_DIM  # int8 bytes resident


def stream_ceiling(mib: float, reps: int) -> float:
    """GB/s this card delivers for a pure read of this footprint, right now.

    Utilization percent is a bad contention probe on a pipeline-parallel lane,
    because those idle by design. Achieved bandwidth is the honest probe: if
    this reads near the production 895 GB/s the card is effectively ours, and
    if it collapses then a neighbour is eating DRAM and no timing here means
    anything. Their bench/sol_copy.cu is the same idea, per access pattern.
    """
    n = int(mib * 2**20 / 2)
    buf = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        buf.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        buf.sum()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    gbps = n * 2 / dt / 1e9
    del buf
    torch.cuda.empty_cache()
    return gbps


def timeit(fn, reps):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    blocks = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps // 5):
            fn()
        torch.cuda.synchronize()
        blocks.append((time.perf_counter() - t0) / (reps / 5) * 1e6)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200000)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--splits", default="1,4,8,16,32")
    ap.add_argument("--max-util", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    contended = preflight(args.max_util, args.force)
    calib0 = stream_ceiling(20.0, 200)
    print(
        f"calibration: a 20 MiB pure read runs at {calib0:.0f} GB/s "
        "(production decode reads at 895, die ceiling 1,593)"
    )
    if calib0 < 0.6 * 895:
        print(
            "WARNING: achieved bandwidth is well below the production rate, so a "
            "neighbour is consuming DRAM. Ratios may still hold, times will not."
        )
    assert os.path.exists(SO), "build the extension first, see pth_attn_gate.py"
    torch.ops.load_library(SO)
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    sel = make_selection(args.rows, args.depth, seed=7)
    q_a, kc, vc, li, bt, t2r, kv, phys = build_arm_a(sel, args.depth)
    b, b_bytes = build_arm_b(sel, kv, phys)

    kv_mib_a = TOPK * NUM_KV_HEADS * HEAD_DIM * 2 * 2 * args.rows / 2**20
    print(
        f"selection: {args.rows} rows x {TOPK} tokens over a {args.depth} logical "
        f"prefix; {int(torch.unique(sel).numel())} distinct tokens\n"
        f"arm A bf16 reads {kv_mib_a:.1f} MiB, resident "
        f"{2 * kc.numel() * 2 / 2**20:.0f} MiB\n"
        f"arm B int8 reads {kv_mib_a / 2:.1f} MiB, resident "
        f"{b_bytes / 2**20:.0f} MiB"
    )

    run_a = functools.partial(
        qsa_ops.qsa_sparse_paged_attention, q_a, kc, vc, li, bt, t2r, False
    )
    orig_cfg = qsa_ops._select_config
    bn, wp, tiles, sp = orig_cfg(args.rows, NUM_KV_HEADS, False, TOPK)
    print(f"arm A shipped config: BLOCK_N={bn} warps={wp} tiles={tiles} splits={sp}")

    def run_b(num_splits):
        ws_o = torch.empty(
            (num_splits, args.rows, NUM_Q_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device="cuda",
        )
        ws_l = torch.empty(
            (num_splits, args.rows, NUM_Q_HEADS), dtype=torch.float32, device="cuda"
        )
        dummy = torch.empty(1, device="cuda")

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
                ws_o,
                ws_l,
                0,
                True,
            ) if num_splits > 1 else torch.ops.pth_flash_attn.pth_flash_attn_fwd(
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
                1,
                1.0 / math.sqrt(HEAD_DIM),
                dummy,
                dummy,
                0,
                True,
            )

        return call

    # Paired blocks: A and B adjacent in time, so both see the same neighbour.
    print("\npaired blocks (us per layer-step, A then B at the same instant)")
    ratios = []
    best_b = {}
    for block in range(4):
        a_us = statistics.median(timeit(run_a, max(20, args.reps // 3)))
        b_us = statistics.median(timeit(run_b(sp), max(20, args.reps // 3)))
        ratios.append(a_us / b_us)
        print(
            f"  block {block}: A {a_us:7.1f}  B(splits={sp}) {b_us:7.1f}  "
            f"A/B {a_us / b_us:5.2f}"
        )

    print("\narm B split sweep")
    for s in [int(x) for x in args.splits.split(",") if x.strip()]:
        us = statistics.median(timeit(run_b(s), max(20, args.reps // 3)))
        best_b[s] = us
        print(f"  splits={s:3d}: {us:7.1f} us")

    print("\narm A split sweep (same forced splits)")
    a_best = {}
    for s in [int(x) for x in args.splits.split(",") if x.strip()]:

        def forced(num_rows, num_kv_heads, use_prefill, num_columns, _s=s):
            return bn, wp, tiles, min(_s, tiles)

        qsa_ops._select_config = forced
        us = statistics.median(timeit(run_a, max(20, args.reps // 3)))
        a_best[s] = us
        print(f"  splits={s:3d}: {us:7.1f} us")
    qsa_ops._select_config = orig_cfg

    calib1 = stream_ceiling(20.0, 200)
    best_s = min(best_b, key=best_b.get)
    best_a = min(a_best, key=a_best.get)
    print(
        json.dumps(
            {
                "contended": contended,
                "ceiling_gbps_start": round(calib0),
                "ceiling_gbps_end": round(calib1),
                "arm_a_best_us": round(a_best[best_a], 1),
                "arm_a_splits": best_a,
                "arm_b_best_us": round(best_b[best_s], 1),
                "arm_b_splits": best_s,
                "speedup_A_over_B": round(a_best[best_a] / best_b[best_s], 2),
                "bytes_ratio": 2.0,
                "note": "ratio is the signal, absolute us are not, "
                f"{'device was contended' if contended else 'device was idle'}",
            },
            indent=2,
        )
    )
    wall = 3 * (a_best[best_a] - best_b[best_s]) * 4 / 1000  # 3 layers/rank, 4 ranks
    print(
        f"if that delta is real: {3 * (a_best[best_a] - best_b[best_s]) / 1000:.3f} ms "
        f"per rank-step x 4 serial ranks = {wall:.3f} ms of a 22.79 ms step"
    )


if __name__ == "__main__":
    main()
