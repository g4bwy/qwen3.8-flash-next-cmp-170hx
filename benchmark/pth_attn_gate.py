#!/usr/bin/env python
"""Run the pth-flash-attn int8 attention op on sm_80 and check it is correct.

Why this exists. Nobody has ever executed this kernel on sm_80. Their own
docs/benchmarks.md:618 says only compute capability 8.6 was measured, and sm80 and
sm89 are build-and-CPU-gate only. So step one is proof it runs here at all.

Step two is the question that decides whether it can serve QSA. Their op has no
per-query index argument, only a positional block table (docs/integration.md:56-63)
and a bottom-right causal mask, "positions j <= limit(r) = seq_len - q_len + tok"
(csrc/pth_attn/mask.cuh:8). Position enters no other way: there is no rotary or
sin/cos term anywhere in mask.cuh, attn_kernel.cuh or cache_geometry.cuh, because
K is cached already rotated. So a block table that lists only the selected pages,
with seq_lens set to the selected count, is a legal encoding of a selected set, as
long as the selection granularity is their page size.

Our selector picks individual tokens (indexer_budget 2048, and it selects tokens,
not pages). Their geometry is `j = block_table[r][i]`, `o = i % BS`, so with BS=1 the
block table becomes a plain per-token list and the selection is expressible exactly,
with no kernel edit. That is what the bs=1 case proves or refutes. Their own planner
only ever emits BS >= 8 (python/pth_flash_attn/planner.py:350), so the small pages
are found here by experiment, which is what their docs ask for.

    PYTHONPATH=/home/gab/test/pth-build/python /home/gab/test/vllm/.venv/bin/python \
        benchmark/pth_attn_gate.py

Correctness needs a few kernel launches and about 2 MiB, so it is safe beside the
running server. Timing is not: use --bench, which refuses a busy card.
"""

import argparse
import json
import math
import os
import subprocess

import torch

SO = "/home/gab/test/pth-build/python/pth_flash_attn/_C.abi3.so"

# Our serving shape, from the checkpoint config.
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 12, and 12 <= 16*wm so wm=1 is legal
TOKEN_TOPK = 2048  # indexer_budget
COMPRESS_RATIO = 4  # a selection entry is 4 consecutive tokens


def quantize_rows(x):
    """Per (token, head) int8 quantisation, their documented vLLM-writer mirror."""
    xf = x.float()
    s = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    q = xf * (1.0 / s).unsqueeze(-1)
    q = torch.where(q >= 0, q + 0.5, q - 0.5).clamp(-128.0, 127.0)
    return q.to(torch.int8), s


def reference(q, k_deq, v_deq, q_len):
    """fp32 attention over the already dequantized cache rows, in table order.

    q [Tq, Hq, D], k_deq/v_deq [N, Hkv, D] where N is the visible token count.
    """
    kf = k_deq.float().repeat_interleave(GROUP, dim=1).transpose(0, 1)  # [Hq, N, D]
    vf = v_deq.float().repeat_interleave(GROUP, dim=1).transpose(0, 1)
    qf = q.float().transpose(0, 1)  # [Hq, Tq, D]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * scale
    if q_len > 1:  # bottom-right causal: query row t sees j <= N - q_len + t
        n = scores.shape[-1]
        lim = n - q_len + torch.arange(q_len, device=q.device)
        mask = torch.arange(n, device=q.device)[None, :] <= lim[:, None]
        scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs, vf).transpose(0, 1).to(torch.bfloat16)


def run_case(page_tokens, visible, q_len, label):
    """One call: build a cache whose pages hold `visible` tokens, attend all of them."""
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(11)
    num_blocks = max(1, (visible + page_tokens - 1) // page_tokens)
    kv = torch.randn(
        (num_blocks * page_tokens, NUM_KV_HEADS, HEAD_DIM), generator=g
    )  # the true rows, in the order the table will present them
    k_int8, k_scale = quantize_rows(kv)
    v_int8, v_scale = quantize_rows(kv + 0.1)
    k_int8 = k_int8.view(num_blocks, page_tokens, NUM_KV_HEADS, HEAD_DIM)
    v_int8 = v_int8.view(num_blocks, page_tokens, NUM_KV_HEADS, HEAD_DIM)
    k_scale = k_scale.view(num_blocks, page_tokens, NUM_KV_HEADS)
    v_scale = v_scale.view(num_blocks, page_tokens, NUM_KV_HEADS)

    # Scattered physical page ids, so a positional assumption would show up as error.
    order = torch.randperm(num_blocks)
    k_cache = k_int8[order].contiguous().to(device)
    v_cache = v_int8[order].contiguous().to(device)
    k_sc = k_scale[order].contiguous().to(device).float()
    v_sc = v_scale[order].contiguous().to(device).float()
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(num_blocks)
    block_table = inverse.to(torch.int32)[None].contiguous().to(device)

    seq_lens = torch.tensor([visible], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    q = torch.randn((q_len, NUM_Q_HEADS, HEAD_DIM), generator=g).to(
        dtype=torch.bfloat16, device=device
    )
    out = torch.empty_like(q)
    ws = torch.empty(1, device=device)  # unused with num_splits=1

    torch.ops.pth_flash_attn.pth_flash_attn_fwd(
        out,
        q,
        k_cache,
        v_cache,
        k_sc,
        v_sc,
        block_table,
        seq_lens,
        query_start_loc,
        q_len,  # max_query_len
        1,  # wm: GROUP=12 <= MMA_ROWS_PER_WARP*1
        1,  # num_splits
        1.0 / math.sqrt(HEAD_DIM),
        ws,
        ws,
        0,  # window
        True,  # causal
    )
    torch.cuda.synchronize()

    # Reference in logical token order, so dequantize the pre-permutation arrays:
    # physical page p holds logical block order[p], and block_table maps
    # logical j back to inverse[j], which round-trips to kv row j.
    n_rows = num_blocks * page_tokens
    k_deq = k_int8.float().reshape(
        n_rows, NUM_KV_HEADS, HEAD_DIM
    ) * k_scale.float().reshape(n_rows, NUM_KV_HEADS, 1)
    v_deq = v_int8.float().reshape(
        n_rows, NUM_KV_HEADS, HEAD_DIM
    ) * v_scale.float().reshape(n_rows, NUM_KV_HEADS, 1)
    ref = reference(q.cpu(), k_deq[:visible], v_deq[:visible], q_len)

    err = (out.float().cpu() - ref.float()).abs().max().item()
    rel = err / max(ref.float().abs().max().item(), 1e-6)
    print(
        f"{label}: BS={page_tokens:4d} visible={visible:5d} q_len={q_len} -> "
        f"max_abs {err:.4f}  rel {100 * rel:.2f}%  {'PASS' if rel < 0.02 else 'FAIL'}"
    )
    return rel < 0.02


def graph_capture_case():
    """Capture the op in a CUDA graph and replay it, which our decode path requires.

    The lane runs -cc.cudagraph_mode=FULL_AND_PIECEWISE, so every decode step is a
    graph replay. A kernel that allocates, synchronises, or reads a host scalar at
    capture time cannot serve there. Their docs claim graph-safety because vLLM
    adaptive decode depth depends on it, so verify it rather than trust it.
    """
    device = "cuda"
    g = torch.Generator(device="cpu").manual_seed(23)
    bs, vis, hkv = 16, 4096, NUM_KV_HEADS
    nb = vis // bs
    kv = torch.randn((vis, hkv, HEAD_DIM), generator=g)

    def quant(x):
        xf = x.float()
        sc = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
        qq = xf * (1.0 / sc).unsqueeze(-1)
        return torch.where(qq >= 0, qq + 0.5, qq - 0.5).clamp(-128, 127).to(
            torch.int8
        ), sc.float()

    k8, ks = quant(kv)
    v8, vs = quant(kv + 0.1)
    args = (
        torch.empty((1, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device),
        k8.view(nb, bs, hkv, HEAD_DIM).contiguous().to(device),
        v8.view(nb, bs, hkv, HEAD_DIM).contiguous().to(device),
        ks.view(nb, bs, hkv).to(device),
        vs.view(nb, bs, hkv).to(device),
        torch.arange(nb, dtype=torch.int32)[None].contiguous().to(device),
        torch.tensor([vis], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    q = (
        torch.randn((1, NUM_Q_HEADS, HEAD_DIM), generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    args = (args[0], q) + args[1:]
    ws = torch.empty(1, device=device)
    fwd = torch.ops.pth_flash_attn.pth_flash_attn_fwd
    for _ in range(3):
        fwd(*args, 1, 1, 1, 1.0 / math.sqrt(HEAD_DIM), ws, ws, 0, True)
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fwd(*args, 1, 1, 1, 1.0 / math.sqrt(HEAD_DIM), ws, ws, 0, True)
    except Exception as e:
        print(f"graph capture: REJECTED {type(e).__name__}: {e}")
        return False
    # Replay with new inputs in the same buffers, the way a decode step does.
    q.normal_()
    eager = torch.empty_like(q)
    fwd(eager, *args[1:], 1, 1, 1, 1.0 / math.sqrt(HEAD_DIM), ws, ws, 0, True)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    err = (args[0].float() - eager.float()).abs().max().item()
    print(
        f"graph capture: PASS, replay matches eager to {err:.4f} "
        f"({'ok' if err < 2e-2 else 'MISMATCH'})"
    )
    return err < 2e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="time instead of checking")
    ap.add_argument("--max-util", type=int, default=5)
    args = ap.parse_args()

    assert os.path.exists(SO), f"missing {SO}, build it first"
    torch.ops.load_library(SO)
    props = torch.cuda.get_device_properties(0)
    print(
        f"loaded {os.path.basename(SO)} on {props.name} "
        f"sm_{props.major}{props.minor} {props.multi_processor_count} SMs, "
        f"free {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB"
    )

    if args.bench:
        util = int(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()[0]
        )
        if util > args.max_util:
            raise SystemExit(
                f"device at {util}% util, timing would measure the neighbor"
            )

    ok = []
    # Small shapes first: does any launch work on sm_80 at all.
    ok.append(run_case(16, 16, 1, "decode  "))
    # Their documented odd page size, to prove BS really is runtime.
    ok.append(run_case(816, 816, 1, "odd BS  "))
    # BS=1 turns block_table into a per-token list, the sparse selector's exact
    # encoding with no kernel edit (mask.cuh:8 leaves position free, and the
    # writer never assumes slot == logical offset). Their planner warns BS >= 8,
    # so find empirically where per-token paging stops working.
    ok.append(run_case(1, TOKEN_TOPK, 1, "bs=1    "))
    ok.append(run_case(2, TOKEN_TOPK, 1, "bs=2    "))
    ok.append(run_case(COMPRESS_RATIO, TOKEN_TOPK, 1, "bs=4=qgrp "))
    ok.append(run_case(8, TOKEN_TOPK, 1, "bs=8    "))
    # Their vLLM floor of 16 tokens per page over the same budget.
    ok.append(run_case(16, TOKEN_TOPK, 1, "bs=16   "))
    # MTP verify length, exercising the bottom-right causal arithmetic.
    ok.append(run_case(16, 4096, 5, "verify 5"))

    ok.append(graph_capture_case())
    print(json.dumps({"all_pass": all(ok), "cases": len(ok)}))
    if not all(ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
