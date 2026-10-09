# Checkpoint build tools

Build a W8A16 dense variant of the official FP8 checkpoint: the 228
main-body dense matrices (QSA q/k/v/o, GDN out_proj, shared experts) move
from bf16 to int8 symmetric group-32. Experts, the PLE n-gram table, and
every sensitive matrix (routers, indexer, GDN in-projections,
hyper-connections, draft layer, lm_head, embeddings, vision) pass through
byte-identical. On Ampere, decode is bandwidth-bound, so the halved dense
bytes buy decode speed; nothing here changes what the model can represent.

Requirements: Python 3.12+, numpy, ~350 GiB free at the working disk plus
read access to `Qwen/Qwen3.8-Flash-Next-FP8`. No GPU, no torch.

## Build

```
python encode_w8a16.py SRC OUT --files a,b,c   # per-batch workers (4 fit an NVMe)
python encode_w8a16.py SRC OUT                 # whole tree, single process
python encode_w8a16.py SRC OUT --assemble-only # config.json + index + aux files
```

`SRC` is the official FP8 directory. The build is resumable: a written
shard that matches its own header is re-derived and skipped.
`ct_config.py` holds the compressed-tensors config template: int8
`pack-quantized` group for the 228, fp8 `float-quantized` block group for
`RoutedExperts` + the n-gram prefix, everything else ignored.
`encode_fp8_dense.py` is the shared helper module (bf16/e4m3 conversions)
and, run directly, the v1 rehearsal encoder that emits fp8-dense instead
of int8. Expect ~6 minutes with four workers and 173 GiB of input.

## Check

```
python resolve_check.py OUT/config.json [OFFICIAL_CONFIG_DIR]   # CPU, needs a vllm tree
python verify_w8a16.py $BATCH                                     # 4 batches, full bytes
```

`resolve_check.py` needs the exact serving vLLM checkout: set `VLLM_TREE`
to a worktree of the base commit from README. It asserts each module
family resolves to the intended scheme: int8 linears to WNA16, MoE and
PLE to float8-block, held and draft modules to unquantized.
`verify_w8a16.py` recomputes every converted tensor from the source and
byte-compares every passthrough tensor: 152,089 checks. Run all four
batches.

## Results from the reference build

Dense rel-RMS: mean 0.64 %, max 0.74 % across the 228. Verification: zero
failures on 152,089 tensors, twice (v1 and v2 layouts). Measured on the
4x-1m lane, two boots per side, against the official FP8 tree: single-stream
decode +3.1 to +5.0 %, prefill (prompt/TTFT) -1.7 to -2.5 %, KV pool
+23,220 tokens, acceptance 3.86-3.95 unchanged. Quality gate
(`benchmark/logprob_capture.py` + `benchmark/quality_diff.py`, captures in
`benchmark/quality/`): top-1 flip rate 8.1 % against a machine
non-reproducibility floor of 7.3 % measured both within and across boots.
The quantization residue is about 0.8 points, roughly one position in 130.
Full tables: [benchmark/results.md](../benchmark/results.md).

## Gotchas

- Target and ignore regexes run against RUNTIME module names:
  `model.layers.N....` after the model's `hf_to_vllm_mapper`, and the
  FUSED names (`self_attn.qkv_proj`, `shared_expert.gate_up_proj`). The
  templates use `.*model\.layers\.` so both spellings match and `mtp.`
  draft layers never do. This cost one boot to learn.
- compressed-tensors MoE scale suffix is `weight_scale`, not the fp8
  method's `weight_scale_inv`. The encoder keeps the ckpt name; the
  rename is header-only (pad to original header length, data never moves).
  The shipped encoder writes `weight_scale` directly.
- Serving this checkpoint needs series patch 10: the PLE method only
  recognized fp8 tables through the Fp8Config gate.
- `pack-quantized` int8 stores `uint8b128` (value + 128), four bytes per
  int32 along the input dim, plus bf16 `weight_scale` [N, K/32] and int64
  `weight_shape` [K, N]. Getting any of that wrong loads garbage silently.
- The official `ple_embedding_dtype` config field is dropped by the
  config parser; do not rely on it. Patch 10 resolves the scheme instead.
- Headers rename in place: after editing, re-check every shard's
  `8 + header_len + max_offset == filesize` and the index against headers.
