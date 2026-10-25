# Checkpoint artifacts

Two builders live here. Both write into a model directory and both
are described at full length below: the draft-vocab pair that feeds
patch 5, and the W8A16 dense tree builder that turns the official FP8
release into an int8-dense variant.

## Draft vocabulary head (patch 5)

Patch 8 makes the MTP drafter score a small head instead of the 248k-row
lm_head per draft step, worth ~9% single-stream. Measured history: 4-stream
deep-fill cells read acceptance 2.2-3.0, first blamed on batch size. The
follow-up (results.md, joining-window section) shows the loss lives in
mixed prefill+decode windows: same batch 4, long decode tail, acceptance
at ceiling (3.98-3.99). Patch 8 v3 keeps both heads: the eager window path drafts with
the full head, graphed uniform decode rounds replay the small one,
so steady multi-stream decode keeps the +9%. It engages only when the
checkpoint directory carries the artifacts, and they take two commands:

1. Build the id list from the model's own outputs. Any corpus of assistant
   text works. On this setup the agent session logs are already that corpus:
   [maki](https://github.com/tontinton/maki), the terminal coding agent
   driving this patchset, keeps JSONL transcripts of every session under
   `~/.local/state/maki/sessions/`, and its replies are the model's own text.
   Another agent harness? Port `extract_maki_corpus.py` to wherever it keeps
   its transcripts, the output contract is one JSON object per line with the
   assistant text under `"output"` (the builder also accepts `text` and
   `content`, so existing JSONL exports may plug in directly):

   ```bash
   ./checkpoint/extract_maki_corpus.py --out draft_corpus.jsonl
   ```

   Copy the JSONL to the serve box. Coverage rule: the list must cover
   ~97% of what the model emits, counted over its own generations. A
   borrowed frequency list covering 92% costs back more in rejections than
   the head saves.
2. Build the artifacts in place (adds ~200 MB to the model dir and one
   index entry, applied AFTER patch 5 or draft weight loading fails):

   ```bash
   ./checkpoint/build_draft_vocab.py \
       --model-path /path/to/Qwen3.8-Flash-Next-FP8 \
       --corpus draft_corpus.jsonl --size 40000
   ```

Startup proof: `MTP drafter scores a 39532-token draft vocabulary`.
Rollback: delete `mtp_draft_vocab_ids.pt`, `mtp_draft_lm_head.safetensors`,
and their index entry. Patch 9's page-cache drop logs its own line,
`Dropped consumed checkpoint pages from the page cache: 132 files, 172.97 GiB`.

## W8A16 dense checkpoint

Build a W8A16 dense variant of the official FP8 checkpoint: the 240 main-body
dense matrices (QSA q/k/v/o, the indexer projection, GDN out_proj, shared experts)
move from bf16 to int8 symmetric group-32. The indexer joined on 2026-10-24, when
upstream folded it into the fused qkv_proj, and a fused module must carry one
scheme across all its shards. Experts, the PLE n-gram table, and the remaining
sensitive matrices (routers, GDN in-projections, hyper-connections, draft layer,
lm_head, embeddings, vision) pass through byte-identical. On Ampere decode is
bandwidth-bound, which is what this aims at, but the runtime W8A8 dense arm of
series patches 12 and 13 gets there better on this box, 22.79 ms per step against
this lane's 23.40 ms, and needs no separate checkpoint. Nothing here changes what
the model can represent.

Requirements: Python 3.12+, numpy, ~350 GiB free at the working disk plus
read access to `Qwen/Qwen3.8-Flash-Next-FP8`. No GPU, no torch.

### Build

```
python encode_w8a16.py SRC OUT --files a,b,c   # per-batch workers (4 fit an NVMe)
python encode_w8a16.py SRC OUT                 # whole tree, single process
python encode_w8a16.py SRC OUT --assemble-only # config.json + index + aux files
```

`SRC` is the official FP8 directory. The build is resumable: a written
shard that matches its own header is re-derived and skipped.
`ct_config.py` holds the compressed-tensors config template: int8
`pack-quantized` group for the 240, fp8 `float-quantized` block group for
`RoutedExperts` + the n-gram prefix, everything else ignored.
`encode_fp8_dense.py` is the shared helper module (bf16/e4m3 conversions)
and, run directly, the v1 rehearsal encoder that emits fp8-dense instead
of int8. Expect ~6 minutes with four workers and 173 GiB of input.

### Check

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

### Results from the reference build

Dense rel-RMS: mean 0.64 %, max 0.74 % across the 228, measured on the reference
build before the indexer fold, the 240-module set has not been re-scored.
Verification: zero failures on 152,089 tensors, twice (v1 and v2 layouts). Measured
on the 4x-1m lane, two boots per side, against the official FP8 tree at bf16 KV:
single-stream decode 216-226 tok/s at MTP-4, where the paired +15 % is this lane at
depth 4 over itself at depth 3 and the edge against the official tree at identical
depth was +2 to +6 %. Prefill (prompt/TTFT) -1.7 to -2.5 % at that base, KV pool
+23,220 tokens at depth 3 (+17k net at depth 4 after block rounding), acceptance
unchanged at depth 3. Re-measured 2026-10-24 on the current series against patch 13
the lane is slower per step, pays 10 % of TTFT and 7.7 % of prefill rate, and gains
4.3 % of pool that does not convert to throughput (`benchmark/results.md`). Quality gate
(`benchmark/logprob_capture.py` + `benchmark/quality_diff.py`, captures in
`benchmark/quality/`): top-1 flip rate 8.1 % against a machine
non-reproducibility floor of 7.3 % measured both within and across boots.
The quantization residue is about 0.8 points, roughly one position in 130.
Full tables: [benchmark/results.md](../benchmark/results.md).

### Gotchas

- Target and ignore regexes run against RUNTIME module names:
  `model.layers.N....` after the model's `hf_to_vllm_mapper`, and the
  FUSED names (`self_attn.qkv_proj`, `shared_expert.gate_up_proj`). The
  templates use `.*model\.layers\.` so both spellings match and `mtp.`
  draft layers never do. This cost one boot to learn.
- compressed-tensors MoE scale suffix is `weight_scale`, not the fp8
  method's `weight_scale_inv`. The encoder keeps the ckpt name; the
  rename is header-only (pad to original header length, data never moves).
  The shipped encoder writes `weight_scale` directly.
- Serving this checkpoint needs series patch 7: the PLE method only
  recognized fp8 tables through the Fp8Config gate.
- `pack-quantized` int8 stores `uint8b128` (value + 128), four bytes per
  int32 along the input dim, plus bf16 `weight_scale` [N, K/32] and int64
  `weight_shape` [K, N]. Getting any of that wrong loads garbage silently.
- The official `ple_embedding_dtype` config field is dropped by the
  config parser; do not rely on it. Patch 10 resolves the scheme instead.
- Headers rename in place: after editing, re-check every shard's
  `8 + header_len + max_offset == filesize` and the index against headers.
