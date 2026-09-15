# Qwen3.8-Flash-Next-FP8 (official checkpoint) on 3x CMP170HX / PP=3 / MTP-3

Patches over vLLM mainline, running with uv venv.
No bullshit slop-wall-of-text, no docker, no opaque scripts, no nonsense.

- Checkpoint: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (official FP8). The serve scripts pull this tag by default. Override with `MODEL=/local/path`.
- Base commit: `995e8581f462a13e32f30cfba946c63d48cf31d7` (vllm-project/vllm main, 2026-09-14)
- 9 patches. After applying them, `git rev-parse HEAD^{tree}` must print
  `d694519b6d8b12499286f788941d6a1e15c645fa`. If it does not, you applied
  something else or onto something else.
- Nothing here runs without the patches. Upstream refuses PP3+MTP+PLE on this
  checkpoint (drafter asserts on the last rank, PLE rejected across pipeline
  ranks, KV allocation dies with a bare `StopIteration`).

## Apply

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout 995e8581f462a13e32f30cfba946c63d48cf31d7
git am --keep-non-patch /path/to/patchset-qwen38-pp/patches/00*.patch
git rev-parse 'HEAD^{tree}'   # d694519b6d8b12499286f788941d6a1e15c645fa
```

To redo after editing a patch: `git am --abort` (or `git reset --hard
995e8581f4`), then re-run the `git am`.

## Build

Python 3.12, `uv` on PATH ([install](https://docs.astral.sh/uv/)).

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

The patches are pure Python. Re-applying an updated series onto the same base
commit needs no rebuild. When you move to a different base commit, re-run the
install line, because the precompiled wheel is pinned to a commit.

Sanity check the tree vLLM actually imports:

```bash
python -c "import vllm; print(vllm.__file__)"   # must be your checkout, not a stale install
```

## Run

Three servers, same tree, same port (one at a time):

| script | context | RoPE | notes |
|---|---|---|---|
| `serve.sh` | 262,144 | none, native window | the safe one |
| `serve-512k.sh` | 524,288 | static YaRN 2.0 | needs patch 9 (RoPE forwarding) |
| `serve-1m.sh` | 1,000,000 | static YaRN 4.0 | vendor recipe |

```bash
./serve-512k.sh          # MODEL=/local/path overrides the checkpoint tag
```

All three: PP3 (`VLLM_PP_LAYER_PARTITION=16,17,15`), MTP-3 spec decode, PLE
n-gram table offloaded to pinned host RAM (`--engram-config
'{"cpu_offload": true}'`), prefix caching on, NCCL P2P/IB off (this box has no
P2P between the cards). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
kills allocator-OOM retries. vLLM only forbids it with KV connectors, and we
run none.

Startup checks for the YaRN lanes:

1. `Using max model len <N>` appears twice, once for the target and once for
   the `Qwen4ExpMTP` draft. A second, smaller number means patch 9 is missing,
   and MTP acceptance rots with context depth.
2. The log prints `Maximum concurrency for <N> tokens per request: X.XXx`.
   Below 1.0 the box cannot serve its own context length, so lower
   `--max-model-len`.

## Measured, this box, FP8 checkpoint, single stream

The cards are CMP 170HX (GA100) running unlocked. The
[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
patched nvidia-open kernel module restores what the CMP firmware restricts:
full SM compute, the full HBM2e geometry (the 64 GB per card this setup
depends on), PCIe Gen 2 speed, and full BAR1. There is no other tuning or
overclocking. Each card is power-capped at 200 W (`nvidia-smi -pl 200`,
re-apply after reboot), so every tok/s number below is at 200 W, not at the
silicon ceiling.

Startup facts, all three configs on base 995e8581f4 + this 9-patch series
(boot logs `logs/boot-256k.log`, `logs/boot-512k.log`, `logs/boot-1m.log`, 2026-09-15):

| config | KV pool tokens | concurrency | available KV/rank | model load PP0/PP1/PP2 |
|---|---:|---:|---:|---|
| 262,144 | 1,150,599 | 4.39x | 13.23 GiB | 43.81 / 46.27 / 45.02 GiB |
| 524,288 (YaRN 2.0) | 1,204,810 | 2.30x | 13.03 GiB | 43.94 / 46.39 / 45.14 GiB |
| 1,000,000 (YaRN 4.0) | 1,208,978 | 1.21x | 12.76 GiB | 44.19 / 46.64 / 45.39 GiB |

- Each YaRN step costs exactly +0.25 GiB/rank of model load. The cos/sin cache
  is `4 x 262144 x factor` fp32 rows, allocated before profiling, so it comes
  out of the KV pool.
- The 7 `expandable_segments: memory mapping failed` warnings per boot all
  fall inside CUDA graph capture, zero after `Application startup complete`.
- At 1M, one full-context request fits with ~200k tokens of pool slack. A
  second one does not.

### Headline results

Full tables, methodology and reproduce commands:
[benchmark/results.md](benchmark/results.md).
All measured 2026-09-15 at the 200 W cap, repetition-task prompts, decode
1024 for the decode numbers (short windows measure drafter warmup, not
steady decode).

- Prefill: 9.2-9.3k tok/s peak on all three context configs. The decay
  tracks prompt depth, not RoPE scaling: 9.1k at 259k tokens, 8.8k at 519k,
  8.3k at 989k. Full-window TTFT: 28.5 / 58.9 / 119.1 s.
- Single-stream decode: flat 163-166 tok/s from 26k to 989k prompt depth.
- MTP acceptance: 3.9-4.0 of the 4.0 ceiling at every depth and stream
  count, including full 1M positions. This is the column that rots with
  depth when patch 9 is missing.
- Batched decode: 314 tok/s at 2 streams and 468 at 4 on shallow prompts
  (1.8x and 2.7x aggregate). At deep prompts the KV pool decides: effective
  concurrent decoding streams = pool tokens / per-stream tokens. At 1M
  (pool 1.209M) two streams share the box only below ~60% fill, and four
  only at 25% fill.
- Oversubscription never preempts on this build: the scheduler gates
  admission, extra streams queue, TTFT grows linearly, zero preemptions in
  all 33 measured concurrency cells.

## Bench scripts

- `benchmark/bench.py`: fill sweep with concurrency. Each (fill, N) cell fires
  N simultaneous greedy streams and diffs `vllm:spec_decode_*` and preemption
  counters from `/metrics`. Reports TTFT mean/p95, aggregate in/out tok/s,
  window-average per-stream decode, sampled sustained decode, and acceptance.
  Flags: `--fills`, `--concurrency`, `--json`.
- `benchmark/sustained_decode.py`: batched-decode capacity. N streams, short
  prompts, long decode, `/metrics` sampled at 0.5 s. Prints the aggregate and
  per-stream plateau.
- `benchmark/mtp_ab_probe.py`: A/B tool for spec-decode acceptance at one
  depth. `--task copy` plants a marker mid-prompt so acceptance stays below
  the ceiling. Run once per build, then `--compare a.json b.json`. It is also
  the shared prompt and metrics library for the other two tools.

## Patches

| # | does |
|---|---|
| 1 | Qwen4Exp MTP forward branches on `intermediate_tensors is None`, not on PP rank, so the last-rank drafter takes the embedding path (upstream #46994 did this for deepseek/qwen3_5 only) |
| 2 | allow PLE when the n-gram layers are on rank 0 under PP |
| 3 | V2: resolve deferred mamba state copies by request slot |
| 4 | mamba: seed the align state column with the real mamba block size |
| 5 | mamba: honor `drop_eagle_block` in MambaManager (MTP + prefix cache) |
| 6 | per-stage KV cache budgets for heterogeneous pipelines |
| 7 | Qwen4Exp: gather pinned PLE rows as raw bytes (Ampere has no `fp8e4nv`) |
| 8 | core: build KV cache tensors from a group's projected layers (fixes `StopIteration` on PP ranks holding no PLE) |
| 9 | let `--hf-overrides` RoPE scaling reach the MTP draft config; mirror `max_position_embeddings` onto the Qwen4Exp wrapper so RoPE standardization can run on it |

## Gotchas

- The PLE table is pinned in host RAM as fp8 (startup log lines
  `float8_e4m3fn`, `pinned=True`). It is several GB, so measure it on your box
  before sizing other host workloads.
- The serve scripts are base-sensitive. The `--hf-overrides` lines match
  995e8581f4, where the YaRN limit rule is `max_position_embeddings` itself
  (#56446). On earlier bases drop that key, because the old rule computed
  `original x factor`.
- `--hf-overrides` for RoPE must nest under `text_config`. The flat form the
  model card shows writes an attribute nothing reads on this multimodal
  checkpoint, and the server runs long with unscaled RoPE. Nesting merges per
  key, so `mrope_section` survives and vLLM builds `MRotaryEmbedding` with
  yarn scaling instead of a plain YaRN module.
- One 1M request nearly exhausts the pool (1.21x). A second long request
  queues behind it, and if both grow past the pool the scheduler preempts
  one, which replays its whole prompt under Mamba align mode, ~2 min. Run 1M
  on its own or keep the other requests short.
- A long prefill that arrives during decode starves every running decode to
  ~1 step/s (under 20 tok/s) until it finishes. Schedule long prefills early,
  not in the middle of decode traffic.
- bench fill 100% targets (ctx - decode) x 0.99. An oversized prompt is a
  plain HTTP 400 from the server.

## Layout

```
patchset-qwen38-pp/
  patches/0001-*.patch .. 0009-*.patch   the series, git am
  benchmark/results.md                   full benchmark tables and how to reproduce
  benchmark/bench.py                     fill sweep x concurrency -> table
  benchmark/sustained_decode.py          batched-decode capacity -> table
  benchmark/mtp_ab_probe.py              acceptance A/B + shared library
  benchmark/*.json                       raw results behind the tables in results.md
  serve.sh                               262,144 native
  serve-512k.sh                          524,288, YaRN 2.0
  serve-1m.sh                            1,000,000, YaRN 4.0
  README.md                              this file
```
