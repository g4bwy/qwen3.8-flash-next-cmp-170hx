# Qwen3.8-Flash-Next-FP8 (official checkpoint) on 3x or 4x CMP170HX / PP=3 or PP=4 / MTP-3

Patches over vLLM mainline, running with uv venv.
No bullshit slop-wall-of-text, no docker, no opaque scripts, no nonsense.

- Checkpoint: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (official FP8). The serve scripts pull this tag by default. Override with `MODEL=/local/path`.
- Base commit: `15859bb3a1d81a709b64eff2a1d1f38a958a362b` (vllm-project/vllm main, 2026-09-21).
- 9 patches. After applying them, `git rev-parse HEAD^{tree}` must print
  `d9dbc9743071da65419c9c18f969db58a1117bd9`. If it does not, you applied
  something else or onto something else.
- Nothing here runs without the patches. Upstream refuses PP3+MTP+PLE on this
  checkpoint (drafter asserts on the last rank, PLE rejected across pipeline
  ranks, KV allocation dies with a bare `StopIteration`).
- Current base status: both 1M lanes boot clean and serve requests, at the
  pools the tables below list. The 256k and 512k lanes, and every tok/s number
  here, come from an earlier base.

## Apply

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout 15859bb3a1d81a709b64eff2a1d1f38a958a362b
git am --keep-non-patch /path/to/patchset-qwen38-pp/patches/00*.patch
git rev-parse 'HEAD^{tree}'   # d9dbc9743071da65419c9c18f969db58a1117bd9
```

To redo after editing a patch: `git am --abort` (or `git reset --hard
15859bb3a1`), then re-run the `git am`.

## Build

Python 3.12, `uv` on PATH ([install](https://docs.astral.sh/uv/)).

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=15859bb3a1d81a709b64eff2a1d1f38a958a362b \
  uv pip install -e . --torch-backend=auto
```

The patches are pure Python, so re-applying an updated series onto the same base
needs no rebuild. A new base does, because the wheel is pinned to a commit, and
that commit must satisfy two rules:

1. The base commit must have a published wheel. The install fetches
   `https://wheels.vllm.ai/<sha>/cu130/vllm/metadata.json` and stops on 404, and
   wheel CI lags main by hours, so main's tip usually has nothing yet. Pick the
   newest commit that answers 200.
2. Pass that 40-hex sha in `VLLM_PRECOMPILED_WHEEL_COMMIT`. If you leave it out
   on a detached checkout, the install falls back to the moving `nightly` alias,
   and the compiled code then comes from a commit you did not choose.

The boot banner proves neither of these, because the install writes
`vllm/_version.py` from your local tree, so a patched tree prints its own head.
To see which wheel supplied the `.so` files, read the install line
`Using precompiled wheel commit <sha> with variant cu130`, or look in the cache:

```bash
find ~/.cache/uv -name "vllm-*.whl" 2>/dev/null | head -3
```

Sanity check the tree vLLM actually imports:

```bash
python -c "import vllm; print(vllm.__file__)"   # must be your checkout, not a stale install
```

## Run

Six launch scripts: three context sizes, two GPU sets. Same tree, same port,
one at a time:

| script | gpus | context | RoPE | notes |
|---|---|---|---|---|
| `serve-3x.sh` | 3 | 262,144 | none, native window | the safe one |
| `serve-3x-512k.sh` | 3 | 524,288 | static YaRN 2.0 | needs patch 9 (RoPE forwarding) |
| `serve-3x-1m.sh` | 3 | 1,000,000 | static YaRN 4.0 | vendor recipe |
| `serve-4x.sh` | 4 | 262,144 | none, native window | booted + benched, see 4-card tables |
| `serve-4x-512k.sh` | 4 | 524,288 | static YaRN 2.0 | booted + benched, see 4-card tables |
| `serve-4x-1m.sh` | 4 | 1,000,000 | static YaRN 4.0 | booted + benched, see 4-card tables |

```bash
./serve-3x-512k.sh       # MODEL=/local/path overrides the checkpoint tag
```

All six: MTP-3 spec decode, PLE n-gram table offloaded to pinned host RAM
(`--engram-config '{"cpu_offload": true}'`), prefix caching on, NCCL P2P/IB
off (this box has no P2P between the cards).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` kills allocator-OOM
retries. vLLM only forbids it with KV connectors, and we run none. The 3x
set runs PP=3 with `VLLM_PP_LAYER_PARTITION=16,17,15`. The 4x set runs
PP=4 with `16,12,11,9`: rank 0 keeps the proven 16-layer PLE prefix, and
the last rank stays lightest because the MTP drafter lives on it.

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
depends on), PCIe Gen 2 speed (x4 link width on this board, ~2 GB/s per
direction per card), and full BAR1. There is no other tuning or
overclocking. Each card is power-capped at 200 W (`nvidia-smi -pl 200`,
re-apply after reboot), so every tok/s number below is at 200 W, not at the
silicon ceiling.

Startup facts, the three contexts on the 3-card set (boot logs
`logs/boot-*.log`, `boot-3x-256k.log`, `boot-3x-1m.log`):

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

4-card set, PP=4 (boot logs `boot-4x-256k.log`, `boot-4x-512k.log`,
`boot-4x-1m.log`):

| config | partition | KV pool tokens | concurrency | per-rank KV PP0..PP3 | per-rank load PP0..PP3 |
|---|---|---:|---:|---|---|
| 1,000,000 (YaRN 4.0) | 16,12,11,9 | 1,568,111 | 1.57x | 12.76 / 24.52 / 27.04 / 28.13 GiB | 44.19 / 34.02 / 31.49 / 30.24 GiB |
| 1,000,000 (YaRN 4.0) | 12,12,12,12 | 2,541,795 | 2.54x | 22.88 / 24.31 / 24.31 / 22.29 GiB | 34.08 / 34.02 / 34.02 / 37.82 GiB |
| 524,288 (YaRN 2.0) | 12,12,12,12 | 2,496,752 | 4.76x | 23.14 / 24.72 / 24.72 / 20.79 GiB | 33.83 / 33.77 / 33.77 / 37.57 GiB |
| 262,144 (native) | 12,12,12,12 | 2,367,797 | 9.03x | 23.23 / n/r / n/r / n/r | 33.71 / 33.64 / 33.64 / 37.44 GiB |

- The model has exactly one PLE layer, at decoder index 1. Rank 0 does not
  need 16 layers, and that stranded budget (0.80 GiB per layer against 2.04
  on the mid ranks) capped the pool. The equal `12,12,12,12` split measures
  +62% pool tokens (2.54x concurrency at 1M, against 1.21x on 3 cards). The
  current cap is rank 3: 12 layers plus the MTP drafter's +3.74 GiB of
  weights. Handing it a layer costs the donor rank more than it gains, so
  the equal split is at its practical optimum. The 3-card split predates
  that finding and stays as measured.
- The 262k boot log starts mid-boot, so the per-rank KV figures for PP1..PP3
  were not captured. Pool and loads are complete.
- The draft printed `Using max model len 1000000` next to the target's:
  patch 9 holds at PP=4.
- The `no KV cache group could be identified as the draft model's` and
  `max_num_scheduled_tokens is set to 2048` warnings are pre-existing on
  the 3-card base (they appear in `logs/boot-*.log` too), not PP=4 issues.

### Headline results

Full tables, methodology and reproduce commands:
[benchmark/results.md](benchmark/results.md).
All measured at the 200 W cap, repetition-task prompts, decode 1024 (short
windows measure drafter warmup, not steady decode). Measured 2026-09-15 on
the 3-card box, 2026-09-19 and 2026-09-21 on the 4-card box.

4-card set, PP=4, layer partition 12,12,12,12:

| metric | 262k native | 524k YaRN 2.0 | 1M YaRN 4.0 |
|---|---:|---:|---:|
| KV pool (tokens / x per context) | 2,367,797 / 9.03x | 2,496,752 / 4.76x | 2,541,795 / 2.54x |
| single-stream TTFT, full window | 22.1 s | 46.0 s | 92.2 s |
| prefill peak | 9.9k tok/s | 10.5k tok/s | 10.3k tok/s |
| single-stream decode | 151-165 tok/s | 163-168 tok/s | 161-169 tok/s |
| 4-stream sustained agg, full fill | 405 tok/s (4 at once) | 367 tok/s (4 at once) | 296 tok/s (2-3 at once) |
| resident streams at full fill | 4 of 4 | 4 of 4 | 2-3 of 4 |

3-card box, PP=3, layer partition 16,17,15 (where patches 1-9 were
published):

| metric | 262k native | 524k YaRN 2.0 | 1M YaRN 4.0 |
|---|---:|---:|---:|
| KV pool (tokens / x per context) | 1,150,599 / 4.39x | 1,204,810 / 2.30x | 1,208,978 / 1.21x |
| single-stream TTFT, full window | 28.5 s | 58.9 s | 119.1 s |
| prefill peak | 9.3k tok/s | 9.2k tok/s | 9.2k tok/s |
| single-stream decode | 163-167 tok/s | 163-166 tok/s | 161-165 tok/s |
| 4-stream sustained agg, full fill | 457 tok/s (4 at once) | ~304 tok/s (2 at a time, rest queue) | 163-164 tok/s (1 at a time) |
| resident streams at full fill | 4 of 4 | 2 of 4 | 1 of 4 |

Full-fill 4-stream 524k never batches: the pool holds two of the four
streams, so the extras queue and TTFT lands in serial multiples (see
results.md). On shallow 8k prompts the 3-card box reached 314 agg at 2
streams and 468 at 4, where residency was never the limit.

- Pool decides batch capacity: resident decoding streams = pool tokens /
  per-stream tokens. At 262k that is nine full streams on 4 cards against
  four on 3, at 1M it is 2.54x against 1.21x.
- Prefill under PP paces at the heaviest stage, not the sum, so four even
  12-layer stages run ~20% faster than the 16/17/15 split. Single-stream
  decode is unchanged between the two widths: the 200 W cap binds. Batched
  decode pays for the longer pipeline: full-fill 4-stream decode at 262k
  drops from 457 (PP=3) to 405 (PP=4) aggregate, 8-11% across warm repeats
  of three methods (bench cells, deep plateau 452 vs 405, shallow plateau
  468 vs 431); at 524k the 2-stream cell repeats 304 vs 265.
- MTP acceptance: 3.7-4.0 of the 4.0 ceiling in every single-stream cell at
  any depth, including full 1M positions. A few short-window 2-stream cells
  read 3.25-3.3 because the window still holds drafter warmup. Acceptance
  decaying with depth is the symptom of a missing patch 9.
- Oversubscription never preempts on this build at either pipeline width:
  the scheduler gates admission, extra streams queue, TTFT grows linearly,
  and preemptions stayed 0 in every 3-card and 4-card concurrency cell.

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
- The serve scripts are base-sensitive. The `--hf-overrides` lines use the YaRN
  rule from #56446: the limit is `max_position_embeddings` itself. On a base
  before #56446, drop that key, because the old rule computed
  `original x factor`.
- `--hf-overrides` for RoPE must nest under `text_config`. The flat form the
  model card shows writes an attribute nothing reads on this multimodal
  checkpoint, and the server runs long with unscaled RoPE. Nesting merges per
  key, so `mrope_section` survives and vLLM builds `MRotaryEmbedding` with
  yarn scaling instead of a plain YaRN module.
- On 1M, transformers warns that the explicit YaRN factor 4.0 does not match
  the implicit ratio 3.81. Nothing is wrong, since 1,000,000 is not
  4 x 262,144 (1,048,576) and the explicit factor wins. The 512k lane stays
  quiet, because 524,288 = 2 x 262,144 exactly.
- The first MTP decode step compiles `_fill_num_accepted_kernel` and stalls. If
  you measure throughput, drop the first cell of a sweep.
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
  serve-3x.sh                            3 gpus, 262,144 native
  serve-3x-512k.sh                       3 gpus, 524,288, YaRN 2.0
  serve-3x-1m.sh                         3 gpus, 1,000,000, YaRN 4.0
  serve-4x.sh                            4 gpus, 262,144 native
  serve-4x-512k.sh                       4 gpus, 524,288, YaRN 2.0, benched
  serve-4x-1m.sh                         4 gpus, 1,000,000, YaRN 4.0, benched
  README.md                              this file
```
