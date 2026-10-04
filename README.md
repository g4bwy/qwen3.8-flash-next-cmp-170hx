# Qwen3.8-Flash-Next-FP8 (official checkpoint) on 3x or 4x CMP170HX / PP=3 or PP=4 / MTP-3

Patches over vLLM mainline, running with uv venv.
No bullshit slop-wall-of-text (almost...), no docker, no opaque scripts, no nonsense.

- Checkpoint: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (official FP8). The serve scripts pull this tag by default. Override with `MODEL=/local/path`.
- Base commit: `155488d853a0bc42df227dbfc74005b3fd488e94` (vllm-project/vllm main, 2026-10-04).
- 7 patches: five ours, two adopted from pending upstream PRs (#56444 as
  patch 2, #58094 as patch 7). The adopted two are experimental until their PRs
  merge or the swap is reverted; the 4x-1m hardware test of #56444 is owed.
  After applying them, `git rev-parse HEAD^{tree}` must print
  `c7cd2ff08a2cd639ec392046be9739258d06d89d`. If it does not, you applied
  something else or onto something else.
- Nothing here runs without the patches. Upstream refuses PP3+MTP+PLE on this
  checkpoint (drafter asserts on the last rank, PLE rejected across pipeline
  ranks, KV allocation dies with a bare `StopIteration`).

## Apply

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout 155488d853a0bc42df227dbfc74005b3fd488e94
git am --keep-non-patch /path/to/patchset-qwen38-pp/patches/00*.patch
git rev-parse 'HEAD^{tree}'   # c7cd2ff08a2cd639ec392046be9739258d06d89d
```

To redo after editing a patch: `git am --abort` (or `git reset --hard
155488d853`), then re-run the `git am`.

## Build

Python 3.12, `uv` on PATH ([install](https://docs.astral.sh/uv/)).

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=155488d853a0bc42df227dbfc74005b3fd488e94 \
  uv pip install -e . --torch-backend=auto
```

The patches are pure Python, so re-applying an updated series onto the same base
needs no rebuild. A new base does, because the wheel is pinned to a commit, and
that commit must satisfy two rules:

1. The base must have a published wheel. The install fetches
   `https://wheels.vllm.ai/<sha>/cu130/vllm/metadata.json` and stops on 404.
   Wheel CI lags main by hours, so walk down from the tip and take the newest
   commit that answers 200. The wheel for this base:
   `0.30.1rc1.dev640+g155488d85`, cp38-abi3, x86_64. Match the
   `+g<sha>` tail to your base. Ignore the leading number, which only tracks
   upstream packaging.
2. Pass that 40-hex sha in `VLLM_PRECOMPILED_WHEEL_COMMIT`. Without it, a
   detached checkout falls back to the moving `nightly` alias, and the compiled
   code comes from a commit you did not choose.

The boot banner proves neither, because the install writes `vllm/_version.py`
from your local tree. To see which wheel supplied the `.so` files, read the
install line `Using precompiled wheel commit <sha> with variant cu130`, or look
in the cache:

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

| script | gpus | context | RoPE | pool headroom |
|---|---|---|---|---|
| `serve-3x.sh` | 3 | 262,144 | none, native window | 4.39x, best multi-stream decode |
| `serve-3x-512k.sh` | 3 | 524,288 | static YaRN 2.0 | 2.30x |
| `serve-3x-1m.sh` | 3 | 1,000,000 | static YaRN 4.0 | 1.21x, one request fits |
| `serve-4x.sh` | 4 | 262,144 | none, native window | 9.24x, best prefill and TTFT |
| `serve-4x-512k.sh` | 4 | 524,288 | static YaRN 2.0 | 4.87x |
| `serve-4x-1m.sh` | 4 | 1,000,000 | static YaRN 4.0 | 2.60x |

Headroom is the measured KV pool divided by one full-context request, so it
counts how many such requests fit at once. `--max-num-seqs 8` limits admission
below it on the 4x 262k row. Choose PP=3 when streams decode side by side. It
pays one fewer hop per step and holds 457 against 405 tok/s at four streams on
262k. Choose PP=4 for long prompts, where prefill reaches 9.9k against 9.3k
tok/s and time to first token drops from 28.5 s to 22.1 s.

```bash
./serve-4x-1m.sh       # MODEL=/local/path overrides the checkpoint tag
```

All six scripts share these settings:

- MTP-3 spec decode, and the PLE n-gram table offloaded to pinned host RAM
  (`--engram-config '{"cpu_offload": true}'`).
- Prefix caching with `--mamba-cache-mode align` and `--prefix-cache-retention-interval
  16000`, plus `--prefix-match-unit 16` on the three 4-card lanes. The 3-card set
  is historical and does not carry the match unit, so its reuse numbers still
  reflect the 1,600-token floor. See Prefix caching.
- NCCL P2P and IB off, because this box has no P2P between the cards.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which stops allocator-OOM
  retries. vLLM forbids it only with KV connectors, and we run none.
- Layer partitions `13,12,12,11` on all three 4-card lanes, and `16,17,15` on
  the 3-card scripts. The 3-card set is kept for historical comparison and is
  not tuned further. See the 4-card notes below.

Startup checks for the YaRN lanes:

1. `Using max model len <N>` appears twice, once for the target and once for
   the `Qwen4ExpMTP` draft. A second, smaller number means patch 7 is missing,
   and MTP acceptance rots with context depth.
2. The log prints `Maximum concurrency for <N> tokens per request: X.XXx`.
   Below 1.0 the box cannot serve its own context length, so lower
   `--max-model-len`.

## Prefix caching

This model mixes linear and full attention, so vLLM runs the Mamba cache in `align`
mode: a later request resumes from a retained state snapshot. Two knobs control
reuse, and they are independent. `--prefix-cache-retention-interval` sets how often
states are stored. `--prefix-match-unit` sets the token boundaries a hit can land
on. vLLM ships both badly defaulted for this model.

**Retention.** The interval defaults to `0`, which keeps almost nothing, and a
growing session then re-prefills most of its history each turn. The scripts set
16,000. A 120,000-token prompt, then the same prompt plus two characters, 4-card
1M lane, before the match unit existed:

| interval | send | cached | recomputed | wall |
|---|---|---:|---:|---:|
| 0 (vLLM default) | 120,000 first | 0 | 120,000 | 10.40 s |
| 0 (vLLM default) | 120,001, plus two chars | 57,600 | 62,401 | 6.28 s |
| 16,000 | 120,000 first | 32,000 | 88,000 | 7.64 s |
| 16,000 | 120,000 again | 96,000 | 24,000 | 2.57 s |
| 16,000 | 120,001, plus two chars | 112,000 | 8,001 | 1.24 s |

**Match unit.** The default is the GCD of the cacheable KV group block sizes, 1,600
tokens here, so no hit could land finer than 1,600. That made prompts under 1,600
dead and left one-shot reuse shallower than the retention grid predicted. The
number came from the fork wtdcode/vllm-backport, which calls the flag effectively
mandatory. `--prefix-match-unit 16` (16 divides the block sizes; 64 is rejected)
drops the floor to 32 tokens with the MTP block drop. Same lane, probe lengths,
after the flag:

| send | cached | recomputed | hit |
|---|---:|---:|---:|
| 9,990 first, cold cache | 0 | 9,990 | 0% |
| 9,990 identical prompt again | 9,968 | 22 | 99.8% |
| 9,991 plus two chars | 9,968 | 23 | 99.8% |
| 22,500 identical prompt again | 22,480 | 20 | 99.9% |
| 22,501 plus two chars | 22,480 | 21 | 99.9% |

- Repeats and appends now hit at 99.8-99.9% at any length above the floor, below
  one retention interval included. The pool line is untouched: granularity changes
  hashing, not allocation. Decode is unchanged (single stream 165.6-165.8 tok/s).
- A grown turn resumes from the previous turn's end junction while that snapshot
  survives pool pressure, recomputing about 20 tokens. When it does not survive,
  the turn falls back to a coarser stored snapshot, about 3,100 tokens, or 0.3 s.
  Worst case is therefore far below the 8,001-token cost of the retention-only fix.
- Dense per-block retention, as the fork pairs with the flag for its LMCache tier,
  is unnecessary without one and unmeasured here.
- Chunk size is still not the lever. `--max-num-batched-tokens 9600` moved no reuse
  number and cost 10% of the pool, so the scripts do not set it.
- `enable_mamba_shared_prefix_checkpoint` stays off and unmeasured. The flag now
  satisfies its stated prefix-unit precondition.

Measure reuse with `./benchmark/prefix_cache_probe.py --lengths 64000`. That counts
tokens, so it cannot say whether the resumed state is right.
`benchmark/prefix_state_check.py` does. It hides two hex needles, one in the first
block and one at half depth, then compares cold, warm and grown answers. A second
phase runs several sessions at once. All lanes pass, and the 4-card 1M lane passed
again with the match unit set: warm 20k rows recompute 17-18 tokens with both
needles intact, and a 4-session round fell from 82,978 recomputed tokens cold to
1,682 warm, zero cross-session leakage.

## Measured, this box, FP8 checkpoint, single stream
The cards are CMP 170HX (GA100). They run unlocked with the
[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) module,
which restores SM compute, the 64 GB HBM2e geometry, PCIe Gen 2 speed and BAR1.
The link is x4 here, about 2 GB/s per card in each direction. Nothing else is
tuned. Each card is capped at 200 W (`nvidia-smi -pl 200`, re-apply after
reboot), so every tok/s figure here is a 200 W figure. Full tables, methodology
and reproduce commands: [benchmark/results.md](benchmark/results.md). Numbers use
repetition-task prompts at decode 1024, because a short window measures drafter
warmup rather than steady decode.

### Pool and weights at boot

| lane | context | KV pool | headroom | weights per rank (GiB) | KV per rank (GiB) |
|---|---|---:|---:|---|---|
| 3 cards | 262,144 | 1,150,599 | 4.39x | 43.81 / 46.27 / 45.02 | 13.23 |
| 3 cards | 524,288 YaRN 2.0 | 1,204,810 | 2.30x | 43.94 / 46.39 / 45.14 | 13.03 |
| 3 cards | 1M YaRN 4.0 | 1,208,978 | 1.21x | 44.19 / 46.64 / 45.39 | 12.76 |
| 4 cards | 262,144 | 2,423,060 | 9.24x | 36.88 / 34.21 / 34.21 / 35.46 | 20.89 / 24.99 / 24.99 / 23.65 |
| 4 cards | 524,288 YaRN 2.0 | 2,550,833 | 4.87x | 37.02 / 34.41 / 34.41 / 35.68 | 20.73 / 24.77 / 24.77 / 23.42 |
| 4 cards | 1M YaRN 4.0 | 2,595,975 | 2.60x | 36.61 / 34.02 / 34.02 / 35.29 | 20.48 / 24.52 / 24.52 / 23.15 |

The 3-card column is one value because the pipeline splits there differ by only a
few blocks. Each YaRN step adds 0.25 GiB per rank of weights. The cos/sin cache is
`4 x 262144 x factor` fp32 rows, allocated before profiling, so it comes out of the
KV pool.

### Throughput

Rates are tok/s unless the cell shows a duration.

| lane | TTFT, full window | prefill peak | 1 stream | 4 streams, full fill |
|---|---:|---:|---:|---|
| 3 cards, 262k | 28.5 s | 9.3k | 163-167 | 457, 4 of 4 resident |
| 3 cards, 524k | 58.9 s | 9.2k | 163-166 | ~304, 2 of 4 |
| 3 cards, 1M | 119.1 s | 9.2k | 161-165 | 163-164, 1 of 4 |
| 4 cards, 262k | 22.1 s | 9.9k | 151-165 | 405, 4 of 4 |
| 4 cards, 524k | 46.0 s | 10.5k | 163-168 | 367, 4 of 4 |
| 4 cards, 1M | 92.2 s | 10.3k | 161-169 | 296, 2-3 of 4 |

- Pool sets capacity. At 262k, 4 cards hold nine full streams against four on
  3 cards, which is the whole argument for the wider pipeline.
- Prefill paces at the heaviest stage, not the sum, so four balanced stages beat
  `16,17,15` by about 20%. Single-stream decode barely moves between the two
  widths, because the 200 W cap binds first.
- Batched decode pays for the extra hop. At 262k full fill, four streams fall from
  457 on 3 cards to 405 on 4. That 8-11% gap repeats across three methods: bench
  cells, deep plateau 452 against 405, shallow plateau 468 against 431. At 524k
  with two streams it repeats 304 against 265.
- 524k on 3 cards never batches four streams, because the pool holds two, the rest
  queue, and TTFT lands in serial multiples. On shallow 8k prompts the same box
  reached 468 aggregate at four streams, where residency is not the limit.
- MTP acceptance holds at 3.7-4.0 of the 4.0 ceiling at every depth, including
  full 1M positions. A few short-window cells read 3.25-3.3, because the window
  still holds drafter warmup. Acceptance that decays with depth means patch 7 is
  missing.
- Oversubscription never preempted at either width. The scheduler gates admission,
  extra streams queue, and preemptions stayed 0 in every concurrency cell.

### Why these layer splits

The pool caps at min_r (rank KV bytes / rank layer count). The model has one
PLE layer, at decoder index 1, so rank 0 needs at least two layers, and a layer
costs rank 0 0.80 GiB of KV against 2.04 GiB on a mid rank. All three 4-card
lanes run `13,12,12,11`. Measured on all three lanes, that split beats an equal
`12,12,12,12` by 2.13% of pool at 1M, 2.17% at 524k and 2.33% at 262k. The
extra layer costs
rank 0 2.53 GiB of weights while rank 3, which also carries the MTP drafter,
gets the room back. Rank 0 is now the binding stage, so a second shifted layer
has nowhere useful to go. Prefill, single-stream decode and four-stream decode
did not move outside their boot-to-boot spread. The 3-card set is historical,
so `16,17,15` stands unmeasured by choice.

Two boot warnings are normal and cost you nothing. The 7
`expandable_segments: memory mapping failed` messages fall inside CUDA graph
capture, none after `Application startup complete`. `max_num_scheduled_tokens is
set to 2048` comes from MTP-3.

## Patches

| # | does |
|---|---|
| 1 | Qwen4Exp MTP forward branches on `intermediate_tensors is None` rather than PP rank, so the last-rank drafter takes the embedding path |
| 2 | upstream PR #56444, adopted: carry PLE input ids (int32, +4 B/token/hop) inside the PP intermediate tensors, so PLE is pipeline-rank-free; replaces our rank-0 confinement; its three new tests got process-isolation guards locally |
| 3 | V2: resolve deferred mamba state copies by request slot |
| 4 | mamba: seed the align state column with the real mamba block size |
| 5 | Qwen4Exp: gather pinned PLE rows as raw bytes (Ampere has no `fp8e4nv`) |
| 6 | core: build KV cache tensors from a group's projected layers (fixes `StopIteration` on PP ranks holding no PLE) |
| 7 | upstream PR #58094, adopted: propagate rope `--hf-overrides` to the MTP draft config. Taken with two fixes: `PretrainedConfig` annotations renamed (NameError at import), and it forwards rope-keyed subsets rather than the full same-checkpoint dict our old patch 7 used |

## Gotchas

- The PLE table sits in pinned host RAM as fp8 (`float8_e4m3fn`,
  `pinned=True`). It is several GB, so measure it before sizing other host
  workloads.
- The `--hf-overrides` lines are base-sensitive. Since #56446 the YaRN limit is
  `max_position_embeddings` itself. On an earlier base drop that key, because
  the old rule computed `original x factor`.
- RoPE overrides must nest under `text_config`. The flat form on the model card
  writes an attribute nothing reads on this checkpoint, and the server then runs
  unscaled RoPE at length. Nesting merges per key, so `mrope_section` survives
  and vLLM builds `MRotaryEmbedding` with yarn scaling.
- On 1M, transformers warns that factor 4.0 does not match the implicit ratio
  3.81. Nothing is wrong: 1,000,000 is not 4 x 262,144, and the explicit factor
  wins. The 512k lane stays quiet, because 524,288 = 2 x 262,144 exactly.
- The first MTP decode step compiles `_fill_num_accepted_kernel` and stalls.
  Drop the first cell of any sweep.
- One 1M request leaves about 200k tokens of pool slack on 3 cards. A second
  queues behind it. If both outgrow the pool, the scheduler preempts one, and
  that request replays its whole prompt under Mamba align mode, roughly 2 min.
- A long prefill that lands during decode starves every stream to under
  20 tok/s until it finishes. Schedule long prefills early, not between decodes.
- A bench fill of 100% means (ctx - decode) x 0.99. Anything larger is a plain
  HTTP 400 from the server.

## Layout

```
patchset-qwen38-pp/
  patches/0001-*.patch .. 0007-*.patch   the series, git am
  benchmark/results.md                   full benchmark tables and how to reproduce
  benchmark/bench.py                     fill sweep x concurrency -> table
  benchmark/sustained_decode.py          batched-decode capacity -> table
  benchmark/mtp_ab_probe.py              acceptance A/B, also the shared lib
  benchmark/prefix_cache_probe.py        prompt reuse per turn -> cached/recomputed
  benchmark/prefix_state_check.py        cache-hit state check, exits non-zero
  benchmark/*.json                       raw results behind the tables in results.md
  serve-3x.sh                            3 gpus, 262,144 native
  serve-3x-512k.sh                       3 gpus, 524,288, YaRN 2.0
  serve-3x-1m.sh                         3 gpus, 1,000,000, YaRN 4.0
  serve-4x.sh                            4 gpus, 262,144 native
  serve-4x-512k.sh                       4 gpus, 524,288, YaRN 2.0, benched
  serve-4x-1m.sh                         4 gpus, 1,000,000, YaRN 4.0, benched
  README.md                              this file
  TODO.md                                open items, in priority order, with the check that closes each
```
