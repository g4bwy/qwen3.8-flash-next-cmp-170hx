# Qwen3.8-Flash-Next-FP8 (official checkpoint) on 4x CMP170HX / PP=4 / MTP-4, optional W8A16 lane

Patches over vLLM mainline, running with uv venv.
No bullshit slop-wall-of-text (almost...), no docker, no opaque scripts, no nonsense.
The thing's basically debugging and optimizing itself, so expect some slop, but trying to avoid it as much as possible.

- Box: 4x NVIDIA CMP 170HX (GA100, unlocked to 64 GB and 74 SM), one NUMA node,
  PLE n-gram table in pinned host RAM. Every number here is at a 200 W per-card
  cap.
- PCIe is the limit. Gen2, GPU1 at x16, GPU0/2/3 at x4, and `nvidia-smi topo -m`
  puts every pair at NODE distance. A ring is only as wide as its narrowest
  card, so budget about 2 GB/s between any two GPUs, not 8. NCCL needs
  `NCCL_P2P_LEVEL=SYS` to use peer DMA here at all.
- Checkpoint: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (official FP8). The serve script pulls this tag by default; override with `MODEL=/local/path`. MTP-4 is the shipped depth; the fast lane builds an int8-dense variant with `checkpoint/` (+6 % on top of depth, +20 % over MTP-3).
- Base commit: `155488d853a0bc42df227dbfc74005b3fd488e94` (vllm-project/vllm main, 2026-10-04).
- 10 patches: eight ours, two adopted from pending upstream PRs (#56444 as
  patch 2, #58094 as patch 7). After
  applying them, `git rev-parse HEAD^{tree}` must print
  `1de2b16fd7b10bd552577667f88399bc3cea22ed`. If it does not, you applied
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
git rev-parse 'HEAD^{tree}'   # 1de2b16fd7b10bd552577667f88399bc3cea22ed
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

One service script, the shipped state:

```bash
./serve-4x-1m.sh       # MODEL=/local/path overrides the checkpoint tag
```

4x cards, PP=4, partition `13,12,12,11`, 1,000,000 context at static YaRN 4.0,
MTP-4 draft depth, KV pool 2.57x one full request. Prefix it with `PROF=1` to
arm the capped torch profiler for step-anatomy runs (benchmark/phase_probe.py).
Retired configs live in [archive/](archive/README.md): the 3-card set, the
262k and 512k lanes, and the rejected TP=4+EP trial, each with its tables.

What the script carries, and why:

- MTP-4 spec decode, and the PLE n-gram table offloaded to pinned host RAM
  (`--engram-config '{"cpu_offload": true}'`).
- Prefix caching with `--mamba-cache-mode align`,
  `--prefix-cache-retention-interval 16160` and `--prefix-match-unit 16`.
  MTP-4 forces a 1616 scheduler block (1600 + 16 per extra draft row) and the
  retention interval is a multiple of it; the match unit drops the hit floor
  from 1,600 to 32 tokens. See Prefix caching.
- NCCL IB off (single node, no fabric). NCCL P2P off as well: peer DMA
  measured 2 to 3% off prefill with decode unchanged, so the shipped path
  stays host-staged.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which stops allocator-OOM
  retries. vLLM forbids it only with KV connectors, and we run none.
- `VLLM_SAFETENSORS_DROP_CACHE=1` (patch 9): consumed shards leave the page
  cache, so the 173 GiB boot read does not evict reusable cache.

Startup checks for the YaRN lane:

1. `Using max model len <N>` appears twice, once for the target and once for
   the `Qwen4ExpMTP` draft. A second, smaller number means patch 7 is missing,
   and MTP acceptance rots with context depth.
2. The log prints `Maximum concurrency for <N> tokens per request: X.XXx`.
   Below 1.0 the box cannot serve its own context length, so lower
   `--max-model-len`.

### Draft vocabulary head (patch 8)

Patch 8 makes the MTP drafter score a small head instead of the 248k-row
lm_head per draft step, worth ~9% single-stream. It keeps the full head
resident too and drafts mixed prefill+decode windows (a stream joining or
growing while others decode) with it, because that is where the small head
costs acceptance: greedy short-tail 4-stream cells read 3.7-4.3 at depth 4
on the fixed gate, versus 2.9 with the small head everywhere. Uniform decode rounds replay
the graphed small head and keep the win. Sampled corpus traffic shows no
measurable effect either way. Artifact commands and the measured story:
[checkpoint/README.md](checkpoint/README.md). Engages only when the
checkpoint directory carries the artifacts. Startup proof:
`MTP drafter scores a 40000-token draft vocabulary`.

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
On the current unlock revision each card exposes the full GA100 die, 74 SMs;
an earlier revision left 4 masked, so the 70-SM tables are marked where they
still stand.
The link is x4 here, about 2 GB/s per card in each direction. Nothing else is
tuned. Each card is capped at 200 W.
Full tables, methodology and reproduce commands: [benchmark/results.md](benchmark/results.md).
Numbers use repetition-task prompts at decode 1024, because a short window measures drafter
warmup rather than steady decode.

### Pool and weights at boot

| lane | context | KV pool | headroom |
|---|---|---:|---:|
| 4 cards, official FP8, MTP-4 | 1M YaRN 4.0 | 2,572,981 | 2.57x |
| 4 cards, W8A16 dense, MTP-4 | 1M YaRN 4.0 | 2,594,720 | 2.59x |

Per-rank weight and KV columns, and the retired 3-card / 262k / 524k rows:
[archive/README.md](archive/README.md). The YaRN cos/sin cache
(`4 x 262144 x factor` fp32 rows) is allocated before profiling and comes out
of the KV pool; each YaRN step adds 0.25 GiB per rank of weights.

### Throughput

Rates are tok/s unless the cell shows a duration. All rows: 74 SMs, partition
`13,12,12,11`, draft-vocab head, decode 1024, repetition-task prompts.

| lane | TTFT, full window | prefill peak | 1 stream | 4 streams, full fill |
|---|---:|---:|---:|---|
| 4 cards, 1M, MTP-3 | 91.9 s | 10.4k | 181-184 | 325, 2-3 of 4 |
| 4 cards, 1M, official, MTP-4 | 90.6 s | 10.4k | 205-215 (median 212) | wall 386 s at 100/4, 2-3 of 4 resident |
| 4 cards, 1M, W8A16 lane, MTP-4 | 93.5 s | 10.1k | 217-226 | 400 s wall at 100/4, batch-neutral |

- The sustained column is a 1.5 s peak that typically lands in the wind-down
  of the run; compare grids on wall or out tok/s. Full grids, repeat runs and
  the salt law: [benchmark/results.md](benchmark/results.md).
- Single-stream decode at MTP-4 sits at this checkpoint's roofline on this
  silicon. The step anatomy (results.md, 2026-10-13) puts 66% of step time in
  GEMM traffic at the measured 580-790 GB/s solo band and the rest in
  per-layer state machinery and glue; the visible NCCL time is structural PP
  bubble that only concurrency reclaims. What remains: batch residency (2-3
  full streams fit at 1M), fewer bytes, or a different checkpoint.
- MTP acceptance: 4.75-4.9 mean length at MTP-4 on pinned salt, 3.7-4.0 at
  MTP-3, stable to full 1M depth. Acceptance that decays with depth means
  patch 7 is missing. Cell-to-cell scatter on rotating prompts is the content
  lottery at depth 4+: pin `--salt` for comparisons, rotate it for ranges.
- Oversubscription never preempted in any grid. The scheduler gates admission,
  extra streams queue, and preemptions stayed 0 in every concurrency cell.

### Why these layer splits

The pool caps at min_r (rank KV bytes / rank layer count). The model has one
PLE layer, at decoder index 1, so rank 0 needs at least two layers, and a
layer costs rank 0 0.80 GiB of KV against 2.04 GiB on a mid rank. The 4-card
lane runs `13,12,12,11`: +2.13% pool at 1M against an equal `12,12,12,12`,
and the same margin repeats at 524k and 262k (archive). The extra layer costs
rank 0 2.53 GiB of weights while rank 3, which also carries the MTP drafter,
gets the room back. Rank 0 is now the binding stage, so a second shifted
layer has nowhere useful to go. Prefill, single-stream decode and four-stream
decode did not move outside their boot-to-boot spread.

Two boot warnings are normal and cost you nothing. The 7
`expandable_segments: memory mapping failed` messages fall inside CUDA graph
capture, none after `Application startup complete`. `max_num_scheduled_tokens
is set to 2048` comes from the MTP drafter's window.

## Optional lane: W8A16 dense checkpoint

`checkpoint/` builds a variant of the official FP8 tree that stores the 228
main-body dense matrices as int8 (group-32, weight-only) instead of bf16.
Serving it needs patch 10 and no other change; point `MODEL=` at the built
directory. Two boots per side on 4x-1m: single-stream decode 216-226 tok/s
at MTP-4 (+15 % over this lane's 190-192 at depth 3; +20 % over the
official FP8 tree's 181-184), prefill -1.7 to -2.5 %, KV pool +17k net at
depth 4, acceptance unchanged. The quality gate (166k scored positions,
two captures per boot, four boots): top-1 flip 8.1 % against a 7.3 %
machine non-reproducibility floor measured within and across boots, so
about one position in 130 moves because of the quantization. Full tables:
[benchmark/results.md](benchmark/results.md).

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
| 8 | vocab-truncated MTP draft head: the drafter scores a ~40k-token head instead of the full 248k-row lm_head per draft step, +8.7% single-stream, verification and output quality untouched. Dual-head: eager mixed prefill windows draft with the full head, graphed uniform decode rounds keep the small one. Needs artifacts, build commands in checkpoint/README |
| 9 | drop each consumed checkpoint shard from the Linux page cache (`VLLM_SAFETENSORS_DROP_CACHE=1`, set by the serve scripts): 173 GiB of read-once weight bytes should not evict reusable cache at every boot |
| 10 | Qwen4Exp PLE embedding honors fp8 n-gram tables inside compressed-tensors checkpoints: resolve the scheme (none -> unquantized, float8 -> fp8, else raise) instead of falling through to the Fp8Config-only gate |

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
  3.81. Nothing is wrong: 1,000,000 is not 4 x 262,144, and the explicit
  factor wins.
- The first MTP decode step compiles `_fill_num_accepted_kernel` and stalls;
  warm up before the first measured cell. Scatter after that is content, not
  the machine: pinned-salt repeats land within ~2%, rotating salts draw the
  prompt lottery at depth 4+ (see the salt law in results.md).
- The 1M pool holds 2.57 full requests. A third queues behind them. If
  residency is exceeded mid-decode, the preempted request replays its whole
  prompt under Mamba align mode, roughly 2 min per 1M.
- A long prefill that lands during decode starves every stream to under
  20 tok/s until it finishes. Schedule long prefills early, not between decodes.
- A bench fill of 100% means (ctx - decode) x 0.99. Anything larger is a plain
  HTTP 400 from the server.

## Layout

```
patchset-qwen38-pp/
  patches/0001-*.patch .. 0010-*.patch   the series, git am
  serve-4x-1m.sh                         the service lane: 4 gpus, 1M, YaRN 4.0 (PROF=1 arms the profiler)
  archive/                               retired lanes, the 3-card set, the TP=4 trial, and their tables
  benchmark/                             measurement tools, raw results, and the full record, see its README
  checkpoint/                            artifact builders: W8A16 dense tree + draft-vocab, see its README
  README.md                              this file
```
