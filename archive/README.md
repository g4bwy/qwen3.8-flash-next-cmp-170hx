# Retired context lanes: 262k and 512k, 3-card and 4-card sets

Kept for reference. Nothing here is maintained: the scripts in this directory
last booted clean on the 9-patch series at base `155488d853` (2026-10-05),
before patch 10 and before the MTP-4/retention-16160 ship state. Expect the
current series to need small flag adjustments on them. The tree pin and apply
flow in the top-level README belong to the active series, not to these.

The service lane is `serve-4x-1m.sh` at the repo root. Full raw grids for
everything below: [../benchmark/results.md](../benchmark/results.md)
(sections "262k (native RoPE, no YaRN)", "524k (YaRN 2.0)", "4-card set",
"3-card" tables) and `logs/boot-256k.log`, `logs/boot-512k.log` on the box.

## The launch matrix these lanes used

| script | gpus | context | RoPE | pool headroom |
|---|---|---|---|---|
| `serve-3x.sh` | 3 | 262,144 | none, native window | 4.39x, best multi-stream decode |
| `serve-3x-512k.sh` | 3 | 524,288 | static YaRN 2.0 | 2.30x |
| `serve-3x-1m.sh` | 3 | 1,000,000 | static YaRN 4.0 | 1.21x, one request fits |
| `serve-4x.sh` | 4 | 262,144 | none, native window | 9.24x, best prefill and TTFT |
| `serve-4x-512k.sh` | 4 | 524,288 | static YaRN 2.0 | 4.87x |

Headroom is the measured KV pool divided by one full-context request.
The 3-card set holds partition `16,17,15` and does not carry
`--prefix-match-unit 16`, so its reuse numbers reflect the 1,600-token floor.
`--prefix-cache-retention-interval 16000` on all of them (the active 1M lane
has moved to 16160 for MTP-4 block coupling).

## Pool and weights at boot

| lane | context | KV pool | headroom | weights per rank (GiB) | KV per rank (GiB) |
|---|---|---:|---:|---|---|
| 3 cards | 262,144 | 1,150,599 | 4.39x | 43.81 / 46.27 / 45.02 | 13.23 |
| 3 cards | 524,288 YaRN 2.0 | 1,204,810 | 2.30x | 43.94 / 46.39 / 45.14 | 13.03 |
| 3 cards | 1M YaRN 4.0 | 1,208,978 | 1.21x | 44.19 / 46.64 / 45.39 | 12.76 |
| 4 cards | 262,144 | 2,423,060 | 9.24x | 36.88 / 34.21 / 34.21 / 35.46 | 20.89 / 24.99 / 24.99 / 23.65 |
| 4 cards | 524,288 YaRN 2.0 | 2,550,833 | 4.87x | 37.02 / 34.41 / 34.41 / 35.68 | 20.73 / 24.77 / 24.77 / 23.42 |

Each YaRN step adds 0.25 GiB per rank of weights. The cos/sin cache is
`4 x 262144 x factor` fp32 rows, allocated before profiling, so it comes out
of the KV pool.

## Throughput (MTP-3 era, decode 1024, repetition-task prompts)

Rates are tok/s unless the cell shows a duration. Rows carry 70-SM numbers on
`12,12,12,12` or `16,17,15` unless noted.

| lane | TTFT, full window | prefill peak | 1 stream | 4 streams, full fill |
|---|---:|---:|---:|---|
| 3 cards, 262k | 28.5 s | 9.3k | 163-167 | 457, 4 of 4 resident |
| 3 cards, 524k | 58.9 s | 9.2k | 163-166 | ~304, 2 of 4 |
| 3 cards, 1M | 119.1 s | 9.2k | 161-165 | 163-164, 1 of 4 |
| 4 cards, 262k | 22.1 s | 9.9k | 151-165 | 405, 4 of 4 |
| 4 cards, 524k | 46.0 s | 10.5k | 163-168 | 367, 4 of 4 |

Findings measured in these lanes:

- Pool sets capacity. At 262k, 4 cards hold nine full streams against four on
  3 cards, which was the argument for the wider pipeline.
- Prefill paces at the heaviest stage, not the sum, so four balanced stages
  beat `16,17,15` by about 20%. Single-stream decode barely moves between the
  two widths, because the 200 W cap binds first.
- Batched decode pays for the extra hop. At 262k full fill, four streams fall
  from 457 on 3 cards to 405 on 4. The 8-11% gap repeats across three methods:
  bench cells, deep plateau 452 against 405, shallow plateau 468 against 431.
  At 524k with two streams it repeats 304 against 265.
- 524k on 3 cards never batches four streams: the pool holds two, the rest
  queue, TTFT lands in serial multiples. On shallow 8k prompts the same box
  reached 468 aggregate at four streams, where residency is not the limit.
- Oversubscription never preempted at either width. The scheduler gates
  admission, extra streams queue, preemptions stayed 0 in every cell.
- Choose PP=3 when streams decode side by side (457 against 405 at 262k,
  four streams); choose PP=4 for long prompts (prefill 9.9k against 9.3k,
  TTFT 22.1 s against 28.5 s).
- The layer-split case: `13,12,12,11` beat an equal `12,12,12,12` by 2.13% of
  pool at 1M, 2.17% at 524k and 2.33% at 262k, measured on all three 4-card
  lanes 2026-09-30.
- On YaRN bookkeeping: at 512k the factor-2.0 scaling matches exactly
  (524,288 = 2 x 262,144), so transformers stays quiet. At 1M the explicit
  factor 4.0 wins over the implicit 3.81 ratio and the warning is harmless.

## TP=4 + EP trial (`serve-4x-1m-tp.sh`, rejected)

Swaps PP=4 for TP=4 with `--enable-expert-parallel`, and loses: 1,434 against
9,012 tok/s prefill, 111.7 against 185.1 tok/s single-stream decode, 1.74x
against 2.60x KV pool. TP puts roughly 100 collectives in every decode step on
x4 links. Peer DMA adds 20% there. At PP=4 it costs 2 to 3% of prefill with
decode unchanged, which is why the shipped path keeps `NCCL_P2P_DISABLE=1`.
Tables: [../benchmark/results.md](../benchmark/results.md)
("TP=4 + EP, 1M: rejected").
