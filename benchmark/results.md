# Benchmark results, full tables

All numbers measured on the 3x unlocked CMP 170HX box (200 W power cap, PP=3,
MTP-3, PLE host offload) running one server at a time, idle and warmed,
greedy, unique prompt per row so every row pays a full prefill. The
"4-card set" section is the same box with a fourth card, PP=4. The 4-card
1M grid runs the current unlock revision, 74 SMs per card; every other table
predates it, at 70. Tools:
`bench.py` (fill sweeps and concurrency cells), `sustained_decode.py` (decode
plateaus), `mtp_ab_probe.py` (acceptance A/B). Everything here was measured
2026-09-15 unless noted. Headlines live in the top-level README.

## Fill sweeps (decode 128)

One greedy stream per row. All three tables use the same script and prompt
task, so they are directly comparable.

### 262k (native RoPE)

Raw data: `bench-262k.json`.

| fill % | prompt tok | TTFT s | prefill tok/s | total s | decode tok/s | mean acc len |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 12,948 | 1.6 | 8,336 | 2.6 | 124.7 | 2.95 |
| 10 | 25,949 | 2.9 | 8,801 | 3.8 | 149.2 | 3.61 |
| 25 | 64,814 | 7.0 | 9,218 | 7.8 | 160.3 | 3.71 |
| 50 | 129,682 | 14.0 | 9,270 | 14.8 | 159.1 | 3.71 |
| 75 | 194,574 | 21.2 | 9,160 | 22.1 | 145.8 | 3.20 |
| 90 | 233,518 | 25.6 | 9,119 | 26.4 | 169.6 | 3.63 |
| 100 | 259,383 | 28.5 | 9,088 | 29.3 | 161.3 | 3.43 |

### 524k (YaRN 2.0)

Raw data: `bench-524k.json`.

| fill % | prompt tok | TTFT s | prefill tok/s | total s | decode tok/s | mean acc len |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 25,927 | 2.9 | 8,848 | 3.7 | 156.9 | 3.71 |
| 10 | 52,006 | 5.7 | 9,131 | 6.5 | 161.9 | 3.71 |
| 25 | 129,728 | 14.0 | 9,244 | 14.9 | 153.2 | 3.42 |
| 50 | 259,416 | 28.5 | 9,093 | 29.3 | 166.7 | 3.49 |
| 75 | 389,353 | 43.5 | 8,941 | 44.2 | 187.1 | 3.58 |
| 90 | 467,174 | 52.8 | 8,845 | 53.4 | 208.9 | 3.51 |
| 100 | 518,587 | 58.9 | 8,802 | 59.6 | 192.5 | 3.58 |

### 1M (YaRN 4.0)

Raw data: `bench-1m.json`.

| fill % | prompt tok | TTFT s | prefill tok/s | total s | decode tok/s | mean acc len |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 49,391 | 5.4 | 9,173 | 6.2 | 148.4 | 3.49 |
| 10 | 99,165 | 10.7 | 9,228 | 11.6 | 141.6 | 3.25 |
| 25 | 247,489 | 27.2 | 9,099 | 28.0 | 162.3 | 3.46 |
| 50 | 495,062 | 56.2 | 8,812 | 56.9 | 187.1 | 3.66 |
| 75 | 742,381 | 86.9 | 8,545 | 87.5 | 195.1 | 3.58 |
| 90 | 890,613 | 106.2 | 8,386 | 106.7 | 259.3 | 3.43 |
| 100 | 988,951 | 119.1 | 8,306 | 119.6 | 236.5 | 3.25 |

Notes on the three tables:

- Prompts are the repetition task, so acceptance sits near its 4.0 ceiling and
  decode reads high versus real traffic. The 524k server logs show 2.3-2.9
  acceptance and 27-105 tok/s on mixed traffic.
- Prefill peaks at 9.2-9.3k tok/s on all three configs and holds within 1% to
  25% fill. The decay tracks absolute depth, not RoPE scaling: 100%-fill rows
  give 9,088 tok/s at 259k tokens, 8,802 at 519k, 8,306 at 989k.
- TTFT scales linearly with the window: 28.5 / 58.9 / 119.1 s at 100% fill. A
  preempted long request loses minutes of prefill.
- Acceptance is flat with depth everywhere, so 2x or 4x YaRN costs speculative
  decode nothing. Without patch 9 the drafter runs unscaled RoPE, and that is
  where this column rots with depth.
- Decode tok/s in these 128-token tables rises with depth (262k: 125 -> 161,
  524k: 157 -> 193, 1M: 148 -> 237), which is the wrong direction for more KV
  reads per step. It is a measurement artifact, not physics: a 128-token
  window is mostly the un-starved burst right after a stream's own prefill, so
  the rate reads the drafter warmup, not steady decode. Re-run at decode 1024
  the depth trend is flat (see below). Treat the 128-token decode column as
  suspect and read the decode 1024 numbers for real rates.

## Long decode recheck (524k, decode 1024)

The same fill sweep with `--decode 1024`, so every column sees a decode
phase long enough to mean something. Raw data: `bench-524k-dec1024.json`,
`bench-524k-conc2-dec1024.json`.

| fill % | conc | TTFT s | win avg tok/s/req | sust agg tok/s | acc len |
|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 2.9 | 164.9 | 172 | 3.96 |
| 25 | 1 | 14.0 | 164.4 | 169 | 3.95 |
| 50 | 1 | 28.5 | 162.8 | 166 | 3.93 |
| 75 | 1 | 43.5 | 165.8 | 167 | 3.97 |
| 100 | 1 | 58.9 | 165.4 | 165 | 3.95 |
| 50 | 2 | 43.4 | 83.6 | 280 | 3.92 |
| 100 | 2 | 89.6 | 78.8 | 265 | 3.90 |

- Single-stream decode is flat at 163-166 tok/s from 26k to 518k prompt
  depth. The depth trend in the decode-128 tables was the warmup window, not
  KV reads.
- Acceptance reads 3.90-3.97 at decode 1024 versus 3.3-3.7 at decode 128 in
  the same configs. The short window undersold it by 0.2-0.5, and the flat
  with depth result holds even more cleanly here.
- Two streams at decode 1024 finally overlap enough to show a real plateau at
  depth: 265-280 tok/s aggregate, about 135 per stream at 518k depth versus
  157 at 8k depth. (Superseded 2026-09-21 by the warm re-test below: 282-304.)
- win avg still runs below the plateau even at decode 1024 (79-84 vs 265/2):
  stream 1 spends most of its decode starved beside stream 2's prefill.
- TTFT and prefill are unchanged from the decode-128 runs, as expected. in
  tok/s reads lower here only because wall now includes a 6-13 s decode tail.

## Concurrency (262k, 1 to 4 streams)

`bench.py --concurrency 1,2,4` fires N streams of the same fill together.
Decode 128 per stream, warm server, 2026-09-15. Preemptions were zero in all
12 cells. Raw data: `bench-262k-conc.json`.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 64,753 | 7.0 | 7.0 | 7.8 | 8,255 | 16 | 157.3 | 69 | 3.61 |
| 25 | 2 | 64,840 | 10.6 | 13.9 | 15.0 | 8,625 | 17 | 85.9 | 120 | 3.63 |
| 25 | 4 | 64,771 | 18.3 | 28.7 | 30.8 | 8,417 | 17 | 43.0 | 101 | 3.68 |
| 50 | 1 | 129,468 | 14.1 | 14.1 | 14.9 | 8,680 | 9 | 158.6 | 79 | 3.58 |
| 50 | 2 | 129,614 | 21.3 | 27.8 | 29.5 | 8,778 | 9 | 68.3 | 94 | 3.21 |
| 50 | 4 | 129,634 | 35.8 | 55.3 | 58.3 | 8,895 | 9 | 42.3 | 90 | 3.59 |
| 75 | 1 | 194,478 | 21.3 | 21.3 | 22.0 | 8,842 | 6 | 180.1 | 32 | 3.91 |
| 75 | 2 | 194,605 | 32.2 | 42.0 | 43.9 | 8,868 | 6 | 85.8 | 61 | 3.66 |
| 75 | 4 | 194,532 | 54.1 | 83.7 | 87.8 | 8,858 | 6 | 44.5 | 90 | 3.34 |
| 100 | 1 | 259,075 | 28.6 | 28.6 | 29.4 | 8,821 | 4 | 155.1 | 74 | 3.33 |
| 100 | 2 | 259,237 | 43.0 | 56.0 | 58.2 | 8,912 | 4 | 90.1 | 63 | 3.61 |
| 100 | 4 | 259,201 | 71.9 | 111.0 | 116.1 | 8,930 | 4 | 45.5 | 71 | 3.58 |

- Prefill does not scale with streams. in tok/s stays at 8.3-8.9k from one
  stream to four at every fill. Concurrency adds a queue: mean TTFT grows
  linearly with N, and the fourth stream at 100% fill got its first token at
  111 s.
- Do not read decode capacity from the win avg column. With decode 128, first
  tokens land 1-111 s apart, so the streams rarely decode together, and each
  decode competes with pending chunked prefills that pace it to ~1 step/s. The
  sust column only catches transient overlap peaks here. The last section
  measures real batched capacity.
- Acceptance ignores batching: 3.15-3.82 in every cell.
- Four concurrent 100%-fill streams (1.036M tokens) fit the 1.15M pool with
  zero preemptions, matching the 4.39x startup number.
- out tok/s counts completions against the whole cell, so it is small when a
  128-token answer sits behind a 65-259k prompt.

### 524k at 2 streams

Same sweep on the YaRN 2.0 config, 2026-09-15. Raw data:
`bench-524k-conc.json`. Zero preemptions: two 100%-fill streams (1.037M
tokens) fit the 1.205M pool.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 2 | 129,834 | 21.4 | 27.9 | 29.5 | 8,804 | 9 | 79.3 | 82 | 3.58 |
| 50 | 2 | 259,356 | 43.1 | 56.2 | 58.3 | 8,893 | 4 | 91.6 | 55 | 3.61 |
| 75 | 2 | 389,021 | 65.5 | 85.2 | 88.1 | 8,832 | 3 | 87.0 | 69 | 3.33 |
| 100 | 2 | 518,443 | 88.3 | 114.6 | 118.2 | 8,769 | 2 | 91.7 | 79 | 3.58 |

Queue behavior matches 262k: aggregate prefill holds ~8.8k tok/s at N=2, and
the p95 TTFT at 100% fill (114.6 s) is 2x the single-stream TTFT (58.9 s).

### 524k at 4 streams, over the pool

Four 100%-fill streams need 2.07M tokens of KV against a 1.205M pool, and
three of them at 75% fill need 1.556M. Raw data: `bench-524k-conc4.json`,
`bench-524k-conc4b.json`, `bench-524k-conc4-dec512.json`.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 4 | 129,776 | 35.6 | 55.1 | 58.1 | 8,937 | 9 | 41.8 | 69 | 3.53 | 0 |
| 50 | 4 | 259,344 | 72.3 | 111.5 | 116.7 | 8,887 | 4 | 41.9 | 73 | 3.28 | 0 |
| 75 | 4 | 389,109 | 108.9 | 167.6 | 174.9 | 8,897 | 3 | 46.9 | 55 | 3.54 | 0 |
| 100 | 4 | 518,536 | 147.0 | 226.1 | 235.6 | 8,802 | 2 | 45.0 | 66 | 3.37 | 0 |
| 100 | 4 | 518,480 | 151.8 | 235.3 | 247.7 | 8,373 | 8 | 46.6 | 198 | 3.85 | 0 |

The last row repeats 100% fill with decode 512, so streams hold their KV
longer and force overlap.

- Zero preemptions in every cell, even over the pool. vLLM does not admit a
  stream it cannot fit, so the pool limit shows up as latency, not eviction.
- TTFTs are exact multiples of the single-stream prefill. At 100% fill the
  four first tokens land at 59.1, 117.6, 176.3 and 234.9 s: each stream waits
  behind a full serial prefill of everything queued ahead of it.
- With decode 512 the multiples drift (59.0, 120.5, 183.0, 244.5 s) because
  the admitted streams also hold blocks. The gap between the last TTFT (244.5
  s) and wall (247.7 s) is the only window where two streams decode together,
  and it is the 198 tok/s sample in that row.

### 1M (YaRN 4.0), 1 to 4 streams, decode 1024

Same sweep on the 1M config at the 1024-token decode. Raw data:
`bench-1m-dec1024-conc.json` (1 and 2 streams) and
`bench-1m-conc4-dec1024.json` (4 streams).

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 246,938 | 27.2 | 27.2 | 33.6 | 7,348 | 160.7 | 164 | 3.90 | 0 |
| 25 | 2 | 247,166 | 41.3 | 54.0 | 62.1 | 7,957 | 91.4 | 306 | 3.95 | 0 |
| 25 | 4 | 247,208 | 71.6 | 113.0 | 125.8 | 7,859 | 44.7 | 456 | 3.93 | 0 |
| 50 | 1 | 494,347 | 56.1 | 56.1 | 62.4 | 7,925 | 164.2 | 163 | 3.96 | 0 |
| 50 | 2 | 494,029 | 85.2 | 111.3 | 121.5 | 8,130 | 78.0 | 265 | 3.93 | 0 |
| 50 | 4 | 494,100 | 145.8 | 226.8 | 242.8 | 8,141 | 47.7 | 265 | 3.93 | 0 |
| 75 | 1 | 741,024 | 86.7 | 86.7 | 97.5 | 7,603 | 95.2 | 172 | 2.14 | 0 |
| 75 | 2 | 741,356 | 131.9 | 172.7 | 183.6 | 8,076 | 163.3 | 164 | 3.93 | 0 |
| 75 | 4 | 741,371 | 222.8 | 345.4 | 365.3 | 8,118 | 162.2 | 164 | 3.91 | 0 |
| 100 | 1 | 989,129 | 118.9 | 118.9 | 125.1 | 7,905 | 164.8 | 163 | 3.95 | 0 |
| 100 | 2 | 988,772 | 179.7 | 234.6 | 247.0 | 8,006 | 163.8 | 163 | 3.94 | 0 |
| 100 | 4 | 988,492 | 301.8 | 466.8 | 491.3 | 8,047 | 163.1 | 164 | 3.89 | 0 |

The pool holds 1.209M tokens, so it fits one full 1M context and a little
over half of a second. The sust column tracks exactly how many streams the
pool can hold at once, and the win avg column shows it too:

- 25% (4 fits): sust 456 at 4 streams, per-user win avg 44.7. Real 4-way
  concurrency.
- 50% (2 fit): sust 265 at both 2 and 4 streams. The third and fourth
  streams cannot be admitted to decode together, so they queue.
- 75 and 100% (1 fits): sust collapses to the single-stream 163-164 tok/s
  even at N=4, and win avg per user is a full 163. There is no batching:
  the scheduler runs each stream alone, head to head, and wall time scales
  linearly with N (100% N=4 wall 491 s is 4x N=1 125 s).
- Zero preemptions in all 12 cells again. Concurrency past the pool is pure
  serialization: you pay N times the latency and gain no decode throughput.
- Acceptance holds 3.89-3.96 across every depth and N, the strongest patch 9
  result yet: at a real 1M-token position the drafter is fully scaled. The
  lone 2.14 entry at 75% N=1 was a one-cell content excursion, not depth: a
  clean immediate re-run read 164 tok/s and 3.95 (`bench-1m-recheck75.json`).

### 262k (native RoPE), 1 to 4 streams, decode 1024 (3-card rerun)

Same grid as the 4-card 262k table, on `serve-3x.sh`, booted 2026-09-21
(`boot-3x-256k.log`): pool 1,150,599 tokens, 4.39x, nightly wheel banner as
everywhere. This fills the decode-1024 gap of the 3-card 262k lane; the
decode-128 table above stays published. Raw: `bench-3x-262k-conc.json`,
rows 25/1 and 100/1 from `bench-3x-262k-recheck25.json` and
`bench-3x-262k-recheck100.json`.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 64,278 | 6.8 | 6.8 | 12.9 | 4,975 | 79 | 166.9 | 172 | 3.96 | 0 |
| 25 | 2 | 64,370 | 10.4 | 13.5 | 20.5 | 6,292 | 100 | 115.8 | 317 | 3.97 | 0 |
| 25 | 4 | 64,296 | 17.7 | 27.7 | 37.7 | 6,826 | 109 | 64.6 | 467 | 3.94 | 0 |
| 50 | 1 | 128,517 | 13.7 | 13.7 | 19.9 | 6,451 | 51 | 165.4 | 171 | 3.93 | 0 |
| 50 | 2 | 128,662 | 20.8 | 27.2 | 35.3 | 7,291 | 58 | 93.2 | 285 | 3.94 | 0 |
| 50 | 4 | 128,690 | 35.0 | 54.2 | 65.1 | 7,911 | 63 | 53.2 | 467 | 3.94 | 0 |
| 75 | 1 | 193,049 | 20.8 | 20.8 | 27.0 | 7,141 | 38 | 163.6 | 171 | 3.92 | 0 |
| 75 | 2 | 193,184 | 31.5 | 41.2 | 48.9 | 7,897 | 42 | 95.5 | 311 | 3.93 | 0 |
| 75 | 4 | 193,112 | 54.2 | 85.1 | 96.9 | 7,969 | 42 | 47.4 | 452 | 3.92 | 0 |
| 100 | 1 | 257,186 | 27.5 | 27.5 | 33.7 | 7,640 | 30 | 165.6 | 169 | 3.96 | 0 |
| 100 | 2 | 257,338 | 42.3 | 55.3 | 63.4 | 8,117 | 32 | 92.0 | 311 | 3.93 | 0 |
| 100 | 4 | 257,305 | 72.9 | 114.4 | 127.1 | 8,096 | 32 | 45.2 | 457 | 3.92 | 0 |

- Single-stream decode 163-167 tok/s at every depth, exactly the band the
  524k long-decode recheck measured, so the 262k lane is no longer n/m.
- First cell of the sweep (25/1, the first request after calibration) read
  win avg 62.2 with prefill 2.8k tok/s; a rerun with the box warm gave 166.9
  and 5.0k. Cold-start artifact, not thermal: the grid's last N=4 cell was
  hotter than this first one.
- Four full 261k streams (demand 1.04M) fit the 1.15M pool and sustain 457
  agg, more than the 4-card set manages (405). One fewer pipeline hop per
  step: at 262k, where residency is equal, PP=3 wins batched decode.
- That gap was re-tested 2026-09-21 with the box dedicated to it, after two
  warmup completions. The 100% N=4 cell repeated at 451 and 451 (original
  457). `sustained_decode.py` plateaus: 468 agg on shallow 8k prompts (the
  number published since 2026-09-15, reproduced to the digit) and 452 agg at
  full 257k prompts with decode 4096. The 4-card set measured identically:
  404, 404, 431 and 405. PP=4 loses 8-11% of batched decode at equal
  residency, shallow and deep alike. Raw: `bench-3x-262k-100n4-rep{1,2}`,
  `bench-3x-262k-sd-{shallow,deep}.json`, and the 4x counterparts.
- Prefill is where 4 cards win: full-window TTFT 27.5 s against 22.1 s, and
  100% N=4 in tok/s 8.1k against 9.9k.
- Acceptance 3.92-3.97 in all clean cells, zero preemptions in 13 runs.

### 524k 2-stream head-to-head (warm, 2026-09-21)

Trigger: the 4-card 524k sweep put 100% N=2 decode at 275 agg, level with the
3-card box's published 265-280, even though the 262k head-to-head had just
shown PP=3 winning batched decode by ~10%. Same dedicated-box method on both
sides: verify the pool from `/metrics` (1,204,810 tokens 2.30x / 2,496,752
tokens 4.76x), two warmup completions, then the same three commands. Raw:
`bench-{3x,4x}-512k-100n2-rep{1,2}.json`, `bench-{3x,4x}-512k-sd-deep.json`.

Full-depth numbers, the ones that hold up:

| test | shape | 3-card agg | 4-card agg |
|---|---|---:|---:|
| bench 100% N=2, rep 1 | 511,370 prompt, 1024 decode | 304 | 265 |
| bench 100% N=2, rep 2 | same | 304 | 265 |
| deep plateau, N=2 | 460k prompt, 20,480 decode | 282 | 270 |

- The 3-card box wins 2-stream full-depth decode at 524k too, so PP=4's hop
  cost is not a 262k-only effect. The margin depends on the test: 304 vs 265
  (+15%) on the bench cell, 282 vs 270 (+4%) on the deep plateau. Both agree
  on sign; neither is tight enough to pin one number, so the safe read is
  "3-card is a few to fifteen percent ahead", not a single figure.
- The deep plateau is the softer of the two this time: on 4 cards the 460k
  prompt prefills in ~41 s per stream (TTFTs 40.6, 82.9), so stream 1 has
  only ~10 s of true 2-stream overlap inside the 20,480-token window before
  it drains, against 3 cards' ~77 s. The bench cell is the one that decides it.
- Two plateau geometry rules: prompt plus decode must fit inside
  `max_model_len` (517k + 12,288 = 529k dies at HTTP 400 on every request),
  and the decode count must exceed the solo rate (~165 tok/s) times the
  TTFT gap, or stream 1 drains before stream 2 starts decoding. The
  20,480-token window was sized for this; the 1,024-token default is not.
- Do not use the shallow plateau at N=2 to compare boxes. It reads 345 agg
  on 3 cards (173/stream) and 379 on 4 (190/stream), and 190/stream is above
  the single-stream ceiling, so the 4-card sample is a transient, not a
  plateau. The earlier note here that "a second stream is free at small
  batch" was built on this noise and is withdrawn; shallow plateaus need the
  longer decode window the deep test uses.

## Sustained decode capacity

`sustained_decode.py` isolates decode: an 8k prompt finishes prefill in ~1 s,
decode 1024 keeps all streams overlapping, and the plateau comes from
`vllm:generation_tokens_total` sampled at 0.5 s. Raw data:
`sustained-decode.json`.

| streams | wall s | first tokens s | agg tok/s | per-stream tok/s |
|---:|---:|---|---:|---:|
| 1 | 7.4 | 1.0 | 172 | 172 |
| 2 | 8.6 | 1.0, 1.9 | 314 | 157 |
| 4 | 13.3 | 1.1, 1.9, 2.8, 3.8 | 468 | 117 |

- Aggregate decode scales 1.8x at 2 streams and 2.7x at 4. Per stream it
  falls 172 -> 157 -> 117 tok/s.
- At 450k prompt depth (524k config): single stream decodes 165 tok/s, so
  depth costs little. With only decode 512 the streams barely overlap, so the
  sampled peak is a tail burst (220 agg). The decode-1024 recheck above gives
  the clean deep plateau instead: 265-280 agg at 2 streams, ~135 per stream.
- These numbers are at the 200 W cap. Scaling at stock power will differ.

## 4-card set (PP=4), decode 1024

### 1M (YaRN 4.0), restored silicon

Re-measured 2026-10-07. The unlock revision returned 4 masked SMs per
card, 70 to 74. Partition 13,12,12,11, tree dev649: draft-vocab head,
`--prefix-match-unit 16`, page-cache drop. KV pool 2,595,975 tokens,
2.60x headroom, cap 200 W. Fresh boot, cold pool; the tool now derives
its prompt salt from the clock, so no row can hit a prefix cached by an
earlier run. The 50/1 sweep row read win avg 45.4 with acc 1.43 right
after the 25/4 cell; an idle rerun gave 184.4 and 3.93 and replaces it,
the same contamination shape as the September 75/1 case. Raw:
`bench-4x-1m-sm74-200w.json` plus that one rerun.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 247,095 | 21.3 | 21.3 | 26.9 | 9,181 | 38 | 181.5 | 187 | 3.92 | 0 |
| 25 | 2 | 247,090 | 32.9 | 43.1 | 50.2 | 9,850 | 41 | 105.0 | 438 | 3.71 | 0 |
| 25 | 4 | 247,286 | 56.3 | 88.2 | 98.7 | 10,022 | 41 | 53.8 | 548 | 3.91 | 0 |
| 50 | 1 | 494,302 | 43.7 | 43.7 | 49.2 | 10,038 | 21 | 184.4 | 187 | 3.93 | 0 |
| 50 | 2 | 494,242 | 66.9 | 87.4 | 95.5 | 10,355 | 21 | 98.5 | 348 | 3.93 | 0 |
| 50 | 4 | 494,315 | 114.8 | 178.9 | 197.4 | 10,016 | 21 | 45.9 | 383 | 2.92 | 0 |
| 75 | 1 | 741,460 | 67.4 | 67.4 | 73.0 | 10,158 | 14 | 183.8 | 182 | 3.92 | 0 |
| 75 | 2 | 741,460 | 102.9 | 134.4 | 144.5 | 10,265 | 14 | 84.7 | 285 | 3.88 | 0 |
| 75 | 4 | 741,400 | 173.9 | 269.7 | 293.7 | 10,096 | 14 | 37.9 | 236 | 2.24 | 0 |
| 100 | 1 | 988,628 | 91.9 | 91.9 | 97.5 | 10,138 | 11 | 183.2 | 183 | 3.85 | 0 |
| 100 | 2 | 988,486 | 140.1 | 183.0 | 193.5 | 10,219 | 11 | 95.3 | 340 | 3.92 | 0 |
| 100 | 4 | 988,394 | 239.5 | 372.6 | 392.9 | 10,063 | 10 | 93.1 | 325 | 2.98 | 0 |

- Single-stream decode 181.5-184.4 against 161.0-169.4 in the September
  grid. The extra SMs did not do it. The draft-vocab A/B measured
  181.4-182.3 on the 70-SM cards with the same tool, so the step is
  patch 8; the silicon change is flat on decode, as expected when HBM
  bandwidth and the 200 W cap bind first.
- Prefill and TTFT unchanged: peak 10.4k tok/s against 10.3k,
  full-window TTFT 91.9 s against 92.2 s. Four more SMs sharing a fixed
  200 W clock lower; compute-bound phases gain nothing measurable.
- 25% 4-stream sust 548 against September's 404. The sust column is not
  a steady-state rate; see the repeat-grid forensics below. End to end: wall
  98.7 and 97.9 s against 105.3, about -7% at 25% fill, flat elsewhere.
- Acceptance holds 3.85-3.93 in every single-stream cell. Batch cells
  dip where a late stream's prefill interleaves another's decode. Zero
  preemptions in all 12 cells.

### 1M grid, September 2026 (70 SMs, 12,12,12,12)

The superseded grid, kept because the 524k and 262k tables below were
measured on that configuration. Measured 2026-09-19, the first 4-card
numbers. Tree at base fc8132a5e5; the compiled artifacts were still the
nightly wheel (`+gdc6954d14`) until the pin is applied. KV pool
2,541,795 tokens, maximum concurrency 2.54x at the 1M window (3-card
box: 1,208,978 tokens, 1.21x). Per-rank KV: 22.88 / 24.31 / 24.31 /
22.29 GiB. Raw: `bench-4x-1m-conc.json`.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 246,938 | 21.4 | 21.4 | 27.7 | 8,910 | 37 | 161.0 | 163 | 3.90 | 0 |
| 25 | 2 | 247,166 | 32.6 | 42.6 | 53.6 | 9,217 | 38 | 68.2 | 335 | 3.25 | 0 |
| 25 | 4 | 247,082 | 59.7 | 92.7 | 105.3 | 9,388 | 39 | 44.5 | 404 | 3.71 | 0 |
| 50 | 1 | 494,227 | 44.1 | 44.1 | 50.4 | 9,801 | 20 | 162.9 | 166 | 3.94 | 0 |
| 50 | 2 | 493,967 | 67.0 | 87.5 | 96.5 | 10,241 | 21 | 87.3 | 303 | 3.93 | 0 |
| 50 | 4 | 494,277 | 115.4 | 181.3 | 196.2 | 10,080 | 21 | 43.6 | 395 | 3.94 | 0 |
| 75 | 1 | 741,484 | 67.5 | 67.5 | 73.5 | 10,083 | 14 | 169.4 | 171 | 3.84 | 0 |
| 75 | 2 | 741,474 | 102.8 | 134.5 | 145.3 | 10,209 | 14 | 77.8 | 265 | 3.92 | 0 |
| 75 | 4 | 741,602 | 173.9 | 269.9 | 287.2 | 10,328 | 14 | 46.2 | 296 | 3.93 | 0 |
| 100 | 1 | 988,429 | 92.2 | 92.2 | 98.5 | 10,037 | 10 | 162.9 | 163 | 3.93 | 0 |
| 100 | 2 | 988,420 | 140.1 | 183.3 | 215.5 | 9,174 | 10 | 23.0 | 331 | 3.26 | 0 |
| 100 | 4 | 988,535 | 239.2 | 371.1 | 392.0 | 10,087 | 10 | 47.5 | 296 | 3.76 | 0 |


- The 75/1 row is a clean rerun. The first attempt read win avg 39.7 and acc
  3.41; an immediate rerun gave 169.4 and 3.84, so the dip was contamination
  in that window, not depth.
- Prefill got ~20% faster: 10.0-10.3k tok/s against 8.3-8.8k at equal depth
  on 3 cards. Under PP the prefill pace is set by the heaviest stage, not by
  the sum of stages, so four even 12-layer stages beat 16/17/15. Full-window
  TTFT 92.2 s against 119.1 s.
- Single-stream decode is unchanged: sust 163-171 tok/s (3 cards: 163-166).
- Batched decode trades a little per-stream speed for the bigger pool: 25%
  N=4 sust agg 404 against 3-card 456 (one more pipeline hop per step). The
  win shows at depth: 100% N=4 sust 296, where the 3-card box could only keep
  one 1M stream resident (163) and the rest queued.
- Acceptance 3.71-3.94 everywhere: PP=4 costs spec decode nothing.
- Zero preemptions in all 12 cells, same admission-gating behavior as 3x.

### 1M, 4-stream repeat grid (metric forensics)

Trigger: 25/4 sust read 404 (September) against 548 (restored-silicon
grid), while wall time differed by only 6%. All four 4-stream cells rerun
once on the same dev649 boot, fresh salts, idle gaps between cells. The
sampler timeseries is in each raw JSON. Raw: `bench-4x-1m-n4-rerun.json`.

| fill % | wall s Sep/grid/rerun | sust Sep/grid/rerun | acc Sep/grid/rerun | rerun peak at |
|---:|---:|---:|---:|---:|
| 25 | 105.3 / 98.7 / 97.9 | 404 / 548 / 545 | 3.71 / 3.91 / 3.92 | 95% of run |
| 50 | 196.2 / 197.4 / 197.0 | 395 / 383 / 493 | 3.94 / 2.92 / 3.69 | 97% of run |
| 75 | 287.2 / 293.7 / 309.2 | 296 / 236 / 284 | 3.93 / 2.24 / 2.44 | 92% of run |
| 100 | 392.0 / 392.9 / 390.2 | 296 / 325 / 291 | 3.76 / 2.98 / 2.93 | 48% of run |

- sust is the best 1.5 s in the cell, and in three of four cells that
  instant sits in the final 8% of wall time: streams finish at staggered
  instants, and whatever is left decodes near single-stream speed. One
  survivor at 183 tok/s plus two finishing bursts writes 545. The metric
  describes the wind-down, not the batch. Use wall or out tok/s to compare
  grids; use sust only inside a cell family.
- The real 25/4 difference is the -7% wall gain of the dev649 tree over
  September's, confirmed twice. 50/4 and 100/4 are flat within 1 s.
- Batch acceptance at deep fills is lower on this tree than September's,
  reproduced across two independent 4-stream runs: 75/4 reads 2.24 and
  2.44 against 3.93, 100/4 reads 2.98 and 2.93 against 3.76. Single-stream
  acceptance is untouched (3.85-3.93). September's grid had isolated batch
  dips (25/2 at 3.25), so some variance is normal; a repeat pattern at 75
  and 100 is new. Candidate causes: the draft-vocab head at deep positions
  under batch, or base drift between the trees. A draft-vocab-off boot at
  these two cells is the separating experiment; wall time is unchanged
  either way, so the effect is on acceptance, where it shows as the draft
  being rejected, not as slower decode.

#### The dip is the joining window, not the batch

Closed 2026-10-10. The separating boot (patch 8 disengaged) had already
said "draft head, not the tree": 75/4 and 100/4 read 3.85 / 3.95 with
the full head. The next question was *what about the head only misbehaves
at batch and depth*. The discriminator is decode length: same fill, same
4 streams, but a tail long enough that pure-decode rounds dwarf the
joining windows. Boot: W8A16 lane, Humming kernel, 40k vocab head. Raw:
`bench-e1-dvon.json`, `bench-e2-dvon.json`; first single-shot reads
(3.68, 3.99) in the session log.

| cell | decode/stream | acc samples | mixed-window share of wall |
|---|---:|---|---:|
| 100/4 | 1024 | 2.98, 2.93, 2.93 | ~60% |
| 100/4 | 8192 | 3.99, 3.99 | ~20% |
| 50/4 | 1024 | 2.92, 3.69 (bimodal) | ~45% |
| 50/4 | 4096 | 3.98 | ~20% |
| 25/4 | 1024 | 3.91, 3.92 | ~15% |
| conc<=2, all fills | 1024 | 3.71-3.95 | rounds stay uniform |

- Acceptance tracks the mixed-window share monotonically at fixed batch.
  Methodology scar: the first E1/E2 captures were contaminated by an
  unrelated prompt on the box (visible only in wall time: 281 s vs 218 s
  on E1) and read 3.63-3.72; the idle-box reruns replaced them, and E2
  then repeated 3.99 exactly twice.
- The structural fit: the drafter takes a CUDA graph only when the whole
  batch is uniform decode (`get_uniform_decode_token_count`); any real
  prefill in the batch forces the eager step-0 draft path. The small head
  degrades only on that eager mixed-window path.
- Shipped as patch 8 v3 (dual-head): the full head stays resident and the
  speculator sets `model.full_head_window = prefill_batch_desc.cg_mode !=
  FULL` before drafting, so joining windows score the whole vocabulary and
  graphed uniform rounds replay the small head (capture always sees the
  flag false). Rank 3 absorbs the 1.27 GiB with rank 0 still binding the
  pool; verify the pool line at the next boot, within 1k tokens of the
  small-head boot.
- Real traffic reading: long-lived sessions pay nothing (rare windows);
  bursts of joins on long contexts draft full-head quality during churn.

#### Corpus acceptance: sampling hides the window (v2 baselines)

`corpus_accept_probe.py` reruns the joining-window experiment on real
maki-session text at the clients' own sampling defaults (T=1.0, top_k=20),
4 streams staggered 1.5 s. Raw: `quality/corpus_v2_*.json`, one seed.

| cell | window depth | join share of span | acc |
|---|---:|---:|---:|
| light join | 45k tok prompts | ~15% | 2.306 |
| long window | 45k, decode 12288 | ~8% | 2.313 |
| heavy join | 250k tok prompts | ~40% | 2.371 |

- On sampled corpus text the acceptance sits at ~2.3 regardless of join
  share: temperature + top-k set the floor and eager-path draft degradation
  does not surface. The synthetic greedy grid, where acceptance is an
  argmax contest, is the regime that exposed the 2.9-vs-3.99 split.
- Practical reading: patch 8's measured harm on this box lives in
  greedy/T0 traffic; agent traffic as configured here samples and is
  unaffected by the window. v3 remains correct to ship (T0 users exist,
  the gate is free), but its expected win on real sessions is ~0, and
  2.31 is the number to hold v3 against, not 3.9.

#### v3 dual-head: measured (numbers voided by f660b7e, see depth ladder)

Boot 2026-10-11 22:08, W8A16 lane, Humming kernel, 40k vocab, tree
9a7a4cd46e. VOID NOTE: this boot carried the broken gate predicate, so
"v3" here behaved as always-full-head; the 3.73-3.92 acceptances match
the dv-off control because they WERE dv-off behavior, and the "2.44
outlier" was the post-idle first-cell artifact identified 2026-10-12.
The gate's real validation is the depth-ladder section below. Pool 2,620,743, identical to v2 to the token: the full head
lives in rank 3's slack, exactly as designed. Raw: `bench-v3-100n4-short
/r2/r3.json`, `bench-v3-50n4.json`, `quality/corpus_v3_*.json`.

| cell | v2 acc | v3 acc |
|---|---|---|
| 100/4 greedy short tail | 2.93, 2.98, 2.93 | 2.44, 3.73, 3.76, 3.92 |
| 50/4 greedy short tail | 2.92, 3.69 | 3.93 |
| corpus heavy join, T=1 | 2.371 | 2.259 |
| corpus light join, T=1 | 2.306 | 2.120 |
| corpus light join, T=0 | (never run) | 3.048 |

- The gate fires where predicted: greedy short-tail acceptance left the
  tight 2.93-2.98 band in three of four samples (3.73, 3.76, 3.92; the
  2.44 outlier was the first heavy cell after boot, kept in the record).
  The v3 band 3.73-3.92 is closing on the dv-off ceiling of 3.95.
- Sampled corpus cells read -0.1 to -0.19 against v2. With one sample
  per cell on v3's side and no v2 greedy-corpus reference, this is
  unresolved rather than a regression; wall and sustained rates are
  unchanged, so no cost materialized in throughput either way.
- Decision: keep v3. The proven regime (greedy churn) improved, the
  pool and decode rates are untouched, and the sampled-corpus spread
  sits inside what five v2 samples showed across similar cells. Open
  if anyone cares to close it: a v2 greedy-corpus baseline and one
  more 100/4 repeat of each tree.

### MTP depth is content-bound: the 4-vs-3 sweep (2026-10-12)

W8A16 lane, v3 dual-head, depth 4 via `num_speculative_tokens: 4`.
Boot gotcha: every extra draft row grows the hybrid scheduler block
size by the match unit (1600 -> 1616 at depth 4, logged as "kv cache
group sizes"), so `--prefix-cache-retention-interval` must track it
(16160). Pool cost: 2,620,743 -> 2,596,273. Raw: `bench-d4-*.json`,
`quality/corpus_d4_*.json`.

| workload | E(d3) | E(d4) | tok/s d3 | tok/s d4 |
|---|---:|---:|---:|---:|
| synthetic repeat 100/1 | 3.92 | 4.88 | 191 | 198.6 |
| synthetic repeat 50/1 | 3.93 | 4.91 | 190 | 198.0 |
| corpus, greedy | 3.048 | 3.006 | (span 45.1 s) | (span 60.1 s) |
| corpus, T=1 top-k | 2.12 | 2.417 | ~58/stream | ~63/stream |

- Per-position acceptance at depth 4 on synthetic: 0.986 / 0.976 /
  0.967 / 0.952, the fourth position holds, +3.7 to +4.2% tok/s.
- On real assistant text greedy, E is FLAT (3.006 vs 3.048): the fourth
  draft token is nearly always rejected while its drafting iteration
  (~4 ms) is always paid. Result: 25-30% slower. Sampled T=1 sits
  between: E +0.3 roughly cancels the step cost.
- Conclusion: fixed depth 4 wins only on highly repetitive content;
  production stays at depth 3. The community MTP6 claim (E 4.46) is
  the synthetic curve, not real text: content decides, and their
  single-run totals cannot show that. Depth 5/6 were not booted: the
  real-content curve already says extra positions cost and do not pay.
- vLLM has `enable_adaptive_verification` (per-request draft budget)
  but it is gated to method=dspark. An MTP early-exit ("stop drafting
  after trailing rejections", the megakernel project does this) would
  be the straight version of this feature; on the local work list, not scheduled.

#### The depth ladder on the fixed stack: ship 4 (2026-10-12)

All numbers post f660b7e (correct gate predicate), Marlin lane, W8A16 +
v3 dual-head + 40k draft vocab, same stack across depths, cooldowns
between cells. Two boots at depth 5, one at 3 and 4.
Raw: `bench-fix-*.json`, `bench-fixd4-*.json`, `bench-d5*.json`,
`quality/corpus-*d5*.json`, `quality/corpus-fixd4-*.json`.

| profile | d3 | d4 | d5 |
|---|---:|---:|---:|
| 100/1 tok/s | 192.2 | 226.2 | 233.6 / 241.0 |
| 50/1 tok/s | 190.2 | 216.6 | 237.9 |
| E synthetic | 3.94 | 4.88 | 5.59-5.80 |
| 100/4 wall s | ~404 | ~400 | 400.5 |
| 100/4 win avg | 45-54 | 48-53 | 66.0 |
| corpus T=1 rates | ~63 | ~64 | ~65 (neutral x3) |
| corpus greedy rates | not measured | 101-124 | 74-113 / 79-100 (mean ~89, x2) |
| pool tokens | 2,619,195 | 2,594,720 | 2,573,208 |

- Depth pays only on greedy/low-temp content; sampled real traffic is
  flat from 3 to 5 (E ~2.1-2.4, rates ~63-65). The gain is not free:
  real-text greedy loses ~15% going 4->5 (fifth iteration never pays on
  content where the chain dies at position 2-3), and each depth costs
  ~1% pool plus the block/ring config dance (d5 needs --block-size 1632
  for the QSA ring, retention 16320).
- Measured control, same session, official FP8 tree at depth 4:
  211.6 / 214.5 / 212.4 tok/s (median of three, E 4.85-4.93, TTFT 90.6 s,
  pool 2,572,981 = lane minus the int8 byte savings, consistent). The
  depth gift transfers: +16 % of the lane's +20 % is depth, the rest is
  int8. Prefill also did NOT regress at depth 4 on the official tree
  (10.35k vs the 10.1-10.2k d3 band), so the lane's "-2 % prefill" is a
  lane-vs-official delta (Marlin unpack), not a depth cost.
  Batch control added same session: 100/4 median of three = wall
  386.3 s (vs 392-393 official d3, 399-405 lane d4), TTFT 235-237,
  acceptance 3.59-4.90 median 4.36 vs the small-head 2.93-2.98. The
  depth gate combination is the best 4-stream wall this box has
  recorded on the official tree.
  Corollary for lane selection: at batch full-fill the official tree
  now beats the int8 lane (386 s vs 399-405 s) - the Marlin prefill
  tax dominates where prefill is the wall and extra resident bytes do
  not pay. Lane profile, stated plainly: it wins single-stream and
  low-concurrency deep context (+6% over official at identical depth),
  and loses ~4% at 4x1M full fill. Choose by workload, not by headline.
- Ship depth 4: best real-content profile at T=0, neutral-to-better
  elsewhere, +15% synthetic single-stream over d3, +20% vs the official
  FP8 tree's 181-184.
- PROTOCOL, new: the first heavy decode cell after a long idle can
  collapse acceptance (d5 boot2: E 1.19 / 51.8 tok/s; v3 boot's 2.44 the
  same species). Immediate rerun restored 241.0 / E 5.80. Discard the
  first cell after long idle; two boots of d5 would have been misread
  without this.

### Budget sweep v1 (2026-10-12): partial by construction, two columns
missing

`logs/budget-d4.json`, min-composed, m=1/5/50. Findings:
- bf16 dense big shapes (12288x2560, 10240x2560, 4608^2, 6144x2560) run
  580-790 GB/s solo = 60-78% of the 1.1 TB/s ceiling: the bandwidth
  regime where halving bytes pays.
- Everything <= ~1.6M elements pins at 24-30 us across m=1..50: solo
  launch-latency floor. In-graph pipelining hides most of this; only a
  production A/B can price L2 policy here.
- The dense fp8-block rows (27-39 GB/s, 2-3x slower than bf16 on the
  SAME tiles) measure a path production never calls (experts run
  grouped). Kept as the grouped kernel's tuning cousin to test, not as
  a budget line.
- Verdict: the step budget cannot be closed from this file. Missing
  columns: grouped-MoE expert GEMM (the actual expert path) and int8
  pack-quantized Marlin (the lane's dense path). Bench extension
  underway; until then L2-vs-kernel priorities are claims without a
  map.


#### L2 experiment 1: QSA KV evict-last (experiments/0009)

The 60/40 split (bytes vs latency/state, from the budget sweep plus
ceiling arithmetic) says attack the 40% in production rather than
extend the microbench. The megakernel recipe inverts on our stack:
their weights lived in their kernels, ours live in cuBLAS/humming/
Marlin where loads cannot be tagged - so the poachable policy is the
opposite end: mark the QSA sparse-paged attention kernel's KV loads
evict_last so the 2048-token working window survives L2 across the
four draft iterations and five verify rows instead of being flushed
by the gigabytes of GEMM streaming around it.

Protocol: apply to the serving tree (python-only; Triton recompiles
at first step), fixed-salt cell x3 with stepwatch, reboot unpatched,
x3 again, compare decode step_ms medians. Upside case: single-digit
percent. Null case, pre-registered: split-KV touches disjoint tiles
and index overlap across draft iterations is too small to matter.

#### L2 experiment 1: FAILED - not a miss, a corruption class. REVERTED

Patched boot, fixed-salt 7770 cells: acceptance 1.59/1.67/1.60
(unpatched reference 4.75-4.91 on identical prompts), decode ~71-74
tok/s while steps ran FASTER (~15 ms) because collapsed acceptance
empties the verify pipeline. A cache hint that changes values is a
driver/hardware story, and this silicon already owns one documented
load-path quirk (Xid 31 first-touch MMU faults on the unlocked GSP
stack). Verdict: eviction-priority PTX hints corrupt loads on this
box; the entire L2 poaching line is CLOSED - KV evict_last, and by
extension the planned hc.py/indexer trials and any evict-first
variant. Revert = checkout qsa.py; post-revert fixed-salt acceptance
must return to ~4.9.
Rule extracted, permanent: on modified-driver GPUs, every kernel-level
trial reads fixed-salt ACCEPTANCE first, timing second - timing can
look great while values rot. Budget consequence: the 40% latency/
state share of the step now has no cheap lever left; the only
informed next move for it is phase-level instrumentation (a real
tool, scheduled deliberately), or nothing.
Capture: logs/sw-l2-patched.jsonl (364 KB, with clocks - first nvml
run).

### Rebase certification (2026-10-13): base 183625d4f9, 7-patch series

Boot banner dev41+g5cddeba8a (base dev34 + 7 local am), fender tree replayed
to the scrubbed patches. One behavior delta from +504 upstream commits: the
pool budgets CUDA graph memory now, 2,572,981 -> 2,423,913 tokens
(2.57x -> 2.42x); --gpu-memory-utilization 0.94 is otherwise unchanged.
Fixed-salt 7770 cells, decode 1024, same protocol as the old-base controls:
single-stream sust 209/212/211 (control 208-215), acceptance 4.93/4.84/4.93
(control 4.75-4.91), cold TTFT 91.8 s and warm 4.8 s repeats (salt law
reproduced). 100/4 triple wall 301/388/388 (cell 1 prefix-cache-warm from
the preceding run), median 388 s against the 386 s control (+0.5%),
sust 300-315, acceptance 4.83-4.86, zero preempts. Verdict: rebased series
performs identically within noise; the smaller pool costs nothing measured
at 1M so far. Raw: logs/rebase-7770-c{1,2,3}.json,
logs/rebase-1004-c{1,2,3}.json. Follow-up same day: utilization raised to
0.9453 (the boot log's CUDA-graph-equivalence suggestion for the binding
rank) to restore the pre-rebase pool. Three full 1M streams resident is
NOT reachable on 4x64 GiB with these weights: the need is ~2.98M tokens
and even 0.99 utilization with a perfect split tops out near 2.9M; 0.9453
buys back the ~2.57x and nothing more. Re-verification cells owed after
the boot: fixed-salt 7770 x3 and 100/4 x3.

### Decode step anatomy (2026-10-13): the 60/40 question is closed

Tooling: PROF=1 gate in serve-4x-1m.sh (torch profiler armed with
max_iterations=150 so the window self-closes, no stacks, frontend ignored),
benchmark/phase_probe.py (runs the salt-7770 100%-fill cell and triggers
/start_profile in steady decode), benchmark/phase_report.py (parses the
four per-rank chrome traces). The runner phase spans came back empty:
serving uses the V2 model runner (boot log "Using V2 Model Runner"), whose
files lack the record_function markers - the gpu_model_runner: ones are V1.
Kernel-level attribution turned out richer anyway. Capture: logs/prof/
(4 rank traces + profiler_out_*.txt), summary at logs/phase-report-7770.txt.

Window facts: 150 worker steps in 0.93 s = 37 real decode steps
(execute_context_0(0)_generation_1(5)) interleaved with 113 empty-batch
draft-wave spans (~3 per step, the drafter's extra worker steps at d4; rank
3 replays 5 CUDA graphs per step, ranks 0-2 one). CUPTI inflates cadence
21.6 -> 25.5 ms; read ratios, not absolutes. Acceptance held 4.98 through
the window: profiling cost nothing measurable.

| per real step | r0 | r1 | r2 | r3 | sum |
|---|---:|---:|---:|---:|---:|
| compute, ms | 6.20 | 5.76 | 6.08 | 8.87 | 26.9 |
| nccl-spin, ms | 19.4 | 33.7 | 28.3 | 16.6 | 98.0 |
| true GPU gap, ms | 0.47 | 0.05 | 0.06 | 0.19 | 0.8 |

1. The step IS the serial sum of per-rank compute (26.9 ms ~ the inflated
   cadence; clean sum ~22.9 vs 21.6 observed). NCCL kernels occupy the
   GPUs to WAIT (one >2 ms broadcast per step per rank is the whole
   bubble; actual send/recv payload measures 0.2 ms/step). The PP bubble
   is structural for single-stream and shrinks with concurrency. There is
   no comms lever. Confirmed dead: any "reduce NCCL overhead" idea.
2. The 40% latency/state side, now attributed (ms/step summed over ranks):
   byte-side gemm-bf16 10.2 + humming 7.6 = 17.8 (66%); the rest is
   qsa-attn 2.3, gdn/state 1.8, moe-routing (gating/align/sort) 1.5,
   norm 1.2, copy/memset 1.0, elementwise+other 2.1, sampler 0.02. No
   single monster - it is the per-layer state machinery and glue, ~40
   kernels x ~5 us per layer.
3. Rank 3 (attention head + drafter) is the critical path: 8.9 vs ~6.0 ms.
   Its excess is draft gemms (ampere 64x64 x36/step) plus double qsa.
   PP re-partition cannot help: wall = SUM of stage compute; balancing
   moves the same total. Only shrinking total compute works.
4. Launch-tax ceiling is modest: 2540 compute kernels/step across the
   pipeline, median 3.2 us, but everything <5 us sums to only 3.9 ms/step
   and <2 us to 1.3. Perfect glue fusion cannot beat ~2-3 ms/step
   (~10%), and it is exactly the kernel-mod class that requires the
   fixed-salt acceptance check.

Concrete microbench targets (the two missing budget-sweep columns, shapes
taken from trace launch args):
- humming grouped GEMM Shape<0,1280,2560> tile 16x256x128, grid 74
  (= SM count, persistent), 95 us, 13/step/rank = 1.25 ms/rank/step, plus
  its Shape<0,2560,640> sibling at 56 us. Replay at true m=5-rows/topk
  expert mix; compare to the 580-790 GB/s solo band.
- ampere_bf16_s16816gemm_64x64_sliced1x2, grid 256, 63 us x 27/step/rank:
  splitK at m=5 smells like a reduction round-trip tax; solo replay.
- _qsa_mqa_paged_uniform grid [1,3939] block 64: 51 us x 13/rank/step at
  1M depth - check against the splitk variant and the byte SOL.

Follow-up, same day (microbench replays, container venv, server idle but
co-resident; raw json logs/bench-micro-ms5.json):
- bf16 splitK suspect: REJECTED. m=1, 5 and 25 timings are identical to
  within 1-2% on every dense shape; big shapes land at 568-767 GB/s, the
  established solo band. cuBLAS handles decode-m fine.
- humming grouped column: CLOSED within physics. The distinct-variant dump
  shows exactly two grouped kernels per layer-step (gate_up 1280x2560 at
  96 us, down 2560x640 at 56 us), both persistent grid=74. At 4.92 MB per
  fp8-block expert, that time budget cannot exceed ~20 unique experts per
  layer-step (512 experts, topk 10, 50 slots; the 5-row greedy draft chain
  evidently overlaps routing heavily). That puts grouped execution at
  325-650 GB/s - consistent with near-SOL, no 20x lever. The 26-35 GB/s
  dense fp8 single-tile rows are the dead path again; do not quote them as
  production. Exact unique-expert count would need an instrumented call on
  the serving tree; the ceiling bound stands on its own.
- qsa replay: deferred; needs the current tree (container vllm is
  pre-patchset).
Net verdict: single-stream 212 tok/s is the roofline of this checkpoint on
4x170HX@200W at d4. The step is 66% byte-side at solo band, 34% diffuse
state/glue (fusion ceiling ~2-3 ms, gated behind the acceptance rule), and
the visible "NCCL" time is structural PP bubble. Remaining levers: batch
concurrency (shipped), byte-reducing quantization (declined below int8),
or a different checkpoint. No further instrumentation owes.

### Salt law (2026-10-12, supersedes "post-idle" and "bimodal machine")

Fixed-salt triples: acc 4.90/4.91/4.75, decode 208-215 (+-1.5%).
Time-salted triples on the same config: acc 3.2-4.9, decode 145-213.
The scatter this session kept blaming on the box (post-idle cells, the
176<->216 flips, the v3 "2.44") is the prompt lottery at depth 4+:
acceptance carries the content, clocks do not. TTFT never moved.
Protocol, final form: comparisons pin --salt; headline ranges rotate
salt and quote the band; identical fixed salt re-uses the prefix cache
(itself now demonstrated: 1M-token re-prompt at 4.8 s TTFT, 103k
tok/s in, vs 90.7 s cold). The old "first cell after idle" folklore is
retired - the cells were never identical to begin with.

Stepwatch block audit of the same six cells (remote 20 Hz): decode
step_ms median 21.3-21.5 in EVERY block, fixed and rotating salt
alike, while acc medians swing 2.75-5.00 with throughput. Machine
acquitted by instrument, not arithmetic. Two open threads from the
capture: a stable p90 ~32 ms second mode in all blocks (suspect poll
aliasing from remote 20 Hz sampling; localhost rerun decides) and an
empty nvml column (--nvml needs pynvml on the serving box; fallback to
nvidia-smi subprocess queued).
  GHOST RESOLVED: localhost 20 Hz capture on a fixed-salt cell reads
  step_ms 21.6 / it_s 46.3 / acc 5.0 / gen 231 - one clean cadence,
  arithmetically closed. The remote p90 ~32 ms was poll aliasing, not
  engine periodicity; the tok_it<=8 filter now drops mixed-edge
  samples from decode stats.

#### Indexer sharing across draft steps: voided, retest pending

Retest 2026-10-12, fixed d4 stack (depth 4, Marlin, 40k head, warmup
cell first): flagged samples read E 4.78 / 4.88 vs control 4.88 / 4.89
with identical TTFT and in-band step rates - no upside anywhere, no
downside beyond noise. Verdict: dead on this config, as the indexer
size math predicted; the flag stays off. The retest also produced a
protocol finding stronger than its answer: two cells flipped between
216 and 176, and 222 and 187, minutes apart on an idle box - bimodal
cells are NOT limited to the post-idle case. Single-cell comparisons
are worthless here; ladder verdicts now require median of three. Three
separate flag-verdicts this week would have been wrong without repeat
cells.

NOTE (2026-10-12): this section's measurements ran on the broken v3 gate
(full head on every round, f660b7e era predecessor). Flag-on 175.9 and
clean-Humming 176.9 were both the bug, not the flag. The conclusion
below stands only as history; a valid retest on the fixed depth-4 stack
is the next queued experiment.

Original text follows.

The upstream `index_share_for_mtp_iteration` flag (hooks implemented in
`qwen4_exp/nvidia/mtp.py`, driven by the MTP speculator) is the free
version of the Qwen-Flash-SM80-170HX overlay's "QSA indexer at draft
step 0 only" trick. Verified via the config classes that
`--hf-overrides` carries it into the draft config, booted at depth 3:

| cell | control (v3 boot) | indexer-share | delta |
|---|---:|---:|---:|
| 100/1 tok/s | 190.9-192.4 | 175.9 | -8% |
| 50/1 tok/s | 189.8-190.7 | 177.5 | -7% |
| E (100/1) | 3.90-3.95 | 3.879 (p2=0.939) | down |

Step time grew ~7% and acceptance sagged: the compact/reuse machinery
costs more than the indexer passes it skips here (12 QSA layers only,
indexer_head_dim 128 at budget 2048 - small fry; on the overlay's
NVFP4/TP2 config the relative cost is clearly different). Flag reverted,
kept out of the scripts. Revisit only if the draft path grows heavier
(deeper MTP with cheap iterations) or on hardware where the indexer is
a bigger share of the step. Raw: `bench-share-100n1.json`,
`bench-share-50n1.json`.


### 524k (YaRN 2.0)

Measured 2026-09-21, same tree. KV pool 2,496,752 tokens, maximum
concurrency 4.76x at the 524k window (3-card box: 1,204,810 tokens,
2.30x). Per-rank KV 23.14 / 24.72 / 24.72 / 20.79 GiB, loads 33.83 /
33.77 / 33.77 / 37.57 GiB (rank 3 again the cap: 12 layers + drafter).
Raw: `bench-4x-512k-conc.json`.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 129,516 | 11.1 | 11.1 | 17.3 | 7,471 | 59 | 164.6 | 168 | 3.95 | 0 |
| 25 | 2 | 129,512 | 16.8 | 21.9 | 29.2 | 8,861 | 70 | 105.4 | 313 | 3.93 | 0 |
| 25 | 4 | 129,354 | 28.9 | 45.3 | 79.7 | 6,490 | 51 | 22.0 | 473 | 3.84 | 0 |
| 50 | 1 | 258,890 | 22.5 | 22.5 | 28.8 | 8,993 | 36 | 163.5 | 169 | 3.92 | 0 |
| 50 | 2 | 258,784 | 34.1 | 44.5 | 52.4 | 9,878 | 39 | 94.2 | 309 | 3.94 | 0 |
| 50 | 4 | 258,857 | 58.4 | 91.2 | 102.8 | 10,072 | 40 | 48.4 | 497 | 3.93 | 0 |
| 75 | 1 | 388,474 | 34.4 | 34.4 | 40.6 | 9,563 | 25 | 164.0 | 169 | 3.93 | 0 |
| 75 | 2 | 388,526 | 52.0 | 68.0 | 76.3 | 10,180 | 27 | 90.1 | 309 | 3.94 | 0 |
| 75 | 4 | 388,444 | 89.6 | 140.7 | 154.5 | 10,055 | 27 | 42.9 | 404 | 3.92 | 0 |
| 100 | 1 | 517,766 | 46.0 | 46.0 | 52.0 | 9,950 | 20 | 168.4 | 166 | 3.98 | 0 |
| 100 | 2 | 517,865 | 70.2 | 91.7 | 101.4 | 10,219 | 20 | 80.4 | 275 | 3.89 | 0 |
| 100 | 4 | 517,707 | 118.3 | 182.9 | 198.1 | 10,453 | 21 | 41.1 | 367 | 3.79 | 0 |

- The 100/1 row is the third sample. The sweep run read TTFT 46.7 but win
  avg 37.8 and acc 3.54; a rerun 100 s after the sweep read 129.9 and 3.28;
  after ~4 minutes idle it hit 46.0 / 168.4 / 3.98, on the flat baseline.
  This cell only reads low right after the N=4 cells, so the box needs a
  few minutes idle between a 4-stream run and a single-stream measurement.
- Full-window TTFT 46.0 s against 58.9 s on 3 cards (-22%), same heaviest-
  stage effect as the 1M lane. Peak prefill 10.2-10.5k tok/s.
- Single-stream decode flat 166-169 tok/s from 129k to 518k depth.
- The 2.50M pool changes the concurrency picture: 4 streams at 50% fill
  (demand ~1.04M) decode together at 497 agg, and even 4 streams at full
  517k prompts (demand ~2.07M, which on 3 cards sat at 2.30x) sustain
  367 agg. Zero preemptions, acceptance 3.79-3.98 everywhere.
- Warm N=2 re-test 2026-09-21: bench cell repeats 265 and 265 (sweep value
  275), deep plateau 270. See "524k 2-stream head-to-head" for the 3-card
  comparison and the discarded shallow samples.

### 262k (native RoPE, no YaRN)

Measured 2026-09-21, same tree. KV pool 2,367,797 tokens, maximum concurrency
9.03x at the 262k window (3-card box: 1,150,599 tokens, 4.39x). Rank loads
33.71 / 33.64 / 33.64 / 37.44 GiB. Raw: `bench-4x-262k-conc.json`; the 75%
row and the 100/1 cell come from reruns (`bench-4x-262k-recheck75.json`,
`bench-4x-262k-recheck100.json`). The 3-card 262k table above ran decode
128 per stream, so it is not directly comparable; the 3-card comparators here
are the 512k decode-1024 rows.

| fill % | conc | each tok | TTFT mean s | TTFT p95 s | wall s | in tok/s | out tok/s | win avg tok/s/req | sust agg tok/s | acc len | preempt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1 | 64,278 | 5.6 | 5.6 | 12.3 | 5,208 | 83 | 151.3 | 269 | 3.71 | 0 |
| 25 | 2 | 64,370 | 8.4 | 10.9 | 18.7 | 6,903 | 110 | 108.6 | 285 | 3.97 | 0 |
| 25 | 4 | 64,296 | 13.9 | 21.5 | 31.1 | 8,261 | 132 | 70.6 | 480 | 3.96 | 0 |
| 50 | 1 | 128,517 | 11.1 | 11.1 | 17.3 | 7,417 | 59 | 163.6 | 169 | 3.92 | 0 |
| 50 | 2 | 128,662 | 16.7 | 21.8 | 29.1 | 8,853 | 70 | 105.8 | 311 | 3.95 | 0 |
| 50 | 4 | 128,690 | 28.7 | 45.0 | 56.6 | 9,097 | 72 | 52.0 | 407 | 3.91 | 0 |
| 75 | 1 | 193,057 | 16.5 | 16.5 | 22.9 | 8,437 | 45 | 161.0 | 169 | 3.84 | 0 |
| 75 | 2 | 193,075 | 25.2 | 32.9 | 40.4 | 9,556 | 51 | 98.7 | 309 | 3.95 | 0 |
| 75 | 4 | 192,972 | 43.3 | 68.1 | 80.2 | 9,628 | 51 | 47.0 | 405 | 3.92 | 0 |
| 100 | 1 | 257,186 | 22.1 | 22.1 | 28.3 | 9,073 | 36 | 164.5 | 169 | 3.95 | 0 |
| 100 | 2 | 257,338 | 35.5 | 46.0 | 60.6 | 8,495 | 34 | 52.3 | 384 | 3.60 | 0 |
| 100 | 4 | 257,305 | 58.2 | 91.5 | 104.1 | 9,886 | 39 | 44.7 | 405 | 3.93 | 0 |

- The sweep's first 75% row read win avg 19.3 / 27.2 / 9.7 with prefill down
  at 2.8-5.9k tok/s while preemptions stayed 0. After ~4 minutes idle the
  rerun above returned to the flat baseline, so the dip was contamination in
  that window, same failure shape as the 524k 100/1 case. The 100/1 rerun
  matches its sweep value (164.5 vs 164.3).
- Single-stream decode 151-165 tok/s, the same band as the 1M and 524k
  lanes. The 25% cell reads lower because a 12 s window includes more
  drafter warmup.
- The pool turns the 262k lane into a batch machine: four full 261k streams
  demand 1.04M against 2.37M, all stay resident, and sustain 405 agg with
  zero preemptions. At 25% fill four streams hit 480 agg. The 3-card pool
  already admitted four full streams (1.04M against 1.15M, 4.39x), and its
  decode-1024 rerun sustains 457 agg at full fill, above this set's 405:
  one fewer pipeline hop per step. The 405 is confirmed warm: cell repeats
  404 and 404, deep plateau (`sustained_decode.py --depth 257000 --decode
  4096`) 405, shallow plateau 431 against 468 on 3 cards. An isolated 449
  from the crashed pre-grid attempt was a cold artifact and is not used.
  The 4-card gain here is headroom (nine full streams) and prefill speed,
  not batched decode.
- Full-window single-stream TTFT 22.1 s; four of them 58.2 s mean, 91.5 s
  p95. Prefill peaks 9.9k tok/s, close to the 1M lane's 10.1-10.5k.
- Acceptance 3.60-3.97 across all 15 runs.
- 100/2's low win avg (52.3) is queue pacing, not slow decode: the first
  stream's decode runs while the second 257k prompt still prefills. The sust
  column (384) is the real number.

## NCCL P2P, and TP=4 + EP, 2026-10-09

The driver was upgraded so the cards answer `cudaDeviceCanAccessPeer`, which
they did not before. Two layouts were measured against the tables above, same
tree (dev649), same 200 W cap. Box state: `nvidia-smi topo -m` puts every GPU
pair at NODE distance (PCIe plus the interconnect between host bridges, one NUMA
node), and link widths are x16 on GPU1, x4 on GPU0/2/3. In a ring every hop is
limited by the narrower end, so all four links sit near 2 GB/s.

### PP=4 1M, peer DMA versus the host-staged path

Nothing had to be done to get P2P at PP=4: NCCL already selects it at its
default level for the 2-rank send/recv comms that pipeline hops use. Reference
column is the restored-silicon grid above (pool 2,595,975; the P2P boot pooled
2,597,523, a 0.06% drift).

| cell | metric | P2P off, Oct 7 | P2P on, Oct 9 | delta |
|---:|---|---:|---:|---:|
| 25/1 | in tok/s | 9,181 | 9,012 | -1.8% |
| | TTFT s | 21.3 | 21.9 | +2.8% |
| | win avg tok/s/req | 181.5 | 185.1 | +2.0% |
| 50/1 | in tok/s | 10,038 | 9,750 | -2.9% |
| | TTFT s | 43.7 | 45.2 | +3.4% |
| | win avg tok/s/req | 184.4 | 186.7 | +1.2% |
| 100/1 | in tok/s | 10,138 | 9,841 | -2.9% |
| | TTFT s | 91.9 | 94.9 | +3.3% |
| | win avg tok/s/req | 183.2 | 183.4 | +0.1% |
| 25/4 | in tok/s | 10,022 | 9,774 | -2.5% |
| | TTFT mean s | 56.3 | 58.0 | +3.0% |
| | wall s | 98.7 | 101.1 | +2.4% |
| | win avg tok/s/req | 53.8 | 53.8 | 0 |

Acceptance 3.77 to 3.95, zero preemptions, four cells. Prefill and TTFT are
consistently 2 to 3% worse with peer DMA, single-stream decode is one or two
points better, batched decode is flat. PP moves one activation per stage
boundary per step, so P2P has little to win and something to lose. Keep P2P
disabled at PP=4, which is what every shipped script does. Raw:
`bench-4x-1m-p2p-n1.json`, `bench-4x-1m-p2p-n4.json`. Transport evidence is in
the untracked `boot-4x-1m-p2p.log`: `isAllCudaP2p 1`, three 2-rank comms
`isAllDirectP2p 1`, and rank 0 `Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM`.

### TP=4 + EP, 1M: rejected, script kept for reference

`serve-4x-1m-tp.sh` runs the same 1M lane with `--tensor-parallel-size 4
--enable-expert-parallel` instead of PP=4. At TP=4 NCCL refused peer DMA for
the collective rings until `NCCL_P2P_LEVEL=SYS` was exported, which is the
NODE-distance fix. vLLM's own custom all-reduce is unavailable at any setting:
`custom_all_reduce.py:236` turns it off for `world_size > 2 and not
fully_connected`, and `fully_connected` is an NVLink mesh test. Both boots:
weights 33.59 GiB on every rank, attention block size 800, KV pool 1,741,935
tokens (SHM) and 1,741,148 (P2P), 1.74x headroom against 2.60x at PP=4, so two
1M requests do not fit. EP did engage: 512 routed experts split 128 per rank,
with humming moving to `grouped_contiguous` GEMMs. Patch 8 was disabled for
both TP rows because its draft head is not TP-safe, so TP decode carries a
handicap of about 8.7%.

| cell | metric | TP, P2P off | TP, P2P on | P2P gain | PP=4, P2P on |
|---:|---|---:|---:|---:|---:|
| 25/1 | in tok/s | 1,195 | 1,434 | +20% | 9,012 |
| | TTFT s | 197.0 | 163.2 | -17% | 21.9 |
| | wall s | 206.9 | 172.3 | -17% | 27.4 |
| | win avg tok/s/req | 103.4 | 111.7 | +8% | 185.1 |
| | acc len | 3.62 | 3.73 | | 3.93 |
| 50/1 | in tok/s | 1,221 | 1,456 | +19% | 9,750 |
| | TTFT s | 393.3 | 327.5 | -17% | 45.2 |
| | win avg tok/s/req | 88.2 | 86.3 | -2% | 186.7 |
| | acc len | 3.25 | 3.18 | | 3.95 |

Zero preemptions. The 100/1 cell on the SHM boot was killed after 5.7 minutes
with no first token. Raw: `bench-4x1m-tp4-n1.json`, `bench-4x1m-tp4-p2p-n1.json`.

- TP loses 6.7x on prefill because it puts about two collectives per layer in
  every step, roughly 100 per decode step, against one hop per stage boundary
  at PP. On a 2048-token chunk each of those moves about 25 MB over a link that
  carries 2 GB/s, which lands almost exactly on the measured 1,434 tok/s. The
  lane is at its wire limit, not at a software limit.
- P2P bought only 20% here. A reference box with the same cards at the same
  NODE distance but Gen2 x16 measured 1.8 to 1.9x faster TP prefill with
  P2P, 2.5k against 4.6-5.0k tok/s. Peer DMA removes the host copy and the CPU
  synchronisation, and it cannot add bytes per second. Three of four cards here
  negotiate x4.
- `serve-4x-1m-tp.sh` exports `NCCL_P2P_LEVEL=SYS` and passes
  `--disable-custom-all-reduce`. Once peer access exists, the CUDA-IPC buffer
  path fails across root complexes with "invalid argument" and a VRAM blowup
  that crash-loops the workers, while NCCL keeps its own P2P path. The flag is
  a no-op for selection at TP=4 and matters at TP=2, where the world-size gate
  passes.
- Revisit only if the links change: check `lspci -vv` and compare `LnkCap`
  with `LnkSta` on GPU0/2/3; if `LnkCap` also reads x4 the slots are wired that
  way and no software change helps. One lever is independent of this
  comparison: the GPC voltage-frequency offset, where +200 MHz at the same 200 W
  is worth about +6% prefill and +7% decode on the reference box, and +250 hung
  a card under real load there. If TP is ever retried, make patch 8 TP-safe
  first (`disable_tp=True`), since both TP rows above run without the draft
  head.

## W8A16 dense lane (4x PP=4, 1M, SM74, 200 W)

Int8 group-32 weight-only on the 228 main-body dense matrices; experts,
PLE and sensitive tiers byte-identical to the official FP8 tree. Built and
verified with `../checkpoint/` (152,089-tensor byte gate, 0 fails; dense
rel-RMS mean 0.64 %, max 0.74 %). Served with series patch 10.
Raw: `bench-w8a16-s1.json` (boot 1), `bench-w8a16-boot2.json`
(boot 2), `bench-official-boot3.json` (official control, same session);
boot-1 reruns and recovered cells are logged in the table.

| fill % | decode official | decode W8A16 b1 | W8A16 b2 | prefill official | prefill W8A16 | acc |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 181.5 | 187.2 / 190.1 | 187.3 | 11,600 | 11,290-11,385 | 3.86-3.93 |
| 50 | 184.4 | 190.7 / 190.5 | 189.8 | 11,310 | 10,932-11,075 | 3.92-3.95 |
| 75 | 183.8 | 191.4 / 191.5 | 191.0 | 11,000 | 10,718-10,815 | 3.93-3.95 |
| 100 | 183.2 | 191.1 / 192.4 | 190.9 | 10,760 | 10,502-10,573 | 3.91-3.94 |

- Decode +3.1 to +5.0 %. Prefill -1.7 to -2.5 %. Pool 2,619,195 vs
  2,597,523 (+23k).
- The prefill loss is the weight-only DEQUANT itself, not the kernel:
  booting with `VLLM_DISABLED_KERNELS=MarlinLinearKernel` swaps every
  dense int8 layer to HummingLinearKernel and moves prefill <0.5 % at
  every depth (same grid, raw `bench-w8a16-humming.json`), while both
  stay ~2 % under cuBLAS bf16. Ampere unpacks int8 in registers per
  element; at saturated prefill issue that competes with MMA whatever
  streams the bytes. Decode side, the same swap read +0.5 to +1.0 % in
  all four cells over Marlin's two boots; one boot, so unconfirmed, and
  an env line away if it ever matters.
- Official side has three boots agreeing (Sept grid, Oct SM74 grid, the
  quality-session boot). W8A16 boot 2's 50/1 and 100/1 read acc 2.37 and
  2.79 mid-grid, recovered to 3.92 on idle rerun: the known
  transient-dip shape, logged from the boot-1 session on the same cells.

### Quality gate

`logprob_capture.py`: 256 corpus windows x 4k chars, top-1 token+logprob
at 166,532 positions, plus 64 greedy tails. `quality_diff.py` compares
captures; raw pairs in `quality/*.jsonl.gz`.

| comparison | top-1 flips | mean abs logprob delta | >0.5 nats | greedy exact |
|---|---:|---:|---:|---:|
| official, same boot x2 | 7.27-7.30 % | 0.0734 | 1.06-1.07 % | 39-48 % |
| official, boot 3 vs boot 4 | 7.36 % | 0.0738 | 1.12 % | 36 % |
| W8A16, same boot x2 | 7.27 % | 0.0734 | 1.07 % | 39 % |
| W8A16 vs official (2 boot pairs) | 8.09-8.18 % | 0.0815-0.0816 | 1.45 % | 34-38 % |

- This box is not temp-0 reproducible: 7.3 % of top-1 tokens move on
  identical weights and code, within or across boots. Any comparison
  without that floor stated is theater.
- Against the floor, the int8 conversion moves ~0.8 points: one extra
  top-1 change per ~130 positions, 1.45 % of positions drifting over 0.5
  nats. Acceptance is behavior-level agreement between drafter and
  target, and it is unchanged; that is the practical pass.

## Step telemetry (CUPTI-free)

`stepwatch.py` samples the engine's own per-step counters (plus optional
NVML clocks/power with `--nvml`, on the serving box) and separates slow
cells into machine (step_ms drift, clocks), draft (acc drift) or
scheduling (run/wait) causes. Intended use is per-cell: start it, run one
bench cell, stop it - windows spanning idle time dilute the acc estimate.

```
python3 stepwatch.py --out cell.jsonl --hz 20 [--nvml] &
python3 bench.py ... ; kill %1
python3 stepwatch_report.py cell.jsonl
```

First live run (official-d4, 25/1): prefill chunks 2048-token steps at
~8 ms, decode steady 21.6 ms median at tok/iter 5, gen-rate arithmetic
closing (21.6 ms x 5 tok = 232 vs measured 230).

## Reproduce

Run with the server idle, after warming it with a few short requests. The
decode-128 fill tables used
`--fills 5,10,25,50,75,90,100 --concurrency 1 --decode 128`. bench.py now
defaults to `--decode 1024` and fills 25,50,75,100 at one stream, because a
short decode window measures drafter warmup, not steady decode:

```bash
./benchmark/bench.py --base-url http://<host>:8000 --ctx 262144
./benchmark/bench.py --base-url http://<host>:8000 --ctx 524288
./benchmark/bench.py --base-url http://<host>:8000 --ctx 1000000
./benchmark/bench.py --base-url http://<host>:8000 --ctx 262144 \
    --fills 25,50,75,100 --concurrency 1,2,4
./benchmark/bench.py --base-url http://<host>:8000 --ctx 1000000 \
    --concurrency 4   # the 1M 4-stream cells (takes ~20 min)
./benchmark/bench.py --base-url http://<host>:8000 --ctx 1000000 \
    --concurrency 1,2,4 --json bench-4x-1m-conc.json   # the 4-card 1M table (~35 min)
./benchmark/bench.py --base-url http://<host>:8000 --ctx 524288 \
    --concurrency 1,2,4 --json bench-4x-512k-conc.json # the 4-card 512k table (~16 min)
./benchmark/bench.py --base-url http://<host>:8000 --ctx 261120 \
    --concurrency 1,2,4 --json bench-4x-262k-conc.json  # the 4-card 262k table (~9 min)
./benchmark/bench.py --base-url http://<host>:8000 --ctx 261120 \
    --concurrency 1,2,4 --json bench-3x-262k-conc.json  # the 3-card 262k decode-1024 rerun (~9 min)
./benchmark/sustained_decode.py --base-url http://<host>:8000
./benchmark/sustained_decode.py --base-url http://<host>:8000 \
    --concurrencies 4 --depth 257000 --decode 4096   # the deep batched plateau
./benchmark/phase_probe.py --base-url http://<host>:8000   # PROF=1 boot required; profiled salt-7770 cell (~3 min)
python3 benchmark/phase_report.py --dir logs/prof          # parse the 4-rank traces into the step anatomy tables
```

### FP8 KV cache on Ampere (2026-10-15): the capacity unlock, 1.93x pool

Base #60602 (in the 183625d4f9 base) lets QSA store e4m3 KV in uint8 buffers
and byte-decode inside its Triton kernels below sm_89. Boot: same 7-patch
mainline, same 0.9453, one extra flag --kv-cache-dtype fp8.

| Metric | bf16 KV boot | fp8 KV boot |
|---|---|---|
| Pool tokens | 2,636,645 (2.64x) | 5,095,522 (5.10x, 1.93x) |
| Acceptance, salt 7770 direct cells | 4.68-4.90 | 4.79-4.88 |
| Warm 1M single-stream cell wall | 9.2-9.7 s | 9.3-9.4 s |
| Cold TTFT, two fresh salts (true cold prefill) | 91.8 s (10,777 tok/s) | 108.8 / 108.7 s (9,095 tok/s, +18.6% TTFT, -16% rate) |
| 100/4 wall / preempt | 298-387 s / 0 | 398.9 s / 0 |
| 3x 1M streams + 3072 decode each | impossible (3rd preempts) | resident, 0 preempts, sust 553 |
| Needles / grown turns (prefix_state_check) | pass | pass, exit 0 |

Quality gates against the engine's own noise floor (the value-jitter
discovery of the same day set this floor): candidate-vs-reference top-1
flips 7.7% / mean KL 0.081 / tails exact 44%, candidate repeat floor 7.3% /
0.074 / 50%, bf16 self-floor 6.8% / 0.070 / 44%. The format signal is under
machine noise; that bounds it, it does not prove it zero. Prefill pays a
real tax: cold 1M TTFT +18.6% (108.8 s vs 91.8 s, reproducible to 0.1 s,
not JIT). Cause: sm_80 stores fp8 in uint8 and loses the fused QSA
prepare writer (Triton cannot lower fp8 pointers below sm_89), so
quantize+norm run as extra passes. Load wall follows it (399 s vs the
bf16 cold-scale cells). Decode itself: unchanged. The fp8 study prior
for head_dim 256 feared 1.6x TTFT; we read 1.20x.

Verdict: ship flag. Three simultaneous 1M contexts stop being a
scheduling fiction. Caveats: indexer_kv_dtype fp8 remains untested on
sm_80 (second lever, bigger pool, separate gate); the bench.py conc=1
warm-cell acceptance column has a known open bug (engine-side direct
counter windows used here instead). Raw: logs/fp8kv-1004-c1.json,
logs/fp8kv-1003-demo.json; refs benchmark/quality/capture_bf16ref_*.jsonl.gz.

### Prefill GEMM shootout, kernel level (2026-10-16, container, GPU1)

Shapes from the checkpoint tooling (hidden 2560; expert I=640 trace-derived);
M=2048 = production prefill chunk; CUDA-event medians of 31; co-resident
load, ratios only. Raw: logs/preshoot-dense.json, logs/preshoot-moe.json.

- Marlin fp8 W8A16 runs 0.40-0.81x of bf16 at prefill batch on this box
  (per-tensor and block-128 within 5%). ERRATUM 2026-10-16, from the first
  profiled prefill (next section): production dense linears DO NOT use
  Marlin. Traces show zero marlin/scaled_mm launches and 96k aten::mm ->
  the fp8 dense weights run as bf16 cuBLAS after a load-time dequant
  (W8A16 by dequant). This row prices a path we do not run.
- CUTLASS W8A8 int8 wins its shapes outright: 1.29-1.68x vs bf16
  INCLUDING per-token activation quant, 2.1-2.4x vs Marlin fp8, on all
  seven dense families plus lm_head (1.51x) and the
  40k draft head (1.52x). Loses only below ~2M elements / skinny-K
  (K<1024): keep those bf16, same guard shape as the M4G4MED plugin.
  Real value vs the ACTUAL bf16-dequant dense path: the 1.3-1.7x column.
- Triton int8 tl.dot: never wins (0.35-0.71x). Fallback only.
- MoE grouped experts at the real condition (E_local=512) are bandwidth
  bound: Marlin fp8 grouped 54.5 TF/s (748 GB/s) is the best available;
  int8 experts carry identical bytes, so a requant buys nothing there -
  and grouped int8 kernels (Triton) are 0.47-0.64x of it. Experts stay
  fp8/Marlin in any lane.
- Wiring gotcha: block scales are [N/128, K/128] for
  prepare_fp8_layer_for_marlin(size_k_first=False); transposed spelling
  = 10% error or hard throw.

Ceiling arithmetic pending one input: the dense-GEMM SHARE of the 91.8 s
bf16-KV cold prefill. Needs one PROF=1 boot (current boot lacks the armed
profiler); /start_profile just before a fresh-salt cold 1M prefill, the
150-iteration window covers ~one prefill.

### First profiled 1M prefill (2026-10-16): the family table, and the crash

Boot: PROF=1 + --kv-cache-dtype fp8 (dev41+gce97fc109 = base+7). Fresh-salt
cold prefill in flight; 150-step CUPTI window self-closed mid-prefill at
~242k tokens; then the four ranks serialized their traces one after
another (~40 s each, visible as cascading trace mtimes 00:47->00:51) while
the lockstep pipeline stalled. EngineCore died with "RPC call to
sample_tokens timed out" at 00:51:40. Data survived complete on all four
ranks. RULE: profile with VLLM_RPC_TIMEOUT raised (or at <=300k context);
flush x 4 ranks needs ~3-4 pipeline-minutes.

Window reality check: 75 steps of context 2048 + 75 of context 1184 (this
conversation's own chunked prompt - co-prefilling, not decode), 163 ms/
step, 9.93k tok/s extrapolating to 99.6 s vs the 108.8 s anchor (+9%,
later chunks pay attention). Device-time shares (4-rank sum, kineto +
trace-verified classification, 135 kernels, 0 unclassified):

| family | share of device time |
|---|---|
| nccl_comm (spin + hops) | 35.6% |
| humming MoE experts | 24.3% (+1.0% routing) |
| dense GEMM (bf16 cuBLAS, post-dequant) | 14.4% |
| QSA attention pair (splitk prefix read + prefill) | 13.6% |
| GDN chunked | 4.3% |
| elementwise/norm/silu/HC | 4.9% |
| PLE pinned lookup (rank0 only) | 0.6% |
| quantization passes (incl fp8-KV write) | 0.5% |
| spec-dec aux + memcpy + rest | ~1.5% |

Reads:
1. The fp8-KV +17 s tax is NOT the quantize-on-write or unfused prepare
   (write kernel = 0.14% of device time). It lives in the attention READ
   path (fp8 dequant inside QSA at head_dim 256, 13.6% base). #60992
   (4-bytes-per-register e4m3 reads, merged upstream 10-14, one day past
   our base) is the named fix candidate: cherry-pick, re-measure cold TTFT.
2. Dense is already bf16 cuBLAS - the checkpoint's fp8 dense weights are
   dequantized at load (zero Marlin launches in traces). CUTLASS int8
   therefore projects 1.3-1.7x on 14.4% of device = ~8-12% wall prefill.
   Real, but not the 2x the shootout's Marlin column suggested.
3. Biggest single prefill number is the 35.6% pipeline spin. No kernel
   swap touches it; the boot itself warns max_num_scheduled_tokens=2048
   (capped by spec settings) is suboptimal. Chunk-size/PP-split tuning is
   the untouched lever.

### Cherry-pick 77154ca (#60992) measured on the live tree (2026-10-16)

User grafted the one-file QSA read kernel fix onto the live serving tree
(dev42-era boot, fp8-KV, pool 5,095,522 unchanged). Cold 1M prefill TTFT:
108.8 -> 103.8 s (103.8/104.0/103.8, first rep post-JIT identical; recovers
5.0 of the 17.0 s tax, remaining +13.1% vs bf16-KV 91.8). Correctness:
prefix_state_check pass all sessions, acceptance 4.78 fresh salt.
Incidental: six 1M contexts of my own salts made the 6th slow (150 s) =
five resident 1M contexts confirmed against the 5.1M pool.
Still-parked decision: 13.1% cold tax vs 1.93x pool, by workload. The
residual tax is the QSA fp8 read pair (13.6% of device time in the
profile); the 35.6% NCCL spin remains the biggest prefill lever either
way.

### PP int8 handoff: production trial PASS (2026-10-16)

Live tree = base+7+77154ca + experiments/pp-int8-hop.patch.v3, env
VLLM_QWEN4EXP_PP_INT8_HOP=1, PLAIN bf16-KV boot (one variable), pool
2,649,068 (+12k, halved recv buffers), utilization 0.9453 unchanged.

| Metric | bf16-hop anchor | int8-hop | delta |
|---|---|---|---|
| Cold TTFT x3 fresh salts | 91.8 s | 79.9 / 79.8 / 79.9 | **-13.0%** |
| Warm cell wall (1024 tok) | 9.2-9.7 s | 9.5 s | none |
| Acceptance direct cells | 4.7-4.9 band | 4.79 / 4.86 | in band |
| Needles / grown turns | pass | pass, exit 0 | - |
| 100/4 wall x2 (fresh salts) | ~386 s cold-grade | 345.6 / 346.0 | **-10.5%**, 0 preempts |
| 100/4 TTFT mean | 235-254 s | 210.5 / 210.4 | -11..17% |

Method note: cold trio with salts totalling > pool evicts the first salt;
warm tests must use the LAST salt of the trio or re-warm first (an 85 s
"warm" cell was self-inflicted eviction, not a regression).
The wire was the wall as predicted; bytes halved bought 13%, not the 25%
upper bound (spin/skew share of the 35.6% survives the halving).
Next stack to try: this + --kv-cache-dtype fp8 (predict cold ~97 s vs
today's bf16 79.9 - the fp8 read tax stays; capacity +93%; at 100/4 five
1M streams resident). Also NCCL_P2P_DISABLE=0 free-ish 2-8% of hops,
one-env boot, and candidate upstream PR with acceptance+eval gates.
Raw: logs/hop-1004-c{1,2}.json. Formalized as patches/0008 (series branch
qwen38-pp-hop8 at f8a579218d; 8-patch replay tree
6b311444b1affc4322ead78478896dcd6622943b, round-trip verified).

### Full stack certified: hop + fp8-KV + 77154ca (2026-10-16)

Live tree re-ammed to the official 8-patch series + 77154ca6 cherry-pick,
booted with --kv-cache-dtype fp8 and the hop env set. Pool 5,098,507.

| Config | Cold 1M TTFT | 100/4 wall | Warm 1024-tok cell | Needles |
|---|---|---|---|---|
| bf16-KV, no hop (original) | 91.8 s | 386 s | 9.2-9.7 s | pass |
| fp8-KV, no hop | 103.8 s | 346 s* | 9.3-9.4 s | pass |
| bf16-KV + hop | 79.9 s | 345.6/346.0 | 9.5 s | pass |
| fp8-KV + hop + 77154ca | 92.0-92.1 x3 | 344.4/353.6 | 10.5-10.8 s | pass |

Acceptance 4.86-4.89 in band throughout; zero preemptions; 4x 1M streams
(3.96M tokens) now resident at once on the 5.1M pool. The fp8-KV capacity
is bought for exactly the time the hop gave back: the stack matches the
ORIGINAL bf16 cold prefill wall while holding 1.93x the context. Warm
cells carry ~+1 s of fp8 read tax at turn boundaries, inside cell noise.
(*fp8 no-hop 100/4 figure was single-cell.)
Recommendation table for ship config: hop is unconditional (flag off =
inert, zero downside); fp8-KV now a free capacity upgrade for agent and
concurrency workloads, still a 12 s/tax per fresh 1M document for bulk
ingest. Raw: logs/stack-1004-c{1,2}.json.

### NCCL P2P A/B verdict (2026-10-16 night)

Boot with script line flipped to NCCL_P2P_DISABLE=0, everything else ship:
cold 1M TTFT 94.7 / 94.9 s clean samples (plus one 31.4 s artifact from a
client-cancelled orphan stream leaving a salt half-cached - do not trust
samples right after interrupted probes), vs ship baseline 91.2-92.1. Warm
cells identical (9.7 vs 9.4-10.8). P2P wins idle pairs and loses the real
4-rank boot: pipeline communicators plus PLE pinned DMA contend worse
through root-complex peer windows than host staging does. Script now keeps
=1 as default via ${NCCL_P2P_DISABLE:-1}, so an outer export A/Bs without
editing. Case closed; revisit only if the PLE ever stops streaming over
the same bridges.

## New-base boot hang: asymmetric JIT warmup round (2026-10-18)

The first production boot on base 9731f30a03 (8-patch series, MTP-4, fp8 KV,
hop on) never finished engine init. Ranks PP0-PP2 logged "Graph capturing
finished"; PP3 stopped after its QSA kernel-warmup lines and printed nothing
more. Killed after ~12 min; the box was reverted to the previous tree.

A py-spy dump of the stuck PP3 worker named the site: Worker.execute_model
-> irecv_tensor_dict -> recv_object, called from _warmup_kernels, itself
called from the second _warmup_kernels round inside warmup_kernels. The new
JIT warmup (upstream #59260, 2026-10-14) fabricates scheduler outputs and
drives real execute_model passes, so every pipeline stage must run the same
number of rounds. The greedy second round exists to prime rejection-sampler
kernels and was gated on model_runner.rejection_sampler, which is built only
on the last pipeline stage. With spec decode at PP>1 the last stage waited
for sends that the earlier stages (already done with warmup) never issued.
Still broken on upstream main as of this writing.

Control boot N3, same tree without --speculative-config: boots clean in ~3
min, pool 2,726,114 tokens at kv_cache_dtype=auto. The drafter flag was the
whole difference, which matches the mechanism exactly.

Series patch 9 moves the gate to model_runner.speculative_config, present on
every stage, so all stages mirror the round. CPU suites test_gpu_warmup_blocks
and test_mixed_warmup_gate green (the adaptive-restore test exercises the
guard; the block-count fakes carry the new attribute). Full series replay
re-verified: 9 patches git am onto 9731f30a03 -> tree
9d66357a6af55e6769986dba80ab47e8fdc1a63c, equal to the branch.

Interim escape hatch without patch 9: --kernel-config '{"enable_jit_warmup":
false}' skips the whole JIT pass (the legacy QSA/GDN kernel_warmup still
runs). Status: fix is hardware-unverified until boot N4 (MTP + hop + fp8 KV
on the 9-patch tree) lands; re-certification (acceptance trio, cold TTFT,
needles, pool drift) follows on the same boot.

## Rebase onto f98ae6a9fd (2026-10-18, +36 commits from 9731f30a03)

Upstream state first: the greedy-JIT-warmup deadlock (patch 9) is unfixed on
main but has an open PR with the identical gate, #62129 (2026-10-17,
chenjr97). Patch 9 stays in the series and retires when that merges.
The N4 hardware boot validated patch 9 on the previous base: engine init
completes with MTP-4 + hop + fp8 KV, all warmups 4/4 ranks, pool 5,077,611
tokens, hop ACTIVE logged, 3 min 15 s to serving.

Rebase mechanics: worktree /home/gab/test/wt-rebase-1018, branch
qwen38-pp-f98ae6a9fd, cherry-pick --strategy-option=diff3 of all 9, zero
conflicts, series delta lines byte-identical to the old range. Replay: 9
patches git am onto pristine f98ae6a9fd -> tree
7256681104425f758647a00fa23f2e0d5d8bfc03, equal to the branch. Wheel:
0.32.0rc1.dev342+gf98ae6a9f cu130 (200 on metadata probe; the 20 newer tips
404). humming-kernels pin unchanged at 0.1.17. CPU suites: 143 passed across
the ship files + warmup gates; test_ple.py 13 failures are GPU/UVA
environmental, identical on the pristine base (verified side by side);
test_qsa_prepare.py skips on this box (fender-stack question stays open).

Semantics in the range that touch our stack:
- b544587408 (#53558) makes KV planning pluggable. Checked: the default
  builder delegates to the module-level get_kv_cache_config_from_groups, so
  patch 3's fix stays on the live path.
- c3de0b37e2 (#61455) replaces the "identify the draft KV group" heuristics
  with explicit is_draft tags, set only on AttentionLayerBase layers. For
  this model the drafter builds a full hybrid decoder layer: its attention
  layer now gets tagged, its GDN layer cannot. The old all-groups fallback
  (which eagle-flagged mamba groups and drove the documented one-block hit
  loss at exact retention boundaries) now no longer triggers. Behavior of
  mamba prefix hits under MTP can therefore differ on the new base. Verify
  at first cert: boot log for the "no KV cache group could be identified"
  warning (its absence means tagging worked), prefix_state_check boundary
  case, acceptance trio on fixed salts.
- 61bb590877 (#61174) clears stale rows from shared spec-decode block
  tables. Same bug class we audited at #56734; pure benefit, no series
  overlap.

## Certification: base f98ae6a9fd + 9 patches, boot N5 (2026-10-18)

Tree dev351+g693f68e2e, pool 5,077,611, hop ACTIVE, draft-vocab live, no
eagle-identification warning, 3 min 38 s to serving. Server idle except
session traffic between cells; background drafts verified zero around the
acceptance windows.

| Metric | band era (f98ae's predecessor chain) | N5 |
|---|---|---|
| Cold 1M TTFT, 3 fresh salts | 92.0-92.1 s | 92.3 / 92.6 / 92.6 s |
| Warm 1024-tok cell | 10.5-10.8 s | 10.3 / 10.5 s |
| Acceptance, decode-phase window | band 4.75-4.91 | 4.995 |
| 100/4 wall, fresh salt | 344.4 / 353.6 s | 356.5 s, 0 preempts, 4/4 resident |
| Needles (prefix_state_check) | pass | pass, exit 0 |

Method break on this base, read before comparing acceptance columns across
boots: the spec-decode counters now also count drafts from prefill-phase
steps. Visible directly (14 drafts before the first token of a warm 256-tok
stream). Effect: bench.py's full-window acceptance column dilutes toward the
prefill ratio (cold cell 1.8, warm 3.4-4.2), and the old era's full-window
numbers are not comparable to decode-window numbers on the new base. The
band's own era files had no prefill-phase drafts, so their full-window
column equalled the decode value. From this boot on, acceptance is certified
by bracketing the counters inside the decode phase of a warm stream.

#61455 follow-up resolved in our favor: with the draft attention layer
explicitly tagged and the mamba groups no longer eagle-flagged by the old
all-groups fallback, the prefix-hit floor halved, 16-17 recomputed tokens at
retention-boundary prompts (era: 32 with the MTP block drop), and the
concurrent 4-session warm round recomputes 82 tokens (era: 1,682). Grown
turns and needles all pass at 16160/32320/64640 boundary lengths.

Verdict: PASS. The new base keeps the ship numbers (cold +0.5%, wall +1-3%
inside sample spread) and buys the mamba prefix improvement plus #61174.
Raw: logs/n5-cold-s9107{1,2,3}.json, logs/n5-warm-s91073.json,
logs/n5-7770-c{1,2,3}.json, logs/n5-acc-{1,2,3}.json,
logs/n5-1004-c1.json.

## Rebase onto 8407bc0bb6 (2026-10-20, +163 commits from f98ae6a9fd)

Patch 9 absorbed: upstream merged #62129 (7c9042c05a, 2026-10-18) with the
identical speculative_config gate plus its own PP-round regression test.
Series is 8 patches again.

Conflict territory was patch 7 vs #61454 (Qwen3.8-Flash-Next HC/PLE cleanup,
494663940b). Upstream added the compressed-tensors "no scheme -> unquantized"
leg we had, but not the float8 leg, so honoring fp8 PLE tables is still
patch 7's payload. Its old else-raise leg now contradicts an upstream test
(test_ple_embedding_falls_back_for_unsupported_quantization_configs:
int8 compressed-tensors schemes must fall back); the leg was paranoia beyond
the patch's purpose (it only ever guarded our own fast-lane builds), so patch
7 was shrunk to unquantized/float8 and anything else falls through to
upstream's warn-and-unquantize tail. Commit message reworded accordingly;
pre-rule agent trailers were scrubbed from all series commits.

Other in-range interactions checked: #61836 wraps patch 6's fadvise loop in a
sys.platform guard (coexists, our call sits inside it plus its own OSError
handler); #52670 makes kv-cache dtypes pluggable via register_kv_cache_dtype
-- kv_cache_dtype=fp8 stays a registered name, and this registry is the
likely home of the parked indexer_kv_dtype fp8 lever; #58984 (DP drafter
padding) auto-merged over patch 5's speculator hunk; #52477 touched warmup.py
outside the absorbed patch 9 region.

Mechanics: worktree /home/gab/test/wt-rebase-1020, branch
qwen38-pp-8407bc0bb6, cherry-pick diff3, one conflict (patch 7), change-line
delta identical to the old series outside the known-delta files. Patch 8's
two helper functions needed CI-version ruff reflow (amended into the commit).
CPU: 147 passed ship+gate suites; tests/models/qwen4_exp 11 failures
identical to pristine base (149 passed, +9 from series tests). Exported
0001-0008 (no agent trailers, adopted-PR authorship preserved), replay
2e6ac5a3fc930e43771b6e274df4371a30a681f0 == branch tree. Wheel
0.32.1rc1.dev133+g8407bc0bb (rc line moved to 0.32.1, dev counter reset;
post-am banner should read dev141). humming-kernels pin unchanged 0.1.17.
Hardware boot + cert pending.

## Certification: base 8407bc0bb6 + 8 patches, boot N6 (2026-10-20)

Tree 0.32.1rc1.dev141+gb63d5db91 (133+8), pool 5,077,611 identical to N5,
hop ACTIVE, draft-vocab live, no eagle warning, 3 min 45 s to service.
Acceptance now certified with the decode-window bracket, saved as
benchmark/accept_window.py (background-draft checks included, zeroed each
trial).

| Metric | N5 (f98ae6a9fd) | N6 (8407bc0bb6) |
|---|---|---|
| Cold 1M TTFT x3 fresh salts | 92.3 / 92.6 / 92.6 s | 92.9 / 93.9 / 93.7 s |
| Warm 1024-tok cell | 10.3 / 10.5 s | 10.1 s (first try 20.1 s: transient co-tenant interference, discarded and re-run clean) |
| Acceptance, decode window | 4.995 | 4.995 / 4.995 warm, 4.888 with cold prefix |
| Acceptance, 200k probe continuity | 4.365-4.401 | 4.256 / 4.300 (pos-shape normal, low depth = lottery-sensitive) |
| 100/4 wall | 356.5 s | 356.8 s, 0 preempts, ttft mean 223.8 |
| Needles | pass | pass, exit 0 |
| Retention-boundary floor | 16-17 tok | 17 / 17 / 17 tok |

Verdict: PASS. Nothing in the +163 range moved a ship number beyond sample
spread; the cold trio drifts +1% (92.0 -> 92.3 -> 92.9 across three base
jumps) and stays inside the historical band's spread. Patch 9's absorbed
form and patch 7's shrunk form both behave: engine init completes with
MTP+PP4 and the PLE fp8 pinned load line is unchanged.
Raw: logs/n6-*.json.
