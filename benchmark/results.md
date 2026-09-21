# Benchmark results, full tables

All numbers measured on the 3x unlocked CMP 170HX box (200 W power cap, PP=3,
MTP-3, PLE host offload) running one server at a time, idle and warmed,
greedy, unique prompt per row so every row pays a full prefill. The
"4-card set" section is the same box with a fourth card, PP=4. Tools:
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
  on sign; neither is tight enough to pin one number, so the honest read is
  "3-card is a few to fifteen percent ahead", not a single figure.
- The deep plateau is the softer of the two this time: on 4 cards the 460k
  prompt prefills in ~41 s per stream (TTFTs 40.6, 82.9), so stream 1 has
  only ~10 s of true 2-stream overlap inside the 20,480-token window before
  it drains, against 3 cards' ~77 s. The bench cell is the load-bearing
  measurement.
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

## 4-card set (PP=4, partition 12,12,12,12), decode 1024

### 1M (YaRN 4.0)

Measured 2026-09-19, the first 4-card numbers. Tree at base fc8132a5e5; the
compiled artifacts were still the nightly wheel (`+gdc6954d14`) until the
pin is applied. KV pool 2,541,795 tokens, maximum concurrency 2.54x at the
1M window (3-card box: 1,208,978 tokens, 1.21x). Per-rank KV:
22.88 / 24.31 / 24.31 / 22.29 GiB. Raw: `bench-4x-1m-conc.json`.

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
```
