# Benchmark results, full tables

Append-only record from 2026-09-15 to 2026-10-25. The configuration changed
several times over that window, so each section states its own setup, and a
section preamble overrides this header wherever the two disagree.

Current lane, which is what the October sections use. 4x CMP 170HX, GA100
unlocked to 64 GB and 74 SM per card, one NUMA node, 200 W per-card cap, PP=4
with partition 13,12,12,11, 1,000,000 context at static YaRN 4.0, MTP-4
speculative depth, fp8 KV cache, one server at a time. The September sections
were measured on three cards at PP=3 with MTP-3 and on shallower contexts, and
tables that predate the SM unlock ran at 70 SMs per card, which matters when a
bandwidth or kernel-occupancy figure is compared across sections. The "4-card
set" section is that three-card box with a fourth card added. PLE host offload
is on throughout.

Cells are greedy at temperature 0 with a unique salted prompt per row, so each
row pays a full prefill unless the section says otherwise. Headlines live in the
top-level README.

Tools. Each row states what the script measures, condensed from its own docstring.

| tool | measures |
|---|---|
| `bench.py` | context fill and concurrency cells |
| `sustained_decode.py` | sustained concurrent decode capacity, isolated from prefill |
| `mtp_ab_probe.py` | MTP draft acceptance at a chosen context depth |
| `accept_window.py` | decode-window acceptance bracket for certification |
| `corpus_accept_probe.py` | acceptance and decode rate on held-out corpus text |
| `itl_probe.py` | whether a starved decode is slow or deferred |
| `phase_probe.py` | arms the profiler around one fixed-salt decode cell |
| `phase_report.py` | attributes decode-step time from torch profiler traces |
| `dense_weight_bytes.py` | dense weight bytes per pipeline rank from safetensors headers |
| `pth_attn_gate.py` | correctness of the int8 attention op on sm_80 |
| `qsa_byte_floor.py` | QSA sparse read floor and split-K sensitivity on one card |
| `qsa_decode_dtype.py` | whether QSA decode cost tracks bytes or elements |
| `qsa_decode_tune.py` | QSA decode tile shape against its gather ceiling |
| `pth_vs_qsa_fp8.py` | equal-bytes A/B, served fp8 kernel against the int8 op |

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
| Cold TTFT x3 fresh salts | 91.8 s | 79.9 / 79.8 / 79.9 | -13.0% |
| Warm cell wall (1024 tok) | 9.2-9.7 s | 9.5 s | none |
| Acceptance direct cells | 4.7-4.9 band | 4.79 / 4.86 | in band |
| Needles / grown turns | pass | pass, exit 0 | - |
| 100/4 wall x2 (fresh salts) | ~386 s cold-grade | 345.6 / 346.0 | -10.5%, 0 preempts |
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

## Rust frontend trial (8407bc0bb6 + 8 patches, 2026-10-20)

Activation: RUST=1 in serve-4x-1m.sh (exports VLLM_USE_RUST_FRONTEND=1).
The Python launcher execs <tree>/vllm/vllm-rs (present from the precompiled
wheel) and hands it the listener fds + ZMQ addresses; zero uvicorn processes,
Rust tracing logs instead, boot log 319 lines vs 533. Live tree unchanged
otherwise (dev141+gb63d5db91).

Parity delta found only at boot (static audit missed both): the frontend
warns that structured_outputs_config and enable_auto_tool_choice "currently
have no effect in Rust frontend". Both are no-ops for this deployment:
function tools + streaming + ~200k-token agent prompts ran native traffic
through it fine. /v1/messages returns 404 (no route; unused here). All
cert counter families present; prefix/probes work unchanged (accept_window
actually races less: faster SSE chunking).

| metric (same engine, same day base = N6) | Python frontend | Rust frontend |
|---|---|---|
| warm 1M cell TTFT | 4.7-4.9 s | 1.4 s |
| warm 1M cell wall | 10.1-10.5 s | 6.9 s |
| append round trip (probe) | 0.17-0.38 s | 0.11-0.14 s |
| cold 1M TTFT | 92.9-93.9 s | 89.8 s (single draw) |
| 100/4 wall / ttft mean | 356.8 / 223.8 s | 349.2 / 218.1 s |
| acceptance (decode window) | 4.995 | 4.978-4.989 |
| needles / boundary floor | pass / 17 | pass / 17 |
| pool at boot | 5,077,611 | 5,053,731 (-0.47%, unexplained; watch next boot) |

Reading: the win is on the request-ingress path (tokenize/validate/broadcast
of huge prompts), exactly where the Python API server spent its CPU; every
session turn starts ~3.5 s sooner. Decode and prefill silicon unchanged,
batch wall neutral-to-better. Keep it on; revert = boot without RUST=1.
Raw: logs/rust-*.json.

## Patch 2 shrunk to guards-only, promoted (2026-10-21)

Pipeline: live probe boot (2026-10-20) proved wire vs stage-local PLE
input ids byte-identical on 768/768 consecutive real-traffic forwards
across PP1-PP3 (prefill chunks, mixed batches, MTP draft rows,
concurrent streams); the only mismatches were fabricated warmup steps,
where stage-local buffers hold stale dummies by construction, plus the
PP3-only greedy round-2 signature. The unwire candidate then booted as
dev142+gdaf777a49 and took the full cert battery: cold 1M 89.7/90.3/90.4 s
(same class as the previous Rust boot), warm cell 7.1 s / sust 199 after
one transient 8.1/168 sample was re-run clean, decode-window acceptance
4.85-5.00 (mean 4.90; the -0.1 against 4.995 is the 1.4 s-TTFT window-
start artifact of the Rust frontend, not the drafter), needles exit 0
with the boundary floor at 17, 100/4 wall 350.6 s / 0 preempts.

Series renumbered in place: patch 2 is now the guard deletions from
#56444 plus retargeted contract tests (intermediate tensors must not
carry ids; non-first ranks feed the runner argument to layers). The
wire carry, its int32 staging buffer, and the consumer shadow are gone
from both backend models. Rebuilt series verified: patch-2-position
surgery + restack 3-8 yields a tree byte-identical to the
hardware-proven tip (f4652439f3869c845b3f537cbc02ba89095f2279), ruff
clean, replay of the eight re-exported patches reproduces it exactly.

Follow-up worth filing on #56444: with #61016 merged, the PR's carry is
obsolete for vLLM-V2 runners; ask the author to reduce to guards or
close. Probe and unwire artifacts stay in experiments/ for provenance.
Raw: logs/unw-*.json.

## Adaptive MTP depth: field trial, REJECTED (2026-10-21)

The B12X acceptance-length ratchet (experiments/0009-adaptive-mtp-depth.patch,
port details in that commit) booted clean twice (window=8, then 32) and
behaved exactly per spec: depth telemetry live (250 transitions at w8,
28 at w32 as the window de-jittered it), pool -0.2% for depth-tier graphs,
boot +50 s capture, needles and all standing gates pass.

A/B on the fixed tool (accept_realistic.py, gated per type, ctok verified,
2 reps each, quiet host): prose/code/json/mixed medians in tok/s -
fixed-4: 139/179/137/130 (avg 146.2); adaptive w8: 144/186/129/126
(avg 141.6); adaptive w32: 132/176/135/128 (avg 143.4). No content family
gained beyond the ±15 tok/s noise floor; the mean sits below fixed on both
windows. One real effect: adaptive compresses variance (code pinned
186-188 vs fixed swinging 165-192) - predictability, not throughput.

Mechanism of the non-win: with the patch-5 vocab-trimmed draft head, a
draft forward is cheap against the 28-layer target verify, so the
controller's cut-first law optimizes a cost term this box barely pays.
B12X's Spark targets have proportionally expensive drafting; the law is
tuned for their economics. Method note: the first probe series was
contaminated - fixed-4 same-seed reps disagreed on acceptance, which is
impossible under greedy+fixed-K, proving co-tenant drafts were inside the
counter brackets; the tool now idle-gates per type and reports tries.

Disposition: revert speculative-config to {"method":"mtp",
"num_speculative_tokens":4}; the commit stays in the live tree inert
(retire at next re-am). Lab branch + patch kept for the smarter horse:
upstream AV (cost-table driven, roadmap #61024) once it lands PP broadcast
and QSA varlen; that design would learn the cheap-draft economics instead
of assuming expensive ones.

## Adaptive depth, margin retune + depth headroom: closed (2026-10-21)

Margin-2.5, window 32, cap 4 boot (experiments 0009 v3, margin made a
config field): tied fixed-4 (prose/code/json/mixed 146/169/158/132 medians,
but json reps split 181 vs 134 on identical config+seed - the value-level
non-determinism surfacing at 2000-token windows, so the +3% mean is not
trustworthy) and held depth at ~4 (8 transitions total, a 4<->3 tie).
Verdict: a wide margin just converges to fixed-4; no throughput case.

Depth headroom past K=4 is unreachable on this cache geometry without
re-basing the whole era. The hybrid block alignment ties legal speculative
depth to the QSA ring and retention (fp8 KV, --block-size 1616, retention
16160):

| cap | attn block | ring | legal? |
|---|---|---|---|
| <=4 | 3232 | 8 | yes (retention 5x3232) |
| 5   | 3232 | 12 | no (3232 %% 12 != 0) |
| 6-8 | 4848 | 12 | no (16160 %% 4848 != 0) |

Only K<=4 or a retention move to 14544/19392 (new mamba snapshot grid, new
certification era, pool shift). The three failed boots (K=6, K=8, K=5) were
that alignment logic and a QSA ring assert, never the adaptive code. Per-
position data (pos3 survival 0.3-0.6) says the next rung earns ~0.3 accepted
tokens/step in top-decile windows - not worth re-basing.

Booked as method wins: the accept_realistic.py tool (idle-gated, ctok-
verified), the co-tenant-detection proof (same seed + greedy + fixed K must
reproduce; disagreement => contamination), and the margin knob for whenever
upstream AV (#61024, needs PP cost/confidence broadcast + QSA varlen) makes
adaptive depth worth reopening. Live tree reverted to stock fixed-4 by the
user; v3 experiment patch left in place inert.


## host_file_gather: PLE rows from the page cache (experiments 0010/0012/0013)

Port of upstream #58815 (Trosfy): with engram `host_file_gather` the 47.6 GiB
FP8 PLE table is never allocated; rows are preadv'd from the mmap'd checkpoint
shards into a device staging buffer per step. Boot trial on fender passed
correctness (needles exit 0; cold-table decode byte-exact), cut weight load
from 1:16 to 0:49, and freed the table from host RSS. Cost: the staging path
syncs the worker's whole stream every step (ids come back D2H before preadv).
Under PP=4 batch prefill went bimodal, 100/4 in 276-380 s vs pinned ~350 s
steady, and cold 1M paid +2% (93.3-94.1 vs 92.3-92.6 tok/s). pidstat says the
table stays page-cache-resident in both regimes (0.8-144 kB/s reads), so the
spread is sync-vs-cadence, not disk.

- 0012: range-aware cache eviction; the loader's `VLLM_SAFETENSORS_DROP_CACHE`
  pass now spares PLE table byte ranges when host_file_gather is on. 5 loader
  tests, CPU-only verification so far.
- 0013 v1 (threaded gather, one commit): failed on hardware. Logprob A/B
  pipeline-vs-sync: 10.5% top-1 flips vs 7.3-7.7% noise floor, union KL 0.195
  vs 0.074-0.081, greedy tails equal 35.4% vs 44-52%. Mechanism: the worker
  waited the previous step's barrier, which in stream order does not delay the
  current forward, so a step could read `_staging` while its own H2D was
  writing it. Self-consistent staleness is invisible to self-comparing gates;
  use a two-path logprob diff for storage or pipeline swaps.
- 0013 v2 (prefill-ahead; shelved 2026-10-22, see the battery below): pure-prefill steps need no GPU
  round-trip; prompt tokens are host-known. Model state mirrors
  `NewRequestData.prefill_token_ids` per request, computes ngram ids on the
  CPU for batches that are prefill-only, draft-free, and mirror-covered, and
  the gather thread orders its H2D behind a submit-time main-stream event
  with the barrier placed before the step's forward. Decode/mixed/draft steps
  keep the lockstep path. CPU tests: mirror-sliced ids == `compute_ngram_ids`
  == `_reference_ngram_ids` (parity test).

Knobs (as shelved): VLLM_PLE_GATHER_PIPELINE defaults 0 (lockstep for all
steps); =1 enables the prefill-ahead fast path for a rerun.
VLLM_PLE_GATHER_SYNC=1 disables the pipeline machinery entirely.

### 0013 v2 hardware battery (2026-10-22, boot-4x-1m.log 11:07, tree 0010+0012+0013)

Correctness gates all pass: logprob A/B vs the sync capture reads 4.41%
top-1 flips, union KL mean 0.048, greedy tails 52.3% equal, per-depth
200k KL 0.0145 - all at or under the noise floor (v1 failed the same
gate at 10.5% / 0.195 / 35.4%). Needles exit 0. Decode-window
acceptance 4.973/5.000 (trial1 dropped: sampling raced the stream).

Perf gates fail. First battery finding: the runner stepped salt bases by
+1, and bench.py stream i uses salt+i, so runs shared 3 of 4 streams
with the predecessor and prefix cache reuse faked a 208-216 s cluster.
Disjoint-base rerun (100 apart, table resident, same boot):

| run | wall s | TTFT mean s | in tok/s |
|---|---|---|---|
| 100/4 clean r92000 | 413.7 | 246.5 | 9,556 |
| 100/4 clean r92100 | 400.5 | 249.8 | 9,873 |
| 100/4 clean r92200 | 389.1 | 246.0 | 10,163 |
| 100/4 clean r92300 | 398.6 | 248.3 | 9,920 |
| cold 1M x4 (96.0/96.3/97.8/98.0) | TTFT vs sync-boot 93.3-94.1 | ~10k | |

The pipeline path is deterministic but slower than lockstep everywhere:
+2.5 s on single-stream cold (about 6 ms per step of worker-side id
compute), and 389-414 s at 100/4 against the sync era's 276-380 spread
and pinned's 350-357 band. The premise is disproved: per-step gather
waiting was never the batch bottleneck, so hiding it cannot recover the
cadence, and the worker CPU cost taxes every prefill step that used to
run clean. Raw: logs/pipe2-*.json.

Decision (2026-10-22): 0013 is shelved, not promoted. The lab branch keeps
the commit as a documented dead end; the export is regenerated with
VLLM_PLE_GATHER_PIPELINE defaulting to 0, so the fast path is off unless a
rerun opts in. Promote 0010+0012 lockstep only: it beat pinned at 100/4
median (~295 vs ~353) while freeing 47.6 GB of pinned RAM and 27 s of boot.
A same-boot lockstep rerun would still pin the exact pipeline-vs-lockstep
delta, since the sync-era 276-380 spread predates the salt-step rule and may
itself be partly salt-inflated. The salt-step rule is now in the bench
protocol notes.

### Promoted 9-patch cert battery (2026-10-22 15:03, idle host, banner dev142+g92c71e87d)

Needles exit 0 and logprob A/B vs the sync capture passed earlier the same
day (4.41% flips / 0.048 KL, verdict ok) while a background CPU job ran;
the timing battery followed on the idle box.

| cell | result | pinned-era band |
|---|---|---|
| cold 1M TTFT x3 (salts 93001-93003) | 92.6 / 92.8 / 92.9 s | 92.9-93.9 (N6) |
| 100/4 wall x4 (bases 93100/93200/93300/93400, 100 apart) | 362.1 / 364.8 / 366.6 / 365.3 s | 350-357 |
| 100/4 TTFT mean | 228.7-229.8 s, preempts 0 | 223.8 |
| decode-window acceptance | 5.000 / 4.973 (trial1 discarded: 81 background drafts detected post-window) | 4.98-5.00 |

Readings:
- Cold 1M on the promoted tree sits at or under the pinned band; this
  morning's 96.0-98.0 was the shelved pipeline's worker-side id cost.
- 100/4 is deterministic to 0.6% (362-367 s) on clean salts. The sync-era
  "bimodal 276-380" was largely the salt-overlap artifact plus the
  post-boot first-touch run; lockstep was always ~365. The +3% against
  pinned is the accepted cost of freeing 47.6 GiB pinned RAM and 27 s of
  boot (loader table-sparing arithmetic confirmed at boot: 125.29 of
  172.97 GiB dropped, 47.68 GiB spared).
- Raw: logs/promo9-*.json, /tmp/promo9_batt.log.

### io_uring gather probe (2026-10-22): dead end, shelved

Question: the file-gather stage does one `os.preadv` per unique row
(~32k per 2048-token step, one PLE layer x 16 heads); can io_uring collapse
those syscalls and recover the GPU0 utilization gap (~90% vs 100% pinned)?

Built a pure-ctypes raw ring (no liburing: fender may not have it; only the
`io_uring_setup`/`io_uring_enter` syscalls + three mmaps; SQE is 64 bytes,
not 128 - the first prototype's phantom NOPs came from that stride).
io_uring_setup and scattered buffered reads work from this container despite
apparmor noise in dmesg; `io_uring_disabled=0` on the kernel.

Measured on the real workload shape (32,768 scattered 2 KB page-cache-hot
reads across 16 shard files, warmed; best of 5 per config; all variants
byte-checked correct):

| path | ms | vs preadv |
|---|---|---|
| current per-row `os.preadv` loop | 47.5 | 1.00x |
| io_uring, per-SQE ctypes fill, naive reap | 100.6 | 0.49x |
| io_uring, numpy-SQE fill, vectorized reap | 61.0 | 0.82x |
| io_uring, contiguous slice fill, sync issue | 56.9 | 0.84x |
| io_uring, IOSQE_ASYNC (default workers) | 47.5 | 1.00x |
| io_uring, IOSQE_ASYNC + fixed-file table | 47.7 | 1.00x |
| io_uring, IOSQE_ASYNC + 16 io-wq workers | 53.8 | 0.88x |
| io_uring, IOSQE_ASYNC + 32 io-wq workers | 55.4 | 0.86x |

The ctypes fill was indeed slow (three rebuilds halved the ring path:
per-SQE field pokes -> numpy structured fill -> contiguous slice writes
with strided CQ views and zero per-row FFI; along the way SQE stride is
64 B not 128 B, CQ fields need stride-4/stride-2 slicing, and resetting
kernel-owned ring pointers between batches silently no-ops the next
submit - never do it). But the optimized ring only TIES the preadv loop,
and extra io-wq workers make it worse (dispatch cost; same-file reads
serialize on io-wq hash-wait). That is the real finding: the syscall
trap is ~0.3 us inside ~1.4 us of unavoidable per-read VFS work (page
lookup + 2 KB copy + file read helpers), which both paths pay identically.
pidstat already showed the table page-cache resident (0.8-144 kB/s in
steady state), so there is no device wait for io_uring to overlap. The
existing FADV_WILLNEED readahead covers the cold-miss case asynchronously.
SQPOLL was not pursued (needs CAP_SYS_NICE; enter is already one call
per batch).

The CPU time in staging is real (~15-20 ms per 2048-token step at batch,
matching the +8 s cold gap vs pinned) but it is memcpy-bound; fewer syscalls
does not touch it. Prototype kept at
`experiments/ple_gather_iouring_probe.py` (runnable, self-checking; rebuild
history in the table).

#### ctypes was the confound - proper C extension re-run (2026-10-22, same day)

The ctypes table above measured Python loop vs Python ring; both carried the
same per-iteration CPython cost, which is why they "tied". Rebuilt the two
gatherers as a raw-syscall C extension (`experiments/ple_gather_c.c`, no
liburing - only the kernel UAPI header, so it builds on the serving host with
`cc -O2 -shared -fPIC`) and re-ran the identical 32,768-row warm read. Two
C-ring bugs had to be fixed first: a static ring left half-drained by a failed
call poisoned the next, and `io_uring_enter` needs its trailing sigset args
passed explicitly (varargs syscall left them garbage -> EINVAL).

| loop, all byte-correct | ms | vs C preadv |
|---|---|---|
| Python `os.preadv` loop (today's shape) | 56.4 | 0.56x |
| C `preadv` loop | 31.7 | 1.00x |
| C io_uring sync (submit-all/wait-all) | 38.3 | 0.83x |
| C io_uring IOSQE_ASYNC | 45.6 | 0.70x |

Two conclusions, from a corrected measurement:
- The real lever is the loop language, not the submission mechanism. Moving
  the per-row gather from Python to a C preadv loop is ~1.8x (56 -> 32 ms)
  because CPython's per-call overhead dominates 32k tiny calls, not the
  kernel. That is the change that would recover GPU0 utilization.
- io_uring, even hand-written in C and batched, does NOT beat a C preadv loop
  for page-cache-resident reads (38-46 vs 32 ms): there is no device latency
  to overlap, so the per-SQE prep and enter just add work over a syscall the
  kernel services in ~1 us of copy.

After `liburing-dev` was installed, pyuring 0.3.3 - the only real Python
io_uring library - was measured on the identical gather through its
`UringCtx` ring: 163 ms, correct (5x slower than the Python preadv loop it
aims to replace). It is a ctypes wrapper over liburing, paying per-op Python
FFI plus one blocking `wait_completion` per CQE, and it has no batch
scatter-read API (high-level surface is `copy`/`write_many`; reads are
sequential `AsyncFile`). Linking the C extension against liburing proper was
considered and skipped: the raw-UAPI ring already submits and reaps with one
enter per window, which is what liburing does, and it still loses to C
preadv. io_uring is closed on four grounds (ctypes, numpy ring, raw C,
library); a C preadv gather helper is the open, worthwhile option.

## mmap beats every syscall flavor: staging access-pattern probe (2026-10-22)

Question: beyond C, is the preadv access pattern itself beatable? Yes.
Same 32,768x2 KB warm scatter gather, all variants byte-checked (script +
table in experiments/ple_gather_mmap_probe.py):

| gather path | ms |
|---|---|
| Python `os.preadv` loop (shipped) | 56 |
| C preadv loop | 32-36 |
| C io_uring / pyuring | 38-46 / 163 |
| C mmap + memcpy loop | 11.3 |
| torch `index_select` from mmap, per-shard `out=` slices | 2.7 |

Mechanism: preadv re-does the kernel's page lookup on every call
(fdget, xarray walk, kmap, ref/unref, fput) for pages the process could
reference directly. One MAP_SHARED mapping per shard + `torch.frombuffer`
+ per-shard `index_select` turns the gather into parallel memory loads of
already-resident pages with multi-threaded copies; syscall and VFS cost
go to zero. This supersedes the C-extension direction: it is pure
Python/torch, so the series stays rebuild-free.

Open risks to settle in a prototype (0014): row order undo pass
(+1x64MB copy), 12 M-page TLB pressure at full table size, mapped-page
reclaim stickiness (why upstream chose preadv), first-touch fault budget
across boot, and interaction with 0012 range-sparing fadvise (unchanged,
different consumer). Estimated per-step staging: 47-56 ms shipped -> 5-10
ms prototype; cold 1M back to pinned parity or better, and the shelved
prefill-ahead pipeline becomes viable again since a 5 ms thread job truly
hides under a 190 ms GPU step. Unmeasured on hardware.

### 0014: mmap gather built as an experiment (2026-10-22)

`experiments/0014-ple-mmap-gather.patch` (branch lab/ple-mmap-staging off the
promoted 9-patch tip, replay tree 09f7f92d, export sha256 17191fc2...):
`host_file_gather_mmap` engram flag; one read-only mmap per shard at bind,
per-shard `index_select` staging, WILLNEED pre-queue kept, preadv path
untouched as default. CPU suites green (new mmap/preadv byte-equality test);
hardware trial not yet run. Boot recipe: reset the fender tree to the 9-patch
series, `git am` 0014, serve with `--engram-config '{"cpu_offload": true,
"host_file_gather": true, "host_file_gather_mmap": true}'`. Gates: logprob A/B
against /tmp/cap_promo9.jsonl (must be tighter than boot-to-boot floor - same
bytes by construction), 100/4 clean salts (hypothesis: at or under pinned
350-357), cold trio, GPU0 btop during prefill, host free -g.

### 0014 hardware battery (2026-10-22 16:59 boot, banner dev143+g5ede4b9cc, mmap gather live)

Boot proof: `PLE file gather serving 320001536 shard rows through 33 file
mapping(s)` on Worker_PP0; engram resolved with host_file_gather_mmap=true;
load 42-61 s, engine init 23.5 s, pool 5,089,552, no errors. The loader's
`Dropped ... 125.29 GiB` line precedes bind, so residency proof comes from
the cold runs instead.

| cell | mmap (0014) | preadv (promo9) | pinned-era |
|---|---|---|---|
| logprob A/B vs preadv boot | 4.44% flips, KL 0.049, verdict ok | — | — |
| needles / accept-window | exit 0 / 4.982 | exit 0 / 4.99 | 4.98-5.00 |
| cold 1M TTFT x3 | 90.9 / 91.0 / 90.8 s | 92.6 / 92.8 / 92.9 s | 89.7-93.9 s |
| 100/4 wall x4 (bases 94100-94400) | 358-362 s | 362-367 s | 350-357 s |

Readings:
- Correctness at the floor exactly as intended: every correct-vs-correct path
  pair measures 4.41-4.44% flips and ~0.049 KL (sync-vs-lockstep this morning
  too), while the broken v1 read 10.5%. Three independent boots now define
  the floor; mmap is another point on it.
- Gains are real but smaller than the microbench promised, and the reason is
  production row counts: per-step cost scales with DISTINCT ids after dedup,
  and the cold delta (~2 s over 485 steps) puts the shipped gather at ~4 ms
  saved per step, not 40. The probe's 32k-worst-case overstated the typical
  step.
- Net position: mmap mode strictly beats preadv mode (cold -1.8 s to pinned
  parity, 100/4 -4 s, same RAM and boot savings) but does not beat pinned at
  100/4 (-1.4% residual); the whole-stream sync per staging step and H2D
  staging remain the gap, not the gather. Raw: logs/0014-*.json.
- Decision: promoted as series patch 10 (keeps the flag A/B-able), tree
  `09f7f92dfcf86bb58ce3621cf6608e94d42104bd`, 10 patches (seven ours, three
  adopted). Serve scripts set `host_file_gather_mmap: true`.

## Protocol battery v2 (2026-10-22)

Six protocol findings since 2026-10-10 all point at the same gap: the rules of
the battery were prose, and prose only binds whoever remembers it. The 216-vs-176
single-cell flips that were read as a bimodal machine (salt law, above), the
f660b7e gate fix, which voided a whole era of accepted numbers, the salt-step
overlap that let the prefix cache fake a warm 208-216 s cluster, the post-idle
first cell, the wind-down `sust` metric that wrote 545 tok/s for a four-stream
cell, and the L2 eviction-priority trial whose steps ran faster while
acceptance collapsed to 1.6. Each one was caught by a person looking back. They are now tool gates:
`bench.py` warms up before the grid, cools down between cells, steps salt bases
100 apart, checks the spec-decode counters before and after every cell for
background drafts, and reports acceptance measured inside the all-decoding
window. The battery below is the certification procedure that follows from them.

### Acceptance column epoch

The `bench.py` `acc` column counts prefill-phase drafts on bases at or after
`8407bc0bb6`, which is the current series base. Window logic and the reason for
it are in the `benchmark/accept_window.py` header. On a deep-fill cell most of
the wall is prefill (~60-65% at 100/4), and drafts issued in a mixed window are
rejected more often than drafts in uniform decode, so the full-window number
dilutes toward the prefill ratio. The effect is modest on the current tree
(100/4, decode 1024: 4.83 full window in `logs/n6-1004-c1.json` against
4.97-5.00 in the decode window) and grows with the prefill share of the cell.
It compounds with history: the 2.93-2.98 values of the 2026-10-10 grids were
the eager small-head window, not dilution, so the same-looking column can
carry three different effects across eras. Consequence:
`acc` columns measured before that base and after it are not comparable. New
records quote `acc win` and the `accept_window.py` bracket. Every historical
table in this file stands as a full-window legacy value from whichever base
produced it.

### The standard battery

Certification on the 4-card 1M lane (`--ctx 1000000`), one server, idle host,
owner's all-clear. Correctness gates run first: a timing number on a tree that
fails a gate is worthless.

```bash
# 1  cache-hit state: two hex needles (block 0, half depth), cold/warm/grown, exits non-zero
./benchmark/prefix_state_check.py --base-url http://fender.lan:8000
# 2  values: capture this boot, diff against its own repeat capture (the floor), then against the stored reference
./benchmark/logprob_capture.py --base-url http://fender.lan:8000 --k 8 --depths 8200,16400 --lp-stride 4 --out cap-boot.jsonl
./benchmark/quality_diff.py cap-repeat.jsonl cap-boot.jsonl
./benchmark/quality_diff.py cap-ref.jsonl cap-boot.jsonl
# 3  real content: greedy and sampled acceptance plus per-stream rates, one invocation
./benchmark/corpus_accept_probe.py --base-url http://fender.lan:8000 --modes greedy,sampled --corpus ../draft_corpus.jsonl
# 4  cold 1M TTFT trio: the 100/1 cell three times, salt bases 100 apart, one run per base
./benchmark/bench.py --base-url http://fender.lan:8000 --ctx 1000000 --fills 100 --concurrency 1 --warmup --cooldown 60 --salt <base> --json bench-1m-cold-r<n>.json
# 5  100/4 wall quartet: the batch cell four times, salt bases 100 apart, one run per base
./benchmark/bench.py --base-url http://fender.lan:8000 --ctx 1000000 --fills 100 --concurrency 4 --warmup --cooldown 60 --salt <base> --json bench-1m-n4-r<n>.json
# 6  decode-window acceptance bracket on a fixed salt
./benchmark/accept_window.py --base-url http://fender.lan:8000
# 7  deep plateau: four 988k contexts resident at once
./benchmark/sustained_decode.py --base-url http://fender.lan:8000 --concurrencies 4 --depth 988000 --decode 8192 --timeout 3000 --json bench-1m-n4-deep.json   # decode 8192, not 4096: see first-run note
# 8  E3 long-decode batch cell
./benchmark/bench.py --base-url http://fender.lan:8000 --ctx 1000000 --fills 100 --concurrency 4 --decode 8192 --warmup --cooldown 60 --salt 95100 --json bench-1m-e3-d8192.json
# 9  oversubscription: six streams against a pool that admits four
./benchmark/bench.py --base-url http://fender.lan:8000 --ctx 1000000 --fills 90 --concurrency 6 --warmup --cooldown 60 --salt 95200 --json bench-1m-oversub.json
```

The three cells that are new to this battery:

- Deep plateau (7). The 100/4 wall cells are ~60-65% serialized prefill (mean
  TTFT ~230 s of ~365 s wall), so they measure the pipeline, not four-resident
  decode. The fp8-KV pool (5,089,552 tokens) admits four 988k contexts plus
  decode tokens each with room to spare, and this cell is the first direct
  measurement of the all-resident decode plateau the fp8 KV lane sells. Quote
  it in tok/s aggregate and per stream. The tool now dumps its 0.5 s sample
  timeseries to the JSON so the plateau can be checked against the wind-down
  trap (see the first-run note below): at 988k depth the four prefills
  serialize at ~87 s each, so stream 4 only reaches first token at ~350 s and
  the all-four-decoding window is the final ~37 s. `--decode 4096` clears that
  window; a longer decode widens it and is cheap, so 8192 is the shipped value.
- E3 long-decode variant (8). Decode 1024 at full fill leaves an all-decoding
  window too short for a stable `acc win` sample. Decode 8192 raises the
  pure-decode share from ~20% of the cell to a majority of it. The E1/E2
  decode-length experiment earlier in this record established the geometry.
- Oversubscription (9). Six 890,096-token prompts plus decode demand against
  the 5.089M pool is ~5.35M of reservation, so admission must queue. Expected
  shape: TTFT stretches in the tail and the preempt column stays 0.

What this battery still cannot price: admission reserves prompt + max_tokens,
so API-driven load cannot force an align-mode preemption, and the ~2 min cost of
replaying a 1M prompt under align mode stays an open measurement. A future
fault-injection experiment is the only way to price it, either shrinking the
pool with `--gpu-memory-utilization` mid-session or routing through a connector.

### Salt bases (2026-10-22)

`bench.py` steps salt +100 per cell from a time-derived base. Two runs started
within (cells x 100) seconds can therefore land a later cell of the second run
on a prompt the first run already served, and the prefix cache fakes a warm
cell: the 208-216 s cluster in the pipeline trial was that artifact. Rule for
certification runs: pass an explicit `--salt` base at least 100 above the
highest base used in the previous 24 h of runs, and keep the bases in the run
names and the result table, as the promo9 and 0014 batteries above do. A state
file that records the last base used would remove the human step. It is future
work and does not exist yet.

### Corpus reference (2026-10-22)

Greedy and sampled corpus cells from `corpus_accept_probe.py` join the battery,
so every certification carries a real-content row next to the synthetic grid.
The two tasks measure different things and headline rates must name which one:
the synthetic repeat-task grid measures the drafter ceiling, where acceptance
sits at ~4.9 and single-stream decode at 205-226 tok/s, while real sampled
content accepts ~2.3 and real greedy content decodes 74-124 tok/s. A number
quoted without its task is not comparable to anything.

### Battery v2 first full run (2026-10-22 18:25-20:50, fender.lan:8000, Rust frontend boot, mmap-gather tree)

All 14 steps exit 0; raw dumps and per-step logs in `logs/battery-v2/`. The
server was the boot already live (owned_by vllm-frontend-rs, model id
qwen3.8-flash-next-fp8, max_model_len 1,000,000). One corpus was generated for
the run (`extract_maki_corpus.py`: 152 sessions, 27,163 records, ~8.6M tokens,
sha 4598072dad02).

Clean timing cells (confirmed-idle windows, bg=0, match column aside):

| cell | readings | comparison |
|---|---|---|
| cold 1M (salts 97100/97200/97310) | TTFT 90.3 / 90.4 / 90.2 s; prefill 10.9-11.0k tok/s (prompt / TTFT-window); decode win avg 183.1 / 190.1 tok/s, sust 199 | promo9 92.6-92.9, pinned 92.9-93.9: best cold trio this lane has logged. Rates are the rotating-salt band; the 205-215 headline was pinned salt |
| 100/4 wall x5 (97400/97500/97600/97700/97610) | 348.5-360.9 s, median 356.7; pipeline prefill 11.2-11.6k tok/s; per-stream win avg 37-52 during the prefill-dominated wall | promo9 362-367, pinned 350-357; TTFT mean 219.8-223.0, preempts 0 |
| accept-window, salt 7770 | 5.000 / 5.000 / 5.000 | at the pinned-era ceiling |
| deep plateau 4x988k, decode 4096 | 507 agg / 127 per-stream tok/s | first direct four-resident decode number; genuine all-decoding window, see below |
| E3 100/4, decode 8192 | wall 571.9 s, win avg 32.3 per stream, acc full 1.62, acc win 2.28, match 82 | long greedy tail drifts, see content note |
| oversub 90%/6 streams | TTFT mean 284.3, p95 464.3, wall 489.9, pipeline prefill 11.1k tok/s, preempt 0, acc 4.58 | admission queues the sixth stream exactly as the battery predicts |

Corpus cells (real content, time seeds, 4 streams x 45k prompts): greedy acc
win 3.043 at 57.7-81.8 per-stream tok/s, sampled 2.218 at 50.0-61.5,
aggregate 212 / 178 tok/s. Within a hundredth of the v2-era baselines (2.3
sampled floor), reproduced by the new dual-mode tool on its first run. These
are the served-traffic rates; the grid's repeat-task cells are the drafter
ceiling, and a drifted salt (97300/97310 here) pulls the rotating single-
stream band to 60-70 win avg, which is content, not the machine.

Quality gates: needles exit 0. Boot-vs-repeat floor 7.759% top-1 flips
(mean KL 0.0669, greedy tails 36.4%) sits a half-point above the historical
7.27-7.30% floor band, same order. The diff against `capture_official_boot4`
is void, not a regression: that reference was captured against a different
`draft_corpus.jsonl` extraction, so it compares different window contents
(99.3% flips is content mismatch). Fixed in tooling while running: captures
now carry a corpus sha fingerprint and `quality_diff.py` prints INCOMPARABLE
when the two sides disagree. Any pre-fingerprint reference must be re-captured
against the current corpus before its flips mean anything.

Two calibration findings from the new gates, both now fixed in tool:

1. The background gate false-positived on its own cell. vLLM flushes step
   stats on a ~5 s interval; the 2 s settle before the after-snapshot missed
   the cell's own tail, and the post-probe counted 77-79 of its drafts as
   foreign (cold-r3, n4-r3). The settle is 6 s now; the reruns read bg=0.
   Related and real: this session's operator model is the model under test,
   so any tool-polling between cells is genuine background traffic on this
   box. Battery runs must come from a host reasoning elsewhere.
2. The content canary learned its discriminator on live data. Five deep cells
   read match below 0.98 with healthy acceptance (n4-r3b 75%/4.28, oversub
   83%/4.58): greedy loops at 988k drift their tails on ordinary salts. Two
   salts (97300, 97310) collapsed match AND acceptance together (56-73%,
   acc win 1.46-2.09), identically on runs 80 minutes apart, while
   accept-window between them read 5.000 and other salts stayed clean:
   salt-deterministic content behavior, not a corruption class (L2 poisoned
   every salt of its boot). The WARNING text now prints acc beside match and
   states the rule. Consequence for the salt law: on this tree at 988k
   single-stream, the rotating-salt acceptance band reaches ~1.5, not the
   3.2 floor recorded 2026-10-12. Pin the salt; when rotating, quote the
   wider band or the drifted tail belongs to the machine by assumption.

The deep plateau number is real, and narrow. The sample timeseries (now
dumped by sustained_decode) shows first tokens at 96/178/263/351 s, the run
median at 10 tok/s while three prefills serialize under the fourth, and the
all-four-decoding window in the final ~37 s: 507 agg / 127 per stream.
Four 1M streams resident do decode together at 3.4x the single-stream
aggregate (concurrency reclaiming the PP bubble, as the step anatomy
predicts), and the measurement only exists because the tool now shows the
window it averaged. Decode 8192 widens the window. The spec line is
updated.

## INT8 draft-screen evaluation (2026-10-22, offline, artifacts in /tmp/int8_screen)

Fork mechanism, corrected from source: NOT int8x int8 - it is Marlin W8A16
(bf16 activations, int8 weights per-row/K128 scales, fp32 accumulate) over the
full 248,320-row head, top-32 per TP shard, exact bf16 rescore, batch=1 only
(`src/draft_int8_runtime.py:18-29`), gated to their TP2/MTP6 stack.

Fidelity on real margins (362,442 positions parsed from /tmp/cap_promo9.jsonl,
plus the real lm_head and 40k artifact from ~/test/Qwen3.8-Flash-Next-W8A16):
screen noise sigma 0.017-0.025 nats, 3-6x finer than the bf16 output grid the
head already rounds to; value-changing argmax flips bounded <= 7e-4/position;
Monte Carlo of their exact algorithm: retention 1.000 even at 8x noise
(the exact rescore needs >32 rivals inside the noise band; real softmax mass
allows that at 0.026% of positions). Effectively lossless for greedy drafting.

But as a patch-5 replacement the cost model rejects it: 647 MB head reads per
pass vs our 207 MB; calibrated at our measured 63% stream-to-wall efficiency
that is ~+4.8% single-stream vs the +8.7% patch 5 already delivers, +616 MiB
extra residency, and their sampled-traffic path caps per-token acceptance at
0.905 (top-8 support q, measured on real top-k mass). Our residual 40k-head
exposure: 2.5% of held-out positions (93.7% of them at 200k depth, high-
confidence), already priced in hardware at 4.99 vs 4.995.

Salvaged idea, strictly better fit: run OUR 40k head itself as CUTLASS W8A8
(our own kernel shootout: W8A8 beats bf16 1.51x on these shapes). Halves the
207 MB to ~103 MB/pass, margin analysis says int8 logit error is far below the
bf16 grid; est ~+1.5-2% decode for one boot to test. Candidate 0015.

## Decode-M int8 head microbench (2026-10-22, GPU2 co-resident with the live
server, ratios only; real draft-40k and lm_head weights; raw
/tmp/head_int8_{no_shim,shim}.json, script /tmp/head_int8_bench.py)

The regime nobody had measured - the draft head at decode sizes, in or near
graph capture - with three backends: bf16 aten mm (status quo), CUTLASS W8A8
(the plugin's op, scales fused, bf16 out), cuBLASLt torch._int_mm TN int8
(int32 out, no scales - shown bare and under the ltautotune shim).

| head | M | bf16 ms | CUTLASS | lt no-shim | lt +shim | bf/CUTLASS | bf/lt+shim |
|---|---|---|---|---|---|---|---|
| draft40k | 1 | 0.141 | 0.101 | n/a (torch needs M>16) | - | 1.39x | |
| draft40k | 4 | 0.161 | 0.103 | n/a | - | 1.55x | |
| draft40k | 16 | 0.172 | 0.107 | n/a | - | 1.61x | |
| draft40k | 70 | 0.234 | 0.140 | 0.128 | 0.105 | 1.67x | 2.21x |
| draft40k | 280 | 0.556 | 0.356 | 0.336 | 0.282 | 1.56x | 1.97x |
| draft40k | 2048 | 2.827 | 1.827 | 2.062 | 1.567 | 1.55x | 1.80x |
| lmfull 248k | 1 | 0.891 | 0.559 | n/a | - | 1.59x | |
| lmfull 248k | 70 | 1.292 | 0.823 | 0.876 | 0.891 | 1.57x | 1.45x |
| lmfull 248k | 2048 | 17.234 | 11.295 | 12.521 | 12.531 | 1.53x | 1.38x |

Findings:
- CUTLASS W8A8 wins EVERY decode-M point for the draft head, 1.39-1.67x, and
  the per-token activation quant is free (+quant within 4us of the raw GEMM).
  0015 premise CONFIRMED. Expected: 4 head calls/step, saving ~0.16 ms/step at
  M=1 (about +1.7% single-stream decode) up to ~0.4 ms/step at batch 70.
  Modest - the head is already only 4x40k bytes wide after patch 5.
- The ltautotune shim is a real effect where cuBLASLt mispicks (40k head,
  M>=17: up to 1.32x self-gain, ending up FASTER than CUTLASS) and a no-op
  where the heuristic was already right (248k head, 0.98-1.00x). It cannot
  serve the draft head in production: torch._int_mm refuses M<=16 (no GEMV
  path at all), it outputs int32 needing a separate scale pass, and it
  requires plain warm-up GEMMs per shape outside graph capture. Keep it as an
  offline kernel-selection probe; no integration.
- Absolute GEMV rates imply ~1.43 TB/s effective head streaming (92% of GA100
  spec) - revises the agent's 1.1 TB/s cost model up ~30%; its ratios stand.
- bf16 here IS the production path (fp8 weights dequantize to bf16 at load).
- Reproduction check: M=2048 CUTLASS/bf16 = 1.53-1.55x matches the Oct-16
  prefill shootout (1.51-1.52x) on the same two heads.

## 0015 + 0016 built: W8A8 int8 for the draft head and the dense linears (2026-10-22)

Branch lab/ple-int8-heads on the 10-patch tip, exports
`experiments/0015-draft-head-w8a8.patch` and
`experiments/0016-dense-linears-w8a8.patch`. Both env-gated, default off,
CPU-verified (22 new tests; test_ple failure set unchanged at the 11 GPU-env
entries; ruff clean).

- 0015 `VLLM_QWEN4EXP_DRAFT_W8A8=1`: swaps the draft head's quant_method to
  per-channel RTN + per-token dynamic quant + `cutlass_scaled_mm`. Directly
  backed by the decode-M microbench (1.39-1.67x at every M, quant free).
  sm_80 has no low-latency skinny-GEMM plans, so the bf16 GEMV replaced is
  exactly what was benched. Expected: ~+1.5-2% single-stream decode.
- 0016 `VLLM_QWEN4EXP_DENSE_W8A8=1`: in-place method swap on per-tensor-FP8
  dense linears (suffix allowlist: qkv/o, GDN in/out, gate_up/down), int8
  cache built at process_weights_after_loading from canonical checkpoint
  bytes before the FP8 kernel rewrites them. Decode M<512 (inside the
  capture range, max capture size here is 80) keeps today's path bit-for-bit;
  only prefill pieces switch kernels. MoE untouched by construction
  (exact-type Fp8LinearMethod claim). Backed by the Oct-16 prefill shootout
  (1.29-1.68x on seven dense families at M=2048). Expected: the bigger lever,
  sized at boot. One open question the boot must answer: which bf16 kernel
  the FP8 selector actually picks on this box (Marlin vs dequant) - the
  startup log names it; the int8 path replaces it either way. VRAM: int8
  copies live next to the FP8 weights (a few GB on the last stage); check
  the KV pool line stays 5.08-5.09M.
- Boot plan (battery v2, owner all-clear required): trees tested separately,
  one env variable per boot. Boot A = 0015 only, Boot B = 0016 only, Boot C
  = both if A and B pass. Gates in battery-v2 order: needles, capture+repeat
  floor + reference diff (note: reference must be re-captured against the
  current corpus - the boot4 reference is INCOMPARABLE), corpus greedy+sampled,
  cold trio (98100/98200/98300), 100/4 quartet (98400-98700), accept window,
  deep plateau, E3, oversub. For 0016 the value gates matter most: dense
  logits change, so quality_diff boot-vs-reference will legitimately read
  above the floor; the acceptance bar there is corpus acc win >= 2.2 sampled
  / 3.0 greedy floor and needles + match canary, not bit-identity.

### Boot A: battery v2 on 0015 (draft-head W8A8), 2026-10-23 00:21-01:40

Tree = 10-patch series + 0015, `MTP draft head scores as W8A8 int8 via
CUTLASS` on PP3, mmap gather live, pool 5,086,567, init 24.5 s. All 17 steps
rc=0, every cell bg=0/0, preempts 0. Raw: logs/battery-v2-boot-a/.

| gate | boot A | reference (first v2 run, same corpus sha) |
|---|---|---|
| needles | pass | pass |
| floor (boot vs own repeat) | 7.743% flips, KL 0.0661 | 7.759% |
| value vs mmap boot | 7.821% flips = floor +0.08pp | - |
| accept-window (salt 7770) | 4.992 (n=2) | 5.000x3 |
| corpus greedy / sampled | 2.991, 3.166 / 2.108, 2.367 (two seeds) | 3.043 / 2.218 inside band |
| cold trio (98100-98300) | TTFT 90.6 / 90.1 / 90.2 s | 90.2-90.4 |
| 100/4 (98400-98700) | 355.1 / 358.5 / 353.8 / 422.9 | median 356.7 |

n4-r4 (salt 98700) is content, not the machine: the canary read 58% match
with acc 3.88 and the WARNING prints the drift rule; same for cold-r3
(match 0.0, 98300). Under greedy, verify masks the int8 drafter exactly as
predicted: the value diff sits at the floor (+0.08pp), acceptance is band-
identical, cold and batch cells match the mmap reference. The expected
+1.5-2% decode micro-gain is NOT resolvable in these cells (E3 and the
plateau vary more with seed content than with head time); it would need a
dedicated fixed-salt single-stream A/B across two boots, env off vs on.

Verdict: 0015 PASS as quality-neutral and regression-free. Wall-clock
justification is micro; its real value is as the safe half of the int8 pair
and as the harness rehearsal for Boot B (0016), where prefill cells should
move directly. Boot B awaits the next owner boot.

### Boot B inert twice: the real root cause is the checkpoint, not the guards (2026-10-23)

The 09:57 boot (export with the block-128 accept fix) again logged
"scored as W8A8 int8 ... 0 of 151 layers", so the block-quant theory was
wrong for both versions. Reading the served config settled it:
`quantization_config` is fp8 block [128,128] with 943 entries in
`modules_to_not_convert`, covering every dense linear (self_attn k_proj /
o_proj on the full-attention layers, all linear_attn.in_proj_* and
out_proj, mlp.shared_expert.*, hyper-connections, routers, lm_head,
embed). The 75,264 `weight_scale_inv` tensors all belong to routed
experts (plus the PLE table). Production dense linears therefore load as
BF16 under `UnquantizedLinearMethod`, and no `Fp8LinearMethod` exists for
them to match, which is why both claim checks returned zero. This also
retroactively explains the Oct-16 shootout note "fp8 dequantize to bf16
then aten::mm": that was plain BF16 unquantized all along, and the Marlin
selection question was moot. The 0016 premise survives unchanged (int8
replaces the dense BF16 GEMM, 1.29-1.68x benched, GA100 int8 tensor cores
run at twice the bf16 rate).

0016 rewritten (lab 8a58ee81, replaces df6d66bb):
`Qwen4ExpDenseW8A8LinearMethod` subclasses `UnquantizedLinearMethod`.
Allowlist, size floors, M>=512 runtime gate and fallback are unchanged.
The int8 cache is built in `process_weights_after_loading` from the bf16
weight, which stays canonical there (the parent rewrites nothing on CUDA),
and the bf16 weight survives for the fallback and the captured graphs.
21 CPU tests replace the FP8-era ones (per-shard / single-scale / block-
grid tests dropped, subclass-avoidance and chunking-equality added).
Re-exported `experiments/0016-dense-linears-w8a8.patch`; replay of series
tip + 0015 + 0016 exports gives tree 1882eb0c, identical to the lab tree.

Boot B retry #3 gates: the line "Dense bf16 linears scored as W8A8 int8
via CUTLASS: N of M layers" must read N > 0 (expected per stage: GDN x3 +
shared MLP x2 per layer + qkv/o on full-attention layers, the fused qkv
falls to BF16 because k/v are ignored, plus the MTP module). Quality bar
unchanged: needles, canary, corpus floors, not bit-identity. VRAM watch:
about +0.2 GB per stage for the int8 copies, note it if the pool line
drops below 5.05M.

### Boot B (0016 dense W8A8, retry #3): PASS, prefill +5-6% (2026-10-23 10:43-12:11)

Tree = 10-patch series + 0016 only (fresh re-am, export sha fc9676cd, boot
log claim lines 39/36/36/33 + MTP 3, pool 4,958,208 = -2.5% vs 5,086,567,
the expected int8-copy cost). All 18 steps rc=0, every cell bg=0/0, zero
preempts including oversub on the smaller pool. Raw: logs/battery-v2-boot-b/.

| gate | boot B | boot A / mmap ref |
|---|---|---|
| needles | pass | pass |
| floor (boot vs own repeat) | 8.725% flips, KL p50 0.0001 | A: 7.743% |
| value vs mmap ref | 9.432% flips = floor +0.7pp | expected shift (dense logits change) |
| accept-window (fixed salt 7770) | 4.998 (n=3) | 4.992 / 5.000 |
| corpus greedy (acc win, two runs) | 2.902, 2.904 | A: 2.991, 3.166; floor >2.9 |
| corpus sampled (acc win) | 2.389, 2.384 | A: 2.108, 2.367; floor >2.0 |
| sampled/greedy ratio | 0.823 | floor >0.65 |
| cold trio TTFT | 85.5 / 85.4 / 85.4 s | 90.2-90.6 |
| 100/4 sustained | 360.9 / 363.9 / 364.7 / 367.2 | ref median 356.7 |
| 100/4 in_tps | 11668-11847 (spread 1.5%) | A: 9352-11173 |

Prefill is the target and it moved: cold TTFT -5.3%, cold in_tps +5-6%,
n4 in_tps +6% vs boot A's clean median with a tighter spread (A's quartet
ranged 9352-11173, B's 11668-11847). Quality controls hold: the
fixed-content accept-window is unchanged, corpus floors pass in both probe
runs, needles pass, KL medians at noise level.

Two observations, neither a gate failure. (1) The within-boot floor itself
reads +1pp vs boot A (8.725 vs 7.743); with dense numerics changed the
repeat-pair flips shift slightly, and the vs-reference number (+0.7pp above
this boot's own floor) is the correct comparison. (2) The third cold salt
drifted via canary (match 0.0, acc 3.72) exactly as boot A's third cold
salt did, and E3/deep read -4.6/-5.8% sustained with acceptance moving 1:1
(4.97->4.55 on E3's salted content). 0016 cannot alter decode numerics
(all captured and plateau batches are far below the M=512 gate), and the
fixed-salt control is identical, so these are content effects. Deep-plateau
level is the one residual to glance at on Boot C.

Verdict: 0016 certified quality-neutral and +5-6% on prefill-dominated
cells. Boot C (0015+0016 together, both envs) awaits the next owner boot.

### Boot C (0015+0016 both envs): PASS, acceptance dips are transient (2026-10-23)

Tree = 10-patch series + 0015 + 0016 (both exports fresh am). Boot log:
dense claim lines 39/36/36/33 + MTP 3/14 and draft-head line
(mtp.py:512), pool 4,958,208 = identical to Boot B, no preemptions
anywhere. Two full battery sessions, same boot: C-1 12:46-14:12
(logs/battery-v2-boot-c/), C-2 14:36-16:00 (logs/battery-v2-c2/), plus
five single-cell probes in between.

| gate | C-1 | C-2 |
|---|---|---|
| needles / steps | pass, 18/18 rc=0 | pass, 19/19 rc=0, bg=0 all cells |
| floor / vs-Boot-B / vs-mmap-ref | 8.597 / 8.500 / 9.372 % | 8.609 / 8.514 / 9.329 % |
| accept-window (fixed salt) | 4.998 (n=3) | 4.995 (n=1) |
| corpus floors | greedy 3.30/2.78, sampled 2.07/2.34, ratio 0.63/0.84 | greedy 2.90/3.08, sampled 2.16/2.31, ratio 0.74/0.75 |
| cold trio acc | 1.67 / 1.44 / 4.38 | 4.83 / 4.84 / 4.75, all match 1.0 |
| n4 quartet acc | 4.77/2.87/4.86/4.34 | 2.41/2.68/4.85/3.40 |
| cold TTFT | 85.2-85.6 s | 85.2-85.6 s |

The verifier is untouched by 0015: Boot-B-vs-Boot-C capture diff (8.50)
sits below this boot's own floor (8.60), exactly as predicted for a
draft-only change (teacher-forced scoring never calls the draft). Cold
prefill gains reproduce to the tenth of a second.

The story of this boot is the free-run acceptance dips. Same boot, same
salt, different runs: 100200 read 1.44 then 4.18; 102400 read 2.41 then
3.99; C-1's collapsed cold trio was fully healthy in C-2 (worst cold cell
of C-1, 100100, re-read 2.11 - still recovering, timing intact); fresh
salts probed mid-session read 4.77-4.79 with match 1.0. Every dip
recovered, sustained prefill throughput never moved with them, and the
fixed-content accept-window never moved either. Deep plateau on the
identical fixed-content cell has now read 488 (A), 459 (B), 423 (C-1),
592 (C-2), so that cell carries environment, not code. Decode numerics
cannot explain the dips either: with both envs on, the int8 branch is
unreachable below M=512, so C's decode path is the same code A ran clean.
This is the box's known bimodality lottery (boot4-era 276-380 n4 spread)
landing on more cells during C's hours than during A/B's.

Verdict: 0015+0016 together certified quality-neutral on every gate that
is deterministic or content-fixed; prefill +5-6% confirmed a second time.
Caveat recorded: battery-v2's free-run cells stay noisy at 1M depth.
Next: promotion decision for 0015/0016 into the series (patches 11/12).

### 0015 and 0016 promoted to series patches 11 and 12 (2026-10-23)

Series is 12 patches on 8407bc0bb6, tree `1882eb0cb1f20493faf233debeac06be108f59eb`
(verified by `git am` of all twelve exports from base in a clean worktree).
Both remain opt-in: `VLLM_QWEN4EXP_DRAFT_W8A8` and
`VLLM_QWEN4EXP_DENSE_W8A8` default off. The serve script is untouched;
flipping production is an owner boot decision.

### Series rebased onto 4bceda5dbf (2026-10-23)

12 patches from base 8407bc0bb6 moved +256 upstream commits to
4bceda5dbf (newest main with a live cu130 wheel, dev390). Zero conflicts;
the series delta is byte-identical except one dead variable ruff flagged in
patch 6's test (squashed into that commit). Replay verified: the twelve new
exports applied to the pristine base give tree
`fff72613e0ec6997fd455d31b14d292d402618c7`, same as the branch. CPU suites:
312 pass, 11 test_ple env failures identical to the pristine base. New
upstream dependency on the serve import path: orjson (comes with the pinned
wheel install). Nothing upstream merged that makes a patch redundant; the
range holds no Qwen4Exp model changes except an MM-encoder PP skip. Fender
keeps running the old-base tree (same content, banner dev145) until the
owner reinstalls on the new base; boot check after that reinstall is the
usual one-line gate plus needles, no full battery.

### New-base boot certified (2026-10-23 16:54, dev402+g3b448e9a1)

Fender reinstalled on 4bceda5dbf and booted the 12-patch am head (banner
dev402 = wheel dev390 + 12 patches). All tree markers in the log: int8
handoff active, dense claim lines 39/36/36/33 + MTP 3/14 with both int8
envs on, draft-head line, mmap gather 320001536 rows / 33 mappings, pool
4,958,208, zero errors, capture normal. Gate per plan, no battery: needles
pass; one cold probe (salt 103100, logs/probe-newbase-cold-103100.json):
TTFT 85.6 s (prefill gain intact), match 1.0, preempts 0, and sustained
239.3 tok/s against the old base's 199.0 band at acceptance 4.31. The
decode side reads faster than the pre-rebase envelope by more than content
explains; single data point, watch it opportunistically on the next
fixed-content accept-window run, no dedicated bench.

### Device ceilings per weight format (2026-10-23, cards idle, 1470 MHz, 200 W cap)

Tooling: `crit eval probe` from crit-labs/crit 0.5.0, patched for Ampere
(`experiments/crit-probe-int8.patch`: int8 format, copy buffer sized from free
memory, `--shapes`, null bandwidth reported instead of crashing). Medians of 3
runs per card, utilization 0% before each, server resident but idle. Raw:
`/tmp/ceil-gpu[0-3].json`.

| card | copy, GB/s | bf16, TFLOPS | int8, TOPS | int8/bf16 | fp8 | nvfp4 |
|---|---:|---:|---:|---:|---|---|
| 0 | 1592 | 180.6 | 238.2 | 1.32 | none | none |
| 1 | 1594 | 181.0 | 233.8 | 1.29 | none | none |
| 2 | 1594 | 182.4 | 232.9 | 1.28 | none | none |
| 3 | 1593 | 185.4 | 232.5 | 1.25 | none | none |
| mean | 1593 | 182.3 | 234.3 | 1.29 | | |

1. The bus sustains 1593 GB/s of copy traffic (read plus written counted). Cards
   0 and 3 could only fit a 0.9 GiB buffer and read the same as the 2 GiB pair,
   so the figure does not depend on the buffer size.
2. The die offers 223 TFLOPS bf16 and 446 TOPS int8 at 74 SMs and 1470 MHz
   (GA100 per-SM rate). cuBLASLt captured 82% of the bf16 rate but 53% of the
   int8 rate, which is why the table shows 1.29x. That int8 row is an algorithm
   selection artifact and not a device ceiling: the same probe run under
   `cmp/int8-gate/shim/ltautotune.so`, which autotunes INT8 TN algorithms, reads
   int8 333.8 TOPS (75% of peak) and a 1.85x ratio, with bf16 unchanged at 0.99x.
   The library's own heuristic never offers the winning algorithm. Reproduced on
   this box at 4096^3: 249.5 TOPS for the library pick and 353.3 through the shim,
   against the 249.8 and 354.4 the shim README published. Quote 1.85x as the int8
   ceiling and 1.29x as what a caller gets by asking the heuristic.
3. FP8 has no tensor-core path on this hardware, and NVFP4 neither. This is the
   cuBLASLt-side confirmation of the 2026-10-16 trace erratum: the checkpoint's
   fp8 weights are widened to bf16 at load and every dense GEMM is a bf16 GEMM.
   Low-bit storage buys bytes here, never FLOPs.
4. The prefill-sized curve is flat: bf16 170-183 and int8 217-232 across
   512x12288x2560, 2048x640x2560, 2048x2560x640, 2048x2560x2560,
   2048x4608x4608, 2048x12288x2560, 2048x2560x12288 and 8192x12288x2560, all
   within 8%. Skinny-K (640) costs bf16 6% and int8 1%.
5. The 1.1 TB/s cost model is dead, and so is the 580-790 GB/s band built on it.
   That band was a solo microbench run on a co-resident card (2026-10-12), not a
   production rate. Dividing exact checkpoint bytes by the trace's own GEMM time
   puts production decode at 895 GB/s, against 1593 GB/s of bus and 984-1070 GB/s
   for an isolated GEMM at the same shape and M. See "Production decode byte rate,
   reconciled" below.

Reproduce:

```bash
SHAPES="2048x12288x2560;2048x2560x12288;2048x2560x2560;2048x4608x4608;2048x640x2560;2048x2560x640;8192x12288x2560"
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu crit eval probe --out ceil-gpu$gpu.json --shapes "$SHAPES"
done
```

### Decode-M streaming at the production layout (2026-10-23, GPU2 idle)

Instrument: `~/test/cmp/int8-gate/bench/decode_gemv.cu`. It builds exactly
`D[M,N] = X[M,K] W[N,K]^T` with every buffer row-major, which is what `F.linear`
hands cuBLASLt: W as (K,N) ld K with opT, X as (K,M) ld K, D as (N,M) ld N. Each
cell is the minimum over 6 rounds of 20 back-to-back launches on a private
stream, best of the algorithms the library offers, per the rule in
`int8-gate/shim/README.md`.

Two layout traps, recorded because both produced wrong numbers on the way here.
`int8-gate/bench/ltbench2` stores D as column-major M x N with ld = M, which is
`W X^T` rather than `X W^T`. It read 340 GB/s at m=5 against 1104 at m=1, a
convincing 3.3x cliff for M that is not a multiple of 8, and the cliff is the
output layout. The crit probe cannot go below M=512 either: it returned 163.8
TFLOPS at M=1, which is 0.38 us for a 63 MB read.

shape qkv / gate_up, 12288 x 2560 (62.9 MB as bf16, 31.5 MB as int8). The GB/s
columns are the streaming rate and are the right measure while the GEMM is
bandwidth-bound. Past M=80 it is compute-bound and only the us columns mean
anything. The crit probe and this bench disagree by 16% at M=2048 on the same
layout (0.746 ms against 0.866 ms), so read ratios within one instrument rather
than absolutes across the two. Every decode row below comes from this bench.

| M | bf16, us | bf16 weight, GB/s | int8, us | int8 weight, GB/s | int8 gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 53.9 | 1167 | 30.2 | 1041 | 1.79x |
| 5 | 62.2 | 1011 | 28.9 | 1087 | 2.15x |
| 8 | 56.8 | 1108 | 29.3 | 1074 | 1.94x |
| 20 | 55.7 | 1129 | 30.8 | 1021 | 1.81x |
| 40 | 53.6 | 1174 | 31.6 | 996 | 1.69x |
| 80 | 73.5 | 856 | 47.4 | 663 | 1.55x |
| 512 | 234.0 | 269 | 160.1 | 197 | 1.46x |
| 2048 | 865.9 | 73 | 619.6 | 51 | 1.40x |

1. bf16 streams a dense weight at 1011-1178 GB/s for every M from 1 to 40, which
   is 63-74% of the 1593 GB/s bus. Single-stream MTP-4 decode runs M=5 (one row
   per request plus four drafts, scheduler `padded_num_tokens = 1 +
   num_spec_tokens`), and the captured decode buckets are 5, 10, 20, 25, 35, 40,
   so M=5 replays unpadded. It is not a bad point on this curve.
2. The byte halving pays most at decode M: 2.15x at M=5 against 1.40x at M=2048.
   int8 holds 1000-1090 GB/s on a half-size weight, while prefill-sized int8 is
   compute-bound and lands at 1.4x.
3. Patch 12's `M >= 512` gate therefore excludes the regime with the largest
   gain. That gate is a CUTLASS kernel constraint, not a device one: cuBLASLt
   offers 2-3 INT8 TN algorithms at every M up to 80 and they run at about
   1.05 TB/s. Patch 11 already scores the draft head as int8 at M=1..80 inside
   the captured graphs, so small-M int8 is not foreign to the decode path.
4. What this table does not price: int8 here stores int32 D, and production needs
   a per-token per-channel dequant to bf16 (61k elements at M=5) plus the
   per-token activation quant (12.8k elements). Both are two orders of magnitude
   below the 28.9 us GEMM, and patch 11 measured the quant as free on the head.
   It also does not price the verifier shift from scoring dense linears as int8
   during decode, which is the gate that decides this.
5. Consequence for the roofline: see the next section. The comparison this section
   invites, isolated 1011-1178 GB/s against the step anatomy's 580-790 GB/s, is
   not a real gap. The 580-790 band was a co-resident solo microbench, and the
   production rate measured against exact checkpoint bytes is 895 GB/s, which is
   84-91% of the isolated rate. The prize here is the byte halving, not better
   bf16 kernels. `ltautotune.so` remains the instrument that shows the library
   leaves 1.43x unclaimed on int8 algorithm choice (GPL-2.0, LD_PRELOAD), not a
   shipping plan.

### Production decode byte rate, reconciled (2026-10-23)

The claim "the step is 66% byte-side at the solo band" compared a trace time
against a microbench number measured under co-resident load. Nobody had divided
the actual bytes by the actual trace time. That division is the whole point of
this section, and it is arithmetic over files already on disk.

Method. `benchmark/dense_weight_bytes.py` reads every shard header of the served
checkpoint, so the dense bf16 matrices a decode step reads are exact: 2-D BF16
weights minus routed experts, minus the PLE n-gram table, minus the token
embedding and minus the vision tower. Assigning layers to ranks under
`13,12,12,11` and dividing by the `gemm-bf16` row of `logs/phase-report-7770.txt`
gives the achieved rate. Isolated rates come from
`int8-gate/bench/decode_gemv.cu` at the same shapes and at M=5, the MTP-4
single-stream decode batch. Captured decode traces carry no tensor shapes, which
is why the byte side has to come from the checkpoint.

| rank | dense MiB per step | gemm-bf16 ms/step | achieved |
|---|---:|---:|---:|
| 0 | 1955.5 | 2.26 | 907 GB/s |
| 1 | 1733.0 | 2.03 | 895 GB/s |
| 2 | 1733.0 | 2.03 | 895 GB/s |
| 3 | 2970.7 | 3.84 | 811 GB/s |
| sum | 8392.3 | 10.16 | 866 GB/s |

1. Ranks 0, 1 and 2 are the trustworthy rows: each matrix is read exactly once per
   step, no output head and no drafter sits on ranks 1 and 2. They land at 895 to
   907 GB/s. Rank 3 also holds `lm_head` (1212 MiB) and the drafter, whose 172.7
   MiB of dense weights and 192 MiB draft head are replayed several times per step
   and are not separated by the trace categories, so its row and the sum carry more
   modelling error than the per-rank rows.
2. An isolated GEMM of the same matrix at M=5 runs at 984 to 1070 GB/s
   (`in_proj_qkv` 984, `q_proj` 1011, `in_proj_z` 1057, `out_proj` 1070).
   Production therefore sits at 84 to 92% of the isolated rate, and at 56% of the
   1593 GB/s bus.
3. There is no large kernel-quality prize hiding in the bf16 path. The 1.13 to
   1.19x distance to an isolated GEMM, applied to 66% of the step, is at most
   about 10%, the same size as the glue-fusion ceiling. The 2026-10-13 verdict
   stands, and the "580-790 GB/s" band it quoted was a solo microbench result from
   a co-resident card rather than a production rate.
4. The bytes are the lever, and they are worth about 2x. At M=5 the same dense
   matrices as int8 take: `q_proj` 62.2 to 28.3 us (2.20x), `in_proj_qkv` 53.3 to
   22.9 (2.33x), `in_proj_z` 29.7 to 13.5 (2.20x), `out_proj` 29.4 to 23.0 (1.28x,
   the K-wide case). Summed over one GDN layer's four large projections that is
   165.7 to 82.3 us, a 2.01x rate gain on 160 MiB.
5. Scaled onto the step, cutting the 10.16 ms bf16 slice to about 5.2 ms moves the
   clean 22.9 ms compute sum toward 18 ms, so up to 1.25x on single-stream decode,
   212 toward roughly 265 tok/s. Two costs are not in that number: cuBLASLt stores
   int32 D and production needs a per-token per-channel dequant to bf16, which at
   M=5 is 61k elements and one more kernel inside every captured bucket, and the
   verifier shift from scoring dense linears as int8 during decode, which is the
   gate that decides it.
6. Kernel library matters here. The numbers above are cuBLASLt int8 TN, which
   offers only 2 to 3 algorithms at decode M and runs them at about 1.1 TB/s.
   Patches 11 and 12 call `cutlass_scaled_mm` instead, and the next section
   measures that arm at the same shapes under graph replay: 1.19 to 1.50x at
   M=5, which is what a patch can actually ship today.

### The dense int8 arm at decode M under graph replay (2026-10-23)

This prices the patch 13 candidate. The bf16 arm is `F.linear`, the deployed
sm_80 dense path. The int8 arm is what patches 11 to 13 call:
`scaled_int8_quant` on the activations plus `cutlass_scaled_mm` with
per-channel weights and a bf16 output. Each arm was captured into one CUDA
graph and the replay timed, minimum of 6 rounds of 20 replays. GPU2, server
resident and idle. Shapes are the checkpoint's own BF16 dense tensors.
Instrument `/tmp/graph_arms.py`.

| shape | M=1 | M=5 | M=20 | M=40 | M=80 |
|---|---:|---:|---:|---:|---:|
| gdn in_proj_qkv 10240x2560 | 1.45x | 1.47x | 1.38x | 1.21x | 1.41x |
| gdn in_proj_z 6144x2560 | 1.51x | 1.47x | 1.20x | 1.16x | 1.18x |
| gdn out_proj 2560x6144 | 1.46x | 1.50x | 1.31x | 0.92x | 1.07x |
| o_proj 2560x2560 | 1.18x | 1.19x | 1.01x | 0.82x | 1.03x |
| qsa q_proj 12288x2560 | 1.52x | 1.42x | 1.29x | 1.20x | 1.37x |

Absolute times at M=5, microseconds: in_proj_qkv 52.7 against 35.8, in_proj_z
29.7 against 20.2, out_proj 32.8 against 21.9, o_proj 17.2 against 14.4,
q_proj 52.5 against 36.9.

1. Measure a decode-path kernel under capture or not at all. The same two arms
   timed eagerly say the int8 arm loses on most shapes, 0.28 to 0.37x on o_proj.
   The reason is `scaled_int8_quant`, which costs about 20 us of launch overhead
   on a 12,800-element input against a bf16 GEMM of 16 to 54 us. Graph replay
   removes that overhead and the ranking flips. Any eager decode-size microbench
   on this box gives the inverted answer.
2. Single-stream MTP-4 decode is M=5, the best column: 1.19 to 1.50x on all five
   shapes. Patch 12's `M >= 512` floor withholds about a 1.4x factor from the
   slice that already moves 895 GB/s.
3. Scaled onto the step. `lm_head` (1212 MiB per step on rank 3) is not claimed by
   the arm, so of the trace's 10.16 ms bf16 slice roughly 9 ms is eligible. At
   1.4x that becomes 6.4 ms, the clean 22.9 ms compute sum moves toward 20.3 ms,
   and single-stream decode predicts about +13%, 212 toward roughly 239 tok/s.
   The prediction is arithmetic on measured rates, not a measured step.
4. The arm is not the ceiling. cuBLASLt int8 measured 2.20 to 2.33x on these
   shapes earlier the same day, about 1.5x again better, but vLLM exposes no
   small-M cuBLASLt int8 binding (`torch._int_mm` refuses M<=16 and stores int32
   behind a separate scale pass), so the shippable column is 1.4x.
5. Two shapes fall under parity at the M=40 bucket, out_proj 0.92x and o_proj
   0.82x. The boot A/B decides whether that costs anything against the M=5 win.
   Cost otherwise: no new memory, because patch 12 already builds the int8 cache
   beside the bf16 weight, and one extra kernel per eligible layer per bucket.
6. Not measured here, and it is the gate: logprob drift from scoring dense
   linears as int8 at decode sizes. Patch 12 certified only at prefill M, where
   the verifier averages over thousands of rows, while decode scores five.

Built as `experiments/0015-dense-w8a8-decode-m.patch` on branch
`lab/dense-w8a8-decode-m`, exported 13-patch series tree `9007320d5d`. The knob
is `VLLM_QWEN4EXP_DENSE_W8A8_MIN_M`, default 512, so nothing changes until a
boot sets it.

### Patch 13 in vivo: decode-M int8, two boots, one variable (2026-10-24)

Same base `4bceda5dbf`, same 13-patch tree, same serve script. Boot A
`VLLM_QWEN4EXP_DENSE_W8A8_MIN_M=1` (23:47), boot B control `=512` (00:02). Each
printed its floor in the claim line on all four ranks at unchanged counts
(39/36/36/33 + MTP 3), zero errors. Pool 4,955,223 against the control's
4,961,194, so the decode arm costs 5,971 tokens, 0.12%.

Cells: `bench.py --ctx 1000000 --fills 50 --concurrency 1 --decode 1024`, one
salt per row, `win avg` is the per-stream decode-window rate. Raw
`logs/{p13,ctrl}-50n1-s*.json`.

| arm | salt | win avg tok/s | sust agg | acc len | TTFT s | wall s |
|---|---:|---:|---:|---:|---:|---:|
| control M>=512 | 104000 | 191.9 | 202 | 4.83 | 40.2 | 45.5 |
| control M>=512 | 104100 | 192.7 | 202 | 4.84 | 40.3 | 45.6 |
| control M>=512 | 104200 | 193.2 | 202 | 4.85 | 40.3 | 45.6 |
| patch 13 M>=1 | 104000 | 202.9 | 209 | 4.89 | 40.3 | 45.3 |

1. The control is reproducible: 191.9 to 193.2 across three salts, 0.7% spread,
   sustained pinned at 202 and acceptance at 4.83 to 4.85. So a single patch 13
   cell is weak evidence, but the direction is not noise.
2. Acceptance is not the explanation for the whole gap, yet it is part of it.
   Patch 13 read 4.89 against a control band of 4.83 to 4.85, about +1%, and
   decode rate tracks acceptance length. Normalising it out, step time is
   acceptance over rate: 4.84/202 = 23.96 ms control against 4.89/209 = 23.40 ms
   patch 13, so the kernel effect is about -2.3% of the step.
3. That fails the prediction. The graph-replay microbench predicted 9 ms of
   eligible dense GEMM falling from 10.16 to about 6.4 ms, which is -12.7% of the
   22.9 ms clean step. Measured is -2.3%, roughly a fifth of the model. The
   isolated arms were right about their own kernels, so what is wrong is the
   transfer: how much of the live decode step those claimed layers actually
   occupy. Two candidates, both checkable only with a paired `PROF=1` boot: the
   layers the arm claims may be a smaller share of a graphed decode step than the
   trace's `gemm-bf16` category suggests, since that category also carries
   `lm_head` and the drafter, and the fused decode path may already serve some of
   those GEMMs from the low-latency kernels, which the exact-type check leaves
   alone by design.
4. Quality is unaffected within resolution. Fixed-seed acceptance at 204,882
   prompt tokens: control 4.319 and 4.336, patch 13 4.282 and 4.392, per-position
   0.937 to 0.949 at position 0 on both arms, so the two distributions overlap.
   Teacher-forced fingerprints: within-boot floors 8.975% (control) and 9.235%
   (patch 13) top-1 flips, cross-arm 9.460% and 9.720%. The excess over the floors
   is +0.2 to +0.7 points, the size boot B's dense W8A8 certification accepted as
   a legitimate shift, and `quality_diff` returns ok on both pairs. Greedy tail
   exactness 40.9% cross against 39.4% at the floor. `prefix_state_check.py`
   passed on the patch 13 boot, four sessions resuming from 82,942 recomputed
   tokens cold to 78 warm with no leakage.
5. Where this leaves patch 13. It is safe, it is quality-neutral, it costs 0.12%
   of pool and about 2.3% of step time in the right direction. Two percent is not
   worth a promotion argument on its own, and the 13% story that justified it is
   dead unless a profiled pair shows where the other 10 points went. Open items,
   cheapest first: three more patch 13 cells to confirm 2.3% rather than one cell,
   then a `PROF=1` paired boot if the 2% is to be defended or rebuilt into
   something larger.

### Patch 13 step anatomy: where the missing points went (2026-10-24, PROF=1 boot)

Same tree as the triple above, profiler armed, `phase_probe.py --salt 7770
--decode 4096` on a 100% fill single-stream cell. Window self-closed at 150
iterations, four rank traces in `logs/prof`, report at
`logs/phase-report-p13.txt`. Cadence read 41.1 ms against a 21.6 ms clean step,
so this window is inflated more than the 2026-10-13 one. Read shares and
cross-category deltas, not absolutes.

The arm is definitely live in the captured graph. Rank 1 and 2 each spend 975 us
per step in 36 launches of `cutlass::Kernel2<enable_sm80_to_sm89<...` at 27 us
each, and rank 0 spends 1061 us in 39 launches. 36 is 12 layers times three
claimed GEMMs, which matches the boot's claim line. `marlin-int8` stays 0.

Per real step, ms summed over the four ranks, this boot against the 2026-10-13
anatomy on the seven-patch tree:

| category | 2026-10-13 | patch 13 | delta |
|---|---:|---:|---:|
| gemm-bf16 (also matches cutlass int8 by name) | 10.16 | 8.64 | -1.52 |
| humming | 7.59 | 7.38 | -0.21 |
| qsa-attn | 2.29 | 2.20 | -0.09 |
| gdn/state | 1.75 | 1.74 | -0.01 |
| moe | 1.48 | 1.50 | +0.02 |
| copy/memset | 1.00 | 1.36 | +0.36 |
| elementwise | 0.67 | 1.00 | +0.33 |
| other | 0.75 | 1.16 | +0.41 |
| reduce/norm | 1.18 | 1.11 | -0.07 |
| compute sum | 26.89 | 26.13 | -0.76 |

1. The byte saving is real and it is almost entirely re-spent. The GEMM category
   drops 1.52 ms, about a 1.4x cut on what the arm claims, which agrees with the
   1.47x measured under graph replay. Over the same step `copy/memset`,
   `elementwise` and `other` grow by 1.10 ms combined, and the compute sum falls
   only 0.76 ms, 2.8%. Kernel launches per step rose from 2540 to about 2780,
   which is the per-token activation quant and its staging copies.
2. The 1.47x was never going to be 13%. Two independent reasons, both visible
   here. First, `gemm-bf16` in a phase report is a name-regex bucket that also
   holds `lm_head`, the drafter and every other cutlass kernel, so it is not the
   arm's reachable slice: the arm claims 36 of 138 linear layers on the middle
   ranks, three GEMMs per layer, and the trace shows exactly that, 0.98 ms per
   rank rather than the 2.5 ms per rank the category total implies. Second, the
   arm carries a per-call overhead that the isolated bench folded into one number
   and the step then pays again as extra kernels.
3. The A/B triple agrees. Patch 13 at salts 104000, 104100 and 104200 reads win
   avg 202.3, 205.7 and 194.4 against the control's 191.9, 192.7 and 193.2.
   Acceptance differs cell by cell (4.87, 4.88, 4.70 against 4.83, 4.84, 4.85), so
   normalize it out: step time is acceptance over rate, 23.30, 22.59 and 22.49 ms
   against 23.91, 23.96 and 23.96 ms. That is -4.8% mean, all three cells in the
   same direction. The anatomy says -2.8%. Take the range as 3 to 5%, so
   212 toward roughly 220 tok/s single-stream.
4. Third cell caveat: patch 13's 104200 cell accepted 4.70 where the control
   accepted 4.85 on the identical prompt, which is the int8 logits steering the
   greedy path elsewhere rather than a machine change. Same-prompt acceptance
   differences of that size are what the values floor already covers.
5. If the 1.4x on the GEMM is to survive into the step, the overhead has to go.
   The cheapest follow-up is one activation quant per layer input instead of one
   per claimed GEMM: three quant passes plus their staging copies currently buy
   what one could, and 1.10 ms of re-spent time is larger than the 0.76 ms saved.
   That is a fused-quant idea, not a gate change, and it is the only version of
   this axis worth a new patch.
6. Not established, and it would need one more profiled boot at MIN_M=512 on this
   same base: the 2026-10-13 column is a different tree, so the category deltas
   above carry base drift. The A/B triple does not have that problem, and it is
   the number to quote.

### W8A16 dense lane against both patch 13 arms (2026-10-24)

Purpose: test whether the per-call activation quant, not the GEMM itself, was what
capped the int8 dense win at decode. The lane halves the same dense bytes with no
activation quant at all, because the weights are int8 in storage and a bf16 GEMM
consumes them after dequant.

Boots, all PP=4, MTP-4, fp8 KV, PP int8 handoff on, PLE mmap gather on:
`ctrl-minm512` official FP8 tree with the dense arm floored at 512, `p13` official
tree floored at 1, and this lane, the repaired `Qwen3.8-Flash-Next-W8A16` tree.
The lane is not a one-variable change: its dense linears go through
compressed-tensors so patches 12 and 13 claim nothing, patch 11's int8 draft head is
off in this boot, and its draft-vocab artifact is the older 40,000-id list against
the official tree's 39,532. Read it as lane versus lane.

| arm | win avg tok/s over salts 104000/104100/104200 | sustained | acc len | TTFT s | in tok/s | pool |
|---|---|---:|---:|---:|---:|---:|
| official, dense arm floored at 512 | 191.9 / 192.7 / 193.2 | 202 flat | 4.83 / 4.84 / 4.85 | 40.2 / 40.3 / 40.3 | 10,851 / 10,837 / 10,832 | 4,961,194 |
| official, patch 13 floored at 1 | 202.3 / 205.7 / 194.4 | 209 / 216 / 209 | 4.87 / 4.88 / 4.70 | 40.1 x3 | 10,944 / 10,962 / 10,892 | 4,955,223 |
| W8A16 dense lane | 200.8 / 198.9 / 198.8 | 209 flat | 4.91 / 4.88 / 4.88 | 44.1 / 44.2 / 44.2 | 10,055 / 10,019 / 10,014 | 5,176,119 |

Step time from acceptance over sustained rate, which removes the acceptance
difference: control 23.94 ms, patch 13 22.79 ms, W8A16 lane 23.40 ms.

1. The hypothesis is refuted. Removing the activation quant entirely gives -2.3%
   against the control, which is worse than patch 13's -4.8%. The per-call quant was
   therefore not the binding cost, and the weight-only dequant path is the slower of
   the two at decode M. This closes the "avoid the quant" idea for a different reason
   than the "dedupe the quant" idea, which was already dead because the three claimed
   GEMMs per layer consume three different tensors.
2. The lane pays for it at prefill: TTFT 44.2 against 40.2 is +10%, and in tok/s
   10,019 against 10,851 is -7.7%. That is the same Marlin weight-only behaviour
   measured on 2026-10-16 at 0.40-0.81x of bf16 for prefill batches.
3. What the lane does buy is capacity. Pool 5,176,119 against 4,961,194 is +214,925
   tokens, +4.3%, and maximum concurrency crosses from 4.96x to 5.18x, so five full
   1M requests become resident instead of four. Since batch residency is what the
   step anatomy names as the remaining lever, the lane is a multi-stream argument,
   not a single-stream one, and pricing it needs a five-resident cell that no lane
   has ever run.
4. Values move further than patch 13 did. Against the control the lane reads 10.199%
   top-1 flips and 0.0978 mean union KL; against patch 13, 10.549% and 0.1027. The
   within-boot floors are 8.975% and 9.235%, so the excess is +1.2 to +1.6 points,
   against patch 13's +0.2 to +0.7. `quality_diff` returns ok on both, and greedy
   tail exactness is 34.8% against a 39.4% floor. Consistent with the lane's own
   2026-10-12 gate, which read 8.1% against a 7.3% floor.
5. Acceptance is unchanged by the dense format: 4.364 and 4.328 at 204,882 prompt
   tokens on the lane, inside the control's 4.319 to 4.336 and patch 13's 4.282 to
   4.392.
6. Net for the patch set. Best measured dense-byte lever at decode stays patch 13's
   W8A8 at about 5% single-stream. The W8A16 lane is not a decode win and costs
   prefill, but it is the only configuration here that crosses the five-resident
   threshold, so it belongs in the queue behind a multi-stream capacity cell rather
   than in the single-stream comparison.
### Five resident at 1M on the W8A16 lane (2026-10-24, 576 s cell)

Question from the lane comparison: the W8A16 tree crosses maximum concurrency
from 4.96x to 5.18x, so can five near-full 1M requests actually decode together
where the official tree can only hold four.

Cell: `sustained_decode.py --concurrencies 5 --depth 995000 --decode 4096`, so
five requests reserve 5 x 999,096 = 4,995,480 tokens, which sits between the
official pool 4,961,194 and the lane pool 5,176,119. That is the only shape that
separates them. Raw `logs/w8a16-plateau5.json`, 1,147 samples.

| streams | wall s | first tokens s | plateau agg | per stream |
|---:|---:|---|---:|---:|
| 5 | 576.4 | 100.1, 186.6, 277.1, 376.8, 475.5 | 640.6 tok/s | 128.1 tok/s |

1. All five were resident at once. The proof is in the timeseries, not the
   headline: aggregate rate is only 10 to 30 tok/s between 100 s and 505 s, then
   reaches 651.7 tok/s at 562.9 s, about five streams' worth. The long flat part is
   the documented starvation effect, each arriving 995k prefill hogs the pipeline
   and holds the already-decoded streams near 2 to 8 tok/s each, which is what kept
   streams one to four alive until stream five finished its prefill at 475 s.
2. Read the 640.6 as a ceiling, not a sustained rate. Only 32 of 1,147 samples sit
   at 80 percent of peak, roughly 16 s of the 576 s cell, because every stream gets
   just 4,096 decode tokens and the prefills consume 82 percent of wall. This is the
   wind-down trap from the protocol note, at five-stream scale.
3. Per-stream cost of the fifth stream is close to nothing: 128.1 tok/s with five
   resident, against the deep-plateau reference of 127 tok/s per stream at four
   resident 988k contexts. Aggregate therefore scales roughly linearly, 507 to 641
   tok/s, +26 percent, on 4 percent more pool. That is the capacity argument, and it
   is real on this lane.
4. What is NOT established: whether the official tree would have failed this cell.
   The arithmetic says it must, 4,995,480 of demand against a 4,961,194 pool, so the
   fifth prefill cannot be admitted until one of the first four finishes, which would
   show as a plateau near four streams, roughly 512 tok/s, and a fifth first-token
   time pinned to some other stream's completion rather than to the prefill queue.
   Confirming needs the same 576 s cell on the official tree.
5. Cost of running this shape at all: 576 s of wall for 20,480 generated tokens, and
   four of the five streams spend most of that time starved. A service that actually
   runs five concurrent 1M sessions on this box gets a short all-five window and very
   poor latency for early arrivals, so the lane's extra residency helps only
   with a scheduling change that decouples prefill admission from decode rate.
### Five resident on both trees: the control (2026-10-24)

Identical cell on the official SHIP tree, patch 13 arm at M=1, pool 4,955,223,
against the lane's 5,176,119. Demand is 5 x 999,096 = 4,995,480 tokens, which the
official pool misses by 40,257 and the lane clears by 180,639. Same
`sustained_decode.py` command, `--timeout 1500` because the default 600 is only
24 s above the lane's own wall and the control needs longer, which is itself part
of the result.

| cell | wall s | first tokens s | plateau agg | per stream | requests per s | out tok/s over wall |
|---|---:|---|---:|---:|---:|---:|
| official, 4 streams | 396.3 | 92.5, 170.7, 261.1, 351.2 | 513 | 128 | 0.01009 | 41.4 |
| official, 5 streams | 899.8 | 92.5, 170.7, 261.1, 351.2, 818.1 | 472 | 94 | 0.00556 | 22.8 |
| W8A16 lane, 5 streams | 576.4 | 100.1, 186.6, 277.1, 376.8, 475.5 | 640.6 | 128.1 | 0.00867 | 35.5 |

1. The pool is the constraint, proven directly. On the official tree the fifth
   stream's first token lands at 818.1 s, 467 s after the fourth, instead of on the
   even 85 to 100 s prefill ramp the other four follow. It could not be admitted
   until one of the first four released blocks. On the lane the fifth arrives at
   475.5 s on the normal ramp, so five really are resident.
2. Five resident on the lane beats five on the official tree by 56 percent in wall
   and 36 percent in aggregate plateau, 640.6 against 472. That is the capacity
   result, and it is real and measured.
3. Five resident on either tree is worse than four on the official tree. Requests
   completed per second: 0.01009 at four, 0.00867 at five on the lane, 0.00556 at
   five on the official tree. End-to-end generated tokens per second tell the same
   story, 41.4, 35.5 and 22.8. Extra residency does not pay for itself here because
   each arriving 995k prefill starves the streams already decoding, which the
   server log shows plainly at 15 tok/s aggregate generation while a prefill runs.
4. So the lane's 4.3 percent more pool is not a usable win at this shape, and the
   binding constraint is scheduling rather than memory. A fifth resident stream is
   only worth admitting if arriving prefills stop grinding decode to a halt, which
   is a scheduler question and not something either checkpoint changes.
5. Consequences for the two open decisions. Patch 13 stays the best measured dense
   byte lever, at about 5 percent single-stream decode, and the lane does not
   displace it, since the lane's step time was worse (23.40 ms against 22.79 ms).
   The lane's one claimed advantage, residency, is now measured and does not
   convert, so there is no configuration in this session that beats the shipped
   official tree with the dense arm at M=1.
### Starvation mechanism: slow, not skipped (2026-10-24)

Question left open by the capacity cells: when three or five 995k prompts arrive
together, decoding streams fall to a few tokens per second. Is decode being held
out of steps, or scheduled into steps that are simply too expensive?

Probe `benchmark/itl_probe.py`, official SHIP tree with patch 13 at M=1, three
streams at 995,000 prompt tokens, 2,048 requested each, salt 106000, wall 329 s.
It timestamps every streamed event per stream and classifies each gap by whether
another stream was still prefilling at that moment.

| regime | events | mean | p50 | p95 | max | share below 0.5x p50 | share above 2x p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| another prefill running | 117 | 670.1 ms | 666.2 | 717.1 | 1007.7 | 0.0% | 0.0% |
| all prefills finished | 1,222 | 314.0 ms | 36.8 | 759.4 | 1485.6 | 0.0% | 40.7% |

1. Decode is slow, not deferred. While prefills arrive the gaps are tight and
   unimodal, p50 666 ms against p95 717 ms with literally no sample below half or
   above twice the median. A scheduler that held streams out of steps would show
   long silences punctuated by bursts, which is what the all-prefills-done column
   actually looks like (p50 37 ms with 41% of events over 2x that, because the last
   streams are finishing).
2. The arithmetic follows. Each stream advances once every four engine steps,
   because the V2 async scheduler stamps `next_decode_eligible_step = step +
   pp_size` (`vllm/v1/core/sched/async_scheduler.py:52-57`, gate at
   `scheduler.py:792-796`) and the stagger is hard-coded to `pp_size` on CUDA. So a
   666 ms observed cadence is a 166 ms mixed step, against the 37 ms pure-decode
   step, a factor of 4.5. Prefill runs 1,616 tokens per step (2048 budget minus the
   Mamba-align clip), which at 166 ms is about 9.7k tokens/s and matches the logged
   ~10k tok/s.
3. So the lever is mixed-step cost, and exactly one knob addresses it on this base:
   `--long-prefill-token-threshold`, honoured at `scheduler.py:820` and `:1276`,
   default 0 meaning off. It also bypasses the 1616 alignment when set below it
   (`scheduler.py:597`). Predicted effect at 512: mixed step 166 to about 51 ms, so
   per-stream cadence during arrivals 666 to about 205 ms, roughly 3x better ITL,
   with prefill throughput held near its bandwidth-bound 10k tok/s. Not yet
   measured, that is the next boot.
4. Ruled out by code reading rather than by experiment. `max_num_partial_prefills`
   and `max_long_partial_prefills` no longer exist in the tree. There is no
   PrefillDelayer, no decode-first policy, and `--scheduling-policy priority` does
   not reorder running requests. DBO and microbatching are refused by the V2 runner
   with pipeline parallelism, speculative decoding or hybrid models. Turning async
   off removes the four-step gate but is an unverified correctness path on V2, since
   the runner has no consumer for the payload the scheduler then emits.
5. Two things this probe did not settle. The Rust frontend batches streamed
   deltas, so the unit measured is an event, not a token; the classification
   survives because batch size is stable within a regime, but the absolute numbers
   should be read as cadence, not ITL. And this cell appeared to show 3.73 M
   prefilled tokens against 2.98 M of prompt content, which would have been a 25%
   silent recompute. It was not: two clean cells in the A/B that follows count
   exactly 2.98 M, so the excess was in-flight work from the 900 s cell that had run
   immediately before, sampled inside my window. See the retraction in the A/B
   section, and the rule that counters must only be read after the engine drains.
### A/B of `--long-prefill-token-threshold 404` (2026-10-24, one boot)

Boot carried `long_prefill_token_threshold': 404` in its config dump, 0 errors,
int8 lane active at `M=1`. Threshold chosen as 404 because it divides the 1616
block size, so chunk ends land on block boundaries every fourth chunk instead of
randomly inside them. Profiler deliberately off, since CUPTI inflates the cadence
this measures.

Single-stream, the shape this deployment actually serves:

| cell | LPT off (certified) | LPT 404 | verdict |
|---|---:|---:|---|
| cold 1M TTFT s | 85.2 to 85.6 | 85.6 | unchanged |
| 50/1 TTFT s | 40.1 to 40.3 | 40.4 | unchanged |
| 50/1 sustained tok/s | 209 to 216 | 212 | unchanged |
| acceptance | 4.83 to 4.89 | 4.82 to 4.85 | unchanged |
| prefix reuse at 64k | 99.8%+, ~20 recomputed | 100.0%, 16 recomputed | unchanged |

This was predicted from code, and it held: the cap is forced to 0 whenever fewer
than two requests are eligible (`scheduler.py:733-739`), and the sub-block chunk
path is explicitly supported (`scheduler.py:588-600`), which the reuse probe
confirms does not cost the align-mode grid anything.

Three concurrent 995k prompts, 2,048 tokens each:

| metric | LPT off (one cell) | LPT 404 (two cells) |
|---|---:|---:|
| wall s | 329.0 | 288.4, 302.1 |
| first tokens s | 91.0, 169.1, 258.1 | 268.5, 270.5, 270.6 and 269.6, 271.0, 271.2 |
| cadence while another prefills | p50 666.2 ms, n=117, unimodal | n=7 and n=4, no usable sample |
| cadence once prefilling stops | p50 36.8 ms, 40.7% of events over 2x that | p50 32.6 and 26.9 ms, 3.3% over 2x |
| engine prompt tokens counted | 3.73 M against 2.98 M expected | 2.98 M exactly, twice |

1. The knob does not speed up the starved regime, it deletes it. With the cap at
   404 the three prefills interleave and all finish at about 270 s, instead of one
   at a time at 91, 169 and 258 s, so there is almost no window in which one stream
   decodes while another prefills. My predicted 666 to 205 ms inside that regime
   was never testable, and the result is a change of shape rather than a
   rate improvement. Measuring the in-regime cadence needs a staggered-start probe,
   which `itl_probe.py` does not do yet.
2. The cost is the first arrival. Without the cap the session that arrived first
   starts answering at 91 s. With it, nobody answers until 270 s. Aggregate cell
   time improves about 10 percent, 288 to 302 against a single 329 s baseline, and
   the trailing stream is no worse off, 258 to 271 s, so this is a fairness change
   that spends the best latency to buy the worst a small improvement.
3. Retraction: the 25 percent prompt over-count reported in the mechanism section
   did not reproduce. Two clean cells here count exactly 2.98 M prefilled tokens and
   exactly 6,144 generated. The earlier excess came from sampling the counters
   immediately after a 900 s five-stream cell on the same boot, so in-flight work
   landed inside my window. It was a measurement error, not silent recompute, and
   the lesson is to sample counters only after the engine has drained.
4. Also corrected: the mechanism section listed 1,170 starved samples. The probe
   printed 117. The distribution shape it describes is unaffected.
5. Disposition. Not adopted as a default. This rig serves one long session at a
   time, where the knob provably does nothing, and under concurrency it trades a 3x
   worse first-arrival time for roughly 10 percent of aggregate. The `LPT` variable
   stays in `serve-4x-1m.sh` as a documented option for a workload that arrives in
   bursts, where equalising may be worth more than the leading edge.

### MoE fan-out measured: the axis is closed (2026-10-24, FANOUT=1 boot)

The number the last byte axis needed. Grouped expert kernels take 148 us per
layer-step, 95 us gate_up plus 53 us down, and each distinct expert costs 4.92 MB of
fp8 weight traffic, so achieved bandwidth is the distinct count times 33.2 GB/s.
That made the count a threshold test: about 20 distinct means 665 GB/s against the
895 GB/s the dense path sustains in the same step, up to 1.4x unclaimed on 27% of
the step, while 27 or more means the kernels are already at the ceiling.

Decode forwards are captured into CUDA graphs, so `select_experts` never runs at
replay and the count cannot be read directly. Prefill forwards do run in Python, and
a window of five adjacent tokens there is a good proxy for a decode step's five rows
(one token plus four drafts), because acceptance length 4.85 of 5 means the drafts
almost always are the next five real tokens. Cell: `bench.py --fills 50
--concurrency 1 --decode 1024 --salt 116000`, 584 reports across four ranks.

| rank | reports | mean distinct | range of report means |
|---|---:|---:|---|
| PP0 | 155 | 37.7 | 29.4 to 43.0 |
| PP1 | 143 | 36.4 | 28.8 to 43.7 |
| PP2 | 143 | 35.9 | 28.3 to 41.9 |
| PP3 | 143 | 35.7 | 29.3 to 41.3 |
| all | 584 | 36.4 | 28.3 to 43.7 |

1. 36.4 distinct experts is 1,212 GB/s, above the 984 to 1,070 GB/s an isolated dense
   GEMV reaches on this box and well above the 895 GB/s measured in the live step,
   approaching the 1,593 GB/s copy ceiling. The grouped expert kernels are not
   leaving bandwidth on the table. The MoE byte axis is closed by measurement.
2. This corrects the 2026-10-13 record, which put grouped execution at "325-650 GB/s,
   consistent with near-SOL" using a ~20 unique-expert estimate, and separately
   bounded "cannot exceed ~20 unique experts per layer-step". The count is roughly
   36, so that range understated achieved bandwidth by about 2x. Its conclusion,
   near-SOL, was right; the number behind it was not, and it was circular, inferred
   from the kernel time it later used to justify that time.
3. Overlap between adjacent tokens' top-10 sets is only 1.4x of the 50 slots a step
   offers, so routing is far from degenerate: five adjacent tokens touch 36 of 512
   experts. That also bounds what any expert-locality scheduling idea could save,
   since a step already has to read 36 distinct expert blocks, about 179 MB.
4. Consequence for the box. Byte reduction is now exhausted on every large slice of
   the decode step. Dense linears are int8 through patch 13, worth 3-5%. Experts are
   fp8 at 1,212 GB/s, at the ceiling. KV is fp8. Hyper-connection hops are int8. NVFP4
   compute does not exist on sm_80. What remains from the original list is batch
   residency, which the five-stream test showed is limited by prefill starvation
   rather than memory, and a different checkpoint.

## QSA sparse attention on sm_80, first measurement (2026-10-24)

The question started as "can the pth-flash-attn int8 kernel replace our fp8 KV read
path". Static reading said no, because their op has no per-query index argument. That
was wrong, and building and testing their kernel on this box proved it wrong while
exposing something more useful: our own sparse attention kernel is far off the
bandwidth this access pattern can reach.

### Their kernel works on sm_80, where nobody had run it

Their `docs/benchmarks.md:618` states only compute capability 8.6 was ever measured
and that sm80 and sm89 are build-and-CPU-gate only. Built here with

```
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 \
  /home/gab/test/vllm/.venv/bin/python setup.py build_ext --inplace
```

which needs no source edits under CUDA 13.4, g++ 16.2 and torch 2.13. The result is
2.94 MB of SASS, 2161 symbols all `compute_80`/`sm_80`, 24 `mma...satfinite.s32.s8.s32`
IMMA instructions, and ptxas reports 0 bytes spilled, 33 registers, 8 KB shared memory.
`benchmark/pth_attn_gate.py` runs 8 shapes on a CMP 170HX and all pass against a
dequantized reference at 0.25 to 0.54% relative error, covering `BS` of 1, 2, 4, 8, 16,
816 and an MTP verify length of 5.

### Token-level sparse selection is expressible, no kernel edit

Their geometry is `j = block_table[r][i]`, `o = i % BS`
(`csrc/pth_attn/cache_geometry.cuh:10-13`). At `BS = 1` every page holds one token, so
`o` is always zero and `block_table[r][i]` is a plain per-token list. Position enters
their math only through the bottom-right causal bound "positions `j <= limit(r) =
seq_len - q_len + tok`" (`csrc/pth_attn/mask.cuh:8`), and there is no rotary term
anywhere in `mask.cuh`, `attn_kernel.cuh` or `cache_geometry.cuh`, because K is stored
after RoPE. So any subset is a legal visible set, and a permuted table is exactly a
selection. The gate test above scatters physical pages deliberately, and the result
still matches the reference, which is the direct proof.

Two details that make this work at our shapes. `block_table` is indexed by request
rather than by query row, and one decode step carries 5 rows in one request, so the
rows go over as 5 pseudo-requests with `query_start_loc` `[0,1,2,3,4,5]`. And our
selector groups are 4 consecutive tokens with `block_topk` 512
(`vllm/models/qwen4_exp/config.py:152`), which is finer than the page size their vLLM
wrapper allows, so the per-token list at `BS = 1` is what closes that gap. Their
wrapper's floor of `BS >= 8` and the multiple-of-16 rule are Python-side refusals on a
kernel that takes `BS` at runtime.

### The byte story I assumed is dead, so this became a kernel story

`serve-4x-1m.sh:138` passes `--kv-cache-dtype fp8` and the boot log confirms it,
"Using fp8 data type to store kv cache" (`boot-4x-1m.log:15`). The served tree supports
it in QSA: `supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]`
(`vllm/models/qwen4_exp/nvidia/qsa.py:154-159` on the 13-patch tip), storing e4m3 bytes
in a uint8 cache and dequantizing on load inside the Triton kernel. One byte per
element already, so int8 buys no bytes, no pool and no capacity. `/home/gab/test/vllm`
holds an older bf16-only variant of this kernel that asserts `torch.bfloat16`, which is
what made the first A/B compare bf16 against int8 and report a 2x byte advantage that
does not exist on the serving lane.

### Measured at equal bytes, their kernel is 1.6x ours

`benchmark/pth_vs_qsa_fp8.py`, one layer-step at the real decode shape, 5 rows times
2048 selected tokens over a 200k logical prefix, D=256, 24 query heads, 2 KV heads,
both arms moving 10.0 MiB. GPU2 beside the live server, so the arms are interleaved in
paired blocks and only the ratio is quoted. Arm A's cache holds e4m3 bit patterns read
through the linear byte decode, so its numerics are meaningless and its traffic is
correct. Arm B's numerics are the ones the gate certified. `logs/pth-ab-fp8-117100.txt`
with salt 117100.

| quantity | value |
|---|---|
| random gather ceiling, 10240 rows of 1024 B | 338 GB/s, 341 at start |
| arm A, our served fp8 Triton kernel | 110.3 us, 95 GB/s, 28% of ceiling |
| arm B at split count 64 | 67.4 us, 156 GB/s, 46% |
| arm B best, split count 4 | 35.9 us, 292 GB/s, 86% |
| ratio with both arms at 64 splits | 1.61, spread 0.023 over 4 blocks |

This run is superseded by the next section. It asked `_select_config` without `is_fp8`,
so arm A was timed at 64 splits, which the served kernel never uses on sm_80, and 64 is
also near arm B's worst setting. The 1.61 is an artifact of that pairing.

The ceiling probe matters more than the ratio. A linear read is the wrong reference for
sparse attention, and `uint8.sum()` is the wrong probe twice over, it is an integer
reduction that reports 88 GB/s on a card doing 895 in production. Gathering bf16 rows
and reducing them measures this pattern at 338 GB/s, which puts our kernel at 28% of
what the access pattern allows. If the card was partly busy both efficiencies are
understated by the same factor, so the ratio stands and the 28% is a floor on how much
headroom exists.

### Share of the decode step

3 full-attention layers per PP rank out of 12, because `layer_types` gives 12
full_attention and 36 linear_attention with `full_attention_interval` 4. Ranks are
serial for one stream. At the corrected 3.1x the saving is 3 layers times 4 ranks times
71.8 us, 862 us of the 22.79 ms step, 3.8%. That is the same order as patch 13, which was
promoted for 3 to 5%. The 2.3% figure this section first carried came from the superseded
1.61x pairing.

### Consequences for the lane

1. Adopting their kernel is possible and worth about 3.8% of the decode step. It is a
   real change: their op cannot read the cache QSA writes today, so a `StreamCache`
   split is needed so the attention rows become int8 with per-token-head scales while
   the indexer keeps its e4m3 byte stream. The earlier size read said about 500 lines
   with about 100 in vLLM and the rest in their package.
2. The defect is in our kernel body, not in a knob. `benchmark/qsa_decode_tune.py`
   swept 30 combinations of `BLOCK_N` from 16 to 256 against split counts from 4 to 128
   on the served tree. Every cell lands between 105.6 and 112.0 us, the best buys 1% of
   this kernel's time and 0.08% of the step, so there is no configuration prize here.
   Recovering the 3.8% means editing the Triton kernel, most likely the e4m3 byte decode
   in the inner loop, which the shipped sm_80 path already works around by widening
   tiles: `if is_fp8 and _is_sm80(): BLOCK_N, num_warps = 128, 4`, commented "Wider
   tiles measured faster for the byte-decode path on A100"
   (`vllm/models/qwen4_exp/nvidia/ops/qsa.py:629-633`). That the override was measured
   on A100 and never on this card is itself a loose end.

### Corrected A/B at the real fp8 config, the number to quote (2026-10-24)

Three fixes at once, all from reading the served contract properly instead of assuming.
The 1M lane runs `--block-size 1616`, so the kernel page is 1616 tokens and not 64. The
packed selection buffer is `indexer_budget + compress_ratio - 1 = 2051` wide plus the
count column, not 2048. And on sm_80 with an fp8 cache the served kernel overrides the
GB300 table with `BLOCK_N=128` and `splits=16`. Arm A's cache is now a view of the
unified `[num_blocks, num_kv_heads, 1616, 512]` buffer split the way
`nvidia/qsa.py:145` splits it, so a token's two KV heads sit 827,392 bytes apart instead
of adjacent. `logs/pth-ab-fp8-unified-118000.txt`, salt 118000.

| quantity | value |
|---|---|
| random gather ceiling, 10255 rows of 512 B | 348 GB/s start, 354 end |
| our served kernel, shipped config | 104.8 us, 100 GB/s, 28% |
| their kernel at 16 splits, the same count | 34.8 us, 302 GB/s, 86% |
| their kernel at its best, 4 splits | 33.0 us, 318 GB/s, 90% |
| paired ratio at 16 splits, four blocks | 3.08, spread 0.076 |

The headline moves from 1.61x to about 3.1x, which is what both the matched pairing at
the config ours really uses (3.08) and each arm at its own best (3.18) give. The value
follows: 3 full-attention layers per rank, 4 ranks serial, 71.8 us per layer, 862 us of
the 22.79 ms step, so 3.8% rather than the 2.3% the earlier pairing implied.

The production cache strides turned out not to matter. Two 256 B runs per token instead
of one 512 B run moved arm A by about 1 us, well inside the spread, so our kernel sits at
28% of the gather ceiling under either layout.

### Sweep of our own kernel, the negative result (2026-10-24)

The shipped decode config on the served tree is `BLOCK_N=128`, `warps=4`, `tiles=16`,
`splits=16`, because the fp8 branch overrides the GB300 table on sm_80. The A/B above
called `_select_config` without `is_fp8`, which returns `BLOCK_N=32` and `splits=64`, so
arm A was timed at a config the server does not use. The sweep makes that mistake
harmless, the whole grid is flat to within 1%, but the label was wrong and is corrected
here.

| BLOCK_N | splits 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| 16 | 109.0 | 111.7 | 108.2 | 108.1 | 107.8 | 212.0 |
| 32 | 110.7 | 108.9 | 107.6 | 107.2 | 106.9 | 107.1 |
| 64 | 111.9 | 108.9 | 108.7 | 107.1 | 109.3 | 108.0 |
| 128 shipped | 107.0 | 106.0 | 109.0 | 108.3 | 108.0 | 108.2 |
| 256 | 109.3 | 107.1 | 106.6 | 105.6 | 106.9 | 105.8 |

Microseconds per layer-step, 10.0 MiB moved, gather ceiling 291 GB/s measured in the
same run, so the best cell still reaches only 34% of it. The `bn=16` cell at 128 splits
is the one clear failure at 212 us.

## QSA decode cost is not bytes, and no launch knob moves it (2026-10-24)

`benchmark/qsa_decode_dtype.py` runs the served kernel on one selection, one stride
pattern and one page size, twice: once with a bf16 cache and once with the uint8 e4m3
cache the lane actually uses. `logs/qsa-dtype-119000.txt`, salt 119000.

| arm | bytes per layer-step | time | GB/s | its own gather ceiling |
|---|---|---|---|---|
| e4m3 in uint8 | 10.0 MiB | 105.0 us | 100 | 236 GB/s, 42% |
| bf16 | 20.0 MiB | 104.3 us | 201 | 342 GB/s, 59% |
| ratio | 2.00x | 0.99x | | |

Twice the bytes, the same time, to within 1%. The kernel is not bandwidth-bound, so the
"28% of the gather ceiling" framing was the wrong diagnostic and the ceiling itself was
too generous, because a 512 B random gather only reaches 236 GB/s here rather than the
348 GB/s measured with 1024 B runs. What the time tracks is elements touched, which is
identical in both arms.

`benchmark/qsa_decode_tune.py --stages` then tests whether it is a pipeline depth
problem, which the dependent index, page, then K address chain in the inner loop would
suggest. `logs/qsa-stages-119100.txt`. Stages 1, 2 and 3 give 111.6, 114.8 and 111.3 us.
Stage 4 and above are rejected with `OutOfResources`. So the kernel is already at the
shared memory edge, deeper pipelining is not available to it, and depth is not the lever.

Combined with the tile sweep, three independent launch knobs are flat: `BLOCK_N` 16 to
256, splits 4 to 128, stages 1 to 3, all inside 105.6 to 118.1 us. The work per
layer-step is 2 rows of arithmetic, 5 rows x 24 heads x 2051 tokens x 256 dims x 2
tensors, which is 2.5 GFLOP, so the kernel sustains about 24 TFLOPS against the 182
TFLOPS bf16 peak measured on this die. It is 7.6x off its own compute ceiling and no
launch configuration gets it there.

Their kernel does the same element count in 33.0 us, about 76 TFLOPS, using int8 tensor
cores for the QK product and folding the value scale into the probabilities so V is
never dequantized. That is the mechanism, and it has a consequence for the plan: int8
MMA needs an int8 cache, so the cache format change is on the critical path for any
route to this win, including rewriting our own Triton kernel to do the same thing.

### CUDA graph capture holds on sm_80

The lane runs `cudagraph_mode=FULL_AND_PIECEWISE`, so every decode step is a graph
replay and a kernel that allocates or reads a host scalar at capture time cannot serve.
Their docs claim graph safety because vLLM adaptive decode depth depends on it, so it is
now tested rather than trusted. `benchmark/pth_attn_gate.py` captures the op, replays it
five times after mutating the query buffer in place, and compares against an eager call:
bit identical, error 0.0000. All 9 gate cases pass.

## Where the QSA decode time actually goes, read from the generated code (2026-10-24)

The dtype A/B said the e4m3 decode costs nothing, which was surprising, so I read the
kernel source and the emitted PTX instead of speculating. `/tmp/qsa-asm/`, compiled from
`wt-p13` at the decode shape.

The decode is already hand-optimized. `vllm/models/qwen4_exp/nvidia/ops/qsa.py:40-53`
is inline PTX that expands four e4m3 bytes to fp16 with one `shf.r.wrap.b32`, one
`lop3.b32`, two `prmt.b32` and two `and.b32`, so about six instructions per four
elements. It exists because sm_80 has no hardware fp8 convert at all, which is also
why the cache is `torch.uint8` holding e4m3 bits rather than a real fp8 tensor, and why
the earlier ceiling probe found no fp8 or nvfp4 formats on this card. There is nothing
left to optimize there, and the A/B already proved it was not the constraint.

Instruction mix in the compiled decode variant, one fully unrolled body, zero loop
labels:

| instruction class | count | reading |
|---|---|---|
| `mma.sync.m16n8k16.f32.bf16.bf16` | 64 | QK product, tensor cores already in use |
| `mma.sync.m16n8k16.f32.f16.f16` | 64 | P times V, in fp16, a different dtype from QK |
| `ld.shared` | 370 | dominates global loads, 10 of them |
| `prmt` / `and.b32` / `cvt.` | 448 / 360 / 430 | the byte decode plus f32 to f16 to bf16 casts |
| `cp.async` | 102 | async copies present |
| `ex2` | 18 | softmax |
| shared memory per CTA | 77,824 B | at most 2 CTAs per SM on this die |

Two things follow. First, our kernel is already using Ampere tensor cores and
`cp.async`, so the gap to their 3.1x is not "missing hardware instructions". Second, at
2.5 GFLOP per layer-step in 105 us we sustain about 24 TFLOPS against a measured 182
TFLOPS bf16 peak, which is 13% of peak, while their kernel at 33 us sustains about 76
TFLOPS against a measured 334 TFLOPS int8 peak, which is 23% of peak.

So both kernels are latency and structure bound rather than at any roofline, and the
decomposition of their 3.1x is roughly half from int8 having 1.84x more compute peak on
this die and half from a better structured inner loop. That matters for the plan in a
way I had backwards earlier in this session. Route 1, matching their speed in our Triton
kernel without touching the cache format, must recover the whole gap as structure,
because bf16 tops out at 182 TFLOPS and no one reaches that on a per-token gather at
decode batch sizes. Route 2 gets 1.84x of it from the format change on day one and then
inherits a kernel whose structure is already measured better.

I corrected this claim twice in one session, so stating the final position plainly: the
e4m3 software decode is not the bottleneck, int8 tensor-core peak is a real and
immediately available advantage, and no launch configuration on the current kernel
closes any material part of the gap.

`ncu` is available in this container, Nsight Compute 2026.3.1, and would give achieved
occupancy, stall reasons and tensor pipe utilization. It locks clocks and replays
kernels, so it cannot run beside the serving stack and belongs in the paused window.

## Eager timings measured the Python wrapper, earlier numbers here are void (2026-10-24)

I fell into the trap recorded in this file twice already, eager timing inverting
decode-path rankings, and I only caught it because a scaling test came out impossible:
growing the selection from 512 to 4102 tokens, eight times the work, made the kernel
faster, 166.1 us down to 112.9 us. Growing rows from 1 to 40, forty times the work,
cost only 1.45x. No kernel does that. A Python wrapper does that, when its fixed cost
dominates everything else.

`benchmark/pth_vs_qsa_fp8.py` and its friends measured `qsa_sparse_paged_attention()`
called eagerly in a loop. That call allocates the split-K partials, `torch.empty` of
fp32 `[S, T, Hq, D]`, runs `_select_config`, does the shape validation and launches two
kernels, every iteration. Measured directly:

| arm | eager | graph replay | difference |
|---|---|---|---|
| ours, `qsa_sparse_paged_attention` | 166.2 us | 24.6 us | 141.6 us of CPU |
| theirs, `pth_flash_attn_fwd` | 38.1 us | 37.9 us | 0.2 us |

The lane runs `-cc.cudagraph_mode=FULL_AND_PIECEWISE` and captures decode at sizes
`[1, 2, 4, 5, 8, 10, 16, 20, 24, 32, 40, ...]`, so at replay time that 141.6 us of CPU
does not exist. Every number in the three preceding sections that came from arm A is
therefore void, including "our kernel sustains 24 TFLOPS", "28% of the gather ceiling",
and "the 30-cell sweep is flat so no configuration buys anything". All three were
measurements of CPython.

### Retimed inside CUDA graphs, both arms at their own best configuration

Same shapes as before, 5 rows, 2051 tokens, page 1616, unified e4m3 cache for arm A and
int8 with `BS=1` per-token pages for arm B, 10.0 MiB per layer-step, 400 replays per
sample, five samples, median.

| arm | config | graph replay |
|---|---|---|
| ours | shipped `(BLOCK_N=128, warps=4, tiles=17, splits=17)` | 24.6 us at splits 16 |
| ours | splits 8 / 32 / 64 | 26.9 / 31.2 / 58.6 us |
| theirs | best, `S=4, wm=1` | 37.9 us |
| theirs | `S=2` / `S=8` / `S=16` | 61.8 / 41.2 / 40.5 us |
| theirs | `S=1` | 113.0 us |

Ours is 1.54x faster than theirs, not 3.1x slower. Under graph timing our kernel is
also strongly config-sensitive after all, 24.6 us at 16 splits against 58.6 at 64, which
is the opposite of what the eager sweep appeared to show. The shipped table already picks
17 splits, so there is no config prize here either way, but for the right reason.

One caveat in their favour, and it is a real one. `BS=1` is the only page size that
expresses a per-token sparse selection, and their loader is built for whole pages, one
`cp.async` row per token instead of one per 64. Their published wins, 2.56x against
Triton `unified_attention`, are for dense and sliding-window attention at `BS=864`, which
is their design centre and is not what we would be asking of it. What this test says is
narrower and still decisive: as a sparse-selection kernel on this shape, it loses to ours.

### What the decode step actually spends on attention

At 24.6 us per layer-step and 3 full-attention layers per rank, the sparse KV read is
about 74 us of a rank-step and about 295 us of the 22.79 ms step, 1.3%. Making it
free would buy 1.3%. That agrees with the profiler, which is a GPU-side measurement and
was never affected by this error: the `qsa-attn` bucket is 0.43 ms per rank-step on ranks
0 to 2, and the kernel names inside it are dominated by `_qsa_mqa_paged_uniform_kernel`
at 327 us and `persistent_topk_kernel` at 315 us per rank-step on rank 3, both of which
are selection, not reading.

So the target in the QSA path is the indexer scan and the top-k pick, roughly 1.4 ms of
the step across four ranks, and unlike the read it grows with context depth. That is
where step 1 of this investigation should have been pointed from the start, and the
lesson is the one already written down twice in this file: on the decode path, time the
graph or do not time at all.

### What survives from the investigation

- Their extension builds for sm_80 with no source edits and runs correct on this card.
  That is genuinely new, their own docs say sm80 was never executed.
- `BS=1` makes their block table a per-token list, so their kernel *can* express our
  selection semantically. Verified by test, with deliberately scattered physical pages.
- Their op captures into a CUDA graph and replays bit-identically, which was the hard
  requirement and is now known to hold.
- The int8 `int8_per_token_head` cache mode already exists in this tree, including the
  store kernel, and our FP8 checkpoint ships no KV scale tensors, so the served e4m3
  cache runs with per-tensor scale 1.0, uncalibrated. That is an accuracy argument for
  per-token-head scales, independent of speed, and it is the only reason left to change
  the format.


## The controlled endpoint profile moves the read figure (2026-10-24)

The live lane answers `POST /start_profile` and `POST /stop_profile`, and its
`torch_profiler_dir` is inside this repo, because the host path `/ssd/ai/workspace` is
this container's `/home/gab/test`. A GPU-side trace of the serving stack therefore needs
no reboot. Read `logs/prof`, 23:39, 36 real decode steps per rank at a 45.2 ms period.

That trace says my "1.3%" for the sparse read was too small. My synthetic shape used 5
query rows, while the lane runs decode batches up to `max_num_seqs` 8 times 5 MTP rows.
Measured on the endpoint:

| kernel | us per rank-step | launches per step |
|---|---|---|
| `_qsa_sparse_paged_gqa_splitk_kernel`, the read | 387 | 3.2 |
| `_qsa_mqa_paged_prefill_kernel`, the scan | 279 | 0.3 |
| `persistent_topk_kernel<512, 4u>` | 173 | 3.2 |
| `qsa-attn` category total, rank 0 | 960 | |

The read is about 387 us of a rank-step, roughly 5x what the 5-row microbenchmark
implied, and across four serial ranks about 1.55 ms of the 45.2 ms step, 3.4%. Selection
is about the same size again, and it is the part that grows with context depth while the
read does not.

Two method notes. The 24.6
us graph-replay number is not wrong, it is a 5-row shape, and the lane is not running 5
rows, so a microbenchmark has to be sized to production before its ratio is quoted even
when it is timed correctly. And the ordering did not change: our kernel is still faster
than the third-party one at equal work, so the read prize is available only by improving
our own kernel, not by swapping it.

The cheapest untested lever is therefore on the selection side. `serve-4x-1m.sh` now
takes `IKD=fp8`, which passes `--attention-config '{"indexer_kv_dtype":"fp8"}'`. The QSA
indexer accepts `bf16` or `fp8` for the compressed key cache
(`nvidia/indexer_qsa.py:152-163`) and the lane leaves it at `auto`, which resolves to
bf16, so those rows are 128 dimensions per group either wide or narrow. fp8 halves the
bytes the scan streams. It changes which blocks get selected, so it is certifiable with
the fixed-salt accept window rather than on speed alone, and the pool line should not
move, because the index cache is not the KV pool.


## `IKD=fp8` cannot boot on sm_80, and the read is probably already at the wall (2026-10-24)

`boot-4x-1m-ikd.log`. The flag reached the engine, the config echo carries
`indexer_kv_dtype='fp8'`, and then the first forward died during
`profile_cudagraph_memory`, before the pool was sized:

```
File "vllm/models/qwen4_exp/nvidia/ops/qsa.py", line 980, in qsa_store_cache_rows
    _store_qsa_rows_kernel[(rows.shape[0],)](
triton.compiler.errors.CompilationError: at 1:0:
def _store_qsa_rows_kernel(
ValueError("type fp8e4nv not supported in this architecture.
            The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

Same Ampere limitation that stopped my own harness earlier today: sm_80 Triton cannot
compile the e4m3 type. `nvidia/indexer_qsa.py:152-163` maps `"fp8"` to
`torch.float8_e4m3fn` and hands that tensor to the store and logits kernels, so nothing
can run. The main cache avoids this by storing e4m3 bits in a `torch.uint8` tensor
and decoding bytes in-kernel, and the indexer side cache has no such path. The comment at
`indexer_qsa.py:150-151`, "the logits kernels dot fp8 x fp8 directly", is a newer-arch
assumption. Do not retry this flag without code.

Fixing it means mirroring the byte trick into the side cache, which on Ampere buys only
the bandwidth, because there is no fp8 MMA. The knob stays in the script as
`bf16|fp8` with a validator, since the intent is sound on sm_89 and above.

Also removed `FANOUT=` from the script. The probe is not in `patches/`, it is
`experiments/0015-moe-fanout-probe.patch` on `lab/moe-fanout-probe`, so on the shipped
tree the knob only exported an environment variable that nothing reads, which implies a
capability the lane does not have. The measurement it produced stands on its own.

### One more method note, because this one nearly cost me the conclusion

`benchmark/phase_report.py --dir logs/prof` globs `*rank*.pt.trace.json*`, and that
directory holds traces from four separate boots on three different trees, Oct 13, Oct 16,
Oct 24 00:25 and Oct 24 23:38, plus both stale `profiler_out_N.txt` and fresh
`dp0_pp*_rank*.profiler_out.txt`. Grepping the wrong one reported 25.1 us per read launch
against 119.1 us from the current window, a 5x error with no warning. Filter by timestamp
before quoting anything from there.

From the fresh window only, rank 0, 36 real decode steps:

| kernel | us per call | us per rank-step |
|---|---|---|
| `_qsa_sparse_paged_gqa_splitk_kernel`, the read | 119.1 | 387 |
| `persistent_topk_kernel<512, 4u>`, the select | 54.5 | 173 |
| `_qsa_mqa_paged_uniform_kernel`, the decode scan | 21.1 | 63 |
| `_qsa_mqa_paged_prefill_kernel`, arriving prefill | 835.8 | 279 |

At the 40-row ceiling a read call moves about 84 MB, which is 119.1 us, so about
79% of the 895 GB/s this card sustains in production. If the decode batch during that
window really was near 40 rows, the read is bandwidth-bound in situ and this entire axis
is closed, by config, by int8, and by any third-party kernel. The trace does not record
shapes, so the row count is not in evidence, and at 8 rows the same arithmetic gives
141 GB/s and the conclusion inverts. Settle it with `vllm:num_requests_running` around the
next capture before treating it as finished.

The number that does not fit is `persistent_topk` at 54.5 us per call, 2.6x the decode
scan that feeds it. Selecting 512 groups should not cost that. That is the next thing.


## Controlled endpoint profile: the read is fine, selection is 3.8x it (2026-10-24 23:55)

`phase_probe.py --salt 121000 --decode 2048`, one stream at 100% fill, 987,972 prompt
tokens, TTFT 85.4 s, acceptance length 4.95, match 100%, zero preemptions, 209 tok/s
aggregate. The profiler armed during steady decode and self-closed at 150 engine steps,
111 kernel calls per rank, which is 37 real decode steps and 3 full-attention layers per
rank, so every count below is one launch per layer per step on the serving lane.

| kernel | us per call | us per rank-step | role |
|---|---|---|---|
| `persistent_topk_kernel<512, 4u>` | 56.9 | 171 | select |
| `_qsa_mqa_paged_uniform_kernel` | 48.2 | 145 | score the prefix |
| `_qsa_sparse_paged_gqa_splitk_kernel` | 22.4 | 67 | read |
| `_qsa_merge_splitk_kernel` | 5.2 | 16 | read combine |

The first finding retracts a number published earlier in this file.

The 387 us per rank-step figure for the read is void. It came from a directory holding
traces from four boots and from a window that contained arriving prefill work, where
`_qsa_mqa_paged_prefill_kernel` averages 836 us over 12 calls and lands in the same
category bucket. Under a controlled single-stream load the read is 22.4 us per call, which
matches my synthetic CUDA-graph measurement of 24.6 us at 5 rows to within 10%. The
microbenchmark was right, the endpoint number was wrong, and the 3.4% step share it implied
is really about 1.2%, read plus combine, 67 plus 16 us per rank-step.

Selection is the target, at 3.8x the read. Score and select cost 105 us per call
against 27.6 us for reading and combining, so 316 us of a rank-step against 83 us. Across
four serial ranks that is about 1.26 ms of the 22.79 ms step, 5.5%, against 1.5% for the
read. `persistent_topk` is now the single most expensive kernel in the QSA path, and this
was measured at the worst case for depth, a 988k context, which is the one axis where the
scan and select grow while the read stays fixed at 2051 tokens.

Two things to hold loosely. The arithmetic does not close: 5 query rows scoring 247k
compressed groups at 128 bf16 dimensions would be 316 MB in 48 us, about 6.6 TB/s, which
this card cannot do, so either the rows share a pass or the logits are not recomputed from
scratch every step, and I have not established which. And a single stream at 988k is the
deep end of the distribution, so the selection share will not look like this on shallow
traffic.

Next question, in priority order: what `persistent_topk<512, 4u>` actually costs per
element and why it is slower than the scan that feeds it, and whether the index cache dtype
knob matters at all given the scan appears to be doing far more work per byte than a copy
would allow.


## Two corrections about the pipeline bubble (2026-10-25)

A static read of the p13 trace and of `vllm/distributed/` corrected two claims of mine,
one in this file and one in the tool.

1. `generation_0(0)` spans are not draft waves. The annotation is
`execute_context_{num_ctx_requests}({ctx_tokens})_generation_{num_gen_requests}({gen_tokens})`
(`vllm/v1/worker/gpu_worker.py:1298-1315`), so `generation_0(0)` is a step with zero
scheduled requests. All 113 of them in that capture launch zero kernels, which is why they
looked like empty spans. The real draft waves are the extra CUDA graph replays on rank 3
inside `sample_tokens`, which is un-annotated: rank 3 replays 5 graphs per step against 1
for ranks 0 to 2. `phase_report.py:6-7` states the wrong interpretation and this file
repeated it in an earlier section.

2. The read bandwidth denominator was wrong. I have been comparing QSA kernels against
895 GB/s all day, which is a copy rate, read plus write. This card is GA100 with 4096
bit HBM2e at 2.9 Gbps, 1.49 TB/s theoretical, and a pure streaming read sustains roughly
1.33 TB/s. That reframes every roofline in the sections above, and in one case it reverses
the conclusion: `_qsa_mqa_paged_uniform_kernel` is not inefficient, it reads 63.2 MB of bf16
index cache in 48.2 us, which is 1.33 TB/s, about 89% of the card's peak read rate. It
is at the roofline and there is nothing algorithmic left in it.

### What `nccl-spin` actually is

One real NCCL device kernel, `ncclDevKernel_Broadcast_RING_LL`, 1 block of 128 threads,
carrying 68 bytes of metadata, the sampled token, accept counts and 4 draft tokens, from the
last pipeline stage to the other three. Not an event wait, not double counted. Ranks 0 to 2
run it as consumers and it stays resident until stage 3 has sampled and drafted, so it
absorbs every millisecond the loop loses anywhere, including host-side stalls. Rank 3 is the
producer and its copy costs 9 us, which is why the spin staircase r0 31.31, r1 27.52, r2
18.01, r3 3.24 ms is one stage of compute wide, and why the `true gap` column is its exact
inverse, a rank that is idle *inside* a spin kernel reads as busy.

It costs essentially no GPU resource, 4 warps on 1 of 74 SMs at zero achieved occupancy, and
it does not even overlap the useful kernels on its own device, `union_nccl` plus
`union_comp` equals `union_all` on rank 0, so the report columns legitimately sum to wall
time. Deleting the kernel would not shorten a step by one microsecond, because the time is a
dependency rather than work.

The part that is not structural: kernel compute is flat at about 26 ms per cycle while the
period grew from 26.7 ms in the best-packed cycles to 41 to 44 ms in steady state, so roughly
14 ms per step is host time that lands in the spin column. On rank 0, once per real step,
`cudaStreamSynchronize` blocks the worker's only CPU thread for 30.5 to 34.9 ms, then about
2.6 ms of per-request Python bookkeeping runs, then a 0.78 ms `cudaGraphLaunch` before the
next forward is even queued.

### Two tooling defects worth fixing

`phase_report.py:145-158` divides whole-capture GPU time by the count of real steps. In the
45 ms capture that produces spin 47.02 ms per step against a 45.23 ms period, and 55.92 ms of
total GPU work per step, which is arithmetically impossible inside a steady cycle. Time
recorded after the last counted step is charged to the counted steps. It needs per-window
attribution, clipping each event to `[step_i, step_i+1]`.

The same file puts every kernel whose name contains `nccl` into one bucket with no duration
filter (`:147-149`), which mixes a 3.8 ms receive wait with a 23 us payload. Splitting the
column into `nccl-wait` above 2 ms and `nccl-xfer` turns 80.09 ms into 79.9 wait plus 0.2
transfer and makes the reading obvious. A third line is worth adding, `period minus the sum of
compute across ranks`, which is the time when no GPU on the box is computing, 18.0 ms per
step in steady cycles against 3.8 ms in the best-packed ones.

Note: `nvidia-smi` GPU-Util answers only "was any kernel
resident during the sample", so a 4-warp spin scores the same as a full-width GEMM. On this
box utilization, memory-used and power draw are not load signals. `/metrics` counters, step
and ITL histograms, and per-kernel durations from a trace are.


## Why the select kernel is slow, and why no cap fixes it (2026-10-25)

`tools/qsa_topk_geometry.py`, in the `wt-int8kv` worktree, reproduces the host-side launch
geometry of `persistent_topk_kernel` from `csrc/topk.cu`, pure Python, no GPU. It is the
first tool in this set that predicts a shape I can check against a measurement, and the
measurement is the 988k single-stream profile, 56.9 us per call.

The important numbers are at the depth the lane actually runs, not at the 247k the first
analysis used. Rows 5, `seq_len` 988000, `stride` 1000304:

| smem cap | CTAs per group | groups | row iterations | barrier round-trips | launch smem | occupancy |
|---|---|---|---|---|---|---|
| 48 KiB, shipped | 86 | 1 | 5 | 30 | 48,608 B | 2 |
| 82 KiB | 49 | 1 | 5 | 30 | 83,744 B | 2 |
| 164 KiB | about 17 | 2 | 3 | 18 | 162,144 B | 1 |

At the shipped cap the whole 148-CTA budget goes to a single group, so the five MTP rows
are selected one at a time, five sequential passes, each paying about six global-memory
spin barriers, thirty in total. At 4.95 MB moved that is 87 GB/s, roughly 17x off the
1.49 TB/s this card can read at, so the time is barrier latency plus seven shared-memory
rescans per element and not data movement.

This kills the cheap fix. The 82 KiB cap that looked strictly better at 247k, halving
iterations at unchanged occupancy, does nothing at depth, same five iterations and thirty
barriers, and it shrinks the group from 86 CTAs to 49, so it is worse. The 164 KiB cap does
reduce barriers to 18 but pays for it by halving occupancy to one CTA per SM. Trading
barrier count against parallelism at a fixed small batch is a coin flip, not a win, so I am
not spending a build on the cap. The experiment patch stays at
`wt-int8kv/qsa-topk-smem-experiment.patch`, gated by `VLLM_QSA_TOPK_SMEM_CAP_KIB`, default
off, for whoever wants the coin flip measured.

The real shape of the problem is that this kernel is built for many short rows and this
workload is a few very long ones. The kernel designed for exactly this case,
`cooperative_topk`, uses TMA cluster loads and a 1024-bin histogram through distributed
shared memory with a hardware `cluster.sync()` instead of global spin barriers, and it needs
sm_90. It cannot run on GA100. A proper fix is a two-phase sampled-then-filtered select for
small row counts, which is a new kernel, about 1.5% of the single-stream step at best.

### A separate real bug, and it costs the short-context case

`vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:504` passes `logits.shape[1]` as
`max_seq_len`. That value is the padded row width, page table columns times page size, so
1000304 on this lane, while the contract at `vllm/model_executor/layers/indexer_topk.py:334-338`
says the argument is the host-side maximum real sequence length across rows. Consequence:
`needs_cooperative` tests `max_seq_len > 32768` and is therefore always true at decode,
even for a 200 token prompt, which selects the multi-CTA radix path with its headroom
penalty and per-call workspace bookkeeping where the single-CTA path would apply.

Not a correctness bug, the kernel clamps reads with `min(stride, max_seq_len)`, and not a
cost at the deep end, where the long row genuinely needs the wide path. It costs the shallow
requests, which is most traffic on a lane capable of 1M. Correct fix is to thread
`compressed_metadata.max_seq_len` into the decode call the way it already is for prefill at
`indexer_qsa.py:471`, with one trap: `_run_qsa` is `@eager_break_during_capture`, and a
per-step tight value may be frozen from capture, so the zero-risk form separates the two
roles, keeping the clamp at `stride` and passing the tight value only as the host hint.

### Remaining cost, ranked

| target | per step | as share of a 45 ms single-stream step |
|---|---|---|
| host-side bubble, `cudaStreamSynchronize` plus bookkeeping plus launch | about 14 ms | 31% |
| MoE and GEMM compute on the critical path, rank 3 heaviest | about 26 ms of GPU work | the serial floor |
| QSA select | 683 us | 1.5% |
| QSA index cache bytes, bf16 to a narrower format | about 290 us | 0.6% |
| QSA sparse read | 269 us | 0.6% |

The QSA path is now fully accounted for and every item in it is small. The largest single
number on this box is not a kernel at all, it is the 14 ms per step the pipeline loses to
the host on the rank-0 handoff, which is consistent with the older finding that four
concurrent streams raise aggregate throughput far more than any kernel change has.

## fp8 KV priced on the shipped tree, 10% prefill for 1.92x pool, decode untouched (2026-10-25)

The README throughput table lists 12.4k prefill on a bf16-KV row and 11.6k on the shipped
fp8 row, which reads as though the lane lost prefill and never got it back. The last
bf16-versus-fp8 A/B ran on the 2026-10-15 and 2026-10-16 bases, before patches 11, 12 and
13, so the tax had only ever been priced against kernels that no longer exist. Measured
again today as a single variable.

Both arms ran the 13-patch series with no diagnostic build in either boot log. Boot A
`./serve-4x-1m.sh --kv-cache-dtype auto`, pool 2,580,745 tokens. Boot B `./serve-4x-1m.sh`,
fp8, pool 4,961,194. Same pinned salts on both, 130100, 130200 and 130300, so the prompt
bytes are identical, and the same cell shape, `bench.py --ctx 1000000 --fills 100
--concurrency 1 --decode 1024 --cooldown 60`, first cell of each boot preceded by
`--warmup`. Per-cell dumps stay local.

| salt | prompt tok | TTFT s, bf16 | TTFT s, fp8 | prefill tok/s, bf16 | prefill tok/s, fp8 | win avg, bf16 | win avg, fp8 | sustained, bf16 | sustained, fp8 |
|---|---|---|---|---|---|---|---|---|---|
| 130100 | 988,214 | 76.4 | 85.5 | 12,138 | 10,902 | 204.9 | 198.4 | 212 | 212 |
| 130200 | 988,845 | 76.4 | 85.5 | 12,131 | 10,913 | 201.4 | 199.2 | 212 | 212 |
| 130300 | 989,439 | 76.3 | 85.4 | 12,141 | 10,931 | 197.9 | 198.7 | 215 | 209 |
| mean | | 76.37 | 85.47 | 12,137 | 10,915 | 201.4 | 198.8 | 213.0 | 211.0 |

Means move 11.9% on TTFT, 11.2% on wall and 10.1% on prefill rate. Decode does not move,
1.3% on window average and 0.9% on the sustained peak, inside the spread of the three salts
themselves. Acceptance is if anything better on fp8, 4.783 against 4.837, so the quantized
cache is not costing output quality on this task. The README note that fp8 KV does not touch
decode is therefore confirmed on current kernels rather than inherited.

Two things fall out that the README could not show.

The remembered 12.4k was never the ceiling of bf16 on this tree. Recomputing the README's own
formula, prompt tokens divided by cold TTFT, boot A gives 12,935 to 12,968, so bf16 today is
about 12.9k and the October trial boot reached only 12.4k because patches 11 and 12 had not
landed. Boot B computes to 11,558 to 11,586, which reproduces the shipped 11.6k to within
0.1% and cross-validates both the method and the table.

The prefill column is not one metric. `in tok/s` from `bench.py` is prompt tokens divided by
wall, which is 12,137 on boot A, while the table column is prompt divided by TTFT, which is
12,943 on the same cell. A 6.7% difference between two rows can therefore be a formula change
rather than a measurement, and the table needs to say which divisor each row uses.

So the answer to why 12.4k and 205-215 were never reached again is that 205-215 is reached
again in every one of these cells, and 12.4k was replaced by a deliberate purchase, about
10% of cold prefill paid for 1.92x KV pool, 2,580,745 to 4,961,194 tokens, which is what
holds four concurrent 1M contexts resident instead of two or three.

## What the fp8 KV tax is not (2026-10-25)

The tax is measured and prefill-only, 76.37 to 85.47 s on a cold 1M TTFT, 9.1 s, with
decode inside the salt spread. Four candidate mechanisms have now been eliminated, three of
them today.

| candidate | how it was ruled out |
|---|---|
| attention read slower on fp8 | measured, fp8 is 4.8 % **faster** at the prefill dispatch under CUDA graph timing, and 1 % faster at decode |
| the fp8-only tile override | it earns 23 % of itself, removing it costs time, measured |
| cache write | 0.14 % of device time in the only capture that measured it |
| GDN and linear-attention state caches | state dtype comes from `mamba_cache_dtype`, not `kv_cache_dtype`, `mamba_mixer2.py:1123-1126`, so fp8 KV never touched the 36 linear layers |

The one dtype-dependent difference still standing is the loss of `use_fused_qsa_prepare`,
`nvidia/qsa.py:500-507`, which is refused when the cache storage is `torch.uint8` because
Triton cannot lower an fp8 pointer below sm_89. That turns one launch into about thirteen
per QSA layer per step, the writer plus the whole indexer prepare, some 36 extra launches
per step per rank and roughly 17,400 over a 1M prompt. Arithmetic caps it at 0.1 to 0.5 % of
the wall, so it is a contributor and not the mechanism.

Two corrections to the record while this is open. `serve-4x-1m.sh:12-14` says fp8 reads cost
about 12 s and the int8 handoff gives back 13 s "so cold prefill wall stays where bf16-KV
was". Those are unrelated levers, the hop quantizes pipeline activations and touches no KV
file, and the 2x2 factorial at `results.md:1341-1346` shows it, hop moves 11.9 s and 11.8 s
independent of dtype, fp8 costs 12.0 s and 12.1 s independent of the hop. Also stale, the
same differential measured today is 9.1 s, not 12 s.

Method note, the hard way. The two-sided profiler comparison that would settle this crashed
the lane on 2026-10-25 at 13:24. The engine's `max_iterations: 150` budget is calibrated for
**decode** windows, a few hundred small kernels per step, and the window was armed inside a
1M **prefill** where every step is 2,048 tokens through CUDA graphs. The cell's TTFT went
from an expected 76 s to 338 s before the EngineCore raised `TimeoutError` waiting on its
workers. The pre-flight also passed while the boot log was already reporting
`expandable_segments: memory mapping failed with OOM ... free: 16056320`, because it checked
request count and config and not device headroom. Do not arm a profiler window on this box
during a prefill, and check free device memory before touching the lane at all.

## The int8 lane booted, and it killed the dtype-hack theory (2026-10-25)

Boot `823f9cfe4` (dev407), the shipped 13-patch tip plus the `experiments/0016` replay
(tree-identical to `lab/qsa-int8-kv` at `b57a32fd3b`), `--kv-cache-dtype
int8_per_token_head`, block 1616, everything else shipped defaults. Clean log: no errors,
capture normal, pool 4,952,238 tokens / 4.95x, `kv cache group sizes (3232, 3232, 3232,
3232, 8, 3232)`.

Cells at the A/B shape `bench.py --ctx 1000000 --fills 100 --concurrency 1 --decode 1024
--cooldown 60`. Salt 130100 discarded (`bg 83`, competing requests landed mid-cell). The
sweep was stopped by the owner after the third cell; three clean cells are enough.

| salt | prompt tok | TTFT s | in tok/s | win avg | sust | acc len |
|---|---|---|---|---|---|---|
| 130200 | 988,845 | 135.5 | 7,032 | 201.3 | 212 | 4.86 |
| 130300 | 989,439 | 136.2 | 7,004 | 199.8 | 209 | 4.88 |
| 140100 | 988,600 | 134.4 | 7,076 | 192.9 | 209 | 4.65 |

Against the same-day fp8 and bf16 means: 85.47 s and 76.37 s TTFT, 10,915 and 12,137 tok/s
prefill, 211 and 213 sustained, acceptance 4.837 and 4.783, pool 4,961,194 and 2,580,745.

Four readings.

1. int8 prefill is not recovering the tax, it roughly doubles it: 135.4 s against 85.5 s on
   fp8 at the same one-byte footprint. The prior put 30% on int8 landing near 76 s. Wrong
   in the informative direction: the fp8 tax is not the `fp8e4nv` bit-trick decode and not
   the refused fused writer, because int8 has neither handicap and is 49 s slower.
2. Decode is unchanged and the numbers are healthy: sustained 209-212 against 211 fp8,
   acceptance 4.65-4.88 against 4.837 fp8, prefix match 100, zero preemptions. A short
   greedy code-generation smoke is coherent. The read path works at lane scale.
3. The pool prediction failed and explains itself: 520-byte cells against 512 were to cost
   1.5%, the boot lost 0.18%. The QSA group was never binding at fp8 widths, the pool sits
   on a dtype-independent group, so the fp8 boot already paid for its "1.92x" only against
   bf16. int8 versus fp8 is an equal-pool comparison, which makes the prefill gap a pure
   compute question.
4. That leaves two int8-only mechanisms for +49 s on ~988k prefill tokens: the fused writer
   computing an absmax and quantizing per (token, KV head) where fp8 did one cast with a
   host scale, and the read kernels loading scale views per token at prefill shapes
   (decode-shape reads measured within 1.23% of bf16, prefill shapes unmeasured). The
   discriminator is standalone kernel timing at the 2048-token chunk, writer first, read
   second, no lane involvement.

The fp8 9.1 s tax itself stays open, and its candidate list is now shorter: not the read
kernel, not the override, not the write cost, not GDN, not fp8-specific decode arithmetic,
not fused-prepare refusal, not extra pool.

## The int8 prefill gap, isolated by measurement (2026-10-25)

The int8 lane runs 1.77x on prefill (135 s against bf16 76 s) and not at all on decode. The
base is provably identical (`wt-int8kv HEAD~3` = wt-p13 = `dcc9c271d8`, tree `9007320d5d`), so
the gap is exactly the three port commits, not a confound. Two new tools price the two QSA
kernels the port touches, standalone on one idle card, no engine:

- `benchmark/qsa_read_prefill_ab.py`, real launcher, prefill shape rows 2048 depth 500k, both
  dtypes take the `is_fp8=False` config (BLOCK_N=32 warps=1 splits=1): **int8 read is 1.07x
  bf16 read**. A kernel 7% slow cannot spend 59 s.
- `benchmark/qsa_store_cost.py`, per-token-head int8 writer against the bf16 cast writer on the
  same (2048, 2, 256) grid: **int8 store is 0.88x** the bf16 store, it is faster because it
  writes half the bytes.

A trap worth recording: "decode is unchanged, so the store is fine" is false reasoning. Decode
writes one token a step, so any per-token store cost is invisible there; the store had to be
measured, and only then is it clear.

That leaves one untested suspect, which is the newest code: the **fused int8 writer** (the
`IS_INT8_PTH` branch in `qsa_prepare`, commit `b57a32fd3b`), a different kernel from the 0.88x
standalone writer, running absmax and quantize at `num_warps=1` inside the one fused launch.
The discriminator is prepared: branch `lab/qsa-int8-kv-nofuse` (`6409f37dfb`) re-adds the gate
clause `not kv_quant_mode.is_per_token_head` so int8 uses the 0.88x unfused writer instead, one
line, everything else identical. A prefill-only cell on that boot against the fused 135 s:
about 85 s means the fused int8 store is the gap and the fix is in its shape or reduction;
about 135 s means it is not the writer either, and the gap is not in any QSA kernel, which
would force a whole-step trace that the live lane cannot give safely.

## The fused writer is innocent too, so the gap is not in QSA compute (2026-10-25)

The last suspect was the fused int8 store (`qsa_prepare` `IS_INT8_PTH` branch, `num_warps=1`,
absmax+quantize per token,head). `benchmark/qsa_fused_store_cost.py` times that exact launch,
bf16 vs int8, same indexer work and same main geometry (24 Q / 2 KV heads, head_dim 256,
2048-token chunk): **int8 fused store 1.06x** the bf16 fused store, +0.0 s amortised over a
1M prefill.

So all three kernels the port can touch are measured at parity: fused store 1.06x, unfused
store 0.88x, sparse read 1.07x. Both dtypes make the identical number of kernel launches (one
fused prepare, one read; splits=1 for the prefill config in both). The base is byte-identical
and the comparison is single-variable on pinned salts. Yet the lane reads 76 s bf16 vs 135 s
int8, prefill only, decode identical.

A port that only touches QSA, whose QSA kernels are all at parity, whose launch structure is
identical, cannot by itself double prefill. The gap is therefore either contention that hit the
int8 boot's measurement window on the compute side (prefill is compute-bound, decode is
bandwidth-bound, so a co-resident process can slow one and not the other), or a whole-step
effect no single-kernel microbenchmark sees. The next instrument is not another kernel timer:
it is the fused-off boot discriminator already staged (`lab/qsa-int8-kv-nofuse`), which if it
still reads ~135 s would prove the writer is not it, and a whole-step nsys trace of one prefill
step on the lane. Recording this so the 2x is not chased in the kernels again.

## The int8 penalty is flat per-token, and no QSA kernel explains it (2026-10-25)

A prefill-only scaling cell on the idle int8 lane (fresh salt, `decode 1`):

| fill | tokens | TTFT s | in tok/s |
|---|---|---|---|
| 25% | 247,329 | 31.6 | 7,837 |
| 50% | 494,937 | 64.7 | 7,650 |
| 75% | 742,117 | 99.2 | 7,483 |
| 100% | 989,900 | 135.9 | 7,286 |

Throughput is ~flat at 7.3-7.8k tok/s and TTFT is near-linear in tokens (ratios 1.0, 2.05,
3.14, 4.30 against token ratios 1,2,3,4; the mild super-linearity is the depth term, ~7%
over 1M). bf16 holds 12,137 tok/s and fp8 10,915. So int8 carries a **constant per-token
prefill penalty of about 38%**, essentially independent of depth.

That signature has two consequences. First, it is per-token work, not a depth-growing read.
Second, it is invisible at decode, because decode pays it on 1-8 tokens inside a 22 ms step
(60-250 ns a rounding error) while prefill pays it on 2048 tokens per chunk. A per-token cost
that prefill shows and decode hides is exactly what a store looks like - but the stores do not
reproduce it:

- fused int8 store (`IS_INT8_PTH` branch, 2048 tokens): 1.06x
- unfused per-token-head int8 writer: 0.88x (faster, half the bytes)
- sparse read int8 at the real block_size 1616 geometry: 1.04x

So int8 prefill is 38% slower per token, and the three int8-specific kernels together account
for about 4-6%. This is a hard, measured contradiction, not a hand-wave: the port touches
nothing else, the base tree is byte-identical (`dcc9c271d8`), and the launch count is the same.
Therefore the 59 s is not attributable to the kernels I can reach, and the next instrument is
mandatory, not optional: a whole-step CUDA kernel trace of one int8 prefill versus one bf16
prefill on the identical tree, which is `benchmark/qsa_prefill_trace.py` and requires the
serving lane stopped (it wants all four cards). Until that trace names the kernel, tuning QSA
kernels is chasing a number that the measurements say is not there.

## The elephant: 520-byte cells killed Triton's vectorization, fixed at 544 (2026-10-25)

The offline whole-step traces (`benchmark/qsa_prefill_trace.py`, one 131k
prefill per dtype, same tree and env, host runs) end the hunt:

| kernel | int8 | bf16 |
|---|---|---|
| `_qsa_sparse_paged_gqa_splitk_kernel` | 20.73 s (234x ~81 ms) | 3.89 s (195x ~19 ms) |
| everything else (nccl, marlin, cutlass, GDN, prep) | ~same | ~same |

The read kernel alone is the entire gap. Standalone, on the same grid
`(6464,2,1)`: 520-byte cell int8 read 80.6 ms (4.92x bf16), 544-byte cell
17.7 ms (1.07x). The cause is the stride: a per-token-head cell packed as
K(256)+4B-scale-pad+V(256)+4B is 520 bytes, so the head stride mod 16 is 8,
Triton drops the divisibility specialization, and every gathered K/V column
loses vectorized loads. bf16 and fp8 cells are 512 (or 256-wide views with
512 stride), always aligned. Decode hides it because one query row costs
microseconds either way, which is why only prefill doubled.

Why my first read microbench missed it: it allocated the int8 cache as its
own contiguous tensor (stride 256, perfectly aligned) instead of views into
the real packed cell. A kernel that takes strided views must be timed with
the production strides.

The fix (lab commit `cd2d7eb305`): the scale slot widens 4 to 16 bytes
(`PTH_SCALE_SLOT_BYTES`, fp32 at the slot front, 12 never-read pad bytes
behind it), cell 544, half-width 272. Layout math now lives in one constant
(spec content, scale-view strides, fused writer V offset, prepare head
width, kernel warmup head width; the kernel takes it as `MAIN_PTH_SLOT`).
Fused-vs-unfused byte parity re-verified on the new layout, 11/11, with the
bf16 and uint8 lanes byte-identical. `experiments/0016` re-exported as the
4-commit lane, replay tree proven identical to the lab tip.

Expected from the rerun boot: cold 1M prefill back near the fp8 class (~85 s,
was 135), decode unchanged (~210), pool unmoved (the QSA group is not the
binding one, 4.6% more QSA bytes against a dtype-independent wall). If it
lands there, int8 is at fp8 speed with native dtypes, a fused writer fp8
cannot use, and the fp8 tax's original 9.1 s remains its own open question.

## The fix boots: int8 prefill 135 -> 89 s (2026-10-25, boot a7909d5fa)

Reinstalled the 4-commit lane, `--kv-cache-dtype int8_per_token_head`, clean log, pool
4,958,208 tokens / 4.96x (vs 520-byte 4,952,238: the wider cell costs 6k tokens, 0.1%, as
predicted, the QSA group is not pool-binding). Cells at the A/B shape, warmup then
`bench.py --ctx 1000000 --fills 100 --concurrency 1 --decode 1024 --cooldown 60`.

| salt | prompt tok | TTFT s | in tok/s | win avg | sust | acc len |
|---|---|---|---|---|---|---|
| 130100 | 988,214 | 88.8 | 10,538 | 204.8 | 215 | 4.86 |
| 130200 | 988,845 | 89.3 | 10,477 | 199.7 | 209 | 4.88 |
| 130300 | 989,439 | 88.9 | 10,477 | 48.4 | 119 | 1.14 |

The prefill gap is closed: 135 s on the 520-byte layout, 88.8-89.3 s now, against fp8 85.5 s
and bf16 76.4 s. The read kernel, not the int8 math, was the whole 46 s, and widening the cell
stride from 520 to 544 restored it. Cells 1 and 2 are clean: acceptance 4.86/4.88 (the fp8
lane's own 4.837), sustained 209/215, prefix match 100, zero preempts. So int8 is now within
3.5 s of fp8 prefill and gives up the 9.1 s fp8 tax almost entirely, the opposite of the 2x it
booted at.

Salt 130300 is a decode-side outlier, not a prefill or dtype problem: its TTFT (88.9) matches
the other two, but the decode window read 48 tok/s with acceptance length 1.14, and the boot
log shows the drafter accepting 0 of 1700 drafts for ~30 s at 22:17 then recovering. That is
the known deep-fill acceptance dip (results.md 2026-10-07, patch 8, net wall nil), a
within-boot transient, and it does not touch the prefill conclusion. The two clean cells carry
it. Worth a repeat of 130300 in isolation to confirm the dip is transient here too.

Open, smaller: int8 prefill (89.0 s mean) sits about 3.5 s / 4% above fp8 (85.5 s). Likely the
config table, not alignment: fp8 takes the sm_80 wide-tile override (BLOCK_N=128 warps=4,
ops/qsa.py:668) that int8 does not (is_fp8=False -> BLOCK_N=32). The read microbench used 32
for both, so fp8's real advantage was never in it. Cheap follow-up: try the same wide-tile
override for int8 prefill on sm_80.

## Why bf16 prefills faster than fp8/int8: the dtype doubles the block, the block splits the chunks (2026-10-25)

Three like-for-like offline traces (same build a7909d5fa+tracetool, same 131k prompt, same
env, `t-bf16/t-fp8/t-int8`), and the answer is not in any compute kernel:

| | bf16 | fp8 | int8 |
|---|---|---|---|
| kv group block sizes | 1616 | 3232 | 3232 |
| dominant prefill chunk (grid rows) | 8080 = 5×1616 | 6464 = 2×3232 | 6464 |
| prefill steps per prompt | ~16 | ~20 (+25%) | ~20 |
| nccl SendRecv total | 14.37 s | 15.53 s | 14.16 s |
| grand GPU total | 36.64 s | 37.47 s | 37.40 s |
| read kernel per row | 2.35 us | 1.82 us | 2.77 us |

Mechanism. One-byte KV halves bytes-per-token, so the hybrid page-size planner doubles the
attention groups' block size to 3232 tokens to stay commensurate with the GDN group. The
mamba align-mode chunk splitter floors every prefill chunk to that block size, so the same
8192-token budget yields 6464-row chunks instead of 8080: 25% more scheduler steps for the
identical prompt. On this pipeline a step costs an nccl round and host work (traces show
nccl at 38-42% of everything), and the fp8 arm's nccl total growing 14.37 -> 15.53 s
extrapolates to its measured 9.1 s at 1M. fp8 pays this even though its read kernel is 22%
FASTER per row than bf16's (the sm_80 BLOCK_N=128 override earning its keep at the lane's
chunk size, where the same override measured slower at my fixed synthetic grid).

int8 = the same chunk penalty plus a 1.52x read per row against fp8, 2.77 vs 1.82 us, measured
at fp8's sm_80 (BLOCK_N=128, warps=4) config which int8 cannot copy: the tile sweep showed
(128,4) is 2x SLOWER for int8 (33.4 vs 17.4 ms at the same grid). The remaining int8-vs-fp8
read cost is the two per-token fp32 scale gathers the dynamic layout needs and fp8's host-scale
path does not, inherent traffic of the design. A first guess, the fp16xfp32 V-descale promotion
pushing the PV dot to tf32, was TESTED by editing the kernel to land V on fp16 before the dot
and REJECTED: it made the read 1.6x slower (28.1 vs 17.4 ms), the change was reverted, tree
back at cd2d7eb305. The promotion is not the cost.

Fix candidates, priced by the same traces:
1. Scheduler-level, no code: raise `--max-num-batched-tokens` to 9728 so chunks floor to
   3×3232 = 9696 rows, 16% below bf16's current step count. Test on the lane.
2. Kernel-level: none found. No tile beats (32,1) for int8, and the V-promotion theory is
   dead; the per-token scale traffic is the price of dynamic per-token-head quantization.
