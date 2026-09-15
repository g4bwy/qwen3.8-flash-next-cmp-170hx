# Benchmark results, full tables

All numbers measured on the 3x unlocked CMP 170HX box (200 W power cap, PP=3,
MTP-3, PLE host offload) running one server at a time, idle and warmed,
greedy, unique prompt per row so every row pays a full prefill. Tools:
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
  157 at 8k depth.
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
./benchmark/sustained_decode.py --base-url http://<host>:8000
```
