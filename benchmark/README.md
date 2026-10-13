# benchmark/

Measurement tools and the record behind the numbers in the top-level README.
Everything here is stdlib Python talking HTTP to a running server, except the
kernel-level benches noted below. Reproduce commands are at the end of
[results.md](results.md).

## The record

- `results.md` - the full chronicle: grids, methodology, verdicts, salt law,
  step anatomy, reproduce commands. README.md summarizes. This file proves.
- `bench-*.json`, `mtp-4x1m-*.json`, `sustained-decode.json` - raw per-cell
  dumps (bench.py `--json`) behind those tables. Names carry the config they
  measured. Stale-looking names are history, kept for provenance.
- `quality/*.jsonl.gz` - raw logprob captures behind the W8A16 quality gate.

## Load cells

- `bench.py` - fill sweep x concurrency -> the table with TTFT, prefill,
  wall, out tok/s, sustained, acceptance, preempts. Salt is time-derived by
  default. Pin `--salt` for comparisons and rotate it for headline ranges (salt
  law). Dumps per-cell JSON and a live progress line.
- `sustained_decode.py` - batched-decode capacity plateaus at chosen depth
  and stream counts.

## Telemetry and step anatomy

- `stepwatch.py` (+ `stepwatch_report.py`) - per-step rate, mean step
  duration, token mix, acceptance from `/metrics` deltas, no CUPTI. Run it
  ON the serving box: a 20 Hz remote poll aliases a 21 ms step into a 32 ms
  ghost. `--nvml` adds clocks and power.
- `phase_probe.py` - runs the fixed-salt 100%-fill cell and fires
  `/start_profile` in steady decode; needs a `PROF=1` boot (capped window,
  self-closing).
- `phase_report.py` - parses the four per-rank chrome traces from that
  window into the decode step budget (cadence, spin vs compute, kernel
  categories, duration histogram). Run where the traces land (logs/prof).
- `humming_decode_bench.py` - kernel-level: dense/humming GEMM shapes at
  decode m, CUDA-event timed, min-composed scoring. Needs CUDA; not part of
  the HTTP tools.

## Acceptance and quality

- `mtp_ab_probe.py` - acceptance A/B at depth; doubles as the shared client
  lib for the other probes.
- `corpus_accept_probe.py` - acceptance over the real-text corpus
  (`../draft_corpus.jsonl`), greedy or sampled.
- `logprob_capture.py` + `quality_diff.py` - the quality gate: top-1 logprob
  fingerprints per tree, diffed into flip rates against the machine floor.

## Prefix caching

- `prefix_cache_probe.py` - cached vs recomputed tokens per conversation
  turn.
- `prefix_state_check.py` - cache-hit state correctness; exits non-zero.

## Conventions

- All HTTP tools default to `http://127.0.0.1:8000`. Always pass
  `--base-url http://fender.lan:8000` when the server is the remote box.
- Do not touch the server (no curls, no scrapes beyond the tool itself)
  while a bench cell is in flight.
- Kernel-level changes read fixed-salt acceptance FIRST, timing second:
  timing can look great while values rot (L2 eviction incident, results.md).
- `sampler.py`, `picked.json`, `decode_probe.py` are untracked local scratch.
