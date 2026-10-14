# KV offload to host RAM (SHELVED 2026-10-14)

Feature: resume an evicted session by DMA-ing its KV back from host RAM
instead of re-prefilling. Proven working for correctness; shelved because
the host DDR budget cannot comfortably hold the regions this model's KV
needs. Everything needed to revive it lives here.

## Why shelved: the capacity math

Measured 2026-10-13/14 on Qwen4Exp at PP=4, block 1616:

- Real KV data: 41 KB/token across all four ranks. A full 1M session is
  ~41 GB. Host RAM on the serve box does not have that to spare per
  region, with the PLE table already pinned at ~47.6 GiB.
- Worse, the upstream region does not use what you give it. The manager
  pool is sized in chunk ROWS, but keys are per (chunk, group), and each
  key allocates a whole row whose cell layout could have held every
  group. With ~5 prefix-cacheable groups the usable window collapses to
  rows/groups chunks: a 51.33 GB region (166 rows at bpc 5) served only
  ~30 chunks ~= 243k tokens. Measured exactly: 150k sessions resumed from
  host (3.8x faster TTFT, byte-identical greedy output), 899k sessions
  loaded 0.00 GB and re-prefilled, twice, quiet box included.
- The eviction-order wrinkle (a 900k resume should at least have hit the
  resident tail; lookup is prefix-first, so a tail window yields zero)
  plus yesterday's unexplained 71% partial load on a 341k session at
  bpc 1: both stay open in case a revival reuses this ground truth.

## What is proven and shipped in these patches

Stack these three on the 7-patch series (they were branched after 0007):

1. `0008-Bugfix-Spec-Decode-Propagate-enable_cumem_allocator-*.patch`
   The drafter rebuilds its ModelConfig field by field; the connector
   compat gate re-validates through that copy and rejected cumem unless
   the flag propagates. Boot died on Worker_PP3 without it.
2. `0009-Bugfix-KV-Offload-Unify-offload-region-geometry-acro*.patch`
   Upstream sizes the shared /dev/shm region from each LOCAL pipeline
   stage's kv bytes per block; the four stages disagree (layer counts,
   PLE group, drafter group) and joiners time out on the creator's
   smaller file. The engine core now stamps the engine-wide max onto
   every stage config; the region's strided views already tolerate the
   local padding.
3. `0010-Bugfix-KV-Offload-Add-pin_region-switch-so-the-CPU-t*.patch`
   `pin_region:false` in kv_connector_extra_config skips the
   cudaHostRegister of the whole region: sparse, lazily faulted, pageable
   transfers, RAM only where stores actually land. Note:
   VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY is for WEIGHT offload and
   does nothing here.

Replay state when shelved: 10 patches, tree `99e57b1e2e02f0d45dc1beece690e297dc244119`
on base `183625d4f9`, series branch `qwen38-pp-183625d4f9` in
/home/gab/test/wt-rebase-1013.

## How to relight

```bash
git am --keep-non-patch /path/to/series/patches/00*.patch \
    /path/to/series/experiments/0008-Bugfix-Spec-Decode-*.patch \
    /path/to/series/experiments/0009-Bugfix-KV-Offload-*.patch \
    /path/to/series/experiments/0010-Bugfix-KV-Offload-*.patch
# serve-4x-1m.sh passes "$@" through, so no script edit is needed:
bash serve-4x-1m.sh --enable-cumem-allocator \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":51539607552,"blocks_per_chunk":5,"pin_region":false,"eviction_policy":"lru","offload_prompt_only":true}}'
```

Invariants the scheduler enforces at boot: blocks_per_chunk x 1616 must
divide the retention interval 16160, so blocks_per_chunk in {1, 2, 5, 10}.
Region geometry sanity: four ranks must log identical Created/Opened mmap
sizes; the creator unlinks the file at once, so /dev/shm looks empty while
Shmem carries the bytes.

## Instrumentation

- `offload_probe.py` (run from the repo root with
  `PYTHONPATH=benchmark python3 experiments/offload_probe.py --length 150000`):
  cold session, settle, GPU flush with per-request
  `max_offload_tokens=0` caps (proven to hold flushers out of the region),
  resume, then per-phase deltas of `vllm:kv_offload_total_bytes_total`
  plus a byte-equality check of the greedy continuation.
- `offload-debug-logging.json`: per-module DEBUG (offloading scheduler +
  kv_offload) via VLLM_LOGGING_CONFIG_PATH. Do NOT use
  VLLM_LOGGING_LEVEL=DEBUG: the IR-op logger reprs CUDA tensors during
  graph capture and kills the boot with cudaErrorStreamCaptureUnsupported.
- Engine INFO lines worth grepping: "KV Transfer metrics",
  "offloading N chunks upto", "hit N offloaded tokens",
  "Host-registered mmap region" (absent when pin_region=false).

## Revival idea: NVMe instead of more DDR

The shelved blocker is DDR price, not bytes. `TieringOffloadingSpec`
(spec_name in kv_connector_extra_config) adds secondary tiers under the
CPU primary tier, including `{"type":"fs","root_dir":"/mnt/kv_cache",...}`
and a KVCR tier, with backpressure and per-tier hit metrics upstream since
~#56318. All GPU<->secondary traffic stages through the CPU tier, so the
design fits a small RAM window plus a multi-TB NVMe tier: capacity stops
being the question, disk latency (~50-100 us/random, GB/s sequential)
still beats 82 s of re-prefill for a 1M session by orders of magnitude.
The row-per-key waste above then costs disk, not RAM. Check upstream
state first; this subsystem moved fast in October 2026.
