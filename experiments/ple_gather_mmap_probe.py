# SPDX-License-Identifier: Apache-2.0
"""PLE staging gather: preadv loop vs mmap-based access (2026-10-22).

Same workload every time: 32,768 scattered 2 KB rows over 16 shard files
(9.4 GiB, warm), byte-verified. Results (ms):

  Python os.preadv loop (current stage_rows)      56
  C preadv loop                                   32-36
  C raw io_uring (submit-all/wait-all)            38 sync / 46 async
  pyuring 0.3.3 UringCtx                          163
  C mmap + memcpy loop                            11.3
  torch index_select from mmap, per-shard out=    2.7

The winner is not a faster syscall; it is not calling one. Map each shard
MAP_SHARED read-only once at bind, wrap with torch.frombuffer, and gather
with per-shard index_select into the pinned staging slices. Kernel work
collapses to (already-resident) page loads; torch parallelizes them.

Production deltas vs this probe:
- ext4 page cache instead of tmpfs: same once resident; first touches
  take minor faults (preadv pays equivalents today anyway).
- rows must land in (token, head) order, not shard order: one
  index_copy_ undo pass (~1 extra 64 MB copy) or pre-sorted staging.
- 47.6 GiB mapping = 12 M pages: expect more TLB pressure than a 9.4 GiB
  probe; still DRAM-bound either way.
- mapped pages resist reclaim (PTE zapping): the reason upstream #58815
  chose preadv; this box keeps the table resident regardless.
"""
