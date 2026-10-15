# W4 schedule serving regression: offline audit

The schedule candidate is not a serving winner. One uncached 8,192-token,
one-output trial increased from 23.032 s (355.68 prompt tokens/s) to 29.818 s
(274.73 prompt tokens/s), a 6.786 s increase. These are individual trials,
despite the benchmark receipt's `median_*` field names.

Source receipts:

- `qsa-resident-v1/cold-benchmark-w4-control-a.json`
- `qsa-resident-v1/cold-benchmark-w4-schedule-a.json`
- `qsa-resident-v1/w4-schedule-up-20480-v1.json`
- `qsa-resident-v1/w4-schedule-down-20480-v1.json`

Both serving trials have zero cached prompt tokens and identical token counts.
Per-rank weight digests and collective configuration digests remain unchanged.
Both start with zero resident prefix checkpoints and end with six per prefix
tier. Their native graph inventory remains five graphs and 70 leases per rank.
These checks exclude obvious cache-count, weight replacement, or graph-count
differences, but do not establish identical allocator state or routing.

## What the component measurements establish

All four ranks passed exact output gates. Rank 0 synchronized component medians:

| Projection / routes | Installed ms | Local-count ms |
| --- | ---: | ---: |
| Gate/up, 5,120 balanced local | 26.054 | 19.256 |
| Gate/up, 5,120 on one expert | 14.127 | 13.193 |
| Gate/up, all 20,480 local | 65.471 | 65.322 |
| Down, 5,120 balanced local | 12.510 | 10.727 |
| Down, 5,120 on one expert | 7.217 | 8.294 |
| Down, zero local | 1.623 | 2.000 |

Total local rows alone cannot select the best schedule for every expert
distribution: the down projection loses on concentrated and empty cases.
However, these measured penalties alone do not explain a 6.8-second serving
regression. A full-run trace is required before attributing the difference to
device arithmetic, host work, allocation pressure, or collective waiting.

## Memory implications and testable hypotheses

The parent observed roughly 47 GB in use on rank 0 versus about 46 GB before,
with rank 2 closer to 45 GB. Device memory totals combine live tensors,
allocator reserves, graph pools, and other runtime allocations. They cannot
identify a leak or prove fragmentation on their own.

The candidate's retained descriptors are seven INT64 values each, capped at
256 entries. Their payload is only 14 KiB at capacity, excluding allocator
rounding. Native output shape and dtype match the installed operator. The
schedule's extra tile buffers reside in on-chip storage. These do not directly
account for a gigabyte of persistent model memory.

One concrete integration difference is asynchronous ownership in the generic
bridge. It captures a vector of tensor references in the queued custom handler,
then clears that vector after native submission. A 20,480-row gate/up call
references about 75 MiB of prepared activation operands plus a 50 MiB output;
the down call references about 18.75 MiB of prepared operands and a 100 MiB
output. Eight delayed or extra retained handler instances could therefore
represent approximately a gigabyte of scratch. This is a hypothesis, not an
observed leak: the bridge explicitly clears its submitted handler's arguments,
and the runtime may release or copy handlers differently.

Synchronized single-operator gates drain those references after every call.
They do not reproduce a mixed prefill stream's queued tensor lifetimes. Faster
device kernels can also change host queue depth and allocator reuse timing;
per-rank memory asymmetry can then amplify allocation retries or collective
waits. Graph recapture can change pool addresses and fragmentation without
changing the graph count.

Useful discriminators, for a future controller-managed experiment:

1. Record allocated, reserved, inactive-split, allocation-retry and peak-memory
   counters per rank around matched cold requests. Compare these with physical
   device usage; do not infer all of them from `npu-smi` alone.
2. Trial the frozen **baseline** schedule through the same generic bridge and
   wrapper. It keeps the original tile policy, isolating bridge/allocation
   behavior from schedule selection. The current deployment factory does not
   enable this mode, so that would require an explicit reviewed experiment.
3. Profile one bounded request and attribute allocator calls, stream waits,
   queued custom-handler lifetimes, kernel durations and cross-rank arrivals.
   Compare overlapping intervals rather than adding host waits to device work.
4. If the bridge is implicated, use a short private sequence with one final
   synchronization, sufficient memory preflight, and retained ownership. A
   per-operator synchronize loop cannot measure the suspected backlog effect.

No hardware action was taken during this audit. Rollback and the separate
finalization trial remain the parent's responsibility.
