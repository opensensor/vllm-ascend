# GLM larger-prefill memory analysis

US English report. [中文报告](README.zh-CN.md).

Historical offline analysis: later hardware trials enabled the 1,280-token
budget with packed state. See the bilingual
[current optimization handoff](../cold-prefill-trace-20261008-v1007/OFFLINE_NEXT.md).
Pending-state statements below describe the original analysis. Larger chunks
and prefill graphs remain unqualified.

Implementation update: the opt-in source candidate is now present and CPU
validated. See [implementation and validation](IMPLEMENTATION.md). The design
and unknown physical bounds below describe the original analysis; no NPU
qualification or service deployment has occurred.

## Finding

The current compressor cache allocates four-token state pages from a global
block-ID pool. Every ID represents **7.96875 MiB per rank**, including attention
backing that the compressor never uses. Increasing the scheduler chunk from 640
to 1,280 tokens adds **1.245117 GiB per rank** to one-request cache admission.
This is separate from larger activations, scratch and graph allocations.

The preferred offline candidate packs eight existing four-token compressor pools
into a **32-token state page**. Its raw FP32 state occupies 32 KiB, within the
existing 40 KiB padded small-page class. Keeping the compression ratio at four
and retaining a conservative 33-token sliding window with MTP1 reduces the
1,280-token admission requirement by **2.178955 GiB per rank**. This layout is
**not implemented in the serving path or hardware validated**.

No server was launched, restored, probed or benchmarked during this analysis.
The last known 1,280-token trial failed startup; an earlier healthy 640-token
snapshot is historical evidence. Pending controller cleanup was not confirmed.

## Exact cache accounting

The deployed geometry has 12 main slots and 12 small slots, including MTP:

| Term | Per-rank bytes |
| --- | ---: |
| One main page: 640 tokens × 512 FP16 elements | 655,360 |
| One indexer page: 160 compressed keys × 128 FP16 elements | 40,960 |
| One global ID: 12 main pages + 12 small pages | 8,355,840 |
| Actual four-token compressor state per layer: 4 × 256 FP32 elements | 4,096 |

For PP1, synchronous scheduling, align-mode KDA, MTP1 and no extra retained
tokens, startup admission for **one maximum-length request** is:

```text
full IDs = ceil(context / 640)
tail IDs = ceil(min(context, 4 + chunk_budget) / 4) + 1
KDA IDs  = 3 groups × (2 + 1 speculative block) = 9
cache bytes = (full IDs + tail IDs + KDA IDs) × 8,355,840
```

The rollover ID is required. Async scheduling would increase in-flight tokens;
it cannot reuse the synchronous calculation. Four configured requests do not
mean four maximum-length requests fit. Null-block overhead, retained prefix
blocks and additional concurrency capacity need separate reserve.

| Context tokens | Current 640 chunk, GiB | Current 1,280 chunk, GiB | Proposed 32-token state page with 1,280 chunk, GiB |
| ---: | ---: | ---: | ---: |
| 65,536 | 2.132 | 3.377 | 1.198 |
| 131,072 | 2.926 | 4.171 | 1.992 |
| 208,640 | 3.868 | 5.113 | 2.934 |
| 311,040 | 5.113 | 6.358 | 4.179 |

These are admission minima, not execution-memory forecasts or allocated KV
budgets. The worker allocates its configured cache budget, which can be larger
than the minimum. Reducing max context alone does not guarantee that allocation
shrinks when the worker still uses a fraction of profiled headroom.

The latest failure reported 4.17 GiB required and 3.72 GiB available. The exact
formula gives 4.171142578125 GiB. Because available memory was logged to two
decimals, treating it as exactly 3.72 GiB would be misleading. Its rounding
interval gives admission context limits of approximately 93,440–94,080 tokens;
the log's 93,440 estimate is consistent. Even the upper interval endpoint leaves
a cache deficit exceeding 0.446 GiB at 131,072 context.

Reducing context from 311,040 to 208,640 offsets the extra cache IDs of a
640→1,280 chunk increase, but leaves no extra allowance for larger execution
buffers. An earlier 65,536-context trial passed some requests and later failed
workspace allocation. Startup admission alone is insufficient.

## Packing the state pages

The first candidate changes state allocation granularity while preserving
four-token compression and all FP32 state values:

1. Declare compressor state block size 32, sliding window 33 for MTP1, and
   retain the current 40 KiB padded page. Attention stays at logical block 640.
2. View each state page as `[32, 256]` FP32 with a **10,240-element page stride**.
   Read the four-member pool starting at `floor(token_offset / 4) * 4`.
3. Update both the ordinary and resident compact writers. Their current
   `[4,256]` checks, `slot / 4` page lookup and row-offset formulas would be wrong
   for this layout. Gather previous state before writing the current tail.
4. Preserve negative-slot masking, request isolation, prefix-hit handling and
   both pools crossing a speculative rejection boundary. Graph metadata must
   keep stable addresses and use the new block size in target and draft paths.
5. Recalculate admission through the real cache specs. Do not merely advertise
   fewer blocks while leaving four-token physical allocation unchanged.

For the proposed layout, tail admission is
`ceil(min(context, 32 + chunk_budget) / 32) + 1`: 42 IDs at a 1,280 budget,
instead of 322. The main/small page sizes and global pool allocator remain
unchanged. At 1,280/311,040, minimum cache plus the identified MoE scratch is
about **0.858 GiB lower** than current 640/311,040 with its scratch. This is an
offline comparison of these two terms, not a total-memory saving measurement.

A more invasive alternative separates compressor physical backing from the
main pool while retaining all 322 tail IDs. It saves 2.358398 GiB at a 1,280
chunk budget, but needs dedicated pools or global-to-local address translation.
Shrinking a tensor descriptor alone is invalid: unmodified kernels still
address arbitrary global IDs. The 32-token state-page candidate captures most
of that potential saving with fewer architectural changes.

## Execution memory that still needs a bound

The archived nine shared A4 MoE storages are reproduced exactly by the planner:

| Chunk budget | Shared MoE scratch, MiB |
| ---: | ---: |
| 640 | 93.016 |
| 1,280 | 170.633 |
| 2,560 | 325.867 |

The 640→1,280 increase is 77.617 MiB, not a per-layer multiplier. Returned FP32
outputs, routing temporaries, KDA/QSA buffers and native operator workspaces are
additional terms. Page-packing savings do not change the math or these buffers.

Historical rank totals are heterogeneous: 45.67/46.15 GiB. The limiting rank
reported 44.61 GiB free at startup; loading weights consumed 34.6368 GiB. The
source prints that weight quantity with a `GB` label but divides by `2**30`.
Use the limiting rank, not marketing capacity or a four-rank average.

The archived 640 snapshot has 40.657 GiB live allocated, 41.938 GiB reserved on
rank 1, and 1.924 GiB minimum reported free across ranks. Its lifetime maximum
allocated/reserved counters are 43.031/43.664 GiB on rank 1. These counters can
include gates and captures; they are not isolated cold-prefill peaks. Reserved
already includes allocated memory. Summing them would double-count it.

The earlier decode captures logged 0.56 GiB, and a later resident recapture
logged a 2.95 GiB allocation delta. Those deltas do not establish an independent
graph-pool upper bound. They cannot be added indiscriminately to a later
allocated snapshot. Likewise, startup non-Torch memory is not a peak operator
workspace bound, and reported free bytes do not guarantee a large allocation.

For 1,280/131,072 under the old layout, historical weights + minimum cache +
identified MoE scratch already total 38.975 GiB. A 0.92 utilization budget on
the historical 45.67 GiB rank leaves about 3.042 GiB for **all** graphs, other
persistent buffers, transient allocations, external workspaces and allocator
slack. This is a partial ledger, not a proof of fit.

## Offline gate and validation

The [CPU-only planner](../../../tools/glm_perf/prefill_memory_budget.py) reports
admission separately from whole-execution feasibility. A per-rank envelope must
provide non-overlapping upper bounds for resident buffers, graph pool, transient
peak, external operator allocations, fragmentation and safety reserve, tied to
the exact candidate/configuration. Unknown or stale bounds block feasibility;
one failing rank blocks the candidate. No launch commands are emitted.

```bash
python -m tools.glm_perf.prefill_memory_budget \
  --chunk-tokens 1280 --context-tokens 131072 --cache-gib 3.72
python -m pytest -q --confcutdir=tests/ut/glm_perf \
  tests/ut/glm_perf/test_prefill_memory_budget.py
```

**45 CPU tests pass**, including production allocator/scratch AST conformance,
archived failure arithmetic, rollover, concurrent batches, page addressing,
MTP1 boundary addressing, invalid inputs, stale-envelope rejection and unknown
peak rejection. A separate CPU check compares 24 cases against the pinned
upstream sliding-window method. Ruff checks and formatting pass.

The addressing tests establish pool isolation and retained-position coverage;
they do not validate scheduler rollback, compression numerics or actual graph
execution for the proposed layout. No new NPU tests or speed claims are made.

[Inputs](INPUTS.json) preserve extracted historical counters and their source
hashes. [Results](RESULTS.json) contain the capacity matrix, scratch breakdowns
and unknown physical bounds. [Latest launch receipt](latest-launch-receipt.txt)
records the failed trial's command, not a successful deployment.

The next offline implementation should implement the 32-token state layout and
exercise the real writers with reference state/prefix/MTP tests. Execution
qualification remains blocked until candidate-specific graph and workspace
bounds complete the per-rank ledger. Another guessed launch is not justified.

## Source basis

- `vllm_ascend/models/glm5next/cache_config.py`: shared pool geometry,
  `_required_scheduler_blocks`, allocation and startup memory requirement.
- `vllm_ascend/core/kv_cache_interface.py`: compressor-state admission.
- `vllm_ascend/models/glm5next/kv_cache.py`: four-token state block and MTP window.
- `vllm_ascend/attention/indexer_kpool.py`: independent state metadata and shape.
- `vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py`: state gathers,
  speculative tail retention and padded row offsets.
- `tools/glm_perf/resident_candidates/kpool_completed_prefill.py`: resident
  compact writer's independent four-token addressing assumptions.
- `tools/glm_perf/glm_fused_moe.py`: shared scratch and routed packing shapes.
- `vllm_ascend/_310p/worker_310p.py`: fraction-based and explicit cache budgeting.
- Pinned upstream `3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`:
  `vllm/v1/kv_cache_interface.py` sliding-window admission and align-mode Mamba;
  `vllm/config/vllm.py` in-flight token calculation.
