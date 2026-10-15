# Fresh QFN graph profile and recovery — October 14, 2026

## Serving state

The single authorized recovery restart completed. The retained combined winner
was restored in `resident-live-v4`, with target general and prefix variants at
capture sizes `[3, 6, 9, 12]`. The model remains `qwen38-flash-next` on
`http://192.168.53.187:8001/v1`; Kilo remains
`vllm-310p-w4/qwen38-flash-next`. Configured context is 262144 tokens, without a
full-context quality claim. The server is unpaused and all profilers are released.

The new log is
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v4/server.log`.
Workers are 1638646–1638649. All four devices were healthy at the final check,
at 65/64/69/68°C. No host reboot or device reset was performed.

## Fresh unprofiled measurements

| Workload | Measurement |
| --- | --- |
| C1, three separate 256-output-token requests | 30.05 HTTP output tok/s median; range 29.96–30.78 |
| Same C1 requests, server decode timing | 31.6 tok/s median; range 31.3–32.2 |
| C4, four simultaneous 256-output-token requests | 61.78 aggregate HTTP output tok/s |
| Same C4 requests, per-request SSE decode estimate | 16.57 tok/s median; range 15.78–17.17 |

These are restored-service checks, not a new speedup. Earlier retained results
were 31.31 C1 HTTP tok/s and 63.04–65.45 C4 aggregate tok/s. The current runs are
slightly lower, and do not isolate the cause. The 40 tok/s C1 goal remains unmet.
C1 uses a short hash-table prompt; C4 uses four distinct short prompts. All
requests returned their exact 256-token budgets. HTTP rates include prefill;
server decode timing excludes it. C4 throughput is not single-request speed.
Cold prefill was not rebenchmarked in this profiling pass.

## All-rank device traces

Three cases each recorded eight engine iterations on all four ranks after eight
warmup iterations: C1 prefix, C4 prefix, and C1 general at 2326 prompt tokens.
The analyzer uses six complete interior iteration intervals and unions device
tasks across streams. Profiling changes timing; these are attribution samples,
not throughput benchmarks. An MTP iteration is not one accepted output token.

| Case | Graph launches per iteration | Eager callbacks per iteration | Rank 0 median iteration | Rank 0 graph-submit API time per iteration |
| --- | --- | --- | --- | --- |
| C1 prefix | 10 | 7 | 86.24 ms | 0.109 ms |
| C4 prefix | 10 | 7 | 149.12 ms | 0.124 ms |
| C1 general | 22 | 19 | 92.49 ms | 0.197 ms |

Counts are consistent across all ranks. Prefix target graphs have two native
segments and one PLE callback. General target graphs have fourteen segments,
with PLE plus twelve QSA selection callbacks. Each of the two draft steps has
four segments and three callbacks: causal-position copy, QSA selection, and
expert dispatch. Graph submission duration alone is a small part of latency.

Rank 0 boundary measurements, averaged over six iterations:

| Boundary | C1 prefix span / no recorded task | C4 prefix span / no recorded task | C1 general span / no recorded task |
| --- | --- | --- | --- |
| Last draft to next target iteration | 13.05 / 8.57 ms | 14.42 / 9.42 ms | 13.31 / 9.20 ms |
| Draft expert dispatch, both steps combined | 8.02 / 1.81 ms | 17.45 / 4.31 ms | 9.16 / 1.68 ms |
| Target QSA selection, twelve breaks combined | Not applicable | Not applicable | 8.92 / 0.26 ms |

These spans include asynchronously executing device work. Neither their full
span nor the absence of a recorded task is a demonstrated latency saving.
The next-iteration boundary contains copies, event synchronization, all-gather,
and metadata operations; this trace does not fully attribute Python or engine
scheduler time. Capturing QSA can reduce submission work even when its current
callback overlaps useful device execution.

The short C1 trace contains an outlier: rank 3 spent 56.99 ms inside one CPU
`aten::copy_` event after its final draft graph. Its iteration grew to 138.65 ms;
other ranks' following intervals grew to 141–143 ms. This is consistent with a
rank-local delay propagating to peers, but the copy event is inclusive and does
not prove a slow DMA transfer or identify the underlying runtime delay. The
outlier is retained in the JSON and means, rather than silently discarded.

Rank 0 summed kernel durations per iteration were 20.66 ms MatMulV2, 4.81 ms
TransData, and 3.65 ms Cast for C1 prefix. At C4, the W4 gate/up matmul category
rose to 28.82 ms, MatMulV2 to 23.42 ms, and TransData to 8.23 ms. These sums
can overlap and are not critical-path savings. Graph closure and projection,
layout, and W4 kernel optimization are complementary work.

## Required end state and implementation order

All remaining eager callbacks are elimination targets. The final architecture
should keep admitted tensor execution, dependency signaling, routing, sampling,
and speculative commit on device, with peer DMA and notifications connecting
rank-local graphs. CPU admission and resource retirement remain cold control
operations. The first bounded chains are an implementation step, not a decision
to preserve per-token CPU orchestration permanently.

1. Close the iteration boundary: retain sampled IDs, accepted lengths, positions,
   KV/recurrent commit masks and next-input construction on device. Establish
   all-rank agreement and exact rejection semantics before chaining iterations.
2. Replace draft host expert-ID readback and Python grouping with graph-compatible
   device routing into fixed, owned expert slots. Preserve current W8A16 math,
   FP32 weighting and reduction order.
3. Absorb PLE device projection/gating/convolution into graphs. Its large host
   embedding table still requires a separate graph-compatible lookup/staging
   design; moving only the device math is an intermediate milestone.
4. Absorb all general and draft QSA selection and position preparation into
   bounded capacity variants, with safe unused page IDs, exact selection ordering
   and independent captured metadata. Existing component results do not yet
   qualify a complete general-model graph.
5. Connect those local graphs with the distributed DMA dependency contract:
   publish payload, signal READY, consume locally, acknowledge final consumption,
   and reuse only the matching generation/sequence slot. Qualify finite chains
   first, then native conditional/device-driven continuation. Measure producer
   skew, hidden waits, accepted-token speed and replay-resource lifetime.

See the updated [distributed graph ledger design](../../tools/sidecard/DISTRIBUTED_GRAPH_LEDGER.md).
Direct DMA already supports the qualified custom SUM path. An autonomous,
whole-model distributed graph is the target, not a completed or fastest-validated
implementation. The graph pool failure below is a lifecycle problem to resolve,
not evidence that the hardware cannot execute the intended design.

## Recovery, fixes, and qualification

An initial diagnostic call used a legacy RPC slot that the current candidate had
repurposed. It failed before starting profiling. A subsequent attempt to install
profiling through model reload stalled during native graph recapture. Rank 3
reported graph pool allocation errors including `pool is nullptr ... id=-1`;
other ranks waited in native operations. Evidence is saved in
`graph-refresh-stall-v1`. The exact resource/lifetime cause remains unresolved.
No temperature or device-health fault was established.

With the user's one-restart approval, only the service's API, engine and four
workers were terminated, including forced termination after the grace period.
This was not a clean native collective close. Fresh workers loaded the same
pinned runtime and rebuilt the retained native resources. W4 finalizer and
schedule, HC combination, K gathering, and QSA geometry passed their existing
all-rank gates. Ten model checks passed, including mixed concurrent contexts,
8K input and return-to-prefix transitions. These are smoke and component gates,
not full model-quality qualification.

Dedicated cold `qfn_native_profile_start`, `qfn_native_profile_stop`, and
`qfn_native_profile_status` RPCs now permit bounded diagnostics without replacing
model code or recapturing graphs. All twelve recording sessions started/stopped
cleanly across the three cases, and final status confirms no profiler ownership.
The combined host regression suite passed 158 tests. Scoped repository hooks
passed; full repository tests and the secret-scan hook were not run.

## Evidence

- `qsa-resident-v1/native-timeline-c4-coverage-v2/summary.json`: all-rank timings,
  per-iteration outliers, device unions, kernel shapes and host API durations.
- The same directory's three `*-receipt.json` files: inventories, balanced
  recording lifecycles, response budgets, unchanged worker/storage identities.
- `boundary-cpu-ops.json`: rank 0/3 post-draft CPU events and the outlier.
- `qsa-resident-v1/decode-benchmark-graph-refresh-v4.json` and
  `concurrency-graph-refresh-c4-v4.json`: unprofiled HTTP measurements.
- `resident-live-v4/requalification/`: component gates and model validation.
- `resident-live-v4/final-serving-status.json` and `benchmark-request-log.txt`.
- Full parsed traces remain under the same artifact path on the hardware host.

Private control credentials are excluded from these artifacts.
