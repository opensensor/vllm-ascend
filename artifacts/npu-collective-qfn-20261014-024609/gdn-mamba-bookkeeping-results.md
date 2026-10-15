# GDN metadata graphs and recurrent-state admission

Both changes are retained on the four-rank QFN service. Model identity remains
`qwen38-flash-next`, API `http://192.168.53.187:8001/v1`, and Kilo entry
`vllm-310p-w4/qwen38-flash-next`. No restart or weight reload was required.
This removes repeated CPU work; it does not complete CPU-independent execution.

## What the Mamba cache means here

QFN uses Gated DeltaNet recurrent layers. vLLM manages their convolution and
recurrent tensors through its generic Mamba cache infrastructure. The live state
is part of model computation: each token updates a compact summary of previous
tokens. Removing that state would require recomputing history or changing the
model. Hybrid attention layers separately retain their attention cache.

Prefix checkpoints are a separate optimization: saved recurrent states permit
resuming a matching prefix instead of recomputing it. Speculative decoding also
needs candidate states so rejected tokens cannot corrupt the accepted history.
The current implementation is not established as optimal.

The qualified geometry has 36 convolution/recurrent pairs per rank, 72 banks in
all. Grouped commit checks whether an accepted sequence crosses a 128-token
checkpoint boundary before copying. It does **not** copy all 72 banks every token.
Host metadata preparation and admission were repetitive even on no-copy steps.

The device checkpoint archive filled during earlier traffic and logged CPU
spills at 18:23:43 and 18:23:57 in the server log. This establishes that eviction
occurred, not its contribution to total latency or whether the cache was thrashing.
A future checkpoint policy comparison must measure saved prefill against copy,
eviction, and restore costs rather than simply disable recurrent state.

## Implemented and retained

- `tools/qwen4exp/gdn_metadata_graph.py`: three retained metadata graph owners per
  rank, each with exact batch sizes 1–4 for uniform three-token MTP2 decode.
  Graphs update the existing model-visible buffers for state indices, masks,
  query boundaries, and accepted counts. Unsupported layouts retain the original
  builder. CPU request admission, graph launch, and allocation remain.
- `tools/qwen4exp/iteration_transition_runtime.py`: validate common request
  geometry once, then validate the union of all possible acceptance-dependent
  accesses. Previously the entire validator ran four times, for counts 0–3.
  This retains pre-submission bounds checking without reading acceptance to CPU.
- Drained controls preserve owners and existing model captures. The original
  admission function remains retained for comparison/rollback in this process.

## Validation

- 179 focused host tests passed across transition, commit, metadata, queue,
  observer, history, and compact-table code; scoped Ruff passed.
- Randomized admission equivalence covers batches 1–4, block sizes 1/2/3/128 and
  INT32 maximum, checkpoint boundaries, invalid IDs, table geometry/dtype, and
  overflow limits. No writes are issued by these host tests.
- GDN metadata qualification: 204 exact comparisons per group per rank,
  comparing all six captured metadata fields to the original builder.
- Admission qualification: 202 additional acceptance/commit checks, 192 next-input
  comparisons, and 216 state-bank comparisons per rank, all passed.
- Each change passed the ten-case hardware model suite, including checkpoint
  crossings, 8K inputs, mixed C4 contexts, and return to short requests.
- The retained combination passed all twelve serving checks: small token budgets,
  penalties, logprobs, uneven C4 completion, structured JSON, stream abort, and
  successful generation after abort.
- Full repository tests and full-context model-quality validation were not run.

## CPU evidence

The new short-request CPU trace contains one prefill and 52 decode iterations;
its earlier comparison contains one prefill and 53 decode iterations.

| Operation | Earlier trace | Retained trace |
| --- | ---: | ---: |
| Common commit validator calls | 213 | 53 |
| Original GDN builder calls | 162 | 3 (prefill only) |
| Admission cumulative time/call, four-rank range | 0.470–0.499 ms | 0.228–0.230 ms |
| GDN builder/wrapper cumulative time/call | 0.735–0.746 ms | 0.214–0.225 ms |

GDN has three builder calls per model iteration. New wrapper timings include
private graph dispatch. These profiled wall times are not additive savings in
end-to-end latency. In particular, parent attention-metadata attribution rose
from 2.895 to 3.976 ms on rank 0 despite the lower leaf timings; this profile does
not establish a reduction in the entire attention-metadata critical path.

A separate local CPU microbenchmark measured admission at 78–190 us before and
30–55 us after across C1/C4 no-copy and boundary cases. That is isolated host work,
not a hardware inference speed claim.

See [CPU attribution](resident-live-v6/gdn-admission-cpu-summary.json) and raw
`resident-live-v6/cpu-scheduling-gdn-admission-v1/rank*.pstats`.

## Sequential serving measurements

C1 is median HTTP throughput across three 256-output-token requests. C4 is one
finite simultaneous batch of four 256-token requests, including prefill and tails.
These small trials have no confidence intervals.

| A/B configuration | C1 HTTP tok/s | C4 aggregate HTTP tok/s |
| --- | ---: | ---: |
| GDN metadata graphs off | 32.319 | 66.329 |
| GDN metadata graphs on | 32.658 | 66.405 |
| Consolidated admission off, GDN graphs on | 32.517 | 67.613 |
| Consolidated admission on, GDN graphs on | 32.266 | 65.310 |

The admission trial did not show an end-to-end speedup. Both changes remain
retained per the user's explicit preference to keep verified reductions in CPU
work. No throughput improvement is claimed for admission. The interrupted
`gdn-metadata-*-v2` trial overlapped external traffic and must not be used.
The clean GDN comparison uses `*-v3`; admission uses `*-v1` in `qsa-resident-v1`.

## Retained state and next targets

[Final inspection](resident-live-v6/final-gdn-admission-status.json) reports health
200, unpaused serving, unchanged worker PIDs, no dirty model graphs, both changes
enabled, and all profilers released. GDN owners each report 2,247 replays and
204 checked replays. Verification mode is off. Device temperatures at inspection
were 79–86 C with all four devices reporting OK; benchmark load has ended.

Remaining CPU work includes recurrent-state mapping/allocation, shared attention
metadata, PLE host lookup, lifecycle/termination handling, and graph submission.
A larger reduction requires admitting a stable execution window once, with
versioned device-owned mappings and acceptance state throughout that window.
Correctness must cover checkpoint ownership, request turnover, rejection,
cancellation, and safe retirement before mappings can be reused.

To reproduce from the qualified v6 service, deploy the two tool modules into the
resident overlay, then use `qfn-gdn-metadata-control.py install` and
`qfn-commit-admission-control.py install` with fresh labels. Installation is once
per retained owner; subsequent controls use `on`, `off`, or `verify` while drained.
Run hardware validation before disabling verification. These hot controls are
not automatically restored by every historical launcher.
