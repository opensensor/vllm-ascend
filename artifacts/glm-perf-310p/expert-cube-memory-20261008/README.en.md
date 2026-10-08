# GLM expert Cube and memory experiments

The qualified KDA score-row batching and permanent FP16-scale checkpoint are
retained. This batch adds two independent native expert options, builds and tests
all three combinations, and fixes allocator cleanup between resident graph
captures. It does not change the quantizer or the expert accumulation order.

## Native changes

`--active-cube-rows` rounds the populated paired A4 rows up to M16 blocks, using
M16/M32/M48/M64 instead of always M32/M64. Activation packing stays unchanged;
Cube readback, casts and single-row gather offsets follow the physical result
stride. Two extra offset tables cost 1,024 bytes of UB per core. The option
requires paired M32 prefill with the qualified strided readback. It preserves the
A8 and decode row selection. Counts 1–8 and 17–24 reduce Cube/readback rows;
counts 9–16 and 25–31 keep the existing geometry.

`--direct-w4-l1` sends prepared W4 codes from GM directly into L1. Cached
projections skip the UB write and UB-to-L1 copy; uncached W4 tiles load L0B from
the retained L1 tile. W2/W3 reconstruction, scale arithmetic, activation
quantization, barriers and native INT4 products retain their existing behavior.
The option requires the prepared layout and existing projection-cache allocation;
it adds no GM, UB or L1 allocation. Both options default off and their provenance,
compiler defines and real-weight gate requirements are recorded by the builder.

Versions 979/980/981 respectively select rows only, rows plus direct W4, and direct
W4 only. Serving substitutes the prefill native object; decode remains v964.
The frozen binaries and exact preformat compiler sources are included. The
committed C++ differs from those executed sources only in formatting.

## Arithmetic and graph checks

- v979: 30 independent arithmetic cases, 12 real-weight cases, and 108 paired
  real-weight boundaries against v965, with three changed graph replays each.
- v980: 30 independent, 12 real-weight and 60 paired boundaries on **each of four
  existing worker devices**. Every paired result is byte-identical to v965.
- v981: the same 30/12/60 full gate on a free device, followed by the standard
  12-case admission on every serving rank.
- Real fixtures are checkpoint expert 0 at layers 10/11/33 (W3/W4/W2), A4 and A8.
  Boundary rows cross the M16/M32/M48/M64 edges, the 31-row expert batch and a
  640-token expert workload. Replays change input, route weights, all-hot routing
  and all-peer routing; peer-only results are zero.

These checks certify the tested arithmetic and graph behavior. They are not a
language-quality evaluation or exhaustive coverage of every expert.

## Serving measurements

All cold requests use the same token IDs and reset scheduler prefix state. They
measure first visible text from full OpenAI-compatible requests. Each series runs
640, 1,280, 6,400, then repeats cold 640 and 1,280 tokens. Separate server processes
and first-use allocation effects limit cross-series comparisons. No statistically
established speedup is claimed for these new options.

| Variant | Cold 640 first / repeat (s) | Cold 1,280 first / repeat (s) | Cold 6,400 (s) |
| --- | --- | --- | --- |
| Rows only (v979) | 5.71 / 5.41 | 11.79 / 11.46 | 66.08 |
| Rows + direct W4 (v980) | 5.75 / 5.43 | 11.80 / 11.47 | 65.31 |
| Direct W4 (v981) | 5.71 / 5.37 | 11.63 / 11.45 | 65.30 |

The previous KDA series measured 5.40/5.36 s, 11.68/11.43 s and 65.36 s. Those
numbers are archived comparisons. The combined candidate's 65.31 s large prompt
retains the cumulative KDA improvement; the new expert changes are neutral within
the observed variation. The individual 640-token hot-expert profile also shows no
clear gain: W4 A4 gate/up is 15.07 versus 15.08 ms for direct-only, and 15.07 versus
15.23 ms for the combined option. Reducing transferred bytes does not alone
establish a shorter critical path. All per-stage timing samples are saved.

Final serving selects direct-only v981 with v964 decode. C1 generation
through request completion measures **9.64 / 9.45 / 9.89 tok/s**; C4 full-request
throughput is **10.34 aggregate tok/s** including prefill.
A real-text completion returns HTTP 200 and a nonempty photosynthesis explanation.
All four ranks report clean graphs, unchanged weight-storage digests across the
final swap and zero expert fallback. Health/models endpoints return HTTP 200 and
the server is unpaused. Current log: `expert-cube-final-server-20261008.log`.

## Graph recapture recovery

Two recaptures stalled while device memory was near capacity, after the arithmetic
checks had passed. The first followed the full in-worker qualification; the second
followed bounded admission and another variant swap. They are recorded as serving
failures, not silently omitted or accepted as successful performance runs.

`ResidentWorkerExtension.resident_apply` now releases unused allocator cache after
discarding graphs and collecting their final references, before replacement
capture. Live weights and KV buffers stay allocated. Mode-only changes do not
release cache or recapture. A regression check enforces cleanup before capture
and preserves the live tensor pointers and contents. After cleanup, prefill capture completed, but subsequent decode capture still
reported an explicit 242 MiB allocation failure. Cache release is useful cleanup,
not a demonstrated cure for all recapture memory pressure. The final recovery
admits all resources and selects its prefill kernel before one replacement capture,
avoiding an intermediate baseline capture. Failure and final live evidence are
saved separately; the full root cause of memory retention is unresolved.

Only the paused, identity-checked GLM trees were stopped. The final launch preserves
all prior eight fusions, the selected KDA OPP, prepared disk checkpoint, TP4,
EP/FlashComm1, target/draft A4, MTP1, prefix caching, 640-token chunks, four slots,
311,040 configured context and decode graph sizes 2/8. Multimodal requests remain
disabled. The tested context is 6,400 tokens; this is not maximum-capacity proof.

## Local checks and artifacts

1,521 focused CPU tests pass, including 17 Cube/builder tests and the new
allocator-order/pointer regression. Scoped file hooks pass.

Required `bash format.sh ci` reports pre-existing repository-wide findings; its
unrelated formatting changes were restored in the isolated checkout. Scoped hooks
pass for the owned changes. Default local pytest also encounters the existing
upstream runtime conftest dependency failure, so the CPU gate uses the performance
directory's `--confcutdir`. Hardware checks use actual compiled kernels and weights.

`measurements/` contains all gate receipts, admission hashes, switch/status records
and client timing results. `bundles/` freezes the three append-only native packages.
`sources/` holds exact compiler-source archives and launch/candidate sources.
`protocols/` saves executed gate, recovery and client scripts; `logs/` preserves
compressed build, test and serving output. `SHA256SUMS.json` binds the evidence.
See [the bilingual runbook](RUNBOOK.md) and [the Chinese report](README.md).
