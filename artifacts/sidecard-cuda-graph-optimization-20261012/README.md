# GLM CUDA graph optimization

The matched decode median rises from **10.56 to 12.42 tokens/s**, a **17.6%**
improvement. This iteration profiles the qualified direct-DMA service and extends CUDA
capture to hot/shared experts and the fixed math around DSA attention. The
runtime uses one RTX PRO 6000 and physical Ascend chips 2 and 3 on one card.
The other Ascend card, clocks, ECC, and power settings were untouched.

## Implementation

`HotDecodeGraphs` captures the original local hot-expert kernels and shared MLP
for each middle layer. The stream-aware native CUDA ABI and its existing shared
scratch remain serialized on the caller's stream. Active Ascend workers receive
their commands before local work replays. Reply waiting, write-visibility
flushes, and ordered cold partial reduction remain outside the graph.

`DsaDecodeGraph` captures fixed-shape projections and mHC math before and after
attention for all 11 DSA layers. Its second graph also includes local expert
and shared MLP execution in the seven CUDA-owned DSA layers. The four cold
layers produce routes and dispatch outside capture. Cache append, sparse pool
updates and selection, variable-length attention, masking, and length mutation
remain eager. Multi-row prefill uses the original path. The arithmetic keeps
its original shapes, weights, reductions, and precision.

Both `direct_dma.cuda_hot_graphs` and `direct_dma.cuda_dsa_graphs` default to
false. DSA capture also requires `cuda_decode_graphs`. Graph buffers stay owned
until completed consumers drain; they are released before native CUDA banks
are closed. Imported DMA aliases retain the qualified synchronization and
mapping-release protocol.

## Profiling

The updated profiler accepts direct DMA and decodes its 12-byte control tags.
It joins parent observations and worker receipts by rank, sequence, layer, and
row count. Profiling adds no activation or routing tensor readback. Direct-mode
route counts are unknown, rather than estimated from a different workload.

The matched 2,048-token baseline profile measured a mean 92.70 ms per decode
forward. Disjoint parent host partitions include 40.03 ms waiting for replies,
7.36 ms in hot/shared submission and reply merging, 5.04 ms in input readiness,
12.64 ms in the middle backbone, 13.30 ms in the prefix, and 9.96 ms in the
suffix. The four eager middle DSA layers each span about 3 ms on the CUDA
stream; captured KDA layers span about 0.34 ms each.

Worker wall-time sums of per-layer rank maxima average 38.59 ms in native
expert execution and 45.41 ms across all worker phases. These overlap CUDA and
parent waits. They must not be added to the parent latency waterfall. This
profile is a baseline observation, not a profile of the final candidate.

## Validation

Startup checks real resident banks, serialized/chunked parity, three complete
64-step repeat traces, and every full-vocabulary logit against the authenticated
socket reference. DSA capture adds an eager-versus-captured comparison on a
2,048-token prompt plus eight outputs. The gate compares all prefill and decode
logit hashes, exercises sparse indexer selection, and restores graph owners on
success or failure. It passed for all 11 captured DSA layers.

All 103 executed Python source files match the delivered source bytes; the
runtime source hashes and archive are included. The previous service and the
hot-only candidate both closed their workers normally, with unchanged poisoned
lifetime guards. The kernel transport was retained without rebuild or reset.

The sidecard unit suite passes 755 tests in both the source and fork trees.
Scoped manual hooks pass. Full `format.sh ci` ran in isolated source and fork
worktrees and reports the same seven existing
failed hooks: Ruff check, Ruff format, codespell, typos, clang-format,
markdownlint, and forbidden imports. Its automatic rewrites were not copied
into the delivery. Unit and CI logs are included as compressed receipts.

## Measurements and limits

`summary.json` records the three matched before, hot-only, and final API runs.
Each completes 2,048 uncached prompt tokens and 64 greedy outputs with EOS
ignored. Decode throughput excludes time to first text. Hot/shared capture
alone stays within the observed run-to-run variation; DSA capture provides the
measured improvement. The generated text is identical across those runs.

| Configuration | Decode median (tokens/s) | Median time to first text (seconds) |
| --- | ---: | ---: |
| Previous qualified DMA service | 10.56 | 27.25 |
| Hot/shared capture only | 10.72 | 27.45 |
| Hot/shared and DSA math capture | 12.42 | 26.92 |

The final decode range is 12.42–12.55 tokens/s. Observed token latency falls by
14.17 ms. The completed 8,192-token prompt plus eight outputs reaches 12.33
decode tokens/s, with 96.28 seconds to first text. Prefill improvement is not
established. All six API cases pass, covering concurrent isolation, streamed
equivalence, cancellation at both boundaries, a 32K output budget, and context
overflow rejection.
An arithmetic request after the 8K run also passes, verifying reset back to a
short request after sparse attention.

The serving endpoint is `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
The runtime is
`/srv/ai/storage/artifacts/sidecard-cuda-expert-opt-20261012/dsa-server`.
The architecture is shown in [design.png](design.png). Detailed measurements,
API checks, worker receipts, and capacity scope are in `summary.json` and
`validation.json`.

Near-capacity prefill completion, broad model quality, and automatic fault
recovery remain unqualified. Dynamic sparse attention and native cold experts
still offer optimization opportunities. The deterministic Ascend decode path
uses 17-row padding in the pinned v1140 helper; this iteration does not replace
that kernel. The earlier nine-row description was incorrect; both runtime
expert-repeat receipts record 17 rows.
