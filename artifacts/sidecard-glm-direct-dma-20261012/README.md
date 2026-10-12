# GLM over direct CUDA ↔ Ascend DMA

**GLM is running end to end through the direct transport** at
`http://192.168.53.187:8000/v1`, model `glm-sidecard`. The
[live receipt](live-attestation.json) reports a healthy 65,536-token endpoint,
`expert_transport: direct-dma`, and two active bridge references for the two
assigned Ascend workers. Physical chips 0,1 remain available for other work.

Use the [integration runbook](../../tools/sidecard/GLM_DIRECT_DMA.md) and
[design PNG](design.png). The final tested runtime is
`/srv/ai/storage/artifacts/sidecard-glm-direct-runtime-v2`.

## Architecture and ownership

CUDA retains all 45 backbone layers, caches, routing, shared experts, complete
expert banks outside layers 18–32, and 22 hot experts per middle layer. Two
independent Ascend workers retain cold banks. No HCCL or weight migration is
introduced. Original route weights, activation clipping, and fixed owner
reduction order are preserved.

Each privileged worker owns dedicated CUDA input/output allocations and their
NVIDIA DMA-BUF exports. The ordinary CUDA coordinator shares them through
CUDA IPC memory and borrowed Torch views. After packing on GPU, it completes
a source event before sending a 12-byte sequence/layer/row identity. Ascend
initiates DMA directly into the worker's Torch HBM input buffer, computes
native experts, and initiates the return DMA directly from the native FP32
output tensor. A matching completion follows CUDA visibility flushing.
Both cold workers can overlap CUDA hot experts and the shared MLP.

Hidden activations and expert outputs are never staged in host DRAM along this
path. Two rank-admission flags reach the CPU. A worker reads back only bounded
routing metadata to validate the DMA-delivered expert IDs and weights; native
kernels consume the original HBM views. The socket carries control and initial
ownership descriptors. CUDA IPC event import returned status 205 across this
privilege boundary; completed local source events provide the qualified order.

## Completed hardware gates

- The [real-weight dispatch gate](dispatch-gate.json) passes **64 cases** across
  two layers, hot-only, cold-only, mixed owners, and row counts 1,4,17,32.
  Repeated outputs are bitwise equal and match the frozen CPU quantizer within
  its existing tolerances. Both [dispatch worker receipts](dispatch-dma-worker-rank0.json.gz)
  and [second worker receipt](dispatch-dma-worker-rank1.json.gz) record clean
  shutdown, 128 direct transfers each, and unchanged poisoned lifetime guards.
- [Transport parity](transport-parity.json) checks all **154,880 logits** against
  the authenticated socket reference for the prompt and 63 subsequent forced
  decode steps. Full vectors and greedy token IDs are bitwise equal.
  [Three repeated 64-step traces](decode-repeat-gate.json) also agree.
- [API validation](api-validation.json) passes concurrent request isolation,
  streaming equivalence and usage, cancellation before prefill and during
  decode, the 32,000-token output budget, and over-context rejection.
  Following cancellation, an isolated request completed in **0.84 s** and
  **0.91 s**, respectively, with unchanged output.
- [Model validation](model-validation.json) completes all five arithmetic
  prompts, three repeated 64-token completions, the matched 2K/64 workload,
  and **8,192 prompt tokens plus eight output tokens**. The 8K completion
  takes 88.52 s including prefill. General long-context quality and a complete
  65K prefill are not established. The near-capacity request is a cancellation
  gate, not a completed capacity claim.

At the live snapshot the two workers report **50,944 completed direct DMA
transfers**: [rank 0](dma-worker-rank0.json.gz), [rank 1](dma-worker-rank1.json.gz).
Their per-call receipts validate native completion identity and geometry;
serving does not checksum every payload. Numeric and full-model evidence
comes from the separate gates above. The initial full-model DMA deployment
also shut down cleanly with unchanged guards:
[rank 0](initial-dma-worker-rank0.json.gz), [rank 1](initial-dma-worker-rank1.json.gz).
No vendor-driver replacement, card reset, BAR resize, topology/IOMMU/ACS change,
or clock/power/ECC change was made.

## Matched performance

All measurements complete the same uncached 2,048-token prompt and 64 output
tokens with EOS ignored. Decode excludes prompt processing. These are individual
completed runs, not a statistical throughput study.

| Transport | Decode tokens/s | Time to first text |
| --- | ---: | ---: |
| [Saved socket baseline](socket-baseline-decode2048.json) | 9.64 | 27.06 s |
| [Initial DMA integration](initial-decode2048.json) | 8.15 | 27.86 s |
| [Final direct Torch input/output path](model-validation-decode2048.json) | 8.58 | 27.69 s |

The final path improves on the first DMA integration but remains **10.9% slower**
than the saved socket baseline. No model speedup is claimed. Removing activation
and result staging proves the capability; it does not remove native expert
execution, CUDA submission, mapping/lifetime setup, and completion costs.

In the bounded final worker window, median one-row upload is 0.22–0.24 ms,
routing-metadata readback/validation is approximately 0.35 ms, native expert
execution approximately 2.5 ms, and return DMA approximately 0.20 ms. These
worker spans overlap CUDA and the other worker and must not be added to a
model latency waterfall. Persistent registration, asynchronous queues, and
CPU/native launch overhead are further optimization targets.

## Reproduction and limits

The [runtime specification](runtime.json) retains paths to existing private SDK
and CUDA environment files; their contents are not copied or published. The
[source manifest](source-sha256.json), [executed source archive](executed-source.tar.gz),
and [parity receipt](execution-source-parity.json) establish byte-identical
published sources for all **97 source files in the frozen runtime tree**. The unchanged native
bridge binary matches the prior [hardware attestation](../sidecard-direct-dma-20261012/module-attestation.json).

All **691 sidecard unit tests pass**, including 48 integration tests, and all
scoped manual hooks pass: [unit log](unit-tests.log.gz), [scoped log](scoped-ci.log.gz).
The required repository-wide `bash format.sh ci` retains the same seven failing
hooks as the prior committed baseline: [full log](full-ci.log.gz),
[validation receipt](validation.json). Repository-wide checks are not reported
as passing.

A clean release drains CUDA consumers, acknowledges native quiescence, closes
IPC imports, and then releases the original worker allocations. Lost release
acknowledgement or uncertain DMA fails closed and retains the original owners;
automatic fault recovery is unqualified. Requests serialize and native DMA
ioctls block. Only the tested host, kernel/driver stack, and assigned chips are
qualified. The service remains on the direct backend so further optimization
can measure the actual integrated path.
