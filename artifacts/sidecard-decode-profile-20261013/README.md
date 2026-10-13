# GLM sidecard decode attribution after 16 MiB prefill

This study profiles the qualified 1024-row cold window with 32-row CUDA math
and direct cross-vendor DMA. One RTX PRO 6000 and physical Ascend chips 2 and 3
on one Atlas 300I Duo perform inference. The other Ascend card remains outside
this runtime. Profiling changes observation, not model placement or arithmetic.

## Method

`tools.sidecard.expert_profile --decode-only` disables stage observation during
cold prefill, keeping the original cold-window geometry. It then records every
single-token forward, CUDA stream event pair, disjoint parent host span, and
exact private worker invocation. The final authenticated full-vocabulary
transport reference must match before workloads begin.

The 2K, 8K, and complete 23,820-token Code request reset model state, disable
prefix snapshots, and require zero cached prompt tokens. Their prompt hashes
and all 64 full-vocabulary output hashes must match the authenticated 16 MiB
benchmark. No prompt or tools are shortened. Logit observers and event-duration
reads occur outside the timed forward, following its existing completion sync.
No new device barriers are inserted between observed stages.

A fourth 2K workload uses 32 outputs with Nsight Systems capture enabled only
for its 31 decode forwards. CUDA graph node tracing exposes individual kernels.
Its additional overhead makes those timings unsuitable for production speed
claims. The three preceding workloads run with kernel activity capture off;
coarse observers still impose overhead. Production throughput is measured
separately after restoring the qualified service.

Worker records join by rank, sequence, layer, and row count. The summary rejects
missing or duplicate invocations, incomplete layer coverage, invalid host
partitions, missing decode steps, and failed numerical parity. CUDA stream spans
include launch and idle gaps, and overlap worker execution and parent waits.
Neither device spans nor rank maxima can be added to the parent waterfall.

The private captured request, environment files, and generated text are excluded
from published receipts. The full Nsight trace stays on the hardware host; the
public report records its checksum and small derived summaries.

## Measured decode costs

Each primary workload contains 63 observed single-token forwards after cold
prefill. The transport reference and all workload vocabularies match bitwise.
These are instrumented forward timings, not serving throughput measurements.

| Prompt | Mean forward | Waiting for replies | Input readiness | Backbone host submission |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 89.05 ms | 46.35 ms | 6.33 ms | 27.51 ms |
| 8,192 tokens | 88.56 ms | 46.06 ms | 6.33 ms | 27.40 ms |
| 23,820 tokens | 83.51 ms | 41.88 ms | 6.27 ms | 26.61 ms |

Backbone host submission combines prefix, middle backbone, and suffix. It
includes submission and any blocking inside those operations, so it is not a
measurement of pure CPU computation. Reply waiting accounts for 50–52% of the
observed forward. The fuller Code prompt decodes faster in this run despite
its context length; token-dependent expert work differs between fixtures.
Do not interpret this as a causal benefit from longer context.

The sum of each layer's maximum native worker span is 38.46, 38.07, and 33.87 ms
for the three fixtures. Across all worker phases it is 48.77, 48.30, and
44.01 ms. At 2K, upload contributes 3.55 ms, metadata validation 3.87 ms, and
return/completion publication 3.41 ms as separate rank-max stage sums. These
host spans include launches, DMA ioctl completion, and synchronization. Maxima
for individual stages can come from different ranks. They are attribution
indicators, not an additive critical-path reconstruction.

The four middle DSA backbone layers span about 2.3–2.5 ms each on the CUDA
stream; the 11 KDA layers span about 0.29–0.42 ms. The eager sparse indexer,
cache updates, selection, masking, and attention remain outside the existing
DSA graphs. Fixed projection and mHC portions are already captured.

See [complete host and worker summary](summary.json),
[raw observations](probe/parent-profile.jsonl.gz), and
[design and cost PNG](design.png).

## CUDA node trace

The separate 31-forward capture records 210,696 CUDA kernels, about 6,797 per
token. Runtime calls include roughly 1,498 eager `cudaLaunchKernel` calls,
72 `cudaGraphLaunch` calls, and 326 `cudaMemcpyAsync` calls per token. Captured
graph nodes account for many kernels without individual host launches.

Kernel execution totals 21.59 ms per token. The union of kernels, memcpy, and
memset covers 21.96 ms per token within a 107.39 ms-per-token kernel timeline
extent. CUDA is active about 20% of that traced interval. This is a heavily
observed run, not an estimate of uninstrumented utilization. Nsight increases
mean observed forward time from 89.05 to 106.24 ms at 2K; generation throughput
also includes event queries, full-vocabulary hashes, and other observer work.

The largest kernel families include FP16 cuBLAS GEMV, WMMA GEMM, and resident
quantized expert projections. The top GEMV variant takes 4.55 ms per token;
the top WMMA variant takes 2.58 ms. No single CUDA kernel dominates the entire
forward. Many small operations and host submission gaps remain.

The trace confirms 15 two-byte CUDA-to-host admission flag reads per token.
Direct Ascend DMA is outside CUPTI memcpy accounting; missing PCIe DMA events
in this trace do not imply host-staged activations. The average 2K step calls
29.68 chip invocations. At 8,320 input bytes, 16,384 result bytes, and an
8-byte completion publication per invocation, payload volume is about
0.70 MiB per token. Decode is dominated by short-call latency and expert work,
rather than the size of the available 16 MiB slot.

[CUDA attribution](cuda-attribution.json), kernel/API CSV summaries, and
[trace identity](trace-metadata.json) are published. The 8.1 MiB Nsight report
and SQLite export remain in the diagnostic directory on the hardware host.
OS runtime totals aggregate multiple threads and must not be treated as a
single-token latency waterfall.

## Opportunities, in order

1. Profile the Ascend native graph internally, then build a deterministic
   single-row cold expert reducer. The current implementation pads one decode
   row to 17 rows to select fixed-slot FP32 reduction and avoid nondeterministic
   atomics. A specialized path should preserve that reduction order while
   reducing padding, scratch setup, and launch overhead. Padding does not imply
   17 times the useful arithmetic. This pass measures the native call boundary,
   not individual Ascend kernels, so its internal breakdown remains unresolved.
2. Reduce eager CUDA DSA submission. A fused Hadamard butterfly for the sparse
   indexer can retain the existing FP32 operation order and BF16/FP16 rounding
   while replacing repeated add/subtract/stack operations. Further capture must
   handle changing pool counts, tail lengths, and top-k boundaries explicitly.
   Full-vocabulary parity is required before any deployment.
3. Reduce dispatch administration: the two-byte route readback, source-event
   readiness, metadata validation, and host acknowledgments. Keep invocation
   authentication, completion ordering, and owner lifetime checks. Simply
   removing socket waits would not remove the Ascend compute dependency; CUDA
   already waits on DMA-published completion words.

These are evidence-backed targets, not implemented speedups. Increasing decode
slot capacity or bulk bandwidth is lower priority than reducing per-layer work
and submission latency. No checkpoint, native kernel, placement, or math changed
in this profiling study.

## Reproduction

Use an isolated source snapshot and the same qualified native consumers and
weights. Stop the serving parent and verify both workers have closed, their
poisoned guards remain exact, and the bridge refcount is zero before loading
the diagnostic model. Set `prompt_prefix_cache` to false in its runtime config.

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --wait=all --output /ARTIFACT/cuda-decode \
  /CUDA_VENV/bin/python -m tools.sidecard.expert_profile \
  --config /ARTIFACT/runtime.json --reference /QUALIFIED/decode-repeat-gate.json \
  --decode-only --chat-request /PRIVATE/full-code-request.json \
  --benchmark-reference /QUALIFIED/cold-benchmark.json \
  --benchmark-sha256 AUTHENTICATED_BENCHMARK_HASH --capture-cuda
```

After normal diagnostic closure, verify worker guards and mapping release before
restoring the qualified 16 MiB service. Run the startup gates and a fresh 2K API
request to check readiness and production decode performance.

## Restoration and validation

The diagnostic process and both Ascend workers closed normally with exact
poisoned guards and no remaining bridge references. The qualified 16 MiB
service restarted at `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
Every serving startup gate passed. A fresh 2,048-token API request completed
64 outputs with zero cached tokens, first text in 9.982 seconds, and decode at
12.01 tokens/s. This is one serving measurement, consistent with the previous
11.74 tokens/s result; no decode acceleration is claimed.

[API receipt](api-cold-2048.json) and [health](health.json) record the check.
All 1,273 standalone sidecard unit tests pass. Changed-file manual checks pass.
Full repository manual checks run in an isolated worktree and report the seven
existing failed hooks: Ruff check, Ruff format, codespell, typos, clang-format,
markdownlint, and forbidden imports. Unrelated automatic edits are discarded.

The [executed source archive](executed-source.tar.gz) freezes the diagnostic
sidecard snapshot. Native helper sources remain frozen from the qualified
service and their retained identities are recorded separately. The delivered summary adds stricter decode coverage checks after
capture; that postprocessor does not execute in the model process. Private
runtime environment files and checkpoint weights are excluded.
