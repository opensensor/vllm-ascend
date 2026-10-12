# Complete expert-sidecard latency profile

This profiles the **new CUDA-backbone / Ascend-cold-expert implementation**,
rather than the historical TP2 Ascend backbone. Both chips of the assigned
card execute resident cold experts, while all model state stays on RTX.

The diagnostic passes a 64-step full-vocabulary byte comparison against
production and reproduces the exact 2K completion text. The verified
serving source was automatically restored afterward into
`/srv/ai/storage/artifacts/sidecard-expert-runtime-20261012-v3`.
It is healthy at `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
Its fresh [2K/64 confirmation](restored-decode2048.json) completes at
**9.64 decode tokens/s**, with **27.06 seconds to first text**.
The previous full 65K capacity and operational gates remain in the
[implementation report](../sidecard-expert-runtime-20261012/README.md).

See the [latency PNG](latency.png) and [complete summary](summary.json).

## Where decode time adds up

The 2K/64 diagnostic records 63 one-token model forwards after prefill.
These **mean host partitions are disjoint** and sum to 108.21 ms per step:

| Host partition | Mean ms/step | Meaning |
| --- | ---: | --- |
| Blocked receiving Ascend replies | 48.65 | GPU hot/shared work can execute during this wait |
| Issuing CUDA backbone work | 37.52 | Prefix, middle preparation, and suffix Python/launch spans |
| Issuing hot/shared and reply/merge work | 13.96 | Includes final middle-layer output submission |
| Preparing host inputs and sending sockets | 5.72 | Includes input readiness; socket sends alone are 0.38 ms |
| Embedding, head, final completion | 2.35 | Remaining forward span, including observer bookkeeping |
| **Total** | **108.21** | Instrumented model forward; excludes sampling/tokenization |

Socket reception accounts for about 45% of the instrumented forward time.
This is not pure NPU compute: GPU work overlaps it, and device transfers,
worker dispatch, and reply arrival affect its duration.

The worker event data independently measures stream spans. Summing the
larger active rank duration at each of the 15 middle layers gives mean
native-expert sections of **44.48 ms/step**, input sections of **3.51 ms**,
and output sections of **4.27 ms**. Taking each layer's larger complete
rank span gives **52.00 ms/step**. These are overlapping device observations,
not extra terms to add to the host table; launch/idle gaps are included.

CUDA backbone stream observations are 14.38 ms for prefix, 16.89 ms for
middle preparation, and 10.81 ms for suffix. CPU submission is close to
these spans. Within the middle block, captured KDA preparation averages
about **0.34 ms/layer**, while eager DSA preparation averages **3.28 ms**
(range 3.06–3.38). Preparation includes mHC and routing, so this is not an
attention-only comparison. This identifies eager DSA preparation and
its surrounding mHC/router launches as a substantial optimization target.

The actual new-runtime route observation has **96.12% cold routes** on the
2K decode fixture and **29.78 chip calls per token**. The short Python
fixture has 83.58% cold routes and 29.48 chip calls; its instrumented model
forward averages 85.67 ms. The static hot set rarely removes an entire
layer's cold dependency.

The diagnostic itself generates at 9.22 tokens/s versus restored
production's 9.64, approximately 4% tracing cost. Device-event recording,
Python timing, route observation, and event queries contribute. No extra
CUDA or NPU synchronization barriers are added. Do not present diagnostic
throughput as an optimization or subtract component spans to predict a
future token rate.

## Communication and available parallelism

The current bridge uses **private Unix-domain sockets and pinned host RAM**.
There is no configured direct GPU-to-NPU DMA, NCCL/HCCL mixed group, or
network transport. One payload carries FP16 x, INT32 global IDs, and FP32
cold weights: 8,256 bytes per row per active chip. The reply is one weighted
FP32 partial: 16,384 bytes per row per chip. A prompt chunk has at most
32 rows. Bytes cross GPU/host and NPU/host boundaries separately.

Both chips are dispatched before either reply is awaited. GPU hot experts
and the shared MLP launch during those cold calls. These independent terms
already overlap. Replies copy back to CUDA asynchronously and merge in
fixed GPU/rank-0/rank-1 order. Successive layers of a single decode token
need the preceding result; future tokens additionally need sampling.

The parent currently waits for CUDA input staging with a device-wide
synchronize. Each worker performs three blocking input `.to()` calls,
then queues output copy and synchronizes before replying. Those boundaries
are opportunities for event-driven execution; replacing the socket alone
would not remove native expert or CUDA launch costs.

Potential improvements, **not implemented in this profiling change**:

- Pack each worker input into one device transfer with typed views, rather
  than three separate transfers. Use copy streams and readiness events.
- Use bounded double buffers for input, output, and native scratch. Protect
  every buffer until its consuming stream finishes. Separate asynchronous
  readers and writers to avoid blocking send/reply cycles.
- Pipeline prefill chunks as a wave through layers: RTX can prepare a later
  chunk's earlier layers while Ascend handles an earlier chunk's experts.
  Maintain token order in every KDA/DSA cache and separate deferred state.
- Capture CUDA mHC/router and hot/shared/merge spans around eager DSA and
  cold IPC. Dynamic DSA/indexer updates can stay eager. The profile supports
  prioritizing this launch reduction alongside native expert optimization.
- Batch independent requests/expert rows to amortize launches and copies.
  This requires per-request cache ownership and a scheduler; the current
  endpoint serializes requests and does not implement batching.

Overlap must respect actual dependencies. Shared buffers, mutable native
scratch, and recurrent state cannot be reused concurrently without ownership
and completion events. The current 15-layer decode chain retains a cold
latency floor even with perfect transfer overlap.

## Reproduce and provenance

A private profile snapshot sets `expert_worker_profile: true` in its
specification. Use the same placement, kernels, graph flags, and isolated
Python environments as the qualified service. Stop only its exact owned
processes before loading this second full model, and arrange restoration
before starting. Then run:

```bash
python3 -m tools.sidecard.expert_profile \
  --config /absolute/path/to/profile/runtime.json \
  --reference /absolute/path/to/production/decode-repeat-gate.json
python3 -m tools.sidecard.expert_profile_summary \
  --root artifacts/sidecard-expert-profile-20261012 \
  --output /absolute/path/to/summary.json
```

Workers record device events at existing upload/native/download boundaries
and read durations after the existing synchronization. Parent socket
wrappers observe exact invocation headers and fragmented receives.
The summary rejects missing/duplicate workers, incomplete layer traces,
non-accounting host spans, and failed vocabulary parity. **613 portable
sidecard tests pass**, including these failure modes and parallel-span
accounting. Hardware evidence is separate from those portable tests.

The [exact executed archive](executed-source.tar.gz) matches every byte in
[source-sha256.json](source-sha256.json). The published driver additionally
closes workers when rejecting an incompatible graph configuration; its
qualified execution AST otherwise matches, as recorded in
[execution-source-parity.json](execution-source-parity.json).
Raw JSONL receipts are compressed without editing; `profile-run.json`
derives model text hashes from [raw output](profile-run.json.gz).
The restoration result is in
[supervisor-receipt.json](supervisor-receipt.json).

The diagram script is [draw_latency.py.txt](draw_latency.py.txt).
NPU per-phase maxima are observations, not sums of parallel rank times.
General model accuracy, context recall, batching speed, and the proposed
asynchronous improvements are not established by this profile.

The subsequent [PCIe bus qualification](../sidecard-pcie-bridge-20261012/README.md)
implements and hardware-tests a separate asynchronous transport library.
The GLM endpoint still uses the socket path profiled here.
