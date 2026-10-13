# GLM with direct CUDA ↔ Ascend DMA

The `cuda-backbone-ascend-experts` runtime supports `expert_transport` set to
`direct-dma`. The CUDA coordinator runs all 45 backbone layers, caches, routers,
shared experts, full CUDA expert banks outside layers 18–32, and the configured
hot experts within those layers. Physical Ascend chips 2 and 3 execute their
independent resident cold expert banks. HCCL is not involved.

## Payload and control

Each privileged expert worker owns dedicated CUDA input/output allocations,
their NVIDIA DMA-BUF exports, a private Torch HBM input slot, and an
unrelated poisoned ACL host lifetime guard. The coordinator imports the CUDA
allocations through CUDA IPC and borrows them as Torch tensors. The CUDA
allocator does not own or free these slots. Slots default to 2 MiB; cold
windows of 256, 512, and 1024 rows derive 4 MiB, 8 MiB, and 16 MiB slots. The
coordinator and workers bind descriptor sizes and guard lengths to that same
runtime geometry. Separate completion control leaves the entire result slot
available for FP32 rows. Older completion binaries fail admission before any
owner allocation or CUDA future wait.

The coordinator packs FP16 hidden rows, int32 expert IDs, and FP32 original route
weights into GPU memory. It synchronizes a source event on the current CUDA
stream before sending a 12-byte invocation identity. The worker's Ascend DMA
engine transfers that GPU payload directly into the Torch HBM input buffer.
The native expert output tensor supplies the return DMA source, retaining its
storage until completion. The weighted FP32 partial returns directly to the
worker-owned GPU output slot. The worker flushes CUDA
write visibility across contexts before returning the matching identity.

CUDA combines hot results and cold rank 0, then cold rank 1, in the same order
as the qualified socket runtime. Original route weights and clipping kernels
are unchanged. Both Ascend workers can run while CUDA computes hot and shared
experts. There is one reusable slot per worker; invocation and request locks
prevent scratch, cache, and slot reuse across outstanding work.

Hidden rows and expert outputs do not pass through host DRAM. Two rank-admission
flags reach the CPU. Each worker reads back only bounded route metadata for
validation; native kernels consume the DMA-delivered IDs and weights in HBM.
The socket carries
startup ownership descriptors and invocation/completion control. CUDA IPC
memory works across the worker privilege boundary on this host; CUDA IPC event
import returned status 205. The qualified path therefore completes a local
source event before socket admission and does not import CUDA IPC events.
The CUDA IPC memory mappings on both ends use the same configured native
runtime library; borrowed Torch views share those exact pointers.

## Launch

Build and load the [restricted kernel bridge](pcie/DIRECT_DMA.md). Keep its
source, kernel headers, and Ascend symbol versions matched. Use the same
authenticated checkpoint, placement, clipped native kernels, CUDA kernel
library, SDK environment, and Python interpreters as the existing expert
runtime. Add these fields to its specification:

```json
{
  "expert_transport": "direct-dma",
  "transport_reference": "/absolute/path/to/qualified/socket/decode-repeat-gate.json",
  "transport_reference_sha256": "SHA256_OF_THAT_FILE"
}
```

The ordinary CUDA process launches root expert workers through `sudo -n`.
Workers restore the existing private CANN environment before importing Torch
NPU, force visibility to physical 2,3, and authenticate their private Unix
socket peer PID/UID and session nonce. The API itself does not need root.
Never print or publish the private environment files.

```bash
python3 -m tools.sidecard.layer_runtime \
  --output /absolute/path/to/new-runtime \
  --specification /absolute/path/to/specification.json \
  --cuda-python /absolute/path/to/cuda-venv/bin/python \
  --port 8000
```

The snapshot includes the native Python transport package and records source
hashes. Startup authenticates resident weights and placement, runs real native
expert repeat gates, compares serialized/chunked execution, and performs three
complete 64-step teacher-forced traces. Direct mode also requires bitwise
equality of full-vocabulary logits and identical greedy tokens against the
authenticated socket reference. A real text generation completes before the
API exposes readiness. `/health` reports `expert_transport: direct-dma`.

## Ownership and shutdown

The coordinator waits for source completion before each request. That event
also covers earlier GPU consumers of the reusable output slot. Completed DMA
and matching control identities precede CUDA output consumption. Source events
are process-owned and explicitly record the calling ASGI thread's CUDA stream.

A clean close drains CUDA consumers, asks each worker to acknowledge native
quiescence, closes the coordinator's IPC mappings, and then sends a release
acknowledgement. Only then does each worker free its original GPU allocations,
HBM slots, DMA-BUF exports, and guard. Workers ignore ordinary termination while
inside this ownership protocol; the coordinator never automatically kills them.

Uncertain native DMA, malformed completion, or lost ownership acknowledgement
makes the runtime fail closed. A worker with uncertain DMA or unreleased IPC
ownership remains alive with its original CUDA/HBM allocations and writes a
`dma-quarantine-rank*.json` receipt. Do not force-kill, reset devices, or unload
the bridge as automatic recovery. Safe fault recovery is still a separate
unqualified feature. Normal serving and clean shutdown do not require resets.

Streaming disconnects cancel between complete model chunks or decode steps;
an in-flight expert invocation finishes before its slots can be reused. A new
request resets model caches under the same serialized generation lock.

## Reducing dispatch overhead

The optional `direct_dma` specification selects independent optimizations:

```json
{
  "direct_dma": {
    "cuda_pack_graphs": true,
    "cuda_hot_graphs": true,
    "cuda_dsa_graphs": true,
    "npu_expert_graphs": true,
    "persistent_gpu_mappings": true
  }
}
```

Each flag defaults to `false`. CUDA packing graphs own one-row scratch per
middle layer and pack both private input slots. Only active ranks receive
commands. Ascend expert graphs capture the existing padded fixed-slot reducer
for one-row decode; larger batches use the original eager path. External DMA,
source events, metadata validation, and write flushes remain outside capture.
Graph scratch is released after completed consumers and before its underlying
slots or imported mappings are freed.

CUDA hot graphs capture the existing local hot-expert kernels and shared MLP
together for each of the 15 middle layers. They require the stream-aware CUDA
expert ABI and are prepared after resident banks and DMA slots are ready.
Invocation first admits active Ascend workers, then copies one-row graph inputs
and replays the local work. Reply waiting, CUDA write flushes, and fixed-order
cold reduction remain outside capture. Multi-row prefill and expert calls
without a shared MLP use the eager path. All graphs share the existing native
CUDA expert scratch under the serialized dispatch contract.

With `cuda_decode_graphs` enabled, CUDA DSA graphs capture fixed one-row
projections and mHC math before and after attention. For CUDA-owned layers,
the second graph also captures the local experts and shared MLP. For cold
layers, it produces routes and leaves expert dispatch outside capture.
Cache append, variable-length dense attention, sparse indexer updates and
selection, causal masks, and cache-length mutation remain eager. Multi-row
prefill keeps the original path. This preserves the original tensor shapes
and reduction arithmetic without padding the attention window for capture.

Persistent GPU mappings use kernel UAPI version 5. Rebuild and load the matching
module before launching this Python version. Each device fd retains at most
two static DMA-BUF mappings, identified by exporter object and physical chip.
The Python peer holds original GPU allocation leases until clean fd close.
HBM lifetime descriptors and physical address queries remain per transfer;
uncertain DMA retains all cached mappings and the descriptor.

`cuda_prefill_conv` defaults to false. Enabling it batches the four-tap KDA
convolution across prefill rows on CUDA. Each window still accumulates in FP32,
rounds to FP16, and then applies SiLU; single-row decode keeps its existing path.
The persistent convolution history is updated in the same order.

`prefill_chunk_size` defaults to 32. Direct DMA also admits bounded candidates
of 64 or 128 rows. Both coordinator and workers validate the choice before
allocating scratch. The existing 2 MiB output slot holds exactly 128 FP32 rows;
the routing metadata allocation scales with the admitted chunk. Cancellation
is checked between completed chunks. Larger batches can change floating-point
results through CUDA projection or attention batch geometry, so admission is
not evidence of numerical qualification.

Before API readiness, either choice runs `prefill-parity.json`: original
32-row prefill versus the candidate on 2,048 prompt tokens and 64 subsequent
teacher-forced output steps. It compares all 154,880 logits bitwise at every
shared boundary and fails closed on a mismatch. For larger chunks, shared
boundaries are the final prefill result and every decode step. Smaller context
limits reduce the test prompt to leave room for the output budget. Consult the
measured prefill report before selecting a larger chunk.

The coordinator checks and enables `SYNC_MEMOPS` on each imported CUDA alias
before any payload use. Setting this attribute on the worker's original
allocation did not enable it on the imported aliases on the tested host.
The startup audit is `ipc-sync-memops.json`. This complements source completion
and returned-write flushes; it does not replace either barrier. NVIDIA describes
the required CUDA/RDMA ordering in its
[GPUDirect RDMA guide](https://docs.nvidia.com/cuda/gpudirect-rdma/).

Workers validate routing metadata with a bounded scalar CPU loop. Running
receipts update every 1,024 completed calls, while final close and quarantine
receipts are immediate. This avoids repeatedly serializing a full 4,096-record
timing window in the critical path every 64 calls.

## Validation receipts

`transport-parity.json`, `decode-repeat-gate.json`, `pipeline-parity.json`, and
`startup-inference.json` record startup gates. Each worker periodically records
completed direct transfers, byte counts, and a bounded window of transport and
native-expert timings in `dma-worker-rank*.json`. Its final close receipt checks
that the unrelated host lifetime guard remained poisoned.

Measure performance through completed OpenAI requests with
`tools.sidecard.decode_benchmark` and use the same fixed prompt/output budget
as the socket baseline. Raw PCIe latency alone does not determine model speed.
Long-context admission is distinct from completing a long prompt or evaluating
long-context model quality.

`tools.sidecard.expert_profile` also accepts a fresh direct-DMA snapshot. Run it
with the same private CUDA environment and an authenticated decode reference
while the serving process is stopped. It writes bounded parent observations and
closes both workers normally. `tools.sidecard.expert_profile_summary --root`
joins the final worker receipts to those observations by rank, sequence, layer,
and row count. CUDA event spans overlap host waits; DMA worker timings are host
wall spans including completion waits. Route counts remain unknown in direct
mode because profiling does not add routing tensor readback.

## Logical slots above the native transfer bound

`pcie.segmented` admits a 16 MiB logical slot while keeping each UAPI 5
registration and transfer within 8 MiB. NVIDIA exports two disjoint 8 MiB
regions of the dedicated contiguous CUDA allocation. Each region has a
separate private kernel endpoint, with at most two persistent GPU mappings
(input and result). The CUDA IPC descriptor still identifies the complete
16 MiB input allocation; result registration sends both region descriptors
before the separate control descriptor. All endpoints derive those counts
from the same admitted cold window and capacity.

The Ascend worker splits its contiguous HBM payload into bounded regions and
completes the corresponding DMA calls in order. A 1024-row FP32 result uses
two 8 MiB return transfers. Its FP16 input plus routing and invocation record
also crosses the 8 MiB boundary. The final 8-byte sequence publication occurs
only after the complete result payload finishes. CUDA consumes the original
contiguous result through one ordered merge and retains 32-row backbone math.

Region lease counts also retain the full owner allocation. An uncertain piece
retains its original HBM tensor, CUDA export, guard, source event, and kernel
endpoint; the worker and coordinator keep their owners alive. All region
mappings must close before any exported descriptor or full owner is released.
The lifetime guard covers the whole logical HBM slot and remains poisoned.
No kernel module replacement or driver configuration change is needed.

The [16 MiB study](../../artifacts/sidecard-prefill16-20261013/README.md) records
raw transfer qualification and the matched GLM comparison. This is a larger
logical batch delivered in bounded pieces, not a single 16 MiB ioctl.
