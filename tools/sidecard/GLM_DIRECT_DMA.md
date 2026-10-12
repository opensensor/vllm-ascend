# GLM with direct CUDA ↔ Ascend DMA

The `cuda-backbone-ascend-experts` runtime supports `expert_transport` set to
`direct-dma`. The CUDA coordinator runs all 45 backbone layers, caches, routers,
shared experts, full CUDA expert banks outside layers 18–32, and the configured
hot experts within those layers. Physical Ascend chips 2 and 3 execute their
independent resident cold expert banks. HCCL is not involved.

## Payload and control

Each privileged expert worker owns dedicated 2 MiB CUDA input/output allocations,
their NVIDIA DMA-BUF exports, a private 2 MiB Torch HBM input slot, and an
unrelated poisoned ACL host lifetime guard. The coordinator imports the CUDA
allocations through CUDA IPC and borrows them as Torch tensors. The CUDA
allocator does not own or free these slots.

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

The optional `direct_dma` specification selects three independent optimizations:

```json
{
  "direct_dma": {
    "cuda_pack_graphs": true,
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

Persistent GPU mappings use kernel UAPI version 5. Rebuild and load the matching
module before launching this Python version. Each device fd retains at most
two static DMA-BUF mappings, identified by exporter object and physical chip.
The Python peer holds original GPU allocation leases until clean fd close.
HBM lifetime descriptors and physical address queries remain per transfer;
uncertain DMA retains all cached mappings and the descriptor.

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
