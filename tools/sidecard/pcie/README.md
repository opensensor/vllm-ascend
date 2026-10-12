# CUDA–Ascend PCIe bus library

The first backend moves bytes between dedicated CUDA allocations and Ascend
device allocations using native SDK streams and completion events. Each
Ascend worker imports GPU buffers through **CUDA IPC**. Two bounded slots
permit transfers to overlap without tensor payloads crossing a socket.
This is a standalone transport library; the GLM server still uses its qualified
socket transport. There is no measured GLM speedup from this library yet.

See the [hardware report](../../../artifacts/sidecard-pcie-bridge-20261012/README.md)
and [design PNG](../../../artifacts/sidecard-pcie-bridge-20261012/design.png).

## Qualified host-staged backend

On this 310P, registering arbitrary host memory through `aclrtHostRegister`
failed. Registering an ACL-allocated pinned buffer with CUDA also failed.
The implemented backend therefore uses separate native pinned allocations:

```text
CUDA allocation / imported IPC allocation
    -> CUDA D2H -> CUDA-pinned slot -> event complete
    -> local CPU memcpy -> ACL-pinned slot
    -> Ascend H2D -> event complete -> Ascend allocation
```

The return direction reverses this chain. Each direction has two PCIe DMA
legs and one local CPU payload copy. A socket can carry only descriptors,
invocation identities, and completion notifications. Buffer handles are
exchanged once; tensor bytes remain on the native transfer path.

The CUDA and Ascend accelerators are behind different host root complexes.
No cross-vendor peer-DMA path is established. CUDA peer APIs and ACL peer APIs
do not themselves provide a mixed-vendor transport. This backend makes no
claim of direct device-to-device DMA or shared registration.

## Ownership and asynchronous API

`Runtime` binds the installed CUDA or ACL C ABI through `ctypes`, with no
PyTorch dependency. Use one worker process per Ascend chip, one creating
thread, and fixed device contexts. This isolates Python framework versions.
Native runtime initialization is process-local; closing the library does not
reset devices or globally finalize runtimes.

```python
from tools.sidecard.pcie import Bus, Runtime

cuda = Runtime("cuda", "/usr/local/cuda/lib64/libcudart.so")
ascend = Runtime(
    "ascend", "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so", 0
)
source = cuda.import_memory(descriptor)  # Dedicated producer cudaMalloc buffer.
ready = cuda.import_event(ready_handle)  # Producer records before notifying us.
destination = ascend.allocate(264192)
bus = Bus(cuda, ascend, capacity=524288, slots=2)
ticket = bus.submit("cuda_to_ascend", source, destination, 264192, source_ready=ready)
while not ticket.done:
    bus.progress()  # Nonblocking event queries; no device-wide synchronize.
# Destination bytes are ready. Schedule the consumer before reusing its buffer.
bus.release(ticket)
bus.close()
ready.close()
source.close()
destination.close()
```

The assigned card is physical chips 2 and 3; workers must receive
`ASCEND_RT_VISIBLE_DEVICES=2,3`, making their local device IDs 0 and 1.
The benchmark explicitly enforces this isolation.

Tickets retain local allocation and event leases. Pending or completed but
unreleased tickets keep their slot. Closing leased allocations, re-recording
leased events, overlapping in-flight writes, incorrect runtime endpoints,
and out-of-bounds transfers fail. Independent read-only sources may be shared.
A native error poisons the bus and retains buffers: pending DMA may exist.
The owning worker should exit, rather than freeing or recycling uncertain
resources. Its supervisor can restart it after closing the failed channel.

Exporters must keep allocations and IPC events alive until importers finish
and acknowledge shutdown. The producer must not overwrite a submitted source
or re-record its event until its completion identity returns. Cross-process
leases are enforced by that control protocol, not by local Python references.
The reference benchmark uses monotonically increasing invocation IDs.

IPC export only accepts library-owned **dedicated allocations at offset zero**.
It deliberately does not export arbitrary PyTorch allocator pointers or VMM
suballocations. Future model integration must pack its three input tensors
into a dedicated device buffer and copy the returned F32 partial into its
owned CUDA result tensor. Consumer-stream dependencies and native MoE scratch
ownership must also be connected before enabling the backend in GLM.

`progress()` does not launch a background thread. Call it from the worker's
event loop while other slots or device work are in flight. Completion queries
may incur native driver overhead. The local host copy and native submission
calls take CPU time; no claim of zero CPU work is made.

## Reproduce the raw benchmark

With the GLM server running, this test uses only a few MiB per device. It
does not reload weights, stop the server, or change device settings.

```bash
python3 -m tools.sidecard.pcie.benchmark \
  --npu-environment /absolute/path/to/private/npu-environment.json \
  --npu-python /absolute/path/to/npu-venv/bin/python \
  --output /absolute/path/to/new/benchmark-directory \
  --iterations 50 --warmup 10
```

CUDA runs in the invoking environment; ACL workers receive the private
existing NPU environment without printing its contents. Both chip ranks are
tested separately. Each fixture tests one or two simultaneous buffers and
compares with sequential socket payloads through driver-pinned host buffers.
The clock covers control notification, device-to-host-to-device transfer in
both directions, and completion notification. Producer upload and byte
verification are outside the timed interval. Input readiness is completed
before each timed invocation; this benchmark does not establish overlap with
a live CUDA producer kernel or native MoE execution.

Payload sizes cover an 8,256-byte expert input row, 16,384-byte output row,
their 32-row chunks, and 1 MiB. Each individual test returns the same byte
count as it sends; it does not simulate the asymmetric input/output sizes
of a full GLM expert invocation. Every warmup and timed buffer is verified.

Portable ownership and failure tests:

```bash
pytest --noconftest -q tests/ut/sidecard/test_pcie_bus.py
```

## Next integration gates

Connect dedicated packed GPU inputs and per-chip output buffers to the cold
dispatcher, then connect source-ready events and native consumer completion.
Preserve original expert IDs, weights, fixed reduction order, and all existing
placement and kernel authentication. Qualify byte-exact dispatch, full-vocab
teacher forcing, repeated decode, 65K context, cancellation, and queued-request
isolation before switching the endpoint. Reprofile the full model: improving
raw transport alone does not remove the measured native expert and CUDA
backbone launch costs.
