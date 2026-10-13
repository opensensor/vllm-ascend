# DMA-driven handoff prototype

The CUDA half of a device-driven handoff is hardware-qualified on the RTX
PRO 6000 Blackwell and both 310P chips on the assigned Ascend card. A CUDA
stream waits for a future 64-bit completion value, then runs its already
queued consumer. Ascend DMA writes the payload, completes that transfer,
then writes the completion value. Two independent slots can finish in
reverse submission order. No socket command or host flush call releases the
CUDA consumer in the qualified owner-context probe.

**DMA submission and Ascend execution remain host-driven.** The optional
[GLM completion integration](../GLM_COMPLETION.md) now registers
coordinator-owned results and prequeues rank-ordered consumers behind DMA flags.
The default transport retains its CUDA IPC output views and flush calls. Both
paths use blocking DMA ioctls and socket commands. Neither submits asynchronous
DMA or makes an arbitrary running GPU polling kernel coherent.

See the [hardware evidence](../../../artifacts/sidecard-device-contract-20261012/README.md)
and [design PNG](../../../artifacts/sidecard-device-contract-20261012/design.png).

## Completion ordering and ownership

1. Allocate every slot, export, stream, and completion event before queuing
   a future wait. Preload the consumer with `cudaFuncGetAttributes`.
2. Register each owned, aligned CUDA completion allocation with
   `CudaCompletionWait`. Imported IPC aliases are rejected.
3. Queue `cuStreamWaitValue64_v2` with equality on the next nonzero sequence.
   Queue all consumers, then seal their private completion event.
4. Complete payload DMA, then descriptor DMA with an unready completion word.
   Submit the final completion-word DMA separately. This establishes ordering
   without assuming byte order within a DMA job.
5. Reuse the slot only after the sealed consumer event completes and its
   ticket is released. Increasing sequences prevent stale completion reuse.

The RTX reports 64-bit stream memory operations and native RDMA write ordering
for the allocation's owner context. The no-flush probe requires those
capabilities and checks that every exported allocation belongs to that context.
Imported output aliases retain the legacy flush path. The optional GLM
integration uses separately qualified coordinator-owned results and controls,
with foreign DMA-BUF registration rather than imported result views. See NVIDIA's [GPUDirect RDMA ordering documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/).

The first probe stalled because a consumer launch lazily loaded its kernel
behind an unsatisfied stream wait. Preloading removes that initialization from
the waiting phase. NVIDIA documents this hazard in
[CUDA Lazy Loading](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/lazy-loading.html).
No device reset or driver unload was used for recovery; the evidence retains
the failed trial and its explicit completion/quiescence audit.

Any uncertain DMA or pending consumer retains its original allocations,
events, descriptors, and device fd. A timeout does not authorize releasing
those owners. Closing a queue with active or uncertain tickets fails.

## Invocation record

`Invocation` provides a 64-byte little-endian codec. The hardware probe
transfers it into CUDA memory and validates it in the queued CUDA consumer.
The probe rejects stale epochs/sequences, foreign ranks/layers, changed
geometry, and reserved fields. An Ascend device dispatcher remains future work.

| Field | Bytes | Meaning |
| --- | --- | --- |
| Magic | 8 | `SCMAIL01` |
| Epoch, sequence | 8 each | Nonzero session and monotonic invocation identity |
| Layer, rows, slot, rank | 4 each | Registered cold layer and bounded endpoint |
| Input, output bytes | 8 each | Derived from rows and the existing GLM wire geometry |
| Reserved | 8 | Must be zero |

Registered endpoints determine addresses; records do not carry arbitrary
pointers. Decode rejects a foreign rank, old epoch/sequence, invalid slot,
changed geometry, or nonzero reserved field. The target dispatcher must also
enforce slot ownership on device before executing work or publishing completion.

## Ascend dispatcher remains future work

Both assigned 310P chips return `ACL_ERROR_RT_FEATURE_NOT_SUPPORT` (207000)
for `aclrtValueWait` and `aclrtValueWrite`, even for an already-satisfied wait.
Their presence in the CANN headers does not establish device support.
Ascend therefore needs a separately qualified dispatcher or supported native
queue mechanism. A persistent polling kernel cannot be assumed to coexist
with the current Cube kernels or provide ordered DMA visibility.

The [GLM integration](../GLM_COMPLETION.md) qualifies coordinator-owned CUDA
result slots and completion words, registered once with the Ascend workers.
CUDA hot/shared work is scheduled while the Ascend workers compute cold experts;
ordered consumers follow the hot result on the CUDA stream. The existing FP32
rank addition order is preserved. The next boundary is replacing Ascend host
dispatch and blocking DMA submission with a qualified device queue. This requires measured scheduling, memory visibility,
and failure recovery; the CUDA probe alone does not prove that full path.

## Reproduce the bounded probe

Compile `mailbox_consumer.cu` with the installed CUDA compiler:

```bash
nvcc -shared -Xcompiler -fPIC --cudart shared -arch=sm_120 \
  -o libmailbox_consumer.so tools/sidecard/pcie/mailbox_consumer.cu
```

Use an authenticated runtime configuration with `mailbox_consumer_library`,
its `mailbox_consumer_sha256`, and `mailbox_native_owner_ordering: true`.
The existing private runtime environments remain on the hardware host.
Run from the frozen source directory using the existing NPU Python:

```bash
sudo -n /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  -m tools.sidecard.pcie.mailbox_probe --bootstrap \
  --config /path/to/runtime.json --rank 0 --output /path/to/rank0.json
```

Repeat for rank 1. The bootstrap restricts physical chips to 2 and 3.
Twelve successful waves exercise two slots at 8,256, 16,384, 264,192, and
524,288 bytes. Six additional waves test invalid records. Each wave completes
the second slot first. For accepted records, the consumer XORs each byte to
prove that queued device work consumed the transferred payload; rejected
records produce a fixed rejection marker without consuming the payload.
