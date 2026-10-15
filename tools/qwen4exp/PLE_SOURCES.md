# PLE sources: shared host RAM and filesystem storage

The target is device-owned PLE row selection, transfer submission, and
completion, using four Ascend 310P NPUs. The RTX is outside this design.
CPU setup may register storage, pin memory, and admit requests. It must not
hash token IDs, gather rows, or submit a transfer for each steady-state lookup.

## Source selection

| Backend | Cold admission | Intended steady-state path | Current implementation |
| --- | --- | --- | --- |
| Shared host RAM, preferred when memory permits | Load one immutable table, share its FD across ranks, register and pin pages for DMA | Device row IDs → registered host pages → PCIe DMA → NPU row buffer | Sealed shared table and CPU reference reader; small native DMA gate passed on four NPUs |
| Filesystem storage | Register shard files and tensor offsets; admit bounded page-cache capacity | Device row IDs → device cache → storage miss requests → registered NPU buffers | File-relative page planner and filesystem-aware CPU reference reader |

The actual FP16 table contains 320,001,536 padded rows of 320 bytes:
102,400,491,520 bytes, or 95.367889 GiB. A token selects sixteen rows,
5,120 bytes. Sharing must be physical: four ranks must attach to one object,
not each allocate a 95-GiB table. The sample registration gate covers all
128 shards. The complete table has not been loaded or pinned by these tests.

`ple_sources.SharedRamSource.load` checks a caller-specified budget and reserve,
copies tensor payloads into one memfd in bounded chunks, checks source file
identity, and seals the result against writes and resizing. Reader processes
attach through an inherited FD or SCM_RIGHTS. Each attachment owns its own FD
and mapping; closing the creator does not invalidate another attachment.

Populating a memfd does **not** make it permanently resident or DMA-ready.
Native page pinning, IOMMU mappings, and transport leases are separate gates.
Keep the configured host reserve during admission and reject insufficient RAM
before publishing a partial table. The filesystem backend is an explicit
alternative; a failed or in-flight DMA must never silently change backends.

`ple_sources.FileSource` opens registered files read-only and executes the
same row mapping through `pread`. It validates open-file identity, bounds,
and read lengths. Its planner deduplicates page reads and preserves duplicate
rows and destination order. This is a correctness oracle, not autonomous SSD
I/O. The current reference lookup bound is 192 rows; larger prefill batches
need explicit chunking or a separately admitted larger window.

## Transfer and completion contract to implement

Both sources need the same bounded device request envelope: source generation,
request sequence, slot, row count, row IDs, and admitted destination range.
Use offsets into registered objects rather than arbitrary physical pointers.
The backend translates those offsets into its owned DMA mappings.

1. Device hashing selects the exact FP16 rows, including EOS/history behavior.
2. Each available slot owns a fixed output buffer and completion record.
3. The producer publishes descriptors before advancing the submission sequence.
4. The DMA backend writes all row bytes before publishing completion with the
   matching generation and sequence. Hardware ordering must be proven, not
   inferred from Python or C memory ordering.
5. The NPU consumer acquires completion before projection and acknowledges
   consumption before the producer can reuse that slot.
6. Drain pending DMA before unmapping or releasing pins. Errors retain leases
   until recovery establishes that no device still owns the memory.

Cold CPU admission and teardown remain valid. A C thread performing each
lookup or writing each DMA doorbell would still be host bookkeeping.

## Hardware evidence, 2026-10-14

`ple_host_dma_probe.cpp` builds against the installed ACL and driver headers.
It creates private device allocations and calls:

```text
drvMemConvertAddr(source, destination, bytes, descriptor)
halMemcpySumbit(descriptor, MEMCPY_SUMBIT_SYNC)
aclrtMemcpy(readback, ..., ACL_MEMCPY_DEVICE_TO_HOST)
drvMemDestroyAddr(descriptor)
```

On all four NPUs, shared writable mappings, shared read-only mappings,
sealed read-only memfds, and native ACL host allocations produced exact
320-byte copies across a host-page boundary. These calls are **host-submitted**.
They establish that registered host pages can supply DMA payloads; they do
not establish device-originated fetches or full-table registration capacity.

The real-weight gate loaded a 2.5-MiB sample, taking 64 rows from each of the
128 actual shards. Both sources matched 640 selected rows against independent
safetensors reads. All four native processes attached to the same memfd inode
and each verified five DMA transfers, including boundary, middle, last, and
duplicate row selections. No model-serving code was switched.

The corrected `aclrtGetMemcpyDescSize` query used
`ACL_MEMCPY_INNER_DEVICE_TO_DEVICE` (kind 6). Every NPU returned status 207000,
with no descriptor allocation. Earlier probes using kind 3 for
`aclrtMemcpyAsyncWithOffset` returned a parameter error and are not evidence
of hardware support or lack of support. The
[official offset-copy API documentation](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2011.html)
also restricts that API to inner-device copies and lists Atlas inference
products as unsupported.

Driver inspection identifies the missing submission step:
`pcie_host/devdrv_dma.c:devdrv_dma_ch_sq_submit` updates the channel SQ tail
through `devdrv_set_dma_sq_tail`. Prepared DMA descriptors alone do not ring
this doorbell. Device submission requires a channel owned by this transport,
device-accessible submission/completion resources, and proven ordering.
The live driver's channel cannot be borrowed by overwriting its ring indices.

## Device selector and driver extension

`ple_request.py`, `native_ple_request.cpp`, and `ple_request_integer.h` now
implement device-only row hashing and RAM byte offsets. The immutable geometry
is checked during admission. Token IDs, packed request boundaries, EOS, and
history remain on device during execution. Malformed dynamic inputs produce
failure receipts and invalid offsets; the future DMA consumer must check those
receipts before submission.

The native graph gate used actual checkpoint multipliers, vocabulary sizes,
and offsets. Across four NPUs it checked 452,992 exact row IDs and the same
number of exact byte offsets, including changed inputs during replay, EOS,
empty/ragged requests, and a 2,048-token prefill shape. All 72 malformed-input
checks passed. Decode-shape graph replay was approximately 26–31 microseconds;
2,048-token selector replay was approximately 228–229 microseconds. These are
isolated selector timings including graph dispatch/synchronization overhead,
not full PLE or model throughput.

A private custom AI CPU function also executed successfully. Its loaded
`drvMemConvertAddr` symbol returned success with an empty descriptor, and its
32 instruction bytes exactly matched the SDK stub. No device-side DMA was
submitted. Read-only inventory confirmed a separate ARM Linux 6.6 environment
with native custom-task libraries; the host runs x86 Linux 7.0.0-38.
The absence of `/dev/devmm_svm` in that process is not evidence that no device
memory driver exists: its mappings include `/dev/npu_device_cust`.

A separate host query, `halHostRegisterCapabilities`, returned status zero
and `HOST_MEM_MAP_NOT_SUPPORTED` for AI CPU, AICore, AIVector, PCIe DMA, and
SDMA on each of the four ranks. This queries direct host-memory mapping; it
does not contradict the successful host-submitted DMA copies above.
An indirect runtime D2D descriptor probe also failed in its private stream
with a firmware illegal-parameter error. That probe does not establish a
general hardware impossibility. Its V2 variant was not exercised.

The user authorized extending the driver fork. The working candidate is in
`/home/matteius/ascend-310p-host-driver`, documented in `DEVICE_DMA.md`.
The regular driver loads with every channel available. A model loader can
acquire, query, and release an FD-owned channel lease through the versioned
`/dev/ascend_device_dma` ioctl interface. There is no reservation module
parameter. Admission atomically requires a drained queue and preserves a host
traffic channel; a busy queue returns `EBUSY` for retry. Closing the FD releases
an idle lease, while reset invalidates old ownership. Shared host dispatch skips
leased channels. A fixed instance targeting one returns `EBUSY` to preserve
instance identity and ordering, so those callers need coordination.

All 31 bare-metal modules build and pass kernel-symbol/CRC verification against
7.0.0-38. Nine CPU tests pass, including dynamic ioctl lifecycle checks and
87,264 successful host dispatches. The local base also includes the installed
host's hrtimer setup fix from `453e106`, preserving that runtime correction.
`ple_driver.py` supplies a startup-only client with 16 additional passing tests.
The driver candidate is now installed on the hardware host. All 31 loaded
module source versions match the bundle, and the initramfs and DKMS source
tree were updated with rollback copies retained. Live reload hit a firmware
boot-state timeout; one authorized host reboot recovered all four NPUs.
Automatic lease acquisition remains disabled in serving.

Hardware tests passed simultaneous leases on all four NPUs, query, duplicate
owner rejection, stale-generation rejection, explicit release, reacquisition,
and last-close cleanup. Startup and final sysfs snapshots show no reservations.
Each lease leaves two host traffic channels available. Sixteen ordinary DMA
copies passed with reservations disabled. Acquiring reservations before ACL
context initialization exposed the expected fixed-instance conflict: a vendor
host copy returned `EBUSY` and ACL startup failed with 507033. All leases were
closed; transparently coexisting with fixed-instance clients remains unresolved.

Reboot also exposed a collective bootstrap bug: memory and Notify exports can
have identical key strings because their SDK namespaces are separate. The
collective adapters now reject aliases within each namespace, with 180 passing
unit tests. The four-rank service successfully booted with the corrected check.

The reservation currently reports `capabilities=0` / `device_submit=0`.
Device-visible queue mapping, producer lifetime, doorbell access, and completion
ordering remain implementation gates. A host fork extension does not by itself
replace the separate on-device HAL or kernel.

## SSD-specific requirements

File-relative offsets remain authoritative until a backend obtains an actual
filesystem mapping lease. A saved extent list or file metadata check does not
pin filesystem blocks. A raw NVMe backend also needs queue ownership, valid
DMA mappings, supported PCIe routing, and completion handling. No device was
unbound and no raw namespace writes were attempted.

A 320-byte row can require one or two 4-KiB page reads. The registration sample
measured 12.18 times as many planned file bytes as useful row bytes; that is
descriptor accounting, not measured SSD traffic or inference performance.
The SSD backend therefore needs page deduplication and a bounded device cache.
Warm host RAM is the preferred source when its capacity can be reserved.

## Validation and remaining gates

- Host tests: 62 source/selector tests and 16 driver-client tests passed using the focused test directory as `--confcutdir`.
  The repository-wide conftest cannot load in this local environment because
  its vLLM installation lacks `vllm.third_party.flash_linear_attention`.
- Native probe: compiled with `-Wall -Wextra -Werror`; four NPUs passed.
- Real-weight source gate: 640 exact source rows and 20 exact native DMA
  transfers from one shared object passed.
- Post-probe serving check: `qwen38-flash-next` on port 8001 returned HTTP 200
  and eight completion tokens, beginning with the correct answer to the
  France-capital prompt. No service restart or source-backend switch occurred.
  The model endpoint continued to advertise a 262,144-token limit.
- Driver deployment: four-NPU lease lifecycle and ordinary DMA passed; active
  reservation coexistence failed as described above. Service startup after the
  reboot reports 1,068,936 cache tokens and 4.08 concurrency at 262,144 tokens.
- Device hashing and graph replay passed the isolated four-NPU gate.
- Full-table pinning, autonomous submission/completion,
  autonomous SSD I/O, serving integration, and inference speedup remain
  unqualified. There is no end-to-end throughput claim for this work.

Receipts are in
[`artifacts/npu-collective-qfn-20261014-024609`](../../artifacts/npu-collective-qfn-20261014-024609/):
`ple-storage-registration-v1.json`, `ple-host-dma-probe-rank*-v2.jsonl`, and
`ple-shared-source-real-weights-v1.json`, `ple-request-rank*-v1.json`,
`ple-aicpu-stub-comparison-v1.json`, `ple-aicpu-inventory-v2.txt`, and
`ple-host-capabilities-rank*-v1.jsonl`.
