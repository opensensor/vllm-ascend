# Direct CUDA VRAM ↔ Ascend HBM DMA

Direct cross-vendor DMA is **hardware-validated on this host**, using the
Ascend DMA engine against NVIDIA-exported GPU memory. Both assigned chips
pass repeated transfers in both directions. This backend does not stage
tensor payloads in host DRAM. The first API performs a blocking ioctl;
GLM can use it through the [worker-owned serving integration](../GLM_DIRECT_DMA.md).

See the [hardware report](../../../artifacts/sidecard-direct-dma-20261012/README.md)
and [design PNG](../../../artifacts/sidecard-direct-dma-20261012/design.png).

## How the direct path works

1. CUDA owns a dedicated `cudaMalloc` allocation. `Driver.export()` checks
   device memory type, absence of managed memory, and RDMA capability; enables
   allocation-specific `SYNC_MEMOPS`; and exports an NVIDIA DMA-BUF fd.
2. The kernel attaches that fd to the assigned Ascend PCIe requester and maps
   its scatterlist into that device's DMA address space. It uses the exporter
   mapping directly, without guessing physical addresses or remapping it.
3. Ascend retains its HBM allocation through
   `hal_kernel_svm_dma_desc_create`. The descriptor references an unrelated
   ACL host allocation as a lifetime guard. **Its SQ is never submitted.**
   This retains the HBM range without requiring HBM export through a BAR.
4. The kernel queries local HBM addresses from the owner's Ascend page cache,
   constructs bounded physical DMA nodes, and calls
   `devdrv_dma_sync_link_copy_extend`. Both directions are driven by Ascend.
5. Completed transfers release their mapping and lifetime descriptor.
   The CUDA owner flushes GPUDirect writes before consuming a returned result.

```text
GPU VRAM -- NVIDIA DMA-BUF mapping -- Ascend PCIe DMA engine -- local HBM
                  ↔ direct payload transfers in both directions ↔

CPU: descriptors, source/completion control, and allocation lifetime
     (no payload bounce through host DRAM)
```

The ordinary `hal_kernel_p2p_get_pages` probe failed because its HBM-to-host-BAR
translation could not represent the allocated pages. Ascend can nevertheless
access those pages locally. Choosing Ascend as the DMA initiator avoids that
BAR-export requirement. The poisoned guard test distinguishes the guard
from the transferred GPU or NPU payload.

## Build and reproduce

The experimental module targets the assigned 310P card at `0000:49:00.0`,
physical chips **2 and 3**. It refuses requests for physical 0/1 before any
device access. This fixed host binding is deliberate during qualification.
The installed vendor ABI declarations are in `kernel/vendor_abi.h`; the build
checks imported symbol versions against the matching Ascend `Module.symvers`.

```bash
make -C tools/sidecard/pcie/kernel
sudo insmod tools/sidecard/pcie/kernel/sidecard_peer.ko
python3 -m tools.sidecard.pcie.peer_benchmark \
  --npu-environment /absolute/path/to/private/npu-environment.json \
  --npu-python /absolute/path/to/npu-venv/bin/python \
  --output /absolute/path/to/new/results-directory \
  --iterations 50 --warmup 10
sudo rmmod sidecard_peer
```

Use this sequence only after a clean successful diagnostic. The diagnostic
requires root workers through `sudo`, since the device is mode 0600 and its
operations require `CAP_SYS_RAWIO`. The controller passes the private SDK
environment through a worker exec without printing its contents. No device
reset, IOMMU/ACS change, BAR resize, clock/power/ECC change, or vendor-driver
replacement is required on the qualified host.

Native source and installed kernel/driver versions must match before loading.
This is an experimental module, not a generally enabled vLLM backend. A new
host or driver version needs its own capability and correctness gates.

## Python API and ownership

```python
from tools.sidecard.pcie import Driver, KernelPeer
from tools.sidecard.pcie.peer import CUDA_TO_ASCEND

driver = Driver()
gpu_export = driver.export(gpu_allocation)
peer = KernelPeer(physical_chip=2)
receipt = peer.copy(
    CUDA_TO_ASCEND,
    gpu_export,
    npu_allocation,
    acl_host_guard,
    payload_bytes,
    source_ready=cuda_source_event,
)
peer.close()
gpu_export.close()
```

Allocations and events use the native objects from `pcie.runtime`. An HBM
allocation is private to its worker, and its ACL guard must cover its full
pin extent. Dedicated CUDA buffers must satisfy the CUDA driver's export
alignment and size requirements; the diagnostic uses 2 MiB allocations.
Arbitrary framework allocator pointers and imported/VMM suballocations are
not accepted by this initial export API.

The call waits for the source event, performs the kernel request, checks the
completion identity and mapping geometry, and releases local leases only
after a trustworthy completion. The reverse direction flushes writes in
the CUDA owner's context. There is no device-wide synchronization call.
The native ioctl currently blocks for DMA completion; persistent mappings,
asynchronous completion queues, and multiple in-flight slots are future
extensions, not qualifications from this test.

Every device fd belongs to its creating process, and requests serialize
through a per-file mutex. The module bounds byte counts and DMA-node counts,
validates endpoint binding, and rejects unknown versions/operations. Python
also checks runtime ownership, capacity, source events, and exported-buffer
lifetimes. Native setup failures release resources without claiming a copy.

If native DMA returns an error, quiescence is not established. The module
retains mappings, the HBM lifetime descriptor, and a module reference; Python
quarantines allocation/event leases. The diagnostic remains alive. Do not kill
that owner or force removal/reset as automatic cleanup. Recovery must first
establish completion or stop DMA while preserving those allocations. This
recovery path is not implemented, so the backend is not enabled in serving.

## Validation and integration boundary

The final hardware gate performs **1,440 verified direct copies**, including
warmups, over 4,096; 8,256; 16,384; 264,192; 524,288; and 1,048,576 bytes.
Both chips are tested separately. Every copy checks the payload and all
remaining bytes in its 2 MiB allocation. An unrelated 2 MiB host guard stays
poisoned throughout. Sixteen invalid-request cases are rejected across the
two workers. CPU initialization and readback are outside the timed transfers.

Before enabling this in GLM, pack inputs into dedicated GPU transfer buffers,
connect buffer fds and source/completion control to the independent expert
workers, and preserve per-request state, native scratch ownership, placement
authentication, expert IDs/weights, and fixed reduction order. Repeat dispatch
and full-vocabulary parity, decode, context, cancellation, and queue-isolation
gates, then profile the model. Direct DMA qualification establishes a transport
capability; it does not establish a GLM throughput improvement.
