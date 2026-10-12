# Direct CUDA VRAM ↔ Ascend HBM DMA qualification

**Direct cross-vendor DMA works on the tested host.** The Ascend PCIe DMA
engine reads and writes an NVIDIA-exported GPU DMA-BUF, with local HBM as
its other endpoint. Both assigned Ascend chips pass repeated bidirectional
copies. No tensor payload is staged in host DRAM along this transfer path.

The [implementation](../../tools/sidecard/pcie/DIRECT_DMA.md) includes a restricted
kernel bridge and native Python API. See the [design PNG](design.png).
The final gate verifies **1,440 direct copies**, including warmups, plus
**16 rejected invalid requests**. Every transfer verifies varied payload bytes,
all untouched red-zone bytes, and an unrelated poisoned host lifetime guard.

The [raw results](direct.json) preserve every measured duration. Fifty
measured repetitions follow ten verified warmups per size and direction on
each chip. Chips run separately; simultaneous throughput is not established.

## Median transfer latency

These are one-way milliseconds, including source-event wait, Python request
preparation, DMA-BUF mapping, HBM lifetime-descriptor creation, native DMA,
completion validation, and (for the return) CUDA-owner write flushing.
CPU payload initialization and verification readback are outside timing.

| Bytes | Chip 2, GPU → HBM | Chip 2, HBM → GPU | Chip 3, GPU → HBM | Chip 3, HBM → GPU |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.224 | 0.191 | 0.226 | 0.195 |
| 8,256 | 0.261 | 0.220 | 0.234 | 0.208 |
| 16,384 | 0.252 | 0.216 | 0.230 | 0.200 |
| 264,192 | 0.303 | 0.244 | 0.286 | 0.243 |
| 524,288 | 0.325 | 0.262 | 0.346 | 0.274 |
| 1,048,576 | 0.421 | 0.346 | 0.434 | 0.354 |

For the 8,256-byte expert input, transfer is approximately **0.23–0.26 ms** across
the two chips. For a 16,384-byte weighted output, transfer is **0.20–0.22 ms**.
The 1 MiB cases complete at approximately **0.35–0.43 ms** per direction.
These include registration/setup on every call; persistent mappings and
asynchronous direct queues are not implemented yet. No full-model token rate
can be inferred by subtracting these observations from the earlier profile.

## Mechanism and hardware gates

CUDA reports GPUDirect RDMA and DMA-BUF support, and a real CUDA allocation
exports successfully: [capability receipt](cuda-capabilities.json). The
export API additionally verifies device memory type, unmanaged memory, and
pointer RDMA capability, then enables allocation-specific `SYNC_MEMOPS`.

The module attaches NVIDIA's DMA-BUF to the Ascend requester and consumes
the exporter's DMA-mapped scatterlist. It queries physical local HBM addresses
from the owner's Ascend cache and submits physical nodes through
`devdrv_dma_sync_link_copy_extend`. The nodes directly name GPU and HBM
addresses. The vendor PCIe driver programs its DMA queue from those nodes.
There is no CPU payload-copy operation in that path.

The ordinary Ascend peer-export API could not translate the fresh HBM pages
into its host BAR aperture. The selected path uses a local HBM address,
avoiding that export requirement. An unsubmitted native descriptor referencing
an unrelated ACL host allocation retains the HBM lifetime. Its host guard is
filled with poison and remains unchanged; the guard's SQ is never submitted.

The [topology receipt](../sidecard-pcie-bridge-20261012/topology.json) puts GPU
at `0000:01:00.0` under root `0000:00`, and the assigned Ascend at
`0000:49:00.0` under root `0000:40`. Actual DMA succeeds across these roots
on this chipset. This does not establish support on other PCIe topologies.

Kernel gates reject unsupported version/operation, physical card 0/1,
zero/oversized length, nonzero reserved fields, and an invalid DMA-BUF fd.
The other card is rejected before device access. The device requires root
and `CAP_SYS_RAWIO`, binds an fd to its creating process, serializes requests,
and limits DMA nodes and allocation extents. Uncertain native completion
retains mappings and allocation leases; fault recovery is not qualified.

No DMA error or quarantined state occurred in the final hardware gate.
The [module attestation](module-attestation.json) records zero references
at completion and a successful clean unload. No vendor-driver replacement,
IOMMU/ACS change, BAR resize, card reset, or clock/power/ECC change was made.

## Scope and reproducibility

**This is a raw transport qualification, not a GLM transport switch.** The
[post-test service receipt](service-confirmation.json) records a healthy
65,536-token endpoint and HTTP 200 returning `4` for `2 + 2` after the module
was unloaded. The last fixed 2K/64 full-model benchmark remains
[9.64 decode tokens/s](../sidecard-expert-profile-20261012/README.md) through
its qualified socket path. No direct-DMA model speedup is claimed.

Use the [direct backend runbook](../../tools/sidecard/pcie/DIRECT_DMA.md) to build
against the installed kernel headers and Ascend `Module.symvers`, load the
experimental module, and run the diagnostic with the private existing SDK
environment. The tested host runs kernel `7.0.0-38-generic`, open NVIDIA
615.71.09, and CANN 9.1. Each diagnostic worker sees only physical chips 2,3.

The [executed archive](executed-source.tar.gz) preserves the source snapshot.
All kernel and Python source files match the published bytes, recorded in
[execution-source-parity.json](execution-source-parity.json). The raw source
manifest also hashes generated build metadata; its exact bytes are preserved
as [generated module source](generated-module-source.c.txt).

Portable tests verify compiled C/Python UAPI agreement, source-event and write
flush order, endpoint/capacity checks, allocation/export leases, failed native
setup, and quarantine on corrupted/uncertain completions. **643 sidecard unit
tests pass**, including 17 direct-DMA tests. Hardware-specific claims come
from the separate real-device gate above.

Before GLM enablement, dedicated packed transfer buffers and fd/completion
control must be connected to the independent expert workers, followed by
full-vocabulary parity, repeated decode, context, cancellation, and queue
isolation gates. General framework-pointer export, asynchronous direct queues,
persistent registration, fault recovery, and other host/hardware support remain
unqualified. The measured native expert and CUDA launch costs also remain
optimization targets.

## Publication checks

The clean fork-main publication tree passes all **643 sidecard unit tests**
([log](unit-tests.log.gz)) and every scoped manual hook, including Markdown
([log](scoped-ci.log.gz)). The required repository-wide `bash format.sh ci`
was run in an isolated worktree and still fails the same seven hooks as the
[previous committed baseline](../sidecard-pcie-bridge-20261012/full-ci.log.gz).
Its [full log](full-ci.log.gz) and [validation receipt](validation.json) preserve
the result. These repository-wide failures are not reported as passing checks.
The [hardware build log](hardware-build.log.gz) records the module build against
the installed kernel and vendor symbol versions.
