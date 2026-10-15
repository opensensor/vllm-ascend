# Device-initiated DMA: reservation foundation

This candidate prepares exclusive PCIe DMA queue ownership for Qwen PLE
lookups on four Ascend 310P NPUs. It is **build-validated, not installed or
hardware-qualified**. It does not yet expose a device submission interface.

## Evidence and intended data path

Private-buffer tests in the vLLM Ascend workspace established that the host
can convert ordinary shared read-only pages, sealed memfds, and ACL host
allocations into DMA descriptors and copy exact data to all four NPUs.
The AICore PLE selector also passed exact row/offset and graph replay tests.

A custom AI CPU function executed on device, but its loaded
`drvMemConvertAddr` entry matched the SDK device stub byte for byte: it
returned success without constructing a descriptor. A symbol lookup or a
zero return code is therefore insufficient capability evidence. Device-side
inventory showed a separate ARM Linux environment, not the x86 host kernel.
Changing this host fork alone does not replace that device library or driver.

The intended RAM path is:

```text
cold host setup:
  shared immutable table -> pin + per-device DMA mappings
  reserve queue + fixed destinations + generation-scoped ownership

steady-state device work:
  token/history -> AICore row IDs -> source mapping lookup
  -> device submission -> PCIe DMA -> device completion -> PLE projection

cold teardown:
  revoke producer -> drain/reset with proof -> unmap -> unpin
```

The 95.37-GiB table must remain one shared physical object across ranks.
Per-device DMA translations may differ. Neither a host virtual address nor a
host driver's opaque descriptor handle is a portable device address.

## Implemented dynamic reservation

The regular driver loads with **all channels available to ordinary clients**.
There is no reservation module parameter. A model loader opens
`/dev/ascend_device_dma` and explicitly acquires a reservation through an ioctl.
The node is initially root-only; deployment may delegate access through its
normal device-permission policy. The existing vendor HAL ABI is unchanged.

The 64-byte, versioned UAPI is in `inc/driver/ascend_device_dma.h`:

| ioctl | Purpose |
| --- | --- |
| `ASCEND_DEVICE_DMA_ACQUIRE` | Reserve one idle traffic channel on the requested physical 310P |
| `ASCEND_DEVICE_DMA_QUERY` | Report the current lease on this open file |
| `ASCEND_DEVICE_DMA_RELEASE` | Release the matching device/channel/generation lease |

One lease belongs to one open file description; dup/inherited FDs share its
lifetime. Each physical NPU permits one lease. Keep its FD open throughout
model use. An explicit release validates the device, hardware channel, and
monotonically increasing generation. Last close releases an idle lease.
Reset detaches the device and invalidates the old FD before freeing ring
storage; reusing a device ID does not revive that FD. An unexpectedly nonempty
queue on close is quarantined until reset rather than returned to host use.

Admission selects the final traffic channel, leaves at least one other traffic
channel, and rejects overlap with another pool. It locks CQ processing and
submission in that order, verifies that completed SQ head equals submitted SQ
tail, and marks ownership atomically. If a transfer or completion callback is
still active, acquisition returns `EBUSY` with no reservation; the loader may
retry. It never waits for hardware while holding a spinlock. Other chip
families, virtual modes, unavailable devices, invalid ABI fields, duplicate
owners, and stale releases are rejected. A failed acquisition reply copy
releases the newly acquired lease.

Host routing ranges stay immutable. Round-robin dispatch skips leased
channels; normal exhaustion still returns `ENOSPC`. **An explicit fixed
instance mapping to the leased channel returns `EBUSY`** rather than silently
moving ordered work to another queue. Fixed-instance users of that traffic
channel therefore need coordination before live deployment. Direct physical,
virtual-channel, and small-packet submission guards reject a leased channel.
Host completion processing and its periodic guard skip leased channels.
SR-IOV reassignment is serialized with admission and refuses an active lease.
The steady-state host dispatch path does not take the lease registry mutex.

The read-only `device_dma` attribute alongside `dma_info` reports the selected
logical device, reserved hardware channel, available traffic channels, and
queue depths. Like existing PCI diagnostics, `davinci_dev_id` selects the
logical device. It performs no firmware RPC and exposes no physical addresses.

This version always reports `capabilities=0` in the ioctl response and
`device_submit=0` in sysfs. **It reserves an idle channel but does not expose
its memory, doorbells, or a device producer.** The existing fault/reset machinery
remains in place. Adding producer activation requires a separate revocation,
drain, and memory-lifetime contract; closing a reservation FD alone must not
be treated as proof that a future device producer has stopped.

The vLLM workspace has a startup-only client in `tools/qwen4exp/ple_driver.py`:

```python
from tools.qwen4exp.ple_driver import DeviceDMALease

with DeviceDMALease.acquire(driver_device_id) as lease:
    info = lease.query()
    # ABI v1 returns capabilities=0; no PLE transport may be activated yet.
    # Keep this scope alive throughout the eventual model/transport lifetime.
```

There is no automatic acquisition in the current serving runtime. The live
server continues using the installed driver without reservations.

## Validation, 2026-10-14

- `make all KERNEL_UNAME=7.0.0-38-generic CC=x86_64-linux-gnu-gcc DKMS_JOBS=8`
  passed for the bare-metal profile.
- Nine CPU regression tests passed. The new C test executes the actual
  planner, admission, release, ioctl, and host dispatch functions with mocked
  kernel services over channel counts 1 through 32. They cover default startup,
  busy queues, pool overlap, range overflow, competing owners, stale generations,
  copy faults, close cleanup, reset/reacquisition, quarantine, and 87,264
  successful host dispatches. Reserved fixed instances are rejected and shared
  dispatch never submits to a leased channel. These tests do not prove hardware
  concurrency. The startup client has 16 additional passing offline tests.
- `scripts/verify-modules.py 7.0.0-38-generic --profile bare-metal` passed all
  31 modules: architecture, kernel release, strong symbols, version CRCs,
  namespace imports, and isolated dependency resolution.
- The existing installed driver and model service were left unchanged.
  No reservation was enabled on live hardware and no speedup is claimed.

The local base is `99b8459` (including the separate virtualization port).
The installed host uses `453e106`, whose hrtimer setup fix is included in this
candidate, together with its regression test. This preserves the installed
fix instead of packaging the older local timer callbacks. The current work
adds no new timer behavior beyond that already running on the host.

The module bundle and manifest are under
`build-artifacts/7.0.0-38-generic/bare-metal/`. New source changes in this
candidate have not been committed. Rebuild all modules together because an
internal shared channel structure changed; replacing only one module can
leave other modules with an incompatible layout.

## Next implementation gates

1. Establish a real device-side control interface. The custom AI CPU context
   can run native code, but its HAL conversion stub cannot submit DMA. Identify
   the device driver mapping/queue mechanism and its source or supported ABI;
   a host-only capability change must not advertise device access.
2. Extend the implemented FD-owned, generation-scoped reservation with exact
   source/destination region bounds and producer activation. Hold long-term pins and per-device DMA mappings until producer revocation
   and DMA drain are proven. Process exit/reset must invalidate old requests;
   no raw physical addresses or arbitrary MMIO ranges should be caller supplied.
3. Expose only the reserved queue and its required doorbells to that owner.
   Prove device descriptor publication, queue-tail visibility, payload-before-
   completion ordering, CQ wrap, timeout, and reset behavior on small buffers.
   A host C polling/submission thread does not satisfy device initiation.
4. Exercise page-crossing 320-byte rows, repeated rows, variable row sets,
   and concurrent use with the normal host DMA pools. Requalify driver/HAL
   startup, DMA, reset, and unload before admitting the full table.
5. Connect the already-qualified device row selector, benchmark complete
   PLE fetch/projection, then integrate into serving graphs. The SSD backend
   additionally requires filesystem/block lifetime and storage queue ownership;
   saved file extents alone are insufficient.
