# Lease-owned host source registration: v3

The driver now supports startup-time registration of immutable host source
pages under an existing dynamic DMA queue lease. It is installed and the
private-buffer registration/coexistence gates passed on all four NPUs.
Device-initiated DMA and full-table PLE integration are not implemented yet.

## Implemented contract

- Additive 64-byte ABI-v1 ioctls: register `0xc04044d3`, query `0xc04044d4`,
  release `0x404044d5`. Capability bit 1 reports source registration;
  bit 0 still reports host-routing isolation. Neither advertises submission.
- One source per lease, checked lease generation and monotonically increasing
  region handle. Stale releases cannot unmap a newer source.
- `FOLL_LONGTERM` pins, locked-memory accounting, and per-device
  `DMA_TO_DEVICE` scatterlist mappings preserve exact byte bounds. Replies
  expose counts and opaque handles, never bus addresses.
- Partial pins, allocation/map failures, failed reply copies, explicit release,
  close, and detach clean up mapping/pin/accounting ownership.
- The 128-GiB range bound is not a full-table qualification. Each rank charges
  its own pin accounting and metadata, even when all ranks share the same
  physical source pages. Applications must populate sources first and keep
  them immutable. A shared sealed memfd is the preferred backing.
- No source address is published to a producer or submitted to a queue. Future
  producer activation must add revoke/drain before unmap; current teardown is
  only valid while registration has no submission interface.

Implementation: driver fork `pcie_host/devdrv_device_dma_source.c`, lease
integration in `devdrv_device_dma.c`, UAPI in `inc/driver/ascend_device_dma.h`.
Startup client: `tools/qwen4exp/ple_driver.py`. Explicit hardware probe:
`tools/qwen4exp/ple_driver_source_probe.py`.

## Validation

- 10 driver CPU tests, including real registration code with injected failures
  and the retained 87,934 host dispatch checks.
- 87 focused client/server/formatter tests pass using `--noconftest`.
  The repository-wide conftest still needs the absent local upstream
  `vllm.third_party.flash_linear_attention` dependency.
- All 31 modules pass kernel-release, architecture, strong-symbol, version CRC,
  namespace, and isolated dependency checks. All 31 loaded source versions
  match the candidate after boot `aad0b818-d05a-4ccf-9812-b035a444bee4`.
- Twelve page-crossing 320-byte source registrations pass across four NPUs,
  sharing one sealed read-only memfd. Locked-memory limit enforcement,
  query, stale-handle rejection, explicit source release, and FD-close cleanup
  pass; locked-memory accounting returns to baseline.
- All 144 ordinary host DMA transfers pass before, during, and after leases.
  During the leased phase, all four source registrations remain held and
  queryable. Source contents remain unchanged. These ordinary transfers do
  not use the new registration handles.

Receipts: [source registration](ple-driver-source-hardware-v3.json),
[DMA coexistence](ple-driver-dma-coexist-v3.json), and
[loaded module versions](ple-driver-loaded-modules-v3.json).

## Device-interface findings

The read-only AI CPU probe identifies the device HAL and device Linux modules.
Fifteen live function prefixes exactly match the HAL extracted offline from
the installed firmware. No firmware file was modified.

- `drvMemConvertAddr` and `halHostRegister` are success-returning stubs.
- `halMemcpySumbitInner` and `halMemcpyWaitInner` return unsupported (`0xfffe`).
- `drvMemcpyInner` calls ordinary `memcpy_s`: the call at PLT `0x7d880`
  resolves through GOT `0x4333e0`. It is not a PCIe fetch mechanism.
- The device kernel's `drv_pcie.ko` exports actual DMA operations. This does
  not establish a user-space interface to those operations or ownership of
  the host-reserved queue.
- The custom AI CPU process can open `/dev/svm0`, but cannot open `/dev/mem`
  or `/dev/davinci_manager`. Opening SVM alone does not expose queue MMIO.

Evidence: [live/firmware comparison](ple-device-interface-evidence-v1.json).
The cleaned `ple_aicpu_probe_v10` diagnostic performs no DMA, MMIO access,
or driver ioctl. No device-initiated source fetch or speedup is claimed.

The user-provided source/research trees led to the original firmware extraction,
including the matching compressed kernel symbol-version archive. All 135
imports of the shipped PCIe module resolve with matching CRCs against that
kernel and three companion modules. The generated kernel build headers remain
missing. See the [source inventory](ple-device-source-inventory-v1.md) and
[offline ABI audit](ple-device-abi-audit-v1.json). The audit's 13 focused tests
and Ruff checks pass; no newly compiled device module has been loaded.

## Deployment and rollback

The previous v2 modules and modified source files are backed up in
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/ple-driver-deployment-v3`.
`ple-driver-deploy-v3.py rollback` restores the previous files and rebuilds
module dependencies/initramfs; reboot afterward. The v2 driver supports queue
leases but rejects the new source-registration capability check.

The serving identity remains `qwen38-flash-next`, port 8001, context 262144.
All retained v10 optimizations have now been restored and requalified on four
workers. The final 12-case serving gate passes, including short output budgets,
penalties, log probabilities, request turnover, structured output, cancellation,
and post-cancellation recovery. Inspection confirms queue depth 2, capture sizes `[3, 6, 9, 12]`, 36
retained engine owners, enabled fused checkpoint preparation/lazy readback,
and an unpaused engine. All four live queue leases are held by API PID 33506;
`device_submit=0` remains explicit. Source registration is not automatically
enabled for the full PLE table. PLE and general-selection eager segments remain.

Receipts: [retained runtime inspection](final-driver-restoration.json) and
[live leases](ple-driver-serving-reservations-v3.json).

Reboot initially auto-started `rtsp-scene-model.service` and
`rtsp-scene-log.service`, occupying NPUs 0/1 and causing the first QFN launch to
fail memory admission. Both services were stopped for the authorized four-NPU
QFN allocation; they remain enabled for future boot. The failed run is archived
as `resident-live-v10-failed-scene-autostart`, and fresh v10 owners were created.

Restoration reached a measured 90 C on one chip. The remaining test sequence was
paced with cooling intervals, while the model service stayed up. The temporary
thermal guard has exited after successful restoration. Performance checks are
paced separately; no driver-submission speedup is implied.

## Post-restoration throughput

The three fixed 256-token C1 HTTP trials measured 32.1253, 32.2785, and
33.1025 output tokens/s, median **32.2785 tokens/s**. The metric includes short
prefill and HTTP overhead, not just device decode time. The prior v9 check was
31.9289 tokens/s; these are short qualification runs, not a controlled A/B test
or evidence of DMA acceleration. The new source-registration handles are still
unused by model execution.

C4 (four concurrent 256-token responses) measured **60.3625 aggregate HTTP
output tokens/s**, with median per-request streaming decode **16.9893 tokens/s**.
The previous v9 C4 check was 63.4324 aggregate tokens/s. The approximately 4.8%
lower result is not sufficient to establish a regression from these single
short runs and different thermal histories; no performance improvement is
claimed. All requests completed and final verification exited successfully.

Receipts: [C1](decode-benchmark-driver-source-v10-v1.json),
[C4](concurrency-driver-source-v10-v1.json), and
[12-case serving gate](scheduling-serving-gate-driver-restored-v1.json).
The live logfile is `/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/server-current.log`.
