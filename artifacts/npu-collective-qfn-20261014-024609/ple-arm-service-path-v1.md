# PLE ARM service investigation

## Operational state

QFN API PID 33506 and its engine/workers shut down after SIGTERM on
2026-10-14 at 22:56:25 UTC because the card inspection was initially understood
to involve the active cards. The user clarified that it is a separate third
card. Hardware work resumed on the installed pair. Fresh run root
`resident-live-v11` uses API PID 117680 and the same `qwen38-flash-next`, port
`8001`, context `262144`. All retained controls and the final serving gate passed.
Retained optimization restoration is recorded in
`restore-retained-v11.log` and `resume-retained-v11.log`; a thermal guard pauses
the restoration supervisor at 84 C and resumes at 79 C. No device firmware or
SELinux configuration was modified.

## Evidence collected offline

The exact extracted firmware contains a plausible user-space service route:

- `var/tsdaemon.yaml` starts TSD as UID/GID 1000.
- `libtsdaemon.so` contains UDF startup, binary-path resolution, subprocess
  manifests, and fork/exec launch machinery. `SetUdfStartInfo` calls
  `GetBinaryPath`, then constructs the subprocess environment and arguments.
- The local CANN GE reference uses `TsdProcessOpen` with `ProcOpenArgs`, a
  UDF process type and optional file path. This is evidence of an API, not proof
  this firmware accepts an arbitrary executable or grants it HDC access.
- The process-launcher configuration mentions `/var/udf.yaml`, but that file is
  absent from this extracted rootfs. Do not assume a configured UDF service.

The previous live custom AI CPU receipt reports UID 1001, GID 1000,
supplementary groups `[1000]`, and no effective capabilities. Its HDC open
returned EACCES. The firmware startup script creates the base group as GID
1102 and assigns `/dev/hisi_hdc` to the base user/group with mode 0660. It adds
the ordinary runtime user to that group, but not the custom AI CPU user.
That strongly suggests a Unix-permissions obstacle. Live node ownership and
the candidate subprocess's actual groups still need verification by a separate
service probe (the custom-process verification follows).

The v12/v13 live access probes subsequently confirmed `/dev/hisi_hdc` and
`/dev/davinci_manager` are UID/GID 1102, mode 0660. The custom AI CPU process
remains UID 1001, group 1000, with supplementary groups `[1000]`. Its live
SELinux context is `system_u:unconfined_r:cust_user_t:s0`. Reading node labels
and the SELinux enforcement flag is denied in that process. DAC is therefore
a confirmed access obstacle; a successful privileged service is still unproven.

Decoded the shipped binary SELinux policy using local libsepol, without loading
or changing policy. Relevant rules:

- `cust_user_t` has attribute `hisi_dev_rw_domain`.
- `hisi_hdc_t` has attribute `hisi_dev_object`.
- The attribute allow rule includes character-device open/read/write/ioctl.
- The custom-domain HDC ioctl allowlist includes `0x4820-0x4829`, covering the
  reference driver's fast-send `0x22`, fast-receive `0x23` and DMA-map `0x24`
  command numbers under magic `0x48`.

Thus the extracted policy does not by itself explain the failed read-only open.
This is not a proof of the live policy/label, supplementary groups, driver
authorization, or successful HDC transfer.

Artifacts: `ple-device-policy-v1.conf`, `ple-policy-read-v1.c`,
`ple-arm-service-evidence-v1.json`, and
`ple-device-subprocess-disassembly-v1.txt`. The JSON records source hashes.

## HDC is not yet a device-selected remote read

The inspected driver fast path resolves a sender's registered TX buffer and a
receiver's RX buffer and then schedules a DMA copy. The user-space fast-receive
API returns the received descriptor; it has no remote source offset argument.
Huawei's [HDC API declarations](https://gitcode.com/cann/driver/blob/master/pkg_inc/ascend_hal_base.h)
also distinguish fast send from fast receive and require HDC-managed memory.

A device service that requests a row and waits for the host to submit fast-send
would still require per-request host work. Do not call that device-initiated PLE
lookup, or claim it uses the v3 source handles. Those handles remain isolated
host registrations with `device_submit=0`.

The target remains a startup-registered host table, device-selected row IDs,
bounded device-side DMA submission, and device-visible completion. An ARM
service is useful only if it can perform that submission or access a reusable
preconfigured transfer mechanism without a host lookup for every row.

## Prepared access probe

`tools/qwen4exp/ple_device_access.h` captures process IDs/groups/capabilities,
SELinux process context/enforcement, node ownership/mode/label, and read-only
open results. It issues no ioctl, mapping, DMA, or privilege changes.

- The existing custom AI CPU probe accepts mode `access`.
- `tools/qwen4exp/ple_device_access.cpp` provides the same diagnostic as a
  standalone binary for a verified device subprocess launch path.
- Build the standalone binary with the matching ARM userspace compiler. It has
  no CANN or kernel-build-header dependency. ARM compilation and local C++
  compilation are validated; standalone launch is not yet validated.

CPU-only validation: 19 tests passed across the access probe, bounded receipt
formatter and existing ABI audit. Use `--confcutdir=tests/ut/qwen38_1m` for
these independent tests. The ordinary repository test bootstrap fails locally
because installed vLLM lacks `vllm.third_party.flash_linear_attention`; that
failure is separate from the diagnostic tests. Ruff passed for the new Python
test file.

## Hardware-qualified progress and remaining deployment obstacle

`ple_tsd_capability.cpp`, compiled against the installed CANN headers and
libraries, returned `level_status=0`, `level=0x0000013f`,
`builtin_udf_status=0`, `builtin_udf=1`. No UDF or DMA was launched by that
query. The public runtime reference used for interpreting these bits is commit
`e25b68947a187fed25a7c714a36b7fff945aa74f` from `cann/runtime`, kept separately
under `/tmp/ple-runtime-reference`.

`tools/qwen4exp/ple_arm_udf` is now a concrete C++ UDF using the same bounded
access report. It compiled and linked successfully for `RESOURCE_TYPE=Ascend`
against the installed device `libflow_func.so`. Its launcher pins the node to
device 0, uses a 64 MiB DataFlow pool, and verifies the returned machine is
AArch64. The private Python 3.11 environment matches the CANN binding; serving
continues in its existing Python 3.12 environment.

DataFlow initialized, built the model package, and created its device queues,
but deployment failed at `TsdFileLoad`: HDC closed the file-transfer session
with error 25 before the sender received the ready acknowledgment. The rejected
target was `.../hdcd/device0/<host_pid>_0_Ascend-flow_model.tar.gz`. Repeating the
same isolated diagnostic as host root produced the same failure. Both attempts
rolled back through DataFlow finalization; no UDF access receipt was produced.

The v13 AI CPU probe read the **live** `/etc/hdcFileTransWhiteList.cfg` and
confirmed the extracted policy: upgrade images and two AI CPU kernel archive
suffixes are allowed, but `_Ascend-flow_model.tar.gz` is absent. This is strong
evidence for a device file-transfer-policy obstacle, although the daemon's
specific rejection log has not been obtained. No attempt was made to disguise
the UDF archive as a kernel/upgrade archive or modify unrelated file permissions.

The initial candidate remedy was to enable the UDF package type in the device
transfer configuration. The subsequent V2 test below provides a documented
alternative to investigate first. Neither route yet establishes access to our
reserved DMA queues or removes the per-request host dependency.

Receipts: `ple-aicpu-access-v12.jsonl`, `ple-aicpu-access-v13.jsonl`, their
decoded `.txt` files, `ple-tsd-capability-v1.txt`, `ple-arm-udf-v5.log`, and
`ple-arm-udf-v6-root.log`. The latter two are failures, not passing service tests.

## Documented V2 file transfer works

The installed driver also exposes `drvHdcGetTrustedBasePathV2` and
`drvHdcSendFileV2`. The public HAL declarations specify a separate CANN transfer
interface without the legacy filename allowlist. Both path queries succeeded:

- Legacy: `/home/HwHiAiUser/hdcd/device0`.
- V2: `/home/HwHiAiUser/hdcd/device0/aicpu`.

An ordinary unprivileged host process successfully sent a unique 52-byte text
diagnostic into the V2 directory (`status=0`). Receipt:
`ple-hdc-v2-transfer-v1.json`. The file is deliberately harmless and its exact
remote path is retained there for later cleanup. No upgrade/kernel filename
was reused and no firmware policy was edited.

This qualifies the V2 file-transfer operation, not an ARM service launch or a
device-initiated row lookup. The existing TSD model loader still constructs the
legacy path and calls `drvHdcSendFile`. A correct adaptation must reconcile the
V2 upload directory with the device's package unpacker and subprocess loader;
simply substituting the function name and reporting success at the old path
would be incorrect. Investigate that loader contract before changing firmware.

Final QFN restoration receipt: `final-driver-restoration-v11.json`. Four
workers passed all enabled-owner checks, graph inventories were clean, queue
depth was two, the engine was not failed, and serving was resumed. The final
scheduling gate passed all 12 cases, including structured output and abort
recovery. No fresh throughput claim is attached to these access experiments.

## Next hardware gate after cooling restoration

1. Verify the supported TSD process type, executable packaging/path constraints,
   installed userspace ABI and bounded process cleanup before launching.
2. Run the same diagnostic in custom AI CPU and the supported ARM subprocess.
   Compare actual groups, HDC ownership, security context and open result.
3. If an authorized service can open HDC, test a private tiny transfer using its
   actual memory-registration and session APIs. Do not borrow management service
   IDs, change all device permissions, or feed host virtual pointers to ARM memcpy.
4. Establish the device-selected transfer interface before integrating PLE.
   If HDC only supports sender-driven transactions, use it for control/setup
   and identify a separate device submission interface; do not disguise host
   round trips as completion of CPU removal.

A card flash dump is useful if it contains device launch configuration,
additional service binaries, or a different firmware partition missing from
the installer. Compare hashes against the evidence manifest before repeating
analysis. It is not a prerequisite for continuing the user-space route.

## V2 loader qualification and firmware omission (2026-10-14 23:40 UTC)

Hardware work continued while QFN stayed up on port 8001. No serving workers
were restarted. The final private chat request returned HTTP 200 with 16
completion tokens (reasoning, length-limited); this is a coexistence check, not
a throughput benchmark. The existing 262144-token context configuration remains.

The device `InotifyWatcher` binary watches both HDC trusted roots, recognizes
`<pid>_<graph>_Ascend-flow_model.tar.gz` in either, and passes the actual watched
root to `OmPackageWorker`. The TSD acknowledgment uses the basename. This proves
why a scoped V2 path adapter can work without changing the device daemon.
`ple_tsd_transfer_v2.cpp` redirects only device 0, this process's graph-0 package,
and the exact old trusted-root destination. Other sends retain the original API.
It is a private diagnostic preload, not installed in QFN or the toolkit.

The v7 hardware attempt successfully uploaded the UDF package and the device
recognized it as package type 5. Device logs also definitively confirm that the
previous legacy uploads failed the filename allowlist. The next failure was
ENOENT creating `/home/HwHiAiUser/hs/0/omfile/<pid>/`: the model parent directory
was missing. Loading the ordinary runtime package returned success but skipped
its already-matching hash and did not create that model directory.

The common-package loader supports `install_path:1` for this model root. A
private configuration overlay and README-only bootstrap were tested: firmware
CMS verification rejected the unsigned archive (ret 103, underlying ret 14).
That attempt did not solve the directory problem. The corrected bootstrap uses
a byte-identical copy of the installed, signed `cann-udf-compat.tar.gz`:
`208bd77db056a34b642d54daa8313ce943d547954f1c8fa02c3ccdcc4f14c317`.
Device logs confirm successful CMS verification and package installation. No
signature check was disabled. This duplicates the vendor runtime files under
the model root; it is a diagnostic workaround, not a production deployment.
The private package metadata and duplicate files are device runtime state, not
a flashed firmware change. Subsequent probes used the original host CANN config.

The v9 UDF attempt then passed directory creation and reached the next missing
firmware dependency: `/var/tsdaemon_check_illegal_link.sh` returned shell status
127. A separate live, read-only v15 AI CPU receipt confirms ENOENT for that
exact helper, while `/bin/sh` and `/bin/bash` exist. `/var` is root-owned 0755;
the custom AI CPU process cannot repair it. Neither the installed signed UDF
compat package nor the legacy runtime package contains the helper. Do not treat
the failed model package as accepted or skip the device link-validation step.
No UDF access receipt or device-initiated PLE DMA transfer has been obtained.
The reserved-queue integration remains `device_submit=0`.

Artifacts: `ple-arm-loader-device-evidence-v2.txt`,
`ple-device-inotify-disassembly-v2.txt`,
`ple-device-package-path-disassembly-v2.txt`,
`ple-arm-udf-v7-v2.log`, `ple-arm-udf-v8-v2.log`,
`ple-arm-udf-v9-v2.log`, `ple-model-dir-bootstrap-v2.log`,
`ple-aicpu-access-v14.txt`, `ple-aicpu-access-v15.txt`, and
`ple-serving-coexistence-v2.json`. Private attempts were bounded by timeouts;
no UDF experiment remains running. CPU validation: 23 tests passed, including
transfer scope and bootstrap byte-preservation/failure cases. Python lint passes.

## Third-card SPI dump received

Inspected `/home/matteius/scriba/build/bin/ascend_A.bin` read-only and retained
an identical read-only artifact copy. Size 16777216; SHA-256
`49c2ca1d31addbca8b9276679ae601ecec83e9a3043344574cc42c1e7e63b56d`.
It contains Huawei LiteOS MCU V23.2.6 strings at offsets 0x2233fd and 0x2633fd,
with a plausible ARM microcontroller vector table at 0x200000 (initial stack
0x20010000, reset 0x080241c9). A 64 KiB code region repeats 0x40000 bytes later.
No ELF, Linux, gzip or SquashFS signatures were found. This points to a board
management MCU image, not an identified copy of the NPU Linux filesystem.
Irregular zero/FF runs and sparse nonuniform bytes outside the code regions mean
read integrity is not established. A separate matching physical read is needed.
See `spi-dump-ascend-A-analysis.json`. Do not flash or publish the raw dump.
