# Device source and ABI inventory

The two user-provided trees supply useful prior research and public driver
sources. They do not contain a complete build tree for the running device kernel.
The earlier statement that no matching symbol-version file was available was
incorrect: the original extracted root filesystem contains a compressed boot
symbol table, rather than a file named `Module.symvers`.

## Sources inspected

- `/run/media/matteius/20TB-drive/ascend-sources/reference/driver`, commit
  `688c802c2ca017ec3cbdf4f0294cf04c063cb4d7`: public CANN source reference.
  PCIe DMA headers and host implementation are present; the matching 310P
  device PCIe agent implementation and generated kernel build headers were
  not found. The reference tree was left unchanged.
- `/run/media/matteius/20TB-drive/open-ascend`: earlier firmware/HAL research.
  `hal-src` is machine-reconstructed C, explicitly described by its own audit
  as unverified scaffolding. Do not treat its signatures as authoritative.
- Its README points to the original extraction at
  `/home/matteius/re-framework/artifacts/ascend-310p-driver-20260911/device-rootfs`.
  This contains the exact-release boot config, System.map, and compressed
  symbol-version archive. The kernel source directory contains only
  `securec.h` and `securectype.h`.

## Verified offline evidence

Device release: `6.6.0-72.0.0.76.h21510.1170.eulerosv2r15.ascend.aarch64`.
The recovered boot table contains 19,148 exported kernel symbol versions.
All 126 kernel imports of `drv_pcie.ko` have matching CRCs. Its other nine
imports match exports in `ascend_kernel_open_adapt.ko`, `drv_dfm.ko`, and
`dbl_runenv_config.ko`. Thus all 135 imports resolve against the supplied
artifacts. This verifies those artifacts, not a newly compiled extension.

SHA-256 comparisons against the independently extracted installed firmware:

- Device HAL: `c655c6dde582d43f7a2c86255c584ddd0822aa107df2ed4c13d94f729a97e0ce`.
- PCIe module: `113c2cdba9704e41bffaad147b9496a913d284806934a4a0403c17a3dcc5e7d5`.

Both match byte-for-byte. No device module was loaded or firmware modified.

Audit implementation: `tools/qwen4exp/ple_device_abi_audit.py`.
Receipt: [ABI audit](ple-device-abi-audit-v1.json). It records source hashes,
import/export CRCs, provider releases, missing build files, and explicit false
flags for build compatibility and device queue access. `pyelftools` is required.

## Remaining work

The matching kernel Makefile, generated headers and module build tools remain
missing. Recovered CRCs do not replace kernel structure layouts or prototypes.
Do not patch version strings, inject CRCs, or force-load an incompatible module.
There is already a concrete prototype discrepancy: `dev_inc_open` declares
`agentdrv_get_ts_dma_chan_info` with five arguments, whereas `dev_inc` and the
public reference declare a device ID plus a six-field structure pointer.
The installed implementation must decide which ABI is valid.

Static disassembly now resolves this specific discrepancy: the installed
`agentdrv_get_ts_dma_chan_info` preserves `w0` as the device ID and `x1` as
the output pointer, then writes six 32-bit fields at offsets 0 through 20.
This agrees with the structure-pointer form. The instruction receipt is
`ple-device-ts-channel-abi-disassembly.txt`; relocation targets in that
unlinked module are not resolved by the plain disassembly. This function
was not called on hardware.

An isolated `genksyms` check of that source declaration, with Linux's
`__u32` to `u32` typedef chain, independently produces `0xfb69bf85`, exactly
matching its shipped export CRC. Ten other isolated DMA declarations did not
reproduce their shipped CRCs. Their complete preprocessed definitions remain
unverified; these partial snippets cannot prove either compatibility or
incompatibility. Receipts: `ple-device-interface-genksyms-v1.c` and
`ple-device-interface-genksyms-v1.json`. No CRCs were inserted into any module.

The next device extension needs a narrow, process-owned interface to the reserved
PCIe DMA channel: checked source descriptors, destination ownership, ordered
submission/completion, and revoke/drain before releasing mappings. Host source
registration alone does not implement that interface. Existing live probes
still report unsupported device HAL submission APIs; queue symbols alone are
not evidence of access from the custom AI CPU process.

The QFN v10 server is restored at port 8001 with context length 262144.
The final 12-case serving gate passed. The restoration supervisor was paced by
`restore-v10-thermal-guard.py` after a 90 C sample; the guard did not stop
serving workers and has now exited. Restoration and performance receipts are
recorded in `ple-driver-source-registration-v3.md`.

## Source-request follow-up

A targeted search of the two supplied trees and the original installer/rootfs
extraction found no kernel-devel RPM, kernel SRPM, generated `autoconf.h` or
`utsrelease.h`, `modpost`, or Linux source tarball. The only additional file
named `Module.symvers` was the existing securec-specific table. Archive metadata
also contained no module build/source links or RPM package database matching
the searched paths. This result is scoped to those local trees.

The [official Ascend notice](https://github.com/Ascend/mind-cluster/blob/master/Third_Party_Open_Source_Software_Notice.md#written-offer)
lists `foss@huawei.com` for source requests. This confirms a Huawei contact;
it does not establish that this separate project's offer covers HDK 26.0.rc1,
or that the exact kernel package is available there. The draft asks for routing
to the responsible device-kernel team.

A [firsthand Kunpeng environment report](https://github.com/vllm-project/vllm-ascend/issues/14799)
records `6.6.0-72.0.0.76.h21510.1170.eulerosv2r15.aarch64`, without `.ascend`.
It supports a related kernel-family lead, not a compatible build tree.
No exact downloadable package was found in the targeted public searches.

Prepared, unsent draft: `huawei-device-kernel-source-request.txt`.
Artifact identity attachment: `ple-kernel-source-request-evidence.json`.
No message was sent and no server or driver configuration was changed.

## AI CPU DMA-path probe v11

A private custom AI CPU kernel was compiled, loaded, executed, and unloaded
successfully while the QFN service stayed up. The probe opens known nodes
read-only and inspects HAL symbol prefixes; it performs no DMA or ioctl.
Seven HDC/SDMA symbol prefixes match the installed firmware HAL exactly.
The custom AI CPU process cannot open `/dev/hisi_hdc` (`EACCES`), and `/dev/sdma`
is absent (`ENOENT`). `/dev/svm0` remains readable. These results describe this
process, not every privileged service running on the device.

The installed `halSdmaCopyInner` disassembly shows fallback to `drvMemcpy` for
lengths at or below `0x9ffff`, an unavailable SDMA FD, or ioctl failure. Ordinary
memcpy must not be tried on an untranslated host virtual address. HDC has real
send/receive interfaces, but their presence does not prove arbitrary remote
host reads or access to our reserved queue. The next investigation is the
supported device-service access and session path, including whether it can
supply device-driven reads without host work per request.

Receipts: `ple-aicpu-interfaces-v11.jsonl`, `ple-aicpu-arm-dma-path-v11.json`,
and `ple-device-sdma-disassembly-v11.txt`. No firmware dump or write was needed.
