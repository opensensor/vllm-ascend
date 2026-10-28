# Where we actually are: the AICPU prize, the circularity, and the spying question

**Date:** 2026-10-28
**Status:** brings together the 2026-10-26 assessment
(`SOFTWARE-ROUTES-20261026.md`), the 2026-10-27/28 analyses, and this
session's hardware tests. Card: `matteius@192.168.53.187`.

---

## 1. The one-paragraph answer

The previous session's assessment stands, and this session confirmed its
weakest link empirically rather than by argument: **the AICPU route needs
exactly one thing — device-side code execution — and every path to it that
we can reach from the host is currently blocked by a vendor signature or by
a circular dependency.** The AI Core route already gives device code
execution, but inside the AI Core's constrained environment, not the device
OS. The gap is not a missing tool; it is a missing *step* in a chain that
starts with a signed package.

On the spying question specifically: the card *is* a PCIe device with DMA
access to host memory, and I confirmed this session that an **ordinary
unprivileged host user** can map host memory into device address space via
the exported `drvHdcDmaMap`. That is a real capability and it deserves
attention on its own terms, separate from the AICPU work. It is also not
the same thing as "the card reads your PC's memory", and the distinction
matters — see §5.

---

## 2. The chain, and exactly where it breaks

```text
host (we control)              device OS (EulerOS aarch64)        MCU (Cortex-M)
--------------                 -------------------------            -------------
npu-smi set -t ssh-enable      sshd (OpenSSH 9.6p1)               SPI flash
  |                              |                                 elabel region
  | HDC over PCIe                 |                                 PPPP region
  |   drv_pcie_hdc_host            |                                 trustlist
  |   hdcd (device agent)         |                                 trust records
  |     |                          |                                 (CRC-16 only)
  |     | file transfer  <-- PROVEN: we can write to
  |     |   (V2, aicpu/ dir)        |   /home/HwHiAiUser/hdcd/device0/aicpu/
  |     |                          |
  |     | package install <-- PROVEN for the stock package:
  |     |                          |   "Load package success, type=9"
  |     |                          |   (CommonSinkPackageWorker)
  |     |                          |
  |     | SANDBOX WRITE    <-- BLOCKED: the .so is inside the signed
  |     |   (SandBoxSo=)            |   payload, so we cannot change it
  |     |                          |
  |     |  ... or DEVICE-SIDE CODE <-- BLOCKED: the package we can
  |     |     EXECUTION              |   deliver is signed; a modified
  |     |                          |   one is rejected
  |     |                          |
  |     +-- drvHdcDmaMap <-- PROVEN: an ordinary host user can map
  |            (host memory ->          |   host pages into device space
  |             device address space)   |
  |                                    +-- ipc_drv.ko (hisi-mailbox) ---+
  |                                    +-- drv_spi_smbus.ko (spi_mcu) --+--> SPI @0xA44B0000
  |                                                                         |
  |                                          forge elabel, recompute CRC-16,
  |                                          write via spi_mcu_reg_write
```

Two things are proven this session that were not before:

1. **The file channel reaches the device and the device acts on what
   arrives there.** The type-9 install completes: "Load package success,
   type=9". The delivery half of route D is closed.
2. **The AICPU scheduler's library path is reachable in principle** — the
   device launches it with `LD_LIBRARY_PATH` containing
   `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/` second, and
   `MoveSoToSandBox` renames the `SandBoxSo=`-named library into exactly
   that directory with no validation of any kind.

And the block is now precisely located rather than suspected:

> the `SandBoxSo=` name lives in `version.info`, which lives inside the
> package payload, which is signed. The signature is RSA-PSS (type 0) or
> PKCS#7 (type 9), both by the vendor. So the library we want the device to
> load is inside the thing we are not allowed to change.

The bypass that would break the circularity is the **`verifyFlag` policy
gate**: `IsVerifyFlagValid` is `verifyFlag <= 7`, and a value of 8 or more
makes `VerifyPkgByCmsEnhance` skip the CMS check entirely, logging
`"verify pkg[%s] no cms just rewrite"`. The flag is a plain integer in
`/var/asdrv/dev<N>/device-sw-plugin/verify/flag` **on the device**. The
310P build compiles out both `dms_custom_host.o` and `dms_custom_device.o`,
so nothing on either side writes it — and the device's default is 1
(`SIGN_FLAG_ONLY_VENDOR`).

**And the HDC channel cannot write it.** Tested this session: sending to
`/home/HwHiAiUser/hdcd/device0/../../../../var/asdrv/dev0/device-sw-plugin/verify`
passes the driver's `strstr` check (rc=25, not 52) but the device
**normalises the path** and writes the file back into the staging tree:

```text
[recv_data] Create file. (dstpth="/home/HwHiAiUser/hdcd/device0/9999_Ascend310P-aicpu_syskernels.tar.gz"; ...)
```

and the policy file is unchanged (`update verify flag old:1, new:1`). The
same held for the V2 base and for the whitelist entry's own
`save_sub_path`. There is no traversal.

## 2a. The one lever that is not the package installer

`CmsCbbVerifyPkg::SetVerifyFlag` → `ParseVerifyFlagFromCfgFile(devId)`
reads that policy file **on the device**. It is a file read by tsdaemon
(uid 1000). If the device's `tsdaemon` could be made to write it, the
circularity would break. Two candidate mechanisms, both device-side and
both requiring device code execution — so neither is a shortcut on its
own, but both are the *right* target once we have execution:

- the `usershadow` route (`USERSSHADOW-ROUTE-20261028.md`): the device's
  `drv_user_cfg.ko` whitelist permits the **`aicpu_scheduler` process** —
  uid 1000, the process our library would run in — to call
  `halSetUserConfig("usershadow", ...)`, and `usershadow` is exactly the
  item the device's `passwd` wrapper writes. So one library, loaded via
  `SandBoxSo=`, would let us set the device password *and* write the flag
  file.
- `/dev/spi_smbus` and `/dev/drv_smbus_intf` as uid 1000 on the device:
  **resolved, and it is bad news for route D.** The device's own dmesg says:

  ```text
  [    9.104433][ T1087] SPI driver spi_smbus has no spi_device_id for spi_mcu
  [    8.715869][  T969] drv_spi_slave: Unknown symbol smbus_slave_query_msg (err -2)
  ```

  `drv_spi_smbus.ko` **failed to bind to the `spi_mcu` device** on this
  card, so the SoC→MCU SPI transport the previous session identified is
  *not* present at runtime. The `spi_mcu` node in the device tree has
  `status = "ok"`, but the driver rejects it.

  What *is* present on the device:

    - `/dev/ascend_manager` (`ascend_manager_cdev_create`)
    - `/dev/drv_smbus_intf` (`smbus_slave_cdev_create`) — and it is *already
      failing* at boot: `smbus_secure_boot_get_flag ...
      Devdrv_driver_get_cert_info failed`,
      `smbus_secure_boot_get_cert ... ret=-27, cert_type=2`, and
      `ascend_manager ... Cert length too large. (cert_length=4294967295)`
    - `spi_slave_prod`'s cdev (the MCU→SoC direction, not the one we want)
    - the `ipcdrv` mailbox, initialised with 15+ channels
      (`MDC_IPC_SI_MBX0_RPID0_Q_RX_ACPU0` …
      `MDC_IPC_LP_MBX1_RPID1_Q_RX_ACPU1`)

  So route D as written is **closed on this card**: the SPI SMBUS transport
  does not bind. The MCU is still reachable in principle — over the
  `ipcdrv` mailbox, and possibly over `drv_smbus_intf` — but not by the
  route the previous session planned, and `drv_smbus_intf` is erroring at
  boot in a way that suggests the cert path is already unhappy.

  The ownership of `/dev/drv_smbus_intf` and `/dev/ascend_manager` is still
  unmeasured; they are created by udev at runtime and are not in the
  shipped rootfs.

## 3. What the AI Core route already gives, and does not

The AI Core route is complete and proven: a custom AscendC kernel written on
the host, shipped by the host's CANN inside normal op dispatch, executed by
the device's AI Core, with numeric proof
(`median(mod/base) = 2.000000`, `max|mod - 2*base| = 0.000000`).

It is worth being precise about what that is and is not, because it is easy
to over-claim:

- it is **real device code execution**, and the AI Core is a separate
  processor with its own memory and its own (restricted) execution
  environment;
- it is **not** code execution in the device OS — it does not give us
  `/dev/spi_smbus`, the mailbox, or a shell;
- the AI Core's own binary is loaded from the device filesystem, and the
  device verifies the *host-shipped* op binaries by a plain SHA-256 in the
  op descriptor — **no signature**. That is why the AI Core route works and
  the AICPU one does not: the AI Core's integrity model is a hash the host
  recomputes, while the AICPU package's is a vendor signature over the
  container.

That asymmetry is the whole story, and it is worth being careful not to
generalise "we have device code execution" from the AI Core result.

## 4. The AICPU integrity model, restated precisely

| | container | signature | verified by | who holds the key |
|---|---|---|---|---|
| AI Core op binary | host-side op descriptor | plain SHA-256, recomputed by the host | host CANN | nobody — it is a hash |
| AICPU type 0 | esbc | RSA-PSS 512 bytes, Huawei CSI CA | **kernel** `soc_verify` → `rsa_verify_soc` | the vendor |
| AICPU type 9 | `ascend_aicpu` | PKCS#7 in a `cms\0` block | tsdaemon user space, `VerifyAicpuPkgByCms` | the vendor |

The type-9 route was the hope — it is the one the device verifies in user
space rather than in the kernel. It is not a bypass: the PKCS#7
`messageDigest` covers the content, and the `ascend_aicpu` container magic
`0x55aa55aa` at the *start* of the file is what makes the device parse the
CMS block at all. This session's three experiments (varying only the first
four bytes) confirmed that a package without the magic is an *error*, not a
"skip".

## 5. The spying question, addressed on its own terms

You said you believe these cards spy on the user's PC. That is worth taking
seriously, and there is a concrete technical reason to.

**What I confirmed this session.** `libascend_hal.so` exports
`drvHdcDmaMap(mem_type, buf, devid)`, and an **ordinary unprivileged host
user** (`matteius`, in group `HwHiAiUser`) successfully called it on a page
of their own memory:

```text
  drvHdcMallocEx(mem_type=0) = 0x7cae00128000
    drvHdcDmaMap(mem_type=0) = 0
    *** a host page is now mapped into device address space ***
  ... (same for mem_type 1, 2 and 3)
```

So: any process on the host that can open `/dev/davinci*` — and those nodes
are `crw-rw---- HwHiAiUser HwHiAiUser`, and `matteius` **is** in the
`HwHiAiUser` group — can map host memory into the device's address space.
The device is a PCIe device; it can read what is mapped to it. The driver
does track the session uid (`HdcPcieGetSessionUid`, "No permission.
(dev_id=%u; uid=%d)"), so there is a notion of per-session identity, but
for *our own* pages it does not restrict us.

**What that does and does not establish.**

- It does *not* by itself show the card reads another process's memory.
  Mapping requires a handle to the page; the interesting question is
  whether the driver lets a process map pages it does not own, or whether
  the device can walk host mappings on its own. I did not test either, and
  the second is the one that would matter for a "the card reads your PC"
  claim.
- It *is* the mechanism by which a hostile card firmware (or a compromised
  device OS) could read host memory: the device's own address space can
  reach anything the host has mapped, and the device's DMA engine is
  driven by the host driver on request. That is inherent to a PCIe
  accelerator, and it is a hardware-trust question rather than a software
  bug — the same is true of a GPU.
- It is also the *reason* the AICPU prize is worth having. A device OS we
  can run code in is a device OS we can audit, or at least observe. A card
  whose device-side firmware we cannot inspect is exactly the situation
  that makes the "spying" concern unanswerable from the outside.

**What I did not do.** I did not attempt to read another process's memory,
and I would not: on a card you own, the useful thing is to establish what
the device *can* do and then get control of it, not to build a working
memory-scraping tool. If the goal is an answer you can rely on — "does this
card read my PC's memory?" — the honest answer today is *"the host driver
gives an unprivileged user DMA access to device space, and the device can
read whatever the host maps; whether it does so is not established"* — and
the way to settle it is to get code running in the device OS and watch,
which is exactly the AICPU work.

## 6. Recommendations, in order

1. **Route D is closed as written.** `drv_spi_smbus.ko` does not bind to
   `spi_mcu` on this card, so the SoC→MCU SPI transport the previous
   session planned is absent at runtime. The remaining MCU paths are the
   `ipcdrv` mailbox (live, 15+ channels) and `/dev/drv_smbus_intf` (present
   but erroring at boot). Re-map the MCU reachability around those two
   before spending more time on the SPI theory.
2. **Measure the ownership of `/dev/drv_smbus_intf` and
   `/dev/ascend_manager`.** They are created by udev at runtime, so they are
   not in the shipped rootfs and their mode has never been checked. If
   either is group-writable by uid 1000, that is the MCU foothold.
3. **Re-read the device's `/usr/lib64/aicpu_kernels/aicpu_package_install.info`
   and the host's `install.sh`.** The host installer creates the device-side
   install tree (`mkdir /usr/lib64/aicpu_kernels/0/aicpu_kernels_device`,
   `touch aicpu_package_install.info`, `chown HwHiAiUser`) — i.e. the *host*
   creates the device's AICPU kernel tree, and the device has an "install
   info" file. If the host can influence what the device installs there, or
   what that file names, there is a route that does not go through the
   signed container at all. This is the most promising untested lead.
4. **Consider whether the AI Core's unsigned-hash integrity model can be
   reached for the AICPU path.** The AI Core op binary is verified by a
   plain SHA-256 the host computes — no key, no signature. The AICPU
   package is verified by a vendor signature. That asymmetry is the whole
   problem; if any AICPU-adjacent artefact uses the AI Core's model, it is
   exploitable.
5. **Decide what the deliverable is.** If the goal is evidence about the
   spying question, the strongest deliverable is not "we got root on the
   card" but "we can read what the card does" — which needs execution in
   the device OS, and therefore the same steps 1-4.

## 7. Card state

- device log level INFO; `/etc/hdcFileTransWhiteList.cfg` and
  `ascend_package_load.ini` restored to their original md5s
  (`7baeaca596f22c7ddb711ba91b2bdb2b`, `baca3a99c7034f140c2e7ce89d42b4c0`);
- host AICPU package still stock (md5 `ccf7ed4c6cef44e7e0dac5c2708cf849`);
- 6/6 chips OK, 0 errors, no NPU processes;
- during this session the device was sent several non-stock packages, all
  rejected before installing anything; the only packages that installed
  were byte-identical to the stock legacy package;
- the stock legacy package was re-delivered at the end so the device's
  check-code state is consistent.
