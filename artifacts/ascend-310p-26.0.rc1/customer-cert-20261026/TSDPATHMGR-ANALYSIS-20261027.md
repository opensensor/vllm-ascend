# TsdPathMgr path resolution and the AICPU signature policy — device-binary analysis

**Date:** 2026-10-27
**Status:** offline analysis complete, no card attached to this host
**Scope:** answer the open question from the previous session — what does
`TsdPathMgr::BuildDriverExtendPackageRootPath` return, and is it writable by
`tsdaemon`? Plus: which file actually controls the AICPU signature check.

Artifacts analysed:

| Path | What it is |
|---|---|
| `/tmp/kilo/devroot/usr/lib64/libtsdaemon.so` | device tsdaemon shared library (ARM64, **not** stripped — full C++ symbols) |
| `/tmp/kilo/devroot/var/drv_pkicms.ko` | device crypto/verification kernel module |
| `/tmp/kilo/devroot/var/drv_user_cfg.ko` | device user-config kernel module (`pss_cfg`) |
| `/tmp/kilo/dev.cpio.gz` | device rootfs (esbc container, 2574 entries) |
| `/home/matteius/ascend-310p-host-driver` | host driver source tree |

---

## 1. The answer to the open question

`BuildDriverExtendPackageRootPath` reads one member and returns one root.
The member table was recovered from the `TsdPathMgr` constructor (each member is
a `std::string` assigned one literal). The constructor is the function at
`libtsdaemon.so+0xd16c8`; it sets `x19 = 0x4b2d50` and stores into
`x19 + <offset>`:

```text
  member +0x020 = 'aicpuExtend_bin_hash.cfg'
  member +0x080 = '/usr/lib64/device-compat-plugin/'
  member +0x0a0 = '/home/HwHiAiUser/hs/'
  member +0x0c0 = 'omfile/'
  member +0x0e0 = 'runtime/bin/'
  member +0x100 = 'runtime/lib64/'
  member +0x120 = 'runtime/'
  member +0x140 = 'opp'
  member +0x160 = 'ops'
  member +0x180 = '/usr/lib64/device-sw-plugin/'
  member +0x1a0 = '/home/HwHiAiUser/device-sw-plugin/'
  member +0x1c0 = '/home/HwHiAiUser/aicpu_kernels/'
  member +0x1e0 = '/home/CustAiCpuUser'
  member +0x200 = 'aicpu_custom_scheduler'
  member +0x220 = '/var/aicpu_custom_scheduler'
  member +0x240 = 'aicpu_scheduler'
  member +0x260 = '/var/aicpu_scheduler'
  member +0x280 = 'queue_schedule'
  member +0x2a0 = '/var/queue_schedule'
  member +0x2c0 = 'hccp_service.bin'
  member +0x2e0 = '/var/hccp_service.bin'
  member +0x300 = 'comop/lib64/'
  member +0x320 = 'sea_'
  member +0x340 = '/usr/utils/rootfs'
  member +0x360 = '/home/HwHiAiUser/inuse/'
```

(Offsets not listed are either not string members or are set elsewhere.
Note `BuildKernelSoRootPath()` does *not* read member `+0x1c0`; it takes a
caller-supplied base and appends `"aicpu_kernels/"` to it.)

`BuildDriverExtendPackageRootPath` (at `libtsdaemon.so+0x14dfe0`) reads
**member +0x1a0 = `/home/HwHiAiUser/device-sw-plugin/`**, appends `<devId>` and
`/`:

```asm
  ldp  x2, x3, [x1, #0x1a0]      ; member +0x1a0  -> "/home/HwHiAiUser/device-sw-plugin/"
  ...
  __to_xstring("%u", devId)
  _M_append(...)
  _M_append("/")
```

so it returns:

```text
/home/HwHiAiUser/device-sw-plugin/<devId>/
```

**It is writable by tsdaemon.** `tsdaemon` runs as uid 1000 / gid 1000
(`/var/tsdaemon.yaml`: `uid: 1000, gid: 1000`), and in the device rootfs:

```text
home/HwHiAiUser   dir  mode=0750 uid=1000 gid=1000
```

So tsdaemon owns its home directory and can create the subdirectory. The
`/usr/lib64/device-sw-plugin/` member (+0x180) is the *other* root — that one
is `usr/lib64` mode `0555 uid=0 gid=0`, i.e. **not** writable by tsdaemon, and
it is also **absent from the device rootfs entirely** (no `device-sw-plugin`
entry under `usr/lib64/` at all).

### Why `originPkg` and `decomPkg` came out empty

Not because of permissions, and not because the path did not exist.

The member table is initialised in one function (`libtsdaemon.so+0xd16c8`), which
sets `x19 = 0x4b2000 + 0xd50 = 0x4b2d50` and then stores each literal into
`x19 + <member offset>`. `BuildDriverExtendPackageRootPath` reads exactly that
object (`adrp x1, #0x4b2000; add x1, x1, #0xd50; ldp x2, x3, [x1, #0x1a0]`), so
the member→literal mapping below is confirmed against the live object, not
inferred.

The two path builders use **different members**:

| Builder | Member | Root |
|---|---|---|
| `BuildDriverExtendPackageDecRootPath` | +0x180 | `/usr/lib64/device-sw-plugin/` |
| `BuildDriverExtendPackageRootPath` | +0x1a0 | `/home/HwHiAiUser/device-sw-plugin/` |

`+0x180` is `usr/lib64` (mode `0555 uid=0 gid=0`) and that directory **does not
exist at all** in the device rootfs — there is no `device-sw-plugin` entry
under `usr/lib64/`. `+0x1a0` is under `home/HwHiAiUser` (mode `0750 uid=1000
gid=1000`), which is writable by tsdaemon.

Now the part that actually explains the empty strings. The workers have a
common vtable (`BasePackageWorker`, 176 bytes, 22 slots). Two slots matter:

| Slot | Method |
|---|---|
| +0x20 | `LoadPackage` |
| +0x60 | `PreProcessPackage` |
| +0x68 | `SetDecompressPackagePath` |

The `originPkg` member is `+0x58` and `decomPkg` is `+0xb8` (confirmed from
the log call site at `+0x135580`: `ldr x7, [x20, #0x58]` is the `originPkg=%s`
argument, `ldr x1, [x20, #0xb8]` is `decomPkg=%s`).

- `+0x58` is written **only** by `BasePackageWorker::DefaultPreProcessPackage`
  (`stp x3, x2, [x19, #88]` at `+0x12dfb4`; 88 == 0x58).
- `+0xb8` is written by each worker's own `SetDecompressPackagePath()`.

Now compare how each worker is driven:

**`AicpuProcessPackageWorker::LoadPackage`** (`+0x138820`) makes three
virtual calls, in this order:

```text
  0x1388dc  slot 0x60  -> blr   PreProcessPackage
  0x1389f0  slot 0x68  -> blr   SetDecompressPackagePath
  0x138ae0  slot 0x88  -> blr   DecompressPackage
```

`AicpuProcessPackageWorker::PreProcessPackage` calls
`BasePackageWorker::DefaultPreProcessPackage` (at `+0x139740`), which fills
`+0x58`. So the AICPU worker's `originPkg` **is** populated, and its
`SetDecompressPackagePath` fills `+0xb8`.

**The driver-extend path** is driven by a local function at `+0x1375d8`
(anonymous namespace, so no symbol). It constructs a `DriverExtendPackageWorker`
and makes exactly **one** virtual call:

```text
  0x1377b8  ldr x2, [x2, #0x20]   -> blr x2   LoadPackage
```

There is no `slot 0x60` and no `slot 0x68` anywhere in that function — the
only `blr` in the whole body is the `LoadPackage` call. So for driver-extend
packages, **neither `PreProcessPackage` nor `SetDecompressPackagePath` is ever
invoked**.

**Therefore `originPkg` is empty because the driver-extend path never calls
the code that fills it.** The two members are only ever written from those two
vtable slots, and the driver-extend driver skips both. The worker's own
`LoadPackage` (`+0x135530`) only *reads* `+0x58` and `+0xb8` to build its log
line — it never populates them.

This is a device-binary behaviour, not a permissions problem, not a missing
directory, and not a delivery problem. `DriverExtendPackageWorker::PreProcessPackage`
and `SetDecompressPackagePath` are fully implemented and would work (the
former chains to `BuildDriverExtendPackageDecRootPath` → the `+0x180` root;
the latter calls `BuildDriverExtendPackageRootPath` → the correct `+0x1a0`
root). They are simply never called on this path.

The practical consequence: **the driver-extend (device-sw-plugin) install path
on this device build cannot work as shipped**, no matter what we deliver and no
matter what the signature policy says. Getting a package into
`/home/HwHiAiUser/device-sw-plugin/<devId>/` is necessary but not sufficient —
the path-computation code is never reached.

---

## 2. The signature policy file path — and why the host cannot help

The previous session assumed the policy file lives at
`/device-sw-plugin/verify/flag` relative to the install root, and that the
host's `/var/asdrv/.../verify/flag` was a *different* file. The first part is
wrong, and the second part turns out to be right for the wrong reason.

`CmsCbbVerifyPkg::GetVerifyFlagFilePath()` (at `libtsdaemon.so+0x18c358`)
builds:

```text
  "/var/asdrv/dev"  +  <devId as %u>  +  "/device-sw-plugin/verify/flag"
```

The literal `"/var/asdrv/dev"` is at `libtsdaemon.so+0x3b4958`; the suffix
`"/device-sw-plugin/verify/flag"` is at `+0x3b4b20`.

And the host driver writes **exactly the same path**:

`/home/matteius/ascend-310p-host-driver/dms/custom/dms_custom_common.h`:

```c
#define CUSTOM_SIGN_CONF_FLAG_PATH "/var/asdrv/dev%u/device-sw-plugin/verify/flag"
```

and `dms_custom_device.c` writes `verify_flag=%u\n` there.

**Important correction to the previous session's framing.** The host and the
device use the same *path string*, but they are **not the same file**, and
`/var/asdrv` is **not** a host↔device mount. The device-side module
(`dms_custom_device.c`, the one that owns the writer) reads and writes the
file with kernel file operations:

```c
  filp = filp_open(file_name, KA_O_RDONLY, 0);
  ...
  result = kernel_read(filp, buf, size, &offset_tmp);
```

`filp_open`/`kernel_read` resolve against the **device's own** filesystem. So
`/var/asdrv/dev<N>/device-sw-plugin/verify/flag` lives on the device, and the
host-side `dms_custom_host.c` writes the same path on the *host's* filesystem —
two independent copies of the same relative path, one per side.

This is consistent with `/var/asdrv` being absent from the device rootfs image
(a runtime mount point, like `/proc` and `/sys`) and with the device's
`drv_pkicms.ko` independently containing:

```text
/var/asdrv/dev%d/device-sw-plugin/verify/%s
flag
verify_flag=%d
"Package verify flag is 0, needn't verify."
```

So there are **two** policy files with the same path, one on each side, and
both the device's crypto module and the host's `dms_custom` code read their own
side's copy. Writing the host's copy would not change the device's behaviour.

### The host is gated out on 310P

`dms/custom/dms_custom.mk`:

```make
ifeq ($(TOP_DIR),)
    ifeq ($(DAVINCI_HIAI_DKMS),y)      # DKMS
        ifneq ($(filter $(TARGET_CHIP_ID), hi1980b),)
            drv_devmng_host-y += custom/dms_custom_host.o
            drv_devmng_host-y += custom/dms_custom_common.o
        endif
        ...
else # for CMake & ctrl cpu open
    ifneq ($(filter $(PRODUCT), ascend910B),)
        ifeq ($(PRODUCT_SIDE), host)
            drv_devmng_host-y += custom/dms_custom_host.o
            drv_devmng_host-y += custom/dms_custom_common.o
        else
            $(MODULE_NAME)-y += custom/dms_custom_common.o
            $(MODULE_NAME)-y += custom/dms_custom_device.o
        endif
        ...
endif
```

(The original uses tabs for indentation; they are spaces here to satisfy the
repository's markdownlint `MD010` rule.)

The 310P build is neither `hi1980b` (DKMS path) nor `ascend910B` (CMake path),
so **neither the host-side nor the device-side `dms_custom` objects are
compiled into the 310P driver.** That is the asymmetry the previous session
identified, now confirmed at source level rather than inferred.

`dms_custom_device.c` — the module that owns the *device-side* writer and that
registers `dms_set_custom_sign_flag` under `main_cmd=0x35,sub_cmd=0x2` — is
also excluded. The 310P device's `drv_devmng.ko` has **no** `custom_sign`
symbols and no `asdrv` strings, confirming the device-side writer is absent
from this build too.

Net effect on the 310P: **nothing on either side can write the policy file.**
The device's `drv_pkicms.ko` still *reads* it, so the read path is live — but
with no writer, the file simply does not exist on the device, and the read
fails. That is a further reason the "turn the signature off" plan was never
going to work: it needed a writer that this build does not contain.

---

## 3. `sign=1` — the previous session's framing is inverted

The device log line was:

```text
[KERNEL] [ascend] [user_cfg_drv] [devdrv_config_get_pss_cfg 4343] <tsdaemon:2861,3048>
    The sign details. (sign=1)
```

`sign` is **not** the signature-check enable flag. It is
`devdrv_config_get_pss_cfg()`'s output, and the host header defines:

`/home/matteius/ascend-310p-host-driver/dev_inc/inc/user_cfg_interface.h`:

```c
#define PKCS_SIGN_TYPE_OFF  1
#define PKCS_SIGN_TYPE_ON   0
```

So **`sign=1` means "PSS code-signing type is OFF"**, i.e. the device is
already configured to *not* use the PSS signature algorithm. It is the *opposite*
of what "sign=1 → signature required" suggests.

This matters because it means the observed `sign=1` is **not** evidence that a
signature blocked us. The actual stop was elsewhere.

### Where `sign` is consumed

`drv_pkicms.ko:soc_verify()` (at `+0x54b4`) calls
`devdrv_config_get_pss_cfg()` at `+0x55c4`:

```asm
  55c0:  sub   x1, x29, #0x3c       ; out-param slot
  55c4:  bl    devdrv_config_get_pss_cfg
  55c8:  cbnz  w0, 57e0             ; tests the RETURN CODE, not the sign value
  55cc:  ldur  w2, [x29, #-60]
  55d8:  bl    image_secure_auth
```

Two things follow.

First, `cbnz w0` at `+0x55c8` tests the **return status**
(`devdrv_config_get_pss_cfg` returns 0 on success, `-EINVAL` if the out-param
pointer is null — see `drv_user_cfg.ko+0x21e0`), not the sign value.

Second, and more decisive: **the sign value written to `[x29, #-0x3c]` is never
read.** A full scan of `drv_pkicms.ko` finds zero reads of that stack slot.
The value is fetched, stored, and discarded.

`devdrv_config_pss_cfg_init()` (`drv_user_cfg.ko+0x956c`) stores the boot-time
value into the module global `g_sign_type` (`drv_user_cfg.ko+0x956c`, called
from `init_module` with `w0 = 1`), and `devdrv_config_get_pss_cfg()`
(`drv_user_cfg.ko+0x2144`) returns a copy of it under a mutex.
`devdrv_config_set_pss_cfg()` (`+0x2228`) accepts only 0 or 1 — exactly the
`PKCS_SIGN_TYPE_ON`/`OFF` domain.

Across the whole device driver set, the sign value is read in exactly one
place that does anything with it: `drv_devmng.ko`'s
`devdrv_manager_ioctl_get_sign_type`, which copies it to userspace via
`copy_to_user_safe`. `drv_upgrade.ko` fetches it and passes it to
`sec_img_verify` as `w3` — but inside `sec_img_verify`
(`hsm_firmware_update.ko+0x15c0`) it is stored into `w26` and then into a
frame slot that is never read either. So on this platform the sign type is
**reported, not enforced**.

The `SIGN_FLAG_*` enum in `dms_custom_common.h` is a *different* thing
entirely — it is the **certificate trust policy** (`verify_flag`), which
selects which of vendor/user/community certificates are acceptable.
`SIGN_FLAG_DISABLED = 0` there means "no certificate required".

So there are two separate mechanisms, and they were conflated:

| Mechanism | Value | Where |
|---|---|---|
| `pss_cfg` / `sign` | `1` = OFF, `0` = ON (`PKCS_SIGN_TYPE_OFF/ON`) | `drv_user_cfg.ko`, algorithm choice |
| `verify_flag` | `0` = `SIGN_FLAG_DISABLED` (no cert required) | `/var/asdrv/dev<N>/device-sw-plugin/verify/flag`, trust policy |

The previous session's plan — "put `verify_flag=0` in the flag file to switch
the signature off" — targets the *trust policy*, which is the right knob for
the CMS/certificate path, but it is **not** the knob that produced `sign=1`.

---

## 4. Why the AICPU install actually stopped

Re-reading the device log chain with the corrected understanding:

```text
hdcd file transfer        OK
inotify watcher sees it   OK
check code == size        OK
path resolution           OK   (originPkg and decomPkg both populated)
PSS signature check      STOP (sign=1)
```

The AICPU package path *was* resolved correctly (`originPkg`/`decomPkg` both
populated). That is consistent with the code:

- `AicpuProcessPackageWorker::SetDecompressPackagePath()` calls
  `BuildAicpuPackageRootPath()`, which reads member `+0x1c0` =
  `/home/HwHiAiUser/aicpu_kernels/` — a path that exists and is writable by
  tsdaemon.
- `AicpuProcessPackageWorker::PreProcessPackage()` **does** chain to
  `BasePackageWorker::DefaultPreProcessPackage()`, which is what fills the
  `originPkg` member.

So the AICPU worker is sound on both counts, and it does not share the
driver-extend worker's defect.

**But the "PSS signature check STOP (sign=1)" label remains an inference, and
it is now a weaker one.** `sign=1` is `PKCS_SIGN_TYPE_OFF` — PSS signing is
*off*. The line is informational, not a stop condition. What we have not
established is what actually terminated the AICPU install after that line. The
device log has more lines after it that were not captured, and the stop reason
is almost certainly in them.

## 4a. Summary of the worker comparison

| | `AicpuProcessPackageWorker` | `DriverExtendPackageWorker` |
|---|---|---|
| driven by | `AicpuProcessPackageWorker::LoadPackage` (3 virtual calls) | local fn at `+0x1375d8` (**1** virtual call: `LoadPackage` only) |
| `PreProcessPackage` (slot 0x60) called? | **yes** | **no** |
| `SetDecompressPackagePath` (slot 0x68) called? | **yes** | **no** |
| root path builder reached | `BuildAicpuPackageRootPath` → member `+0x1c0` = `/home/HwHiAiUser/aicpu_kernels/` | never reached |
| root exists? | yes (`home/HwHiAiUser` mode 0750 uid 1000) | n/a |
| chains to `DefaultPreProcessPackage`? | **yes** (`+0x139740`) | n/a (never called) |
| `originPkg` populated? | yes | **no** |
| `decomPkg` populated? | yes | **no** |

---

## 5. Revised ranking of what's worth doing next

1. **Re-read the AICPU install failure without assuming it was the signature.**
   With `sign=1` = PSS-OFF established, the `sign=1` log line is not a stop
   condition. The real failure is after it, and the device log has more lines
   after it that were not captured. This is now the highest-value next step
   because it is cheap and it may make the rest of this moot.

2. **Stop treating the driver-extend (device-sw-plugin) path as the delivery
   mechanism for AICPU.** On this device build that path is dead: its driver
   makes a single virtual call (`LoadPackage`) and never reaches
   `PreProcessPackage` or `SetDecompressPackagePath`, so `originPkg` and
   `decomPkg` are always empty. This is broken independently of signatures and
   of what we deliver. The AICPU path does not go through it and is sound.

3. **The `ASCEND_AICPU_PATH` route (from the previous session's notes) remains
   valid and is independent of all of the above.** It does not touch the
   signature policy at all.

4. **The policy file is now the wrong target.** `verify_flag` lives on the
   device's own filesystem and is read by `drv_pkicms.ko` on the device side.
   The only way to write it is device-side code execution — which is the
   thing we are trying to achieve. Writing the host's copy does nothing for
   the device. The 310P build also gates out `dms_custom_device.o`, so the
   device has no writer for it either.

### What this analysis does *not* establish

The AICPU install failure is still unexplained. We know:

- the package reached the device, and the AICPU worker's path handling is
  correct (`originPkg`/`decomPkg` both populated);
- `sign=1` is PSS-OFF and is not consumed by anything on this path;
- the driver-extend path is separately broken, but that is a different worker.

So the stop is somewhere we have not read yet. The device log lines **after**
the `sign=1` line were not captured in the previous session, and they are
where the answer is. Re-running the transfer with a fuller log capture is
the cheapest next step, and it needs no new theory.

### Corrections to the previous session's notes

- The device-sw-plugin policy file is at
  `/var/asdrv/dev<N>/device-sw-plugin/verify/flag`, **not** at
  `/device-sw-plugin/verify/flag` relative to an install root. The previous
  session's `/usr/lib64/device-sw-plugin/verify/flag` guess was wrong.
- The host's `/var/asdrv/.../verify/flag` and the device's are the **same
  path string on two different filesystems**, not one shared file. The
  device-side code opens it with `filp_open`/`kernel_read`, so it resolves on
  the device. Writing the host's copy would not change device behaviour.
- `sign=1` means PSS signing is **OFF**, not ON — and the value is not read
  by any module on the AICPU path at all. The stop was misread.
- The empty `originPkg`/`decomPkg` is a device-side defect in the
  driver-extend **driver** (the local function at `libtsdaemon.so+0x1375d8`),
  not in the worker: it makes exactly one virtual call, `LoadPackage` (vtable
  slot 0x20), and never dispatches `PreProcessPackage` (0x60) or
  `SetDecompressPackagePath` (0x68) — the only two slots that ever populate
  those members. The worker's own implementations of both methods are
  present and correct but unreachable. It is **not** caused by a missing
  directory, not by permissions, and not by the `ascend_install_path`
  advertising failure.

---

## 6. Tooling notes

Analysis was done with a small set of scripts written for this session, kept
out of the repo:

- `/tmp/kilo/sym.py` — address→symbol resolution and symbol listing by
  substring for the stripped-or-not ARM64 ELFs.
- `/tmp/kilo/fn.py` — disassembly of a named function with `.rodata` string
  literals annotated inline (resolves ADRP+ADD pairs).
- `/tmp/kilo/callers.py` — BL call-site finder, used to work out which PLT
  stub a `bl` actually targets.
- `/tmp/kilo/members.py` — recovers the `TsdPathMgr` member→literal table
  from the constructor.
- `/tmp/kilo/xref.py` — ADRP+ADD address-materialisation xrefs.

Note for future work: `libtsdaemon.so` reports as "stripped" to `file`, but
its **dynamic and static symbol tables are intact** — the full C++ mangled
names are present (`_ZN3tsd10TsdPathMgr32BuildDriverExtendPackageRootPathB5cxx11Ej`
and so on). `file` under-reports here; `readelf -sW` does not.
