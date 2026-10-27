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

`BuildDriverExtendPackageRootPath` returns **two** candidate roots, and picks
between them. The member table was recovered from the `TsdPathMgr` constructor (each member is
a `std::string` assigned one literal). The constructor is the function at
`libtsdaemon.so+0xd16c8`; it sets `x19 = 0x4b2d50` and stores into
`x19 + <offset>`:

```
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

```
  ldp  x2, x3, [x1, #0x1a0]      ; member +0x1a0  -> "/home/HwHiAiUser/device-sw-plugin/"
  ...
  __to_xstring("%u", devId)
  _M_append(...)
  _M_append("/")
```

so it returns:

```
/home/HwHiAiUser/device-sw-plugin/<devId>/
```

**It is writable by tsdaemon.** `tsdaemon` runs as uid 1000 / gid 1000
(`/var/tsdaemon.yaml`: `uid: 1000, gid: 1000`), and in the device rootfs:

```
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

Now the part that actually explains the empty strings. In
`DriverExtendPackageWorker`:

- `originPkg` is member `+0x58`, `decomPkg` is member `+0xb8` (confirmed from
  the log call site at `+0x135580`: `ldr x7, [x20, #0x58]` is the `originPkg=%s`
  argument, `ldr x1, [x20, #0xb8]` is `decomPkg=%s`).
- `+0x58` is written **only** by `BasePackageWorker::DefaultPreProcessPackage`
  (`stp x3, x2, [x19, #88]` at `+0x12dfb4`; 88 == 0x58).
- `+0xb8` is written by each worker's own `SetDecompressPackagePath()`.

`AicpuProcessPackageWorker::PreProcessPackage` calls
`BasePackageWorker::DefaultPreProcessPackage` (at `+0x139740`) — so the AICPU
worker fills `+0x58` and its `originPkg` is populated.

`DriverExtendPackageWorker::PreProcessPackage` **does not call it**. It calls
`BuildDriverExtendPackageDecRootPath` (the `+0x180` root) and writes the result
into members `+0x128`/`+0x148`/`+0x168` — none of which is `+0x58`. Its
`LoadPackage` (`+0x135530`) only *reads* `+0x58` and `+0xb8` to build the log
line; it never populates them.

**So `originPkg` is empty because the driver-extend worker overrides
`PreProcessPackage` without chaining to the base implementation that fills it.**
That is the actual defect, and it is a device-binary behaviour, not a
permissions problem and not a missing-directory problem.

`SetDecompressPackagePath()` does populate `+0xb8` — but from `+0x128`
(the value `PreProcessPackage` wrote from the `+0x180` root) and from the
package name. Since `+0x128` is rooted at the non-existent
`/usr/lib64/device-sw-plugin/`, `decomPkg` ends up empty too.

The practical consequence: **the driver-extend (device-sw-plugin) install path
on this device build cannot work as shipped.** Getting a package into
`/home/HwHiAiUser/device-sw-plugin/<devId>/` is necessary but, on this
analysis, not sufficient — the `originPkg` member is never filled by this
worker at all.

---

## 2. The more important discovery: the signature policy file path

The previous session assumed the policy file lives at
`/device-sw-plugin/verify/flag` relative to the install root, and that the
host's `/var/asdrv/.../verify/flag` was a *different* file. **That is wrong —
they are the same file.**

`CmsCbbVerifyPkg::GetVerifyFlagFilePath()` (at `libtsdaemon.so+0x18c358`)
builds:

```
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

**So `/var/asdrv` is a shared mount between host and device.** The host's
`/var/asdrv/dev<N>/device-sw-plugin/verify/flag` *is* the device's
`/var/asdrv/dev<N>/device-sw-plugin/verify/flag`. The device's own
`drv_pkicms.ko` confirms it independently:

```
/var/asdrv/dev%d/device-sw-plugin/verify/%s
flag
verify_flag=%d
"Package verify flag is 0, needn't verify."
```

`/var/asdrv` does not exist in the device rootfs image, which is consistent
with it being a runtime mount (like `/proc`, `/sys`, `/dev`), not a baked-in
directory. This is why the device-side `GetVerifyFlagFilePath()` has a
`/var/asdrv/dev` prefix at all — it is reaching into host-provided storage.

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

The 310P build is neither `hi1980b` (DKMS path) nor `ascend910B` (CMake path),
so **neither the host-side nor the device-side `dms_custom` objects are
compiled into the 310P driver.** That is the asymmetry the previous session
identified, now confirmed at source level rather than inferred.

`dms_custom_device.c` — the module that owns the *device-side* writer and that
registers `dms_set_custom_sign_flag` under `main_cmd=0x35,sub_cmd=0x2` — is
also excluded. The 310P device's `drv_devmng.ko` has **no** `custom_sign`
symbols and no `asdrv` strings, confirming the device-side writer is absent
from this build too.

Net effect on the 310P: nothing can write the policy file, and nothing reads
it from the `/var/asdrv` path.

---

## 3. `sign=1` — the previous session's framing is inverted

The device log line was:

```
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

`drv_pkicms.ko:soc_verify()` calls `devdrv_config_get_pss_cfg()` at
`+0x55c4` and passes the result into `image_secure_auth()` at `+0x55d8`.
`devdrv_config_pss_cfg_init()` stores the boot-time value into the global
`g_sign_type` (`drv_user_cfg.ko+0x956c`).

`devdrv_config_get_pss_cfg()` is also called from `pkicms_verify_pkg_cms` path
via `soc_verify`. The `SIGN_FLAG_*` enum in `dms_custom_common.h` is a
*different* thing — it is the **certificate trust policy** (`verify_flag`),
which selects which of vendor/user/community certificates are acceptable.
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

```
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
| root path builder | `BuildAicpuPackageRootPath` → member `+0x1c0` = `/home/HwHiAiUser/aicpu_kernels/` | `SetDecompressPackagePath` → member `+0x1a0` = `/home/HwHiAiUser/device-sw-plugin/`; `PreProcessPackage` → member `+0x180` = `/usr/lib64/device-sw-plugin/` |
| root exists? | yes (`home/HwHiAiUser` mode 0750 uid 1000) | `+0x180` root **does not exist**; `+0x1a0` root exists |
| chains to `DefaultPreProcessPackage`? | **yes** (`+0x139740`) | **no** |
| `originPkg` populated? | yes | **no — never filled** |
| `decomPkg` populated? | yes | empty (rooted at non-existent `+0x180`) |

---

## 5. Revised ranking of what's worth doing next

1. **Re-read the AICPU install failure without assuming it was the signature.**
   With `sign=1` = PSS-OFF established, the `sign=1` log line is not a stop
   condition. The real failure is after it, and the device log has more lines
   after it that were not captured. This is now the highest-value next step
   because it is cheap and it may make the rest of this moot.

2. **Stop treating the driver-extend (device-sw-plugin) path as the delivery
   mechanism for AICPU.** The driver-extend worker on this device build never
   fills its `originPkg` member (it overrides `PreProcessPackage` without
   chaining to the base implementation). That path is broken independently of
   signatures and of what we deliver. The AICPU path does not go through it.

3. **The `ASCEND_AICPU_PATH` route (from the previous session's notes) remains
   valid and is independent of all of the above.** It does not touch the
   signature policy at all.

4. **If the policy file is still needed: determine whether `/var/asdrv` is
   actually a host↔device mount on this platform.** The path match between the
   host writer and the device reader is exact, and `/var/asdrv` is absent from
   the device rootfs. If it is shared, the host *could* write the policy file
   directly — but only if the 310P host driver had `dms_custom_host.o`, which
   the gating excludes. Building that object for the 310P is a small host-side
   change and is the only route to the policy file on this platform.

### Corrections to the previous session's notes

- The device-sw-plugin policy file is at
  `/var/asdrv/dev<N>/device-sw-plugin/verify/flag`, **not** at
  `/device-sw-plugin/verify/flag` relative to an install root. The previous
  session's `/usr/lib64/device-sw-plugin/verify/flag` guess was wrong.
- The host's `/var/asdrv/.../verify/flag` and the device's are the **same
  file**, not parallel ones.
- `sign=1` means PSS signing is **OFF**, not ON. The stop was mis-attributed.
- The empty `originPkg`/`decomPkg` is a device-side defect in
  `DriverExtendPackageWorker::PreProcessPackage` (it does not chain to
  `BasePackageWorker::DefaultPreProcessPackage`, which is the only writer of
  the `originPkg` member), compounded by the worker rooting its decompress path
  at the non-existent `/usr/lib64/device-sw-plugin/` instead of
  `/home/HwHiAiUser/device-sw-plugin/`. It is **not** caused by a missing
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
