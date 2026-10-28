# Where the AICPU route actually fails, and what else is worth probing

**Date:** 2026-10-28
**Status:** offline analysis of the device binaries in `/tmp/kilo/devroot/`. No
card was attached to this host, so nothing here is hardware-validated. Every
offset is given so it can be re-checked.

This supersedes the "the signature blocked us" reading in the previous
session's notes. **The signature did not block us, and the previous session's
plan to disable it was aimed at the wrong knob.** The gate is a package
*name* check, and it is bypassable by construction.

---

## 1. The real gate: a package-name list

`PackageVerify::IsCmsVerifyPackage` (`libtsdaemon.so+0x12d228`) decides
whether CMS signature verification is even attempted. It is a sequence of
`std::string::find` calls — **substring** matches, not equality — against a
hardcoded list of six names:

```text
  +0x12d240  "Ascend-aicpu_syskernels.tar.gz"
  +0x12d274  "_Ascend-runtime_device-minios.tar.gz"
  +0x12d298  "_Ascend-opp_rt-minios.aarch64.tar.gz"
  +0x12d2b8  "Ascend-aicpu_extend_syskernels.tar.gz"
  +0x12d2d8  "Ascend-device-sw-plugin.tar.gz"
  +0x12d2f8  "transformer_tile_fwk_aicpu_kernel.tar.gz"
```

If the package name contains **none** of these, `IsCmsVerifyPackage` falls
through to `PackageProcessConfig::IsConfigPackageInfo(name)`
(`+0x12d30c`), which consults a config-driven list.

The chain is:

```text
  PackageVerify::IsSupportCmsVerify()        -> constant true   (+0x12cb70)
  PackageVerify::IsPackageNeedCmsVerify()     -> IsCmsVerifyPackage()   (+0x12d320)
  PackageVerify::VerifyPackageByCms()        -> VerifyPkgByCmsEnhance() (+0x12cbf8)
  CmsCbbVerifyPkg::VerifyPkgByCmsEnhance()   -> +0x18fa10
```

**Consequence:** a package named something outside that list — for example
`aicpu_custom.tar.gz`, or any name that does not *contain* one of the six
substrings — is not CMS-verified at all.

This is a much more promising direction than anything about signatures,
because it needs no key material, no flag file and no hash collision. It
needs a *different package name*.

### 1.1 What still applies to a non-CMS package

`VerifyPkgByCmsEnhance` also contains the `verify_flag` logic, and this part
is worth stating precisely because the previous session had it backwards:

```text
  +0x18fa9c  bl SetVerifyFlag(devId)          ; -> ParseVerifyFlagFromCfgFile
  +0x18faa4  bl IsVerifyFlagValid()            ; cmp w0,#7 ; cset w0,le
  +0x18faa8  tst w0,#0xff
  +0x18faac  b.eq  +0x18faf0                  ; -> skip verification
```

`ParseVerifyFlagFromCfgFile` (`+0x18c690`) reads
`/var/asdrv/dev<N>/device-sw-plugin/verify/flag` and parses `verify_flag=N`
with `strtoul`. Its return values:

| Situation | `verifyFlag` |
|---|---|
| file missing / no `verify_flag=` line | **1** (`w27 = 1` at `+0x18cb8c`) |
| `verify_flag=N` present | N |

`IsVerifyFlagValid` returns `verifyFlag <= 7`. And `SIGN_FLAG_TYPE` in
`dms/custom/dms_custom_common.h` is:

```c
  SIGN_FLAG_DISABLED       = 0,
  SIGN_FLAG_ONLY_VENDOR    = 1,
  SIGN_FLAG_ONLY_USER      = 2,
  SIGN_FLAG_VENDOR_OR_USER  = 3,
  SIGN_FLAG_ONLY_CMTY      = 4,
  SIGN_FLAG_VENDOR_OR_CMTY = 5,
  SIGN_FLAG_USER_OR_CMTY   = 6,
  SIGN_FLAG_ALL            = 7,
  SIGN_FLAG_MAX            = 8,
```

So the **default when the file is missing is 1 = `SIGN_FLAG_ONLY_VENDOR`** —
i.e. the device already defaults to "vendor-signed only", and the file
existing at all is not required for the default. Writing `verify_flag=0`
would *disable* verification, but the file is on the device's filesystem and
the 310P build compiles out the writer, so that route is closed regardless.

Note the branch polarity, which is the opposite of the previous session's
model: `IsVerifyFlagValid` returning **true** (flag ≤ 7) means the flag is
*usable* and verification **runs**; a flag **> 7** would be "invalid" and
verification would be skipped with the log line
`"verify pkg[%s] no cms just rewrite"`. Since the only way to get a flag
value onto the device is the same blocked route, this path is not the lever.

**But it does not matter for the name-based route**, because that route never
reaches `VerifyPkgByCmsEnhance` at all.

---

## 2. The AICPU install pipeline, in order

`AicpuProcessPackageWorker::LoadPackage` (`+0x138820`) dispatches, in
order:

```text
  +0x138888  slot 0x50  GetAllPackageHashCode
  +0x1388dc  slot 0x60  PreProcessPackage
  +0x1389f0  slot 0x68  SetDecompressPackagePath
  +0x138ac8  slot 0x78  MoveOriginPackageToDecompressDir
  +0x138ae0  slot 0x88  DecompressPackage
```

`AicpuProcessPackageWorker::PostProcessPackage` (`+0x138400`) then does:

```text
  +0x13844c  bl CheckPackageName            ; -> GetInnerPkgName (version.info)
  +0x138514  bl MoveSoToSandBox             ; -> GetSandBoxSoListInVersionFile
```

The vtable layout (recovered from `_ZTVN3tsd17BasePackageWorkerE` at
`+0x48f150`) is:

| Slot | Method |
|---|---|
| +0x50 | `GetAllPackageHashCode` |
| +0x60 | `PreProcessPackage` |
| +0x68 | `SetDecompressPackagePath` |
| +0x78 | `MoveOriginPackageToDecompressDir` |
| +0x88 | `DecompressPackage` |
| +0x98 | `PostProcessPackage` |
| +0xa0 | `Clear` |
| +0xa8 | `IsNeedUnloadPackage` |

The previous session's log showed the failure *after* path resolution, which
now lines up with `CheckPackageName` — the next step after
`SetDecompressPackagePath` and before `MoveSoToSandBox`. The relevant
device log lines are:

```text
  "Get inner package name success, name=%s"                  (+0x14115c)
  "Get inner package name failed, soInstallPath=%s"          (+0x3a2a80)
  "Walk in version file to get inner package name failed, path=%s"  (+0x13af4d0)
  "Package source name is not same as package inner name, srcPkgName=%s, innerPkgName=%s"
  "Check package name failed, ret=%u, soInstallPath=%s, name=%s"   (+0x138478)
```

**This is a strong, cheap hypothesis for the actual failure:** the package
was delivered, the paths were computed, and then the *name* check failed
because the tarball's outer name did not match the inner name recorded in
`version.info`, or because `version.info` was absent.

`CheckPackageName` (`+0x142898`) compares the source package name
(`GetSrcPackageName`, `+0x142428`) with the inner package name
(`GetInnerPkgName`, `+0x140b28`, which reads `version.info`). The
`version.info` path is `<base>aicpu_kernels_device/version.info`, where
`<base>` is `/usr/lib64/aicpu_kernels/<devId>/` — and that directory is
**uid 1000, mode 0750**, i.e. writable by tsdaemon.

So the failure is most likely that our rebuilt tarball did not carry a
`version.info` with a matching inner package name, or carried one under the
wrong path.

---

## 3. The `version.info` format, and the `SandBoxSo=` mechanism

`version.info` is a `key=value` file. Both its keys are now pinned down.

The two callbacks that read it are local (unnamed) functions in
`libtsdaemon.so`:

| Callback | Key literal | Key compared |
|---|---|---|
| `+0x140db8` | `+0x3afb68` | **`SandBoxSo=`** |
| `+0x140fb8` | `+0x3da6a0` | **`Name`** |

Both do the same thing: `find('=')` on the line, take the text before it as
the key and the text after it as the value, and compare the key against the
literal. So the file format is:

```text
Name=<inner package name>
SandBoxSo=<library file name>.so
```

`Name=` is the inner package name, used by `GetInnerPkgName`
(`+0x140b28`) and compared against the outer tarball name by
`CheckPackageName` (`+0x142898`).

`SandBoxSo=` is the sandboxed-library list, read by
`GetSandBoxSoListInVersionFile` (`+0x140c70`).

`AicpuPackageProcess::MoveSoToSandBox` (`+0x1411b0`, 1488 bytes) then, for
each named `.so`:

1. checks the value is non-empty — `"Sandbox so item value is empty, ret=%u."`
2. ensures the sandbox directory exists — `"Check or make sandbox dir failed, ret=%u, path=%s"`
3. **renames** the file into it — `"Rename sandbox so success, orgFile=%s, dstFile=%s"`

I disassembled the whole function: there is **no hash check, no signature
check and no path validation** between the config value and the `rename()`.
The only writer of the hash map is `TsdHashVerify::InsertOrUpdateHashValue`
(`+0x151fc0`), and it is not called from here.

The scheduler side confirms the other half: `libaicpu_scheduler.so` contains
`"Preload aicpu so %s success."` and `"Failed to preload aicpu so %s. Please
check."`, and `libaicpu_custom_scheduler.so` contains `dlopen`/`dlsym`.

**This is a genuine "the config file names the library the device loads"
primitive** — and now that the file format is known, it is writable by hand.
Both halves are located.

---

## 4. The seccomp constraint on what loaded code can do

`libaicpu_custom_scheduler.so` applies a seccomp filter to its threads:
`seccomp_init(0x50001)` (ALLOW) at `+0xcfce4`, then 31 `seccomp_rule_add`
calls in a loop bounded by `cmp x28, #0x1f` at `+0xcfcf4`, each allowing one
syscall with `SCMP_ACT_ALLOW` (`0x7fff0000`).

The allowlist, read from `.rodata` at `0x20c098`:

```text
   29  ioctl              62  lseek              135  rt_sigprocmask
   48  faccessat          63  read               160  uname
   56  openat             64  write              168  getcpu
   57  close              72  pselect6           172  getpid
   93  exit               78  readlinkat         178  gettid
   94  exit_group         79  fstatat            179  sysinfo
   98  futex              80  fstat              215  munmap
   99  set_robust_list    93  exit               222  mmap
  101  nanosleep         94  exit_group         226  mprotect
  115  clock_nanosleep   98  futex              233  madvise
  123  sched_getaffinity  99  set_robust_list    278  getrandom
  134  rt_sigaction     101  nanosleep
  135  rt_sigprocmask   115  clock_nanosleep
  160  uname            123  sched_getaffinity
  168  getcpu           134  rt_sigaction
```

(There is also one non-numeric entry, `0xffffd84a`, which is not a syscall
number and is probably a sentinel.)

**Not allowed: `execve` (221), `socket` (198), `connect` (203), `clone` (220),
`ptrace` (117), `kill`/`tgkill` (129/131), `chmod`/`fchmod` (90/91), `mount`,
`mknod`, `bpf`, `perf_event_open`, `unlink`/`unlinkat` (35), `rename` (82),
`truncate` (45).**

So code in the custom AICPU scheduler cannot fork, exec, open a socket, or
ptrace. It *can* `openat`/`read`/`write`, `mmap`/`mprotect`/`munmap`, and
`ioctl`. That last combination is the interesting one: `/dev/davinci*` and
`/dev/devmm_*` are `openat`-able, and the AICPU process already links
`libascend_hal.so` and `libdp.so`, so the device-driver ioctl ABI is
available without any injection at all.

The filter is applied **per thread** (`"Add the system call fialed, thread
threadIndex[%u]"`), so threads created before the filter is installed, and
the main scheduler process itself, may be unfiltered. Which threads get
filtered, and when, is worth checking separately.

---

## 5. The binary integrity gate

`/etc/mdc/base-plat/process-manager/bin_hash.cfg` (mode `0440`, uid 1000)
is a `key=value` SHA-256 map, and it covers the device's own binaries:

```text
  tsdaemon=060935c536ac176829015612c71e03e4b0a6b8531c2fa142914ab8ce89efaee1
  aicpu_scheduler=beb57bfc5840408176a2f3d58855ec9d449c783540e22c046dbb7a6483c84ebb
  aicpu_custom_scheduler=91c294d21f6c74fe362909f4f55871f9238223c53a2f560d0f242bbd79a5078d
  queue_schedule=4506d5cbc12939cc505a0694e2399d30aad307585dff8c234cdec8acab9ee18c
  hccp_service.bin=1b4c2f8416049a0d2f55d14abbd72547a4878e57530e69ea770a902c29b3078f
  ... (also iammgr, hdcd, dmp_daemon, resource_mgr, adda, adprof, slogd, sklogd, log-daemon)
```

The path is built by `TsdPathMgr::BuildKernelHashCfgPath` (`+0x14ce28`),
which appends member `+0x060` — the hash map itself — to
`aicpu_kernels_device/`. The parse is `TsdHashVerify::InsertOrUpdateHashValue`
(`+0x151fc0`), splitting on `'='` (`+0x152000`).

`TsdHashVerify::VerifyHashMapPostProcess` (`+0x153890`) contains
`"rm -rf /usr/lib64/aicpu_kernels/"` and the string `"/*"` — so on a hash
mismatch the device **deletes the kernel directory**.

This is a fail-closed gate on the *binaries*, and it is the reason we
cannot simply overwrite `aicpu_scheduler` on the device. It also means
**any `.so` we plant must have its hash recorded in this file**, if the file
is covered by the map at all. Whether the map covers `version.info` is one
of the two open questions below.

---

## 6. `LD_LIBRARY_PATH` redirection

`AicpuCustStartManager::BuildAicpuCustStartParam` (`+0x196e98`, 8184 bytes)
builds the custom AICPU subprocess environment and contains:

```text
  +0x197804  "LD_LIBRARY_PATH=" + BuildCompatPluginPackageSoInstallPath(devId)
```

`BuildCompatPluginPackageSoInstallPath` (`+0x14d0b8`) delegates to
`BuildCompatPluginPackageDecRootPath` (`+0x14cf40`), which reads member
**`+0x080` = `/usr/lib64/device-compat-plugin/`** and appends `<devId>/`.

So the custom AICPU process's `LD_LIBRARY_PATH` is
`/usr/lib64/device-compat-plugin/<devId>/`. That directory **does not exist**
in the device rootfs, and `/usr/lib64` is mode `0555 uid=0` — not writable by
tsdaemon. So this is a *named* directory on the device's library search path
but not one we can create ourselves; if it is ever created (by the
driver-extend install, say), it becomes a substitution point.

---

## 7. Probes, in order of cost

### Probe 1 — the package name (cheapest, highest value)

There are two name constraints, and they pull in opposite directions.

**Constraint A — the name must match the regex.** `GetSrcPackageName`
(`+0x142428`) builds a `std::regex` from the literal `"Ascend.*\.tar\.gz"`
(`+0x3af958`) and calls `regex_match` with `match_default` flags (`w4 = 0` at
`+0x14254c`) — a **full-string** match, not a search. So the outer package
name must:

- start with `Ascend`,
- end with `.tar.gz`.

If it does not, the log says `"Package name is invalid, name=%s"` and the
install stops there.

**Constraint B — the name must NOT be in the CMS list.** As established in
§1, `IsCmsVerifyPackage` does a **substring** `find` against six names, and
a name that matches none of them skips CMS verification entirely.

Constraint B is the softer one, because `Ascend.*\.tar\.gz` names that avoid
the six substrings still exist — for example `Ascend-aicpu_custom.tar.gz`
or `Ascend-x.tar.gz`. Neither of those contains any of the six
CMS-triggering substrings, and both satisfy constraint A.

The one thing to check before trying is the *second* half of the fallback:
`PackageProcessConfig::IsConfigPackageInfo` (`+0x148918`). A name not in
the hardcoded six may still be pulled in by the *config* list, so the config
file has to be read too. Its parser is
`PackageProcessConfig::ParseConfigDataFromFile` (`+0x14ad80`, 2580 bytes)
and its writer `SetConfigDataOnHost` (`+0x14a318`, 2660 bytes). The config
lives under `/etc/mdc/base-plat/process-manager/` alongside `bin_hash.cfg`.

**This probe is worth doing first because it costs nothing but a rename.**

### Probe 2 — the `version.info` file

Now that the format is known, the file is writable by hand:

```text
Name=<inner package name>
SandBoxSo=<library file name>.so
```

`Name=` must equal the outer tarball name, because `CheckPackageName`
(`+0x142898`) compares the source name against the inner name and logs
`"Package source name is not same as package inner name, srcPkgName=%s,
innerPkgName=%s"` if they differ.

The `version.info` path is `<base>aicpu_kernels_device/version.info`, where
`<base>` is `/usr/lib64/aicpu_kernels/<devId>/` — and that directory is
**uid 1000, mode 0750**, i.e. writable by tsdaemon. `aicpu_kernels_device/`
does not exist in the shipped rootfs, so it is created by the install.

So the tarball needs, at minimum, a `version.info` carrying a `Name=` equal
to the tarball's own name.

### Probe 3 — the device log

Still the cheapest source of truth, and now much better targeted. The
previous session captured the log only up to the `sign=1` line. With the
pipeline now known, the lines to look for are:

```text
  "Get inner package name success, name=%s"
  "Get inner package name failed, soInstallPath=%s"
  "Walk in version file to get inner package name failed, path=%s"
  "Check package name failed, ret=%u, soInstallPath=%s, name=%s"
  "Package source name is not same as package inner name, ..."
  "Sandbox so item value is empty, ret=%u."
  "Rename sandbox so success, orgFile=%s, dstFile=%s"
  "hash value compare error basevalue:%s, curvalue:%s, procfile:%s"
  "[CMSCBB_VERIFY] verify package[%s] by cms start, isMix[%d]"
  "verify pkg[%s] no cms just rewrite"
```

`"verify pkg[%s] no cms just rewrite"` is the one to watch for — it is the
log line that says the CMS check was *skipped*, which is the success case
for Probe 1.

### Probe 4 — the `SandBoxSo=` channel

Once the package installs, the `SandBoxSo=` entry in `version.info` names a
`.so` for the device to rename into the sandbox and for the scheduler to
preload. Two things need confirming first:

- the exact key for the inner package name in `version.info` (Probe 2), so
  the file we ship is well-formed;
- whether the hash map covers `version.info` (see below).

### Probe 5 — the ioctl surface from inside the AICPU process

Given `ioctl` and `openat` are allowed by seccomp and `libascend_hal.so` is
already loaded, the AICPU process is a privileged device client. Worth
checking whether the *device-driver* path is a shorter route to the MCU's
CRC-only regions than the AICPU route is.

---

## 8. The open questions

One remains, and it is answerable from the code without a card:

1. **What does `PackageProcessConfig::IsConfigPackageInfo` match on?** It is
   the fallback half of the CMS gate. If the config file lists more package
   names than the hardcoded six, renaming alone is not sufficient and the
   config file has to be dealt with too.

Everything else has been pinned down:

- the `version.info` format: `Name=<inner name>` and `SandBoxSo=<file>.so`;
- the outer-name regex: `Ascend.*\.tar\.gz`, full-string match;
- the CMS-trigger list: six names, substring match, with a config-driven
  fallback;
- the `SandBoxSo=` mechanism: a plain `rename()` with no validation between
  the config value and the move;
- the seccomp allowlist on the custom scheduler: 31 syscalls, `ioctl` and
  `openat` in, `execve`/`socket`/`clone`/`ptrace` out;
- the binary integrity gate: `bin_hash.cfg`, covering `tsdaemon`,
  `aicpu_scheduler`, `aicpu_custom_scheduler` and `queue_schedule`.

The one question that would close the loop: **does the hash map cover
`version.info`?** If it does not, the `SandBoxSo=` channel is unverified
even in a fully hash-checked install, and that is the cleanest path to code
execution on the card. `TsdHashVerify::InsertOrUpdateHashValue` (`+0x151fc0`)
is the only writer of the map, and `VerifyHashMapPostProcess` (`+0x153890`)
is where the "rm -rf /usr/lib64/aicpu_kernels/" cleanup on mismatch lives —
reading those two together should settle it.

---

## 9. Corrections to the previous session's notes

- **The signature was not the blocker.** `sign=1` is `PKCS_SIGN_TYPE_OFF`
  and is not read on this path; the device's default `verifyFlag` is already
  1 (`SIGN_FLAG_ONLY_VENDOR`) when the policy file is absent.
- **The policy file is on the device's filesystem**, opened by the
  device-side module with `filp_open`/`kernel_read`. The host's copy is a
  different file, and the 310P build compiles out both writers.
- **Writing `verify_flag=0` would have been the wrong move anyway** — it
  *disables* verification, and the branch polarity is that a flag ≤ 7 means
  "valid, verify now".
- **The driver-extend path is dead** on this build (one virtual call; the
  path-computation code is never reached), so it was never a delivery route
  for AICPU.
- **The empty `originPkg`/`decomPkg` was a separate, unrelated defect** in
  the driver-extend driver, and says nothing about the AICPU path, whose
  paths were populated correctly.
