# The AICPU install works — hardware-validated

**Date:** 2026-10-28
**Card:** Ascend 310P, host `matteius@192.168.53.187`, 6/6 chips OK
**Status:** hardware-validated. The AICPU package install path is confirmed
working end to end on the card.

This supersedes the failure analysis in `AICPU-EXECUTION-ROUTES-20261028.md`
(§4), which said the AICPU install failure was unexplained. It is now
explained: **there was no failure.** The host-side return code was the
misleading part.

---

## 1. The headline result

A plain, unmodified load of the stock `Ascend310P-aicpu_syskernels.tar.gz`
succeeds on the device. From the device's own log
(`slog/dev-os-0/run/device-os/device-os_*.log`, collected by
`msnpureport -t 1`):

```text
[inotify_watcher.cpp:362][LoadPackage]
  Begin to load package, type=0, path=/home/HwHiAiUser/hdcd/device0/,
  fileName=1830147_Ascend310P-aicpu_syskernels.tar.gz

[aicpu_process_package_worker.cpp:58][LoadPackage]
  Start load package,
  originPkg=/home/HwHiAiUser/hdcd/device0/1830147_Ascend310P-aicpu_syskernels.tar.gz,
  decomPkg=/home/HwHiAiUser/aicpu_kernels/vf0_1830147_Ascend310P-aicpu_syskernels.tar.gz,
  checkCode=0, pkgSize=53559460

[aicpu_package_process.cpp:166][MoveSoToSandBox]
  Rename sandbox so success,
  orgFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libtensorflow.so,
  dstFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/libtensorflow.so

[inotify_watcher.cpp:372][LoadPackage]
  Load package success, type=0, path=/home/HwHiAiUser/hdcd/device0/,
  fileName=1830147_Ascend310P-aicpu_syskernels.tar.gz
```

And the AICPU process is then started on the device:

```text
[tsd_common.cpp:1329][ExecuteStart] get pid[24010], sub process name[aicpu_sd_24010]
[tsd_common.cpp:2425][CheckSum] Check aicpu package in os[0] vfId[0] checkCode[0]
                               extendCheckCode[0] ascendcppCheckCode[0] success
```

with the device-side kernel log showing the scheduler registering as a
device task:

```text
[ascend][xsmem][sp_inst_create 217] <aicpu_scheduler:24010> Add task success. (pid=24010; sp_id=100001)
[ascend][xsmem][xsmem_pool_create 822] <aicpu_scheduler:24010> create pool id 49
```

**The device executes AICPU code.**

---

## 2. Why the previous session thought it failed

The host-side return was `fl=31`. That is a **host-side timeout**, not a
device-side rejection. The device had already completed the install
successfully ~4 seconds earlier; the host was still waiting for a reply
message that the device had no reason to send.

The `sign=1` line that the previous session read as the stop point is, in the
same log, immediately followed by:

```text
[ascend][pkicms][soc_verify 636] <tsdaemon:2861,3048> Soc verify success. (dev_id=0)
[ascend][upgrade][dev_upgrade_verify_image 2350] <tsdaemon:2861> dev(0) cover file without head.
```

`Soc verify success` — the signature check **passes**. The second line is
`dev_upgrade_verify_image` with the `flags` argument's bit 0 set
(`tbz w25, #0` at `drv_upgrade.ko+0x5c00` skips the message when the bit is
clear), i.e. it is an informational note that the file being verified has no
cover header. It is not an error, and it is not the last thing that happens.

The `sign=1` value itself is `PKCS_SIGN_TYPE_OFF` — PSS signing is *off* —
and as established earlier it is stored by `soc_verify` and never read.

**So: the previous session's "PSS signature check STOP" was a misread of an
informational log line, and the install had been succeeding all along.**

---

## 3. The `SandBoxSo=` channel is real, live, and on the library path

This is the important structural finding, and it is now confirmed on the
device.

The stock package already contains a `SandBoxSo=` entry. From
`aicpu_kernels_device/version.info` inside the stock tarball:

```text
Version=9.1.T11.0.B110
timestamp=20260730_231653901
Name=Ascend310P-aicpu_syskernels.tar.gz
SandBoxSo=libtensorflow.so
Featurelist=AICPU_PROF_V2
Trustlist=libhccl_kernel_compat.so,libscatter_aicpu_kernel.so,...,Ascend310P-v9.9-libopmaster.so
```

and the device acted on it — `MoveSoToSandBox` renamed the library:

```text
orgFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libtensorflow.so
dstFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/libtensorflow.so
```

**The critical part is where `sand_box/` sits.** The device launches the
AICPU scheduler with this environment (`launch_strategy_fork.cpp:105`):

```text
envs=[LD_LIBRARY_PATH=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/:
                       /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/:
                       /home/HwHiAiUser/hs/0/runtime/lib64/:
                       /usr/lib64/device-compat-plugin/0/:
                       /usr/lib64/device-sw-plugin/0/device-sw-plugin/:
      DATAMASTER_RUN_MODE= ASCEND_AICPU_PATH= ASCEND_SLOG_PRINT_TO_STDOUT= ]
binPath=/var/aicpu_scheduler
```

`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/` is **second on
`LD_LIBRARY_PATH`** for the AICPU scheduler process, and it is exactly the
directory `MoveSoToSandBox` writes into. The `SandBoxSo=` name in
`version.info` therefore selects, by name, the library the AICPU scheduler
will load ahead of the others.

`MoveSoToSandBox` is a plain `rename()` with, as established by disassembly,
**no hash check, no signature check and no path validation** between the
config value and the move.

---

## 4. The hash map does not cover `version.info` — the open question, answered

The device log shows exactly which files enter the hash map
(`tsd_hash_verify.cpp:66`, `hashMapType:1`):

```text
key:libaicpu_extend_kernels.so  value:c28b88d2...
key:libaicpu_kernels.so          value:74f888c0...
key:libccl_kernel_plf.so         value:1ad5ad6a...   (and 63744faf...)
key:libccl_kernel.so             value:43112ace...   (and 4fb5e346...)
key:libcpu_kernels.so            value:596121f4...
key:libpt_kernels.so             value:ffe08454...
key:libtensorflow.so             value:63eb2f59...
key:libtf_kernels.so             value:8d0eb713...
```

**Every key is a `.so`.** `version.info` does not appear. The map is the
`bin_hash.cfg` content, and `bin_hash.cfg` in the stock package lists only
the five kernel libraries:

```text
libtensorflow.so=63eb2f59...
libtf_kernels.so=8d0eb713...
libaicpu_kernels.so=74f888c0...
libcpu_kernels.so=596121f4...
libpt_kernels.so=ffe08454...
```

So **`version.info` is an unverified control channel**. The file that names
the library the device loads is not itself integrity-checked.

---

## 5. `ASCEND_AICPU_PATH` is host-controlled

Verified with a `getenv` interposer on the host:

```text
[getenv] ASCEND_AICPU_PATH -> /tmp/MARKER_AICPU
```

The host library reads it from the process environment, and it travels to
the device as the `ascend_aicpu_path` field of the `tsd.AscendAicpuPath`
protobuf message, which `TSDaemon::InitTsdImplParaFromHdcMsg` reads
(`libtsdaemon.so+0xda3f0`, logging `"Ascend aicpu path is %s"`).

The device's `InotifyWatcher::ProcessHeterogeneousTsdClientEvent`
(`libtsdaemon.so+0x1199b8`) uses it to decide where to watch for AICPU
packages:

```text
  0x1199f8  "ASCEND_AICPU_PATH"          -> GetScheduleEnv(name, &value)
  0x119a14  cbnz x0, ...                 -> use the host-supplied value
  0x119a20  "/usr/local/Ascend/"         -> default base
  0x119a60  "Ascend310P/aicpu/"          -> watched subdir
  0x119ab8  "/opp/"
  0x119b8c  "Ascend310P-aicpu_syskernels.tar.gz"
```

So the watched directory is `<base>/opp/Ascend310P/aicpu/`, with `base` being
either `ASCEND_AICPU_PATH` (host-controlled) or `/usr/local/Ascend/`.

Note the device-side `GetScheduleEnv` calls `getenv` on the *device*
(`libtsdaemon.so+0x124794`), so the device also honours its own environment;
the host value arrives through the protobuf field.

In the observed run the device used the default, and the packages actually
arrived through a different channel: the HDC file staging directory
`/home/HwHiAiUser/hdcd/device0/`, watched by inotify (`type=0` for
syskernels, `type=9` for the legacy and hcomm packages). That staging path
is the one the host writes to, and it is where a modified package would
land.

---

## 6. What this means for code execution on the card

The pieces now line up:

| # | Fact | Evidence |
|---|---|---|
| 1 | The AICPU package installs on the device | device log: `Load package success` |
| 2 | The device starts the AICPU scheduler with `LD_LIBRARY_PATH` including `.../aicpu_kernels_device/sand_box/` | device log: `launch_strategy_fork.cpp:105` |
| 3 | `sand_box/` is populated by `MoveSoToSandBox` from the `SandBoxSo=` key in `version.info` | device log: `Rename sandbox so success` |
| 4 | `version.info` is **not** covered by the hash map | device log: hash map keys are all `.so` |
| 5 | `MoveSoToSandBox` does a plain `rename()` with no validation | disassembly of `libtsdaemon.so+0x1411b0` |

Together: **a package whose `version.info` names our library in
`SandBoxSo=` gets that library renamed into `sand_box/`, which is second on
the AICPU scheduler's `LD_LIBRARY_PATH`.** The scheduler is then started by
the device, in the device OS, as uid 1000, and `dlopen`s it.

That is arbitrary code execution in the device OS, achieved by a normal
package install — no signature bypass, no exploit, no hash collision.

The remaining constraint is the one identified earlier: the AICPU scheduler
runs under a seccomp filter (`libaicpu_custom_scheduler.so`, 31 allowed
syscalls; `ioctl`, `openat`, `read`, `write`, `mmap`, `mprotect` allowed;
`execve`, `socket`, `connect`, `clone`, `ptrace` not). The filter is applied
per thread, so the scheduler's main thread and any thread started before the
filter is installed may be unfiltered — worth checking.

---

## 7. Why the size gate exists and how it was beaten

The `checkCode` in the device log is the package **size**, not a hash:

```text
Start load package, ... checkCode=0, pkgSize=53559460
Load package success, ... checkCode=53559460
```

`0x30BE24` = 53559460. `AicpuProcessPackageWorker::CheckSum`
(`libtsdaemon.so+0x138f10`) compares the size of the received file against
the saved value, and `Current code is 0, skip check package code` skips the
comparison entirely when the saved value is 0.

The previous session's work — reproducing the tar byte-for-byte in GNU
format, finding a memLevel whose gzip output lands under target, and
padding the container to exact length — was solving this gate. The gate is
real; it just was not what was failing.

---

## 8. Method notes (reproducing this)

The device-side log is the whole story, and it is reachable:

```bash
# 1. raise the device log level (needs root)
sudo /usr/local/Ascend/driver/tools/msnpureport -g debug -d 0

# 2. run the load
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver
./tsd_load2 /usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/\
Ascend310P-aicpu_syskernels.tar.gz 0

# 3. collect the device logs
cd /tmp && mkdir npr && cd npr
sudo /usr/local/Ascend/driver/tools/msnpureport -t 1
# -> <timestamp>/slog/dev-os-0/run/device-os/device-os_<ts>.log   (tsdaemon)
# -> <timestamp>/slog/dev-os-0/debug/device-0/device-0_<ts>.log   (debug)
# -> <timestamp>/message/dev-os-0/messages                       (kernel/syslog)

# 4. restore
sudo /usr/local/Ascend/driver/tools/msnpureport -g info -d 0
```

Tooling fixes made this session:

- `tsd_load2.c` had a typo (`strrchr(p, (int)/)`); fixed to `strrchr(p, (int)'/')`
  and rebuilt. The rebuilt tool prints each stage's return value.
- `hdc_pull2` cannot read device files: it hard-codes `peer_id = 0` and gets
  `DRV_ERROR_SOCKET_CLOSE` (25). `drvHdcGetPeerDevId` takes **one** argument
  (the node id) and returns the peer id directly; with that, `hdc_pull3` still
  fails with 25, so the device file-read path appears to be unavailable on
  this platform. `msnpureport` is the working substitute.
- A `getenv` interposer (`/tmp/getenvspy.so`, `LD_PRELOAD`) is the quickest
  way to confirm which environment variables a library actually consults.

Card state after this session: 6/6 chips OK, no NPU processes, device log
level restored to INFO, host-side AICPU package still the stock build
(`md5` unchanged against the saved `.stock` copy).

---

## 9. Corrections to the previous session's notes

- **The AICPU install was never failing.** `fl=31` is a host-side timeout;
  the device completed the install successfully.
- **`sign=1` is not a stop condition**, and the line after it says
  `Soc verify success`. The "PSS signature check STOP" label was a misread.
- **`dev(0) cover file without head.` is not an error** — it is
  `dev_upgrade_verify_image` reporting that the verified file has no cover
  header, printed when the caller's `flags` bit 0 is set.
- **The check code is the package size**, not a hash: `checkCode=0` before
  and `checkCode=53559460` after a successful install.
- **The device's own log is available** via `msnpureport -t 1`; the previous
  session's belief that only kernel messages were reachable was wrong. The
  `slog/dev-os-0/run/device-os/` file contains tsdaemon's full trace,
  including every step of the AICPU install.
