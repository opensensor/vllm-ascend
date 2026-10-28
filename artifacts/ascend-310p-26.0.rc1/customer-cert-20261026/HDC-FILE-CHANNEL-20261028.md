# The HDC file channel: what it will and will not deliver

**Date:** 2026-10-28
**Status:** hardware-validated on the 310P at `matteius@192.168.53.187`.

This note records the results of probing the HDC file-transfer channel, which
is the one delivery path that does **not** go through the package installer
and therefore does not go through the vendor signature check.

---

## 1. The whitelist

`/etc/hdcFileTransWhiteList.cfg` (mode `0444`, root-owned):

```text
//save_filename save_sub_path name_full_check_flag file_max_size file_max_num root_privilege_flag
//save_sub_path=* means to save file int the base path
OnChipRom.bin /upgrade 1 2097152 1 1
HBOOT1_a.bin  /upgrade 1 2097152 1 1
...
Ascend310P-driver_spc_device.tar.gz /upgrade 1 20971520 1 1
_Ascend310P-aicpu_syskernels.tar.gz * 0 104857600 64 0
_Ascend310P-aicpu_extend_syskernels.tar.gz * 0 104857600 64 0
```

Column meanings, confirmed by the driver's own behaviour:

| Column | Meaning |
|---|---|
| `save_filename` | the permitted file name; with `name_full_check_flag=0` a `<pid>_` prefix is tolerated |
| `save_sub_path` | directory under the trusted base; `*` means the base itself |
| `name_full_check_flag` | 1 = exact name, 0 = prefix form allowed |
| `file_max_size` / `file_max_num` | size and count limits |
| `root_privilege_flag` | 1 = written as root, 0 = as the sending user (uid 1000 on the device) |

The trusted base is `/home/HwHiAiUser/hdcd/device0`, reported by
`drvHdcGetTrustedBasePath`. The V2 variant, `drvHdcGetTrustedBasePathV2`,
returns `/home/HwHiAiUser/hdcd/device0/aicpu` — the type-9 staging directory.

## 2. The API

```c
drvError_t drvHdcClientCreate(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag);
drvError_t drvHdcGetPeerDevId(int node, int devid, int *peer);
drvError_t drvHdcSessionConnect(int node, int peer, HDC_CLIENT client, HDC_SESSION *session);
drvError_t drvHdcSendFile (int node, int peer, const char *file, const char *dst_path, void (*notifier)(void *));
drvError_t drvHdcSendFileV2(int node, int peer, const char *file, const char *dst_path, void (*notifier)(void *));
drvError_t drvHdcSendFileEx(int node, int peer, const char *file, const char *dst_path, void (*notifier)(void *), int);
drvError_t drvHdcLoadFileTransWhiteList(const char *path);
```

**The client and session are required.** My first attempts called
`drvHdcSendFile` directly and got `DRV_ERROR_INVALID_VALUE` (3) for every
path — including one that worked. `drvHdcSendFileEx` takes **six** arguments
(recovered by disassembling it in `libascend_hal.so`: it spills `edi`, `esi`,
`edx`, `rcx`, `r8` and `r9`).

`drvHdcLoadFileTransWhiteList` takes the whitelist path as an argument. If it
is not called, the driver's default list is used, and every send is rejected.
**This was the bug that invalidated my first set of results**: I edited
`/etc/hdcFileTransWhiteList.cfg` but never told the library to load it, so
the edits had no effect and every test returned "path illegal".

## 3. What actually works

With the whitelist loaded, using a file named the way the device names its
own transfers (`<hostpid>_Ascend310P-aicpu_syskernels.tar.gz`):

| Destination | Result |
|---|---|
| `/home/HwHiAiUser/hdcd/device0` (the base) | **rc=0 — the file is delivered** |
| `/home/HwHiAiUser/hdcd/device0/aicpu` | rc=25 |
| `/home/HwHiAiUser/hdcd/device0/aicpu/vfid1` | rc=25 |
| `/home/HwHiAiUser/hdcd/device0/vfid1` | rc=25 |
| any absolute path outside the base | rc=52 (`DRV_ERROR_DST_PATH_ILLEGAL`) |
| a `dst_path` that is a bare file name rather than a directory | rc=52 |

The base-directory send is **confirmed to reach the device and be acted on**.
The device's own log records:

```text
[ascend] [upgrade] [dev_upgrade_ioctl_verify_img 3777] <tsdaemon:2861>
    dev_id(0) verify image(/home/HwHiAiUser/aicpu_kernels/vf0_9999_Ascend310P-aicpu_syskernels.tar.gz) fail, ret:270
```

i.e. the device's watcher saw our file, decompressed it to
`/home/HwHiAiUser/aicpu_kernels/vf0_<pid>_<name>`, and routed it to the
AICPU package worker — where the esbc signature check rejected it.

## 4. Why the subdirectories fail

The driver's own log is decisive:

```text
[drv][hdc][drvHdcSendFileParaCheck] Calling strstr failed.
    (dev_id=0; file=".../9999_Ascend310P-aicpu_syskernels.tar.gz";
     dst_path="_Ascend310P-aicpu_syskernels.tar.gz";
     base_path="/home/HwHiAiUser/hdcd/device0")
```

For the *base* directory the `strstr` succeeds. For the subdirectories it does
**not** fail — the error changes to:

```text
[recv_reply] Calling halHdcRecv error. (hdcError_t=25)
[is_recv_side_ready] Calling recv_reply error.
```

So the path check passes and the failure is in the transport: the **device's
`hdcd` agent does not accept a file into a subdirectory.** The device creates
those directories itself (`mkdir -m 750 .../hdcd/$device/aicpu`) and mounts
a tmpfs over `aicpu/vfidN`, but the file-transfer receiver only serves the
base path it was started with.

The host driver also contains the message
`"White list mode not support directory send."`, consistent with this.

## 5. The tmpfs, and the CVE question

The device mounts one tmpfs per VF:

```sh
mount -t tmpfs -o size=50%,uid=$uid,gid=$gid,nosuid,nodev,mode=0700 \
      tmpfs "/home/HwHiAiUser/hdcd/device0/aicpu/vfid$count"
```

with `uid`/`gid` = `HwHiAiUser`. On the **device**, `HwHiAiUser` is **uid
1000** (the card host's `HwHiAiUser` is a different account, uid 1002 — the
two must not be confused). So the staging tree is owned by uid 1000, which is
also the uid `hdcd` and the AICPU scheduler run as.

Two points about the CVE list:

- **The mount has no `noexec`.** The AICPU tmpfs is `nosuid,nodev` only.
  `noexec` would only block `execve()` of a file on that mount, and would not
  affect `dlopen()` anyway — but its absence means a file placed there could
  also be exec'd. Relevant because the AICPU scheduler `dlopen()`s libraries
  from `aicpu_kernels_device/sand_box/`.
- **The permission-bypass CVEs do not apply.** The device kernel is
  `6.6.0-72.0.0.76.h21510.1170.eulerosv2r15.ascend.aarch64`.
  CVE-2022-2590 (Dirty COW) was fixed in 5.16.12/5.15.25 and CVE-2018-18397
  (userfaultfd) in 4.16.9 — both long fixed. CVE-2021-3732 (OverlayFS +
  tmpfs) needs the ability to mount a tmpfs, i.e. `CAP_SYS_ADMIN` or an
  unprivileged user namespace, which is not available. The two 2026 CVEs
  (CVE-2026-89987, CVE-2026-72379) may or may not be backported into this
  EulerOS kernel, but note their direction: 89987 is a data-loss /
  information-disclosure bug (one uid reading another's memory) and 72379 is
  about `vfs_tmpfile()` not validating mapped fsuid/fsgid. Neither is a write
  primitive on its own, and the tmpfs permission boundary here is between
  uid 1000 and uid 1002 — both of which we would be on the friendly side of
  anyway, since the scheduler runs as 1000.

**Conclusion on the tmpfs angle:** the tmpfs is not the obstacle. The obstacle
is that the device's file receiver only serves the base path, and the base
path's contents are fed to a signature-checking installer.

## 6. What this leaves

The HDC channel gives us **arbitrary file delivery to one directory on the
device**, `/home/HwHiAiUser/hdcd/device0/`, and the device's inotify watcher
acts on files that appear there.

That directory is uid 1000, mode 0750, and it is the *staging* area — the
device's own file transfer target. The device then routes by file name:

| Regex | Type | Worker | Check |
|---|---|---|---|
| `^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_syskernels\.tar\.gz$` | 0 | `AicpuProcessPackageWorker` | esbc container + **kernel RSA-PSS signature** |
| `^[0-9]+_Ascend([0-9]{3}(P)?)?-aicpu_extend_syskernels\.tar\.gz$` | 1 | extend worker | same |
| `^[0-9]+_Ascend-runtime_device-minios\.tar\.gz$` | 4 | | |
| `^[0-9]+_[0-9]+_Ascend-flow_model\.tar\.gz$` | 5 | | |
| `^[0-9]+_Ascend-opp_rt-minios.aarch64\.tar\.gz$` | 6 | | |
| `^[0-9]+_Ascend-device-sw-plugin\.tar\.gz$` | 7 | | |
| `^[0-9]+_transformer_tile_fwk_aicpu_kernel.tar.gz$` | 8 | | |
| (the `aicpu/` **directory**) | 9 | `CommonSinkPackageWorker` | **host/device SHA-256 handshake, no signature** |

Type 9 is the signature-free route, and it lives in a subdirectory the file
receiver will not serve. So the two mechanisms are complementary but
currently disjoint:

- the **file channel** can reach the base directory only, and the base
  directory's contents are routed to signature-checked workers;
- the **signature-free worker** only reads the `aicpu/` subdirectory, which
  the file channel cannot write.

Closing either gap is the next step, and the candidates are:

1. **Make the device's own installer put a file in `aicpu/`.** The type-9
   worker extracts with
   `mkdir -p -m 0750 <dir> ; tar --same-owner --same-permissions -ozxf <pkg> -C <dir>`.
   If a type-0 package's *contents* could include `aicpu/...`, the extraction
   would place it. But type-0 packages are signature-checked, so this reduces
   to the same problem.
2. **Find a name that the base-directory watcher routes to type 9.** The
   routing table has a directory rule for `aicpu/` rather than a name rule,
   so this looks closed — but the type-9 worker is also selected by
   `ascend_package_load.ini` on the **host** side, by package *name*
   (`name:Ascend310P-aicpu_legacy.tar.gz`, `install_path:2`). That config is
   host-controlled and unsigned. Whether the device can be made to load a
   host-named package from the base directory is worth testing directly.
3. **Write into the device's `/home/HwHiAiUser/hdcd/device0/` as the *device
   user* rather than through the driver**, e.g. via a command the device
   itself executes. That is circular unless we already have execution.

Option 2 is the cheapest to test and is not yet ruled out.

## 7. Tooling

On the card host, `/var/local/ascend-work/tools/`:

| Tool | Purpose |
|---|---|
| `hdc_plant2.c` | HDC send with the full client+session flow, whitelist loading, and V1/V2/Ex selection |
| `hdc_plant3.c` | the same, using `drvHdcSendFileEx` and the V2 trusted base path |
| `plant_test*.sh`, `wl_test*.sh` | the destination/whitelist probes above |
| `drvlog_test*.sh` | probes that correlate each result with the host driver's own syslog lines |

`/var/log/syslog` carries the host driver's HDC messages with the file, line
and function name — e.g. `hdc_file.c:1416`, `drvHdcSendFileParaCheck` — and
is the fastest way to see *why* a send was refused. `msnpureport -g debug -d 0`
raises the *device* log level; the host driver log level is separate.
