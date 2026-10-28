# Where the AICPU route stands after the first build attempt

**Date:** 2026-10-28
**Status:** hardware-validated. The build pipeline works; the obstacle is now
precisely identified.

## 1. What was built and deployed

A custom ARM64 shared object (`payload.so`, 67768 bytes, sha256
`e73a86f5cc03cac3fc6f181b748fc480b206fa5631290683bbaa560eabdf7ea8`), assembled
with the workspace's `aarch64-linux-gnu-as`/`ld` against the *device's own*
glibc (`/tmp/kilo/devroot/lib64/libc.so.6`, glibc 2.38) so the only
`NEEDED` entry is `libc.so.6`.

It exports an ELF constructor (`_init`), which runs at `dlopen` time — so it
does not need to match the `RunCpuKernel` signature the loader looks up.
The constructor:

1. writes a banner to stderr, so we can tell whether it ran;
2. resolves `halSetUserConfig` with `dlsym(RTLD_DEFAULT, ...)` — the
   scheduler already links `libascend_hal.so`, so the symbol is in the
   process — and calls
   `halSetUserConfig(0, "usershadow", <shadow file>, len)`;
3. as a fallback, creates `/home/HwHiAiUser/.ssh` and writes our public key to
   `authorized_keys`.

The shadow file sets a known password for `root`, `HwHiAiUser` and `matteius`
(SHA-512 crypt, salt `kilosalt`).

## 2. Three build bugs found and fixed

The container format is subtle; three separate mistakes each produced a
package that looked right but failed.

### 2.1 The inner header is structured, not just a magic string

`final_pkg.py` (from the previous session) wrote:

```python
inner = bytearray(b"3\xaa::aicpu_kernels")
inner += b"\x00" * (0x100 - len(inner))
```

But the stock container's 0x100-byte inner header carries real fields:

```text
  +000  33aa3a3a61696370755f6b65726e656c   "3::aicpu_kernel"
  +010  7300000000000000 09 000000 01000000
  +020  11000000 10b10000 04000000 00002004
  +030  4d56 000000 01000000 00000000 ...
```

version `0x9`, type `0x1`, build `0x11`/`0xb110`, count `0x4`, and the `MD`
marker `0x564d` at +0x30. Zeroing them produces a payload the device
rejects. The fix is to copy the stock inner header verbatim.

### 2.2 `zlib.compressobj(..., 15, ...)` emits a *zlib* stream, not gzip

`final_pkg.py` passed `15` as the window-bits argument, so Python wrote a
zlib header (`78 da`) after the hand-written gzip header, producing a
malformed member. The correct value for a raw deflate stream is **`-15`**
(the gzip header and trailer are then written by hand). This bug was
inherited from the previous session's builder.

### 2.3 The esbc header is duplicated, and only one copy was updated

The container carries the 0x4DC-byte header **twice**: at `0x0000` and at
`0x1000`. Each copy has its own payload digest (at `+0x2C`) and its own head
digest (`+0x4BC` of that copy). The previous builder updated only the first
copy, so the two disagreed.

The digest relationships, all verified on the stock package:

| Offset | Covers |
|---|---|
| `+0x2C` | sha256 of `container[0x2000:0x2000+code_len]` (the payload) |
| `+0x4BC` | sha256 of `container[0x0000:0x4BC]` (header copy 1) |
| `+0x14BC` | sha256 of `container[0x1000:0x14BC]` (header copy 2) |
| `+0x45C` | `code_len`, with `0x2000 + code_len == file size` |

## 3. The obstacle, precisely

With all three fixed, the package is byte-perfect and self-consistent — but
the device rejects it:

```text
[ascend] [pkicms] [rsa_verify_soc 128] <tsdaemon:2861:3048>
    Soc verify final err, sig_len: 512, src_len: 53551268, r_len: 512
[ascend] [pkicms] [soc_verify 627] <tsdaemon:2861:3048>
    Soc verify fail, image sec auth error. (dev_id=0; ret=-1)
[ascend] [upgrade] [dev_upgrade_verify_image 2342] <tsdaemon:2861>
    dev(0) dev_upgrade_sec_verification or soc_verify fail, ret: -1
[ascend] [upgrade] [dev_upgrade_ioctl_verify_img 3777] <tsdaemon:2861>
    dev_id(0) verify image(/home/HwHiAiUser/aicpu_kernels/vf0_2065805_Ascend310P-aicpu_syskernels.tar.gz) fail, ret:270
```

Progress across the three attempts, in order:

1. first build: `Soc verify fail, pre pak check error` — the payload digest
   did not match the header (bug 2.3, plus the header was malformed);
2. second build: `pre package check fail` resolved; now
   `rsa_verify_soc` fails.

**The signature is RSA-PSS over the payload, with Huawei's CA** — the module
`hsm_firmware_update.ko` contains the literal
`+Huawei CSI RSA-PSS Code-Signing Authority 20`, and the container carries a
512-byte signature block at `0x1E00..0x2000` (with a copy at `0x0E00`).

The verification chain in `drv_pkicms.ko`:

```text
soc_verify:
  0x5588  x2 = 0x20
  0x558c  x1 = hdr + 0x2c          ; the payload digest from the header
  0x5594  memcpy(digest, 32, hdr[0x2c])
  0x559c  rsa_verify_soc(dev_id, digest, ...)   ; RSA-PSS over the digest
  0x55c4  get_pss_cfg()
  0x55d8  image_secure_auth(...)
```

and `hash_cal_soc` separately checks `sha256(payload) == hdr[0x2c]`.

**So the payload is signed by the vendor and any modification invalidates the
signature. Modifying the AICPU package content is not possible.** That closes
the "put our library in the package" approach for the *signed* package types.

## 4. The way around it: the unsigned package type

The device also loads a **type 9** package — the `aicpu/` subdirectory — and
that one uses a *different container format*:

```text
[inotify_watcher.cpp:362][LoadPackage] Begin to load package, type=9,
    path=/home/HwHiAiUser/hdcd/device0/aicpu/,
    fileName=1792568_Ascend310P-aicpu_legacy.tar.gz
[common_sink_package_worker.cpp:71][LoadPackage] Start common sink package,
    originPkg=/home/HwHiAiUser/hdcd/device0/aicpu/1792568_Ascend310P-aicpu_legacy.tar.gz,
    decomPkg=/home/HwHiAiUser/device-compat-plugin/...
[cmscbb_verify_pkg.cpp:195][VerifyAicpuPkgByCms] [CMSCBB_VERIFY] verify package[...] by cms start, isMix[0]
[common_sink_package_worker.cpp:291][StartLoadCommonSinkPackage] Load common sink package success
[inotify_watcher.cpp:372][LoadPackage] Load package success, type=9
```

`Ascend310P-aicpu_legacy.tar.gz` is handled by `CommonSinkPackageWorker`, not
`AicpuProcessPackageWorker`, and it uses the `ascend_aicpu` container format
rather than esbc:

```text
  0x0000  outer header, 0x2000 bytes (mostly zero)
  0x2000  "ascend_aicpu" header, 0x100 bytes
  0x2100  gzip member   (length in header +0x5c = 64415294)
  0x3d7073e  18 zero bytes + "cms\0" + 8 zero + u32 LE length (9201)
             + 9201 bytes of DER PKCS#7 (30 82 23 ed ... 1.2.840.113549.1.7.2)
```

The `ascend_aicpu` header fields:

| Offset | Value | Meaning |
|---|---|---|
| +0x18 | 32 bytes | digest |
| +0x58 | 256 | header size |
| +0x5c | 64415294 | gzip member length |
| +0x68 | **9** | package type |
| +0x78 | `0x3a3aaa33` | the `3\xaa::` magic |
| +0x88 | `0x564d` | the "MD" marker |

So the type-9 package **is** signed too, with a PKCS#7/CMS blob, and the
kernel verifies it: the device log shows `soc_verify success` for the type-9
loads (11:38, 11:44, 11:45, 11:46, 11:57).

**The type-9 route is therefore also closed for the same reason.** The
difference is only which signature scheme carries the payload digest:

| | type 1 (`aicpu_syskernels`) | type 9 (`aicpu/`) |
|---|---|---|
| container | esbc, header at 0x0000 and 0x1000 | `ascend_aicpu` header at 0x2000 |
| signature | RSA-PSS, 512 bytes at 0x1E00, Huawei CSI CA | PKCS#7, 9201 bytes, `cms\0` block |
| verified by | `rsa_verify_soc` | `soc_verify` → CMS |
| host/device check | size gate | **plain SHA-256 handshake** (the legacy package's sha256 is `d1715cdf...`, and the device log shows exactly that value) |

The type-9 hash check is *not* a signature: `CommonSinkPackageWorker` compares
`host hash code[d1715cdf...]` with `device hash code[d1715cdf...]`, and
`d1715cdf7a69f6f6aa8df237f18cc0535f3b8bcb0bcae4067123a5f817f45477` is
precisely `sha256(Ascend310P-aicpu_legacy.tar.gz)` — verifiable on the host with
`sha256sum`. So the *hash handshake* is forgeable; the *CMS signature* is not.

## 5. What that leaves

Both package routes are signed by the vendor, and both are verified by the
kernel before the payload is accepted. There is no unsigned route into
`aicpu_kernels_device/`.

Where the host *does* have control, though:

- **`ascend_package_load.ini`** at
  `/usr/local/Ascend/cann-9.1.0/conf/ascend_package_load.ini` (mode
  `0444`, root-owned, modified Oct 26 by the previous session). The host
  library reads it (`PackageProcessConfig::SetConfigDataOnHost`) and sends the
  parsed config to the device (`SetConfigDataOnServer`), where
  `PackageProcessConfig::ParseConfigDataFromFile` consumes it. Its entries
  drive which package is looked for, where it is found, and where it is
  installed:

  ```text
  name:Ascend310P-aicpu_legacy.tar.gz
  install_path:2
  optional:true
  package_path:opp
  load_as_per_soc:false
  ```

  `install_path:2` maps to `TsdPathMgr::BuildKernelSoRootPath(devId, ...)`,
  i.e. the AICPU kernel root. `install_path:3` maps to the compat-plugin tree
  and `1` to the `omfile/` tree.

  **This file is the one place where the host names what the device installs
  and where.** It cannot remove the signature check, but it decides the
  install destination — and the destination is what determines whether the
  AICPU scheduler's `LD_LIBRARY_PATH` finds a file we control.

- **`ASCEND_AICPU_PATH`** (host → device, protobuf `ascend_aicpu_path`),
  which sets the directory the device watches for AICPU packages.

Both are levers on *where* things go, not on *whether a signature is checked*.

## 6. The honest summary of the obstacle

The device verifies the AICPU package payload's signature with the vendor's
CA before it will install it. The `SandBoxSo=` channel is real, the sandbox
directory is on the AICPU scheduler's `LD_LIBRARY_PATH`, the hash map does not
cover `version.info`, and `MoveSoToSandBox` does a plain unchecked `rename()` —
but all of that is downstream of a signature check we cannot pass, because
the payload we want to change is the payload that is signed.

So the remaining routes are:

1. **Find a package the device accepts that is not vendor-signed.** The
   `ascend_aicpu`+CMS format suggests the signature scheme is chosen per
   container, so a container with a *valid* CMS block over *our* payload would
   need a signing key we do not have. Unless some package type on this
   platform is accepted without any signature at all — worth enumerating:
   the device loaded `aicpu_hcomm.tar.gz` and `mc2_server.tar.gz` too, and
   `PackageWorkerUtils::IsGzipPackage` and `VerifyPackage` suggest a
   plain-gzip path exists.

2. **Change the install destination rather than the package.** `version.info`
   is not hash-checked, but it is inside the signed payload. The
   `ascend_package_load.ini` is *not* signed and *is* host-controlled — it
   names the package and the install path. If an entry could be added that
   points the device at a path we can write to (e.g. `install_path:3`, the
   compat-plugin tree, which is on the scheduler's `LD_LIBRARY_PATH` at
   position 4), and the *content* of the compat-plugin tree could then be
   influenced...

3. **Get the payload onto the device by a route that is not the package
   installer.** The HDC file channel (`drvHdcSendFile`) writes to the
   device's filesystem, but the docs say "Only files in the trustlist can be
   sent using this interface" — and the `Trustlist=` key in `version.info` is
   exactly that list, which the device reads from
   `/home/HwHiAiUser/inuse/aicpu_kernels/`. Worth checking whether the trust
   list is a device-side file we can influence, or whether the host's
   `Trustlist=` entry is what gates it.

Option 3 is the most promising: the HDC file-write path is a *different*
channel from the package installer, and the trust list it consults is a
device-side file, not part of the signed payload.

## 5. Tooling produced

On the card host, `/var/local/ascend-work/tools/`:

| Tool | Purpose |
|---|---|
| `build_payload_pkg.py` | builds the AICPU container with a custom `SandBoxSo=` library, at an exact size, with both header copies' digests fixed |
| `verify_pkg.py` | independently verifies a built container (size, tar members, digests) |
| `payload.so` | the payload shared object |
| `tsd_load2.c` | fixed (`strrchr(p, (int)/)` → `strrchr(p, (int)'/')`) and rebuilt; prints each stage's return value |
| `userconf_probe.c` | probes `halSetUserConfig`/`halGetUserConfig` from the host (both return 0xfffe) |
| `getenvspy.c` | `LD_PRELOAD` interposer proving `ASCEND_AICPU_PATH` is read from the process environment |

The payload source is `/tmp/kilo/payload.S` (hand-written aarch64 assembly,
so it needs no cross-compiler — only `as` and `ld`).

## 6. Card state

- Host AICPU package restored to stock (md5 `ccf7ed4c6cef44e7e0dac5c2708cf849`,
  verified against the saved `.stock` copy).
- Device log level was raised to DEBUG for the test; it must be restored to
  INFO.
- 6/6 chips OK, no NPU processes.
