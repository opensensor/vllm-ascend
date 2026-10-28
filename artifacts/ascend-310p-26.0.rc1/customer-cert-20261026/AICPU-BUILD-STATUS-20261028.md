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
that one is *not* signed and *not* esbc-wrapped:

```text
[inotify_watcher.cpp:362][LoadPackage] Begin to load package, type=9,
    path=/home/HwHiAiUser/hdcd/device0/aicpu/,
    fileName=1792568_Ascend310P-aicpu_legacy.tar.gz
[common_sink_package_worker.cpp:71][LoadPackage] Start common sink package,
    originPkg=/home/HwHiAiUser/hdcd/device0/aicpu/1792568_Ascend310P-aicpu_legacy.tar.gz,
    decomPkg=/home/HwHiAiUser/device-compat-plugin/...
[cmscbb_verify_pkg.cpp:195][VerifyAicpuPkgByCms] [CMSCBB_VERIFY] verify package[...] by cms start, isMix[0]
[cmscbb_verify_pkg.cpp:231][VerifyAicpuPkgByCms] [CMSCBB_VERIFY] verify pkg[...] by cms end
[common_sink_package_worker.cpp:291][StartLoadCommonSinkPackage] Load common sink package success
[inotify_watcher.cpp:372][LoadPackage] Load package success, type=9
```

`Ascend310P-aicpu_legacy.tar.gz` is:

- a **plain tar.gz** — no esbc container, so `dev_upgrade_verify_image` and
  `soc_verify` never run on it (the container layout differs: `hdrlen` field
  is `0x0`, and the gzip member starts at file offset `0x2100` rather than
  `0x2100` inside an esbc frame);
- handled by `CommonSinkPackageWorker`, not `AicpuProcessPackageWorker`;
- verified only by the **tsdaemon-level CMS name check**, which it passes
  because `Ascend310P-aicpu_legacy.tar.gz` does not contain any of the six
  CMS-triggering substrings;
- gated by a **hash-code handshake** instead — the log shows
  `host hash code[d1715cdf...] device hash code[d1715cdf...]`, i.e. the host
  and device agree on a hash, and there is no vendor signature;
- already carrying a populated sandbox:
  `aicpu_kernels_device/sand_box/libtensorflow.so` is a member of its tar.

**That last point is the whole opportunity.** The type-9 package's tar
already contains `aicpu_kernels_device/sand_box/<lib>.so`, and
`version.info` names it with `SandBoxSo=`. If the type-9 install path copies
the tar's `sand_box/` contents into the device's
`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box/`, then a modified
type-9 package carrying our library gets it into the sandbox with no signature
involved.

Two things to check before that can be relied on:

1. **Where does the type-9 path put the sandbox?** Its `decomPkg` is
   `/home/HwHiAiUser/device-compat-plugin/...`, not
   `aicpu_kernels_device/`. So the type-9 worker may not populate the
   AICPU scheduler's `sand_box/` at all — it installs into the
   *compat-plugin* tree instead. If so, the type-9 route gives us a
   different (also interesting) primitive but not the AICPU scheduler's
   library path.
2. **The hash-code handshake.** The device compares a host-supplied hash
   against its own. We need to know what is hashed — the whole file, or the
   payload — and whether the host's value is what the device computes or
   merely what it is told.

Both are answerable from the code, and both are the next step.

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
