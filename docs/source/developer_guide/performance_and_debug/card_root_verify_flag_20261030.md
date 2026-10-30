# Card-root session report — the verify flag, the pinned rootfs, and where the wall actually is

Date: 2026-10-30 (daytime sessions, rig `matteius@192.168.53.187`)
Status legend: **[HW-validated]** observed on hardware this session ·
**[offline]** derived from binaries/containers on disk · **[hypothesis]** not yet tested.

Companion documents: [the plan](card_root_custom_firmware_plan.md) (§6–§10 carry the
running log) and the Oct-28 analyses in
`artifacts/ascend-310p-26.0.rc1/customer-cert-20261026/`
(`TYPE9-ROUTE-20261028.md`, `CMS-TAIL-SEMANTICS-20261028.md`, `HDC-FILE-CHANNEL-20261028.md`).

---

## 1. What we set out to do

The type-9 CommonSink door is live: `tsd::ProcessModeManager::SendAICPUPackageSimple(dev,
path, name, sink=1)` routes arbitrary files to the device's `common_sink_package_worker`
without reboot, and stock bytes install cleanly **[HW-validated]**. Modified (unsigned)
bytes do not — they die in `VerifyPackageByCms` (`ret=14`/`ret=103`) at the device's
default `verify_flag=1`. The day's goal: find where that flag lives and what it takes to
open the signature-free door, per the Oct-28 prediction
(`IsVerifyFlagValid: value ≤ 7`; beyond that ⇒ skip).

## 2. The package-side gates, fully mapped (morning) **[HW-validated]**

Container geometry (legacy `Ascend310P-aicpu_legacy.tar.gz`, 64,445,730 B): header
`0x0..0x2100` — `ascend_aicpu` tag @`0x2000`, sha256(gzip stream) @`0x2018`, gzip length
u32 @`0x205c`, **total-size u32 @`0x588`**; gzip code `[0x2100, 0x3d7073e)`; `cms\0` +
length descriptor; PKCS#7 @`0x3d70760`; 12,753 B opaque post-CMS blob.

Measured with nine single-byte flips and five rebuilds:

| Gate | Rule (measured) | Evidence |
|---|---|---|
| `GetCmsInfoOffSetFromPkg` "cms pkg len" | file length must equal header `0x588` | v2 (self-consistent resize) failed; size-preserving v3 passed |
| `VerifyIniData` "hash code cmp" | re-hash covers only a **< 16 KB prefix** of the gzip stream | flips at stream-rel 0x4000/0x10000/0x40000/32 MB all passed; rel-0 (magic) failed |
| `CheckCmsDataLenValid` | CMS-internal positions derived from length fields — any shift lands reads inside the DER | v8 (level-6 rebuild, both length fields patched) failed with `taglen` = garbage |
| body integrity | transport checkcode only; failed installs are non-destructive (device hash unchanged) | repeated failed pushes left device hash at stock `d1715cdf…` |

Side finding **[offline]**: Huawei's compressor is reproducible to `0x58f8` bytes by
python zlib **level 6** — the first 22.7 KB of a level-6 re-deflate are byte-identical
to the stock stream. The code region has exactly zero slack (`code_len == stream size`),
which is why every modified-package recipe must either pad to the stock size or patch
and survive `CheckCmsDataLenValid`.

## 3. Where the verify flag actually lives (afternoon) **[offline → static-confirmed]**

From the device `libtsdaemon.so` (extracted from the rootfs container):

- `CmsCbbVerifyPkg::SetVerifyFlag(devId)` → `ParseVerifyFlagFromCfgFile(devId)` →
  `GetHwCertCfgFilePath(devId)`. **`GetHwCertCfgFilePath` is built on the string
  `/var/asdrv/dev`** (refs at `0x18bc48`/`0x18bc90`; the flag path is assembled at
  `0x18c470` as `/var/asdrv/dev<N>/device-sw-plugin/verify/flag`; key parsed at
  `0x18c88c`/`0x18cc80` is `verify_flag`).
- The `old:1, new:1` log line prints on **every** verify — the file is re-read live per
  push, so seeding it at *any* time (even late) changes the next push's behavior
  **[hypothesis, log-consistent]**.
- Value semantics: `0..7` are signer-type policies (1 = vendor-only is the effective
  device default); `>7` ⇒ log line `VerifyFlag is [close], skip package verify`.
- The other candidate base, `/home/HwHiAiUser/device-sw-plugin/`, is referenced only
  from `0xd19a4` (a different component) — **not** the verify path.
- `tsd::TsdExecuteCmd` (popen-based device command channel, `tsd_exec`) returns `-1`
  on this build — dead end for runtime writes **[HW-validated]**.
- `signflag` (`dsmi_cmd_get/set_custom_sign_flag`, DMP `0x35/0x2`) segfaults in the
  post-reload host HAL state; unread today **[HW-validated for the crash, offline for
  the mechanism]**.

Implication: the flag file lives on the device's tmpfs `/var`, outside the HDC file
channel's trusted base (`/home/HwHiAiUser/hdcd/**`), outside the sink packages' install
roots (`/home/HwHiAiUser/**`), and outside the (read-only-signed) rootfs. Every host
mirror guess under `/usr/local/Ascend/latest/` was confirmed ineffective **[HW-validated]**.

## 4. The rootfs is digest-pinned — first real modification attempt on this rig **[HW-validated]**

The boot container `Ascend310P.cpio.gz` (109 MB, esbc) is **not** gzip-from-byte-0:
it is `header 0x2000 | "initrd"-tagged block 0x100 | gzip(cpio) @0x2100`, then the
16-aligned cms/ini/crl slots. The signed `ini` slot (76 B) decodes to

```
initrd, 39634b3e512d477fcc1b7280de792fcc450dfeca2d1d135fb4231d23c8ba0ff3;
```

and `sha256(data[0x2000 : 0x2000+code_len])` **equals it exactly** on stock bytes.

The boot-script (`var/device_sys_init.sh`, 33,902 B — note `var/drv_pkicms.ko` is the
real signed ELF module, 183,453 B; the misnamed script is the *init* orchestrator and
runs before tsdaemon) was extended with a `/var/asdrv/dev{0..7}/device-sw-plugin/verify/flag`
seeder (`verify_flag=8`). The surgical repack (full 364 MB cpio preserved, 2,575 entries,
re-gzip level 6, header digest + code_len rebuilt, slots verbatim) produced a 109.4 MB
container that **failed to boot the device** — twice, including after a clean host
reboot: bringup timed out with `Expected six Ascend device nodes, found 0`; the cards
sat at the PCIe handshake.

Two readings, one still open:
1. the loader enforces the signed ini digest (payload change ⇒ reject), or
2. our rebuilt container broke some geometry assumption (gzip level 6 is well-formed;
   framing verified `1f8b`).

Important correction from this experiment: the on-disk `Ascend310P.cpio.gz` "modified
Oct 26" was **byte-identical to `.orig`** — no modified rootfs had ever been boot-tested
on this rig before today. The `repack_rootfs.py` tool (Oct-28 vintage) also has a live
bug: its `--set` re-extraction path produced a 382-entry, 11 MB container (vs 2,575
entries / 364 MB) — it silently operated on a subset.

## 5. Rig operations — the reload cliff **[HW-validated]**

The rmmod/modprobe "soft reboot" works exactly once from a fresh boot. Chained reloads
degrade fast: `npu-smi` instances get stuck in ioctl (D-state) and pin module refcounts
(`fuser /dev/davinci_manager` to see them); `systemctl start` on the `RemainAfterExit`
`ascend-310p-bringup.service` is a **no-op while active** (must `stop` first); and the
bringup module list needs `drv_davinci_intf_host`, `drv_vpc_host`, `ascend_logdrv`,
`drv_seclib_host` in any manual unload loop. **Host reboot restored 6/6 every single
time today** (three times). Practical rule: one reload per reboot; treat the second
reload as a reboot in waiting.

## 6. State of the rig at end of day

- 6/6 healthy; rootfs = stock (`md5 0c93f734…`, from `.oct30bak`); opp legacy slot =
  stock (`.kbackup`); `hcomm` built-in still the Oct-29 modified copy (`.stockbak` kept).
- Kept for next session: modified container `~/exp-20261030/rootfs_x/surgical2.cpio.gz`
  (proper gzip framing + flag seeder), rebuilt v8 package recipe (level-6, append
  member, `0x588`/`0x205c`/digest patched), `~/exp-20261030/` (slog exports, rootfs_x).

## 7. Where the wall is now

One wall, two layers, same architecture:

| Layer | Pinned by | Bypass today | Needs |
|---|---|---|---|
| type-9 package | cmscbb at `verify_flag=1` | none (modified bytes rejected) | flag file in `/var/asdrv/dev<N>/…` |
| rootfs initrd | signed `ini` payload digest | none (modified container fails boot) | re-signed CMS or boot-time loader bypass |

Unlocked candidates for the flag (in rising cost order):
1. **Device-side `/var` writer**: DMS route now closed — re-probed today:
   `dsmi_cmd_get_custom_sign_flag` returns `rc=8` (stub), confirming the Oct-28 finding
   that `dms_custom_host.o` is compiled out on this 310P build **[HW-validated]**.
2. ~~syskernels may be flag-free~~ — **answered by hardware today, see §7a**: type-0
   gates on `pkicms` RSA-PSS, same wall.
3. **kexec-style boot (plan S4)**: re-signed container with our own cert chain, booting
   a payload that seeds everything else.

## 7a. Late-session follow-up — the type-0 gate, peeled like an onion **[HW-validated]**

Four pushes of a payload-modified syskernels package (add `libkilo_payload.so` member +
`bin_hash.cfg` line, re-gzip, `tsd_load2` name-table route) peeled the gate layer by
layer — each failure taught one field rule:

| Push | Failure | Learned rule |
|---|---|---|
| 1 (whitelist) | `miss the white list` | device file name must end `_Ascend310P-aicpu_syskernels.tar.gz` (host `/etc/hdcFileTransWhiteList.cfg` suffix match) |
| 2 | `pre_package_check: code len 53551268, size 53749169` | container code_len u32 (at `0x45c`, copy `0x145c`) must equal **file size − 0x2000** |
| 3 | `pre_package_check: hash check err` | digest at `0x2c`/`0x102c` must equal **sha256(data[0x2000 : 0x2000+code_len])** |
| 4 | `CmscbbCryptoPssVerifyFinal: TEE Asymmetric Verify Digest failed (0x88200203)`, `rsa_verify_soc … src_len 53740977` | **RSA-PSS (512 B sig) over the whole code region** — the real signature check |

So the syskernels container uses the **rootfs esbc geometry** (img `0x2000`,
code_len `0x45c`, digest `0x2c`, header copies at `0x0`/`0x1000`, total-size u32 at
`0x4e8`) — and it verifies content cryptographically at flag=1, exactly like type-9's
cmscbb. The stock-bytes control push (the Oct-26 `ccf7ed4c` file — effectively unmodified
content, see §4 note) installed successfully at 13:04, proving the channel itself works.

**Consequence:** every host-pushable channel (type-9 CommonSink, type-0 syskernels) is
signature-pinned at the default flag, and no host-visible writer exists for either flag
(file on tmpfs; DMS stubbed; no device shell). The flag question and the re-signing
question are now the same question, which is precisely plan S3/S4 territory: a payload
that boots *before* the flag is needed (kexec / re-signed CMS with our own CA).

---

*One-sentence summary: the flag is a file on device tmpfs that nothing host-side has
reached yet, the rootfs refuses changed bytes the same way packages do, and the rig is
healthy on stock with every artifact needed for the next push already staged.*
