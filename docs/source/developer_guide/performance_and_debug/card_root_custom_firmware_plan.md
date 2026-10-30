# Card Rooting & Custom Firmware Plan (post-FPGA pivot)

**Date**: 2026-10-30 · **Hardware**: 3× Ascend 310P3 (6 chips) @ `192.168.53.187`, driver 26.0.rc1
**Supersedes**: the bring-up motivation of [fpga_card_hacking_plan.md](fpga_card_hacking_plan.md)
(the ALINX Zynq rig is shelved — we do not own an FPGA board today).

Goal: get a shell on the cards ("root") and define what "run custom firmware" can and cannot
mean on this platform. **Long-term target: custom firmware + secure-boot bypass; incremental
steps (shell → runtime ownership → runtime-level bypass) are accepted on the way** — see §8.

---

## 1. Settled constraints (evidence-anchored — do not re-litigate)

Full history: `artifacts/ascend-310p-26.0.rc1/CUSTOM-FIRMWARE-PROGRESS.md` and
`customer-cert-20261026/*.md`. The load-bearing results:

| Fact | Status | Source |
|---|---|---|
| The card OS boots from host-side files (`/usr/local/Ascend/driver/device/Ascend310P.*`) DMA-loaded over PCIe at every boot; flash holds only HBOOT1 a/b + HiLink FW stubs | Observed (host logs; corroborated by [jishuzhan 1964682616751833090](https://jishuzhan.net/article/1964682616751833090)) | boot slog |
| Boot verifies kernel/dt/rootfs esbc containers against the **one** Huawei root baked into `Ascend310P.fd` (BL31); no customer-CA hook | Observed (V/goal/P1 series) | progress log |
| The payload **is** digest-bound: modified `cpio.gz` payload under the stock (Huawei) `ini`+`cms` = load acks fine, OS never starts (V1) | Observed | progress log |
| Customer-signed boot variants (S/T/R/V/goal/P1) all "load OK, never boot" — six independent negatives; boot path is closed | Observed | progress log |
| Runtime package install (tsdaemon, type-0/9) is blocked only by the PKCS#7/CMS check; routing, hash map (`bin_hash.cfg` is *populated from our package*), and sandbox rename all work | Observed | `OTHER-ANGLES-20261028.md` |
| Policy file `/var/asdrv/dev<N>/device-sw-plugin/verify/flag` exists on the device FS; flag ≥ 8 skips verification; the DMS writer (`dms_custom_*.o`) is compiled out on 310P — no host-side writer exists (stubs return 8) | Observed | `OTHER-ANGLES-20261028.md` §1 |
| Custom **device-executed code** is proven free: AI Core kernels (plain SHA-256 in the op descriptor) and AICPU `dlopen` — a modified AI Core kernel was built, deployed, and executed end-to-end (~5 s loop) | **Proven** | `prove_aicore_exec.py` |
| `aicpu_scheduler` runs under seccomp (31 syscalls; `ioctl`/`openat` in, `execve`/`socket` out) as uid 1000 = the device `HwHiAiUser`, whose home `/home/HwHiAiUser` (0750, uid 1000) it owns | Observed | `USERSSHADOW-ROUTE-20261028.md` |
| Device sshd (OpenSSH 9.6p1) is reachable over the vNIC: `HwHiAiUser@fe80::2e0:fcff:fefc:d302%endvnic` answers with its banner; only credentials refused (factory SHA-512 hash; 87 defaults exhausted) | Observed (re-verified 2026-10-30) | this session |
| HDC is a **file-transport** channel today (plant/pull verified); whitelist permissive for `aicpu_syskernels` tarballs; `execve` is not part of what we have exercised | Observed | `HDC-FILE-CHANNEL-20261028.md` |
| Open negative (2026-10-30): stock container RSA-PSS does not verify offline under the obvious rules — do not bet the plan on re-signing before `rsa_verify_soc`/`soc_verify` semantics are pinned | Observed | FPGA plan doc |

**Consequence.** "Custom firmware" on the *boot* path (modified kernel image or modified rootfs
contents) requires a Huawei-root-valid signature we cannot produce: it is digest-locked.
"Custom firmware" on the *runtime* path (device kernels, plugins, `.so` payloads) is already
proven free. Persistence caveat: the rootfs is a RAM image rebuilt from the host file at every
boot, so runtime edits (`/var`, `/etc`, `/home`) vanish on reboot unless re-pushed or cached in
MCU flash (`usershadow` path).

## 2. What "root" means here

Tier 0 (today): boot-time control only — swap host-side image files, reboot, read host slog.
Tier 1: an interactive shell on the card as `HwHiAiUser` (uid 1000) — enough to read `/var`,
write `/var/asdrv/**` (flag ≥ 8!), restart tsdaemons, and drive `spi_mcu` to the MCU.
Tier 2: full root — replace `/etc/shadow` hashes, restart anything, `kexec` an arbitrary
kernel+rootfs pair and thereby escape the boot digest lock at runtime.

## 3. Ranked routes to Tier 1 (no FPGA, no new hardware)

1. **Boot-time payload via the SandBoxSo rename (self-opening route).**
   The stock installer `rename()`s the `SandBoxSo=libtensorflow.so` library into `sand_box/`
   (unverified control channel, 2nd on the scheduler's `LD_LIBRARY_PATH`), and the scheduler
   `dlopen`s it *as part of boot*, before we ever need a channel. Replace the staged `.so` with
   our payload (ctor writes `/home/HwHiAiUser/.ssh/authorized_keys`, mode 0700 dir — owned by
   uid 1000, which the scheduler runs as) and the card opens its own SSH door at every boot.
   The previous "payload sits unexecuted in `sand_box/`" observation was a *runtime* staging
   artifact — boot-time dlopen has not been disproved and is the decisive experiment.
   - Needs: a package (or in-rootfs-adjacent placement) whose content passes the CMS wall —
     candidates in order: a name **outside the six-name CMS-trigger list** (falls to the
     config-package path, possibly signature-free), the type-9 `aicpu/` route, or accepting a
     stock-bytes-only install of the stock `libtensorflow.so` while our payload rides in as an
     additional file the installer does not hash.
   - Acceptance: `ssh -i key HwHiAiUser@fe80::2e0:fcff:fefc:d302%endvnic` works after cold boot.
2. **HDC command-service probe.** `hdcd` is a full client/session RPC
   (`HDCDRV_SERVICE_TYPE_FILE_TRANS 3`, …). Check the device `hdcd` binary and
   `hdcdrv_interface.h` for an exec/shell service type beyond file transfer; if one exists,
   Tier 1 is one ioctl round-trip away and route 1 becomes optional convenience.
3. **Serial console (hardware, still unproven).** The jishuzhan author's efuse/ATU/BAR lines
   look like early BL31-stage output — i.e. someone had a console. Verify whether the 310P3
   card carries an onboard UART/debug header; if it does, we get BL31 boot-log card-time answers
   (which FIP slot boots, which verify strings fire — the rank-1 open question in
   `SECUREBOOT-WEAKNESSES.md`) for free, and console login makes credential recovery moot.
4. **Credential recovery, last.** `usershadow` via `halSetUserConfig` is the sanctioned password
   setter (runtime, RAM-only). With route 1's payload calling it in the ctor, even Tier 1+ (known
   `HwHiAiUser` password) comes for free — then `sudo` (stock-limited to `/bin/ls`) or
   `su` under `pam_wheel` decides Tier 2.

## 4. Custom-firmware strategy once Tier 1/2 is in hand

- **Break the file-write circularity**: write `flag = 8` (or 5/6 + our `user-certificate`) under
  `/var/asdrv/dev<N>/device-sw-plugin/verify/` from the shell. From then on *any* runtime package
  installs without the CMS fight, `bin_hash.cfg` is ours, and the SandBoxSo → payload →
  scheduler loop is a 5-second dev cycle, like the AI Core proof today.
- **Steady state**: boot stock-signed kernel+rootfs+dt (Huawei lock) from host files, then let
  our host-driver fork re-push custom device packages each boot (we already own the driver).
  This is "custom firmware" in every runtime sense: kernels, plugins, schedulers, config.
- **Full-OS escape (Tier 2 only)**: `kexec` a custom kernel + custom rootfs cpio from the stock
  boot — `CONFIG_KEXEC=y`/`KEXEC_SIG=y` with no `KEXEC_SIG_FORCE`-style lock observed, and
  unsigned modules load freely (`MODULE_SIG_FORCE` off, though the `module.sig_enforce=1`
  bootarg from `dt.img` must be re-checked from `/proc/cmdline` first).
- **MCU**: its trust anchor is a hardcoded `sha256(vendor pubkey)`; the customer region is
  CRC-16 only — `WriteCertToFlash`/`customized_*` remain available via the device-side
  `spi_mcu` transport once we have any shell.

## 5. Immediate next actions (all on the existing rig)

1. Probe `hdcd` + `hdcdrv_interface.h` for a non-file service type (offline, 30 min, no card risk).
2. Stage the payload `.so` for boot-time execution and cold-boot one card; check
   `authorized_keys` effect by SSH'ing with our key; grep ctor marker in device slog
   (`msnpureport` level up first). One card only (NPU 9), keep `.orig` backups.
3. Locate/verify a UART header on the 310P3 card (multimeter/known pinout; no soldering if a
   header exists).
4. Re-run today's PSS open-negative only after disassembling `rsa_verify_soc`/`soc_verify`
   offsets in `dev_upgrade.ko` / `libtsdaemon.so`; do not block the plan on it.

**Safety rule unchanged**: verify every artifact offline before the card sees it; the cards stay
recoverable from `.orig` files + power cycle at all times.

---

## 6. 2026-10-30 — Route A (type-9 / extend) executed

**Route A outcome: exhausted and fully understood. Both surviving name-routed channels on the
live 26.0.rc1 tsdaemon (`syskernels` type-0, `extend_syskernels` type-1) terminate at the same
TEE RSA-PSS wall; the signature-free type-9 door only exists on the old rootfs.**

1. **The legacy/CommonSink name is dead.** The device's own `Ascend310P-aicpu_legacy.tar.gz`
   (64 MB, esbc `0x55aa55aa`) sent via V2 into `aicpu/` under its own name is now
   `remove file`d by `inotify_watcher` — the 2026-10-28 type-9 worker is no longer name-routed.
   The `mc2_server` / `aicpu_hccl` names are removed too. Only two names survive:
   `..._aicpu_syskernels` (type=0) and `..._aicpu_extend_syskernels` (type=1, `ExtendProcessPackageWorker`).
2. **`extend` is signature-gated by a *head tag*, not a size/CMS error.** `package_verify.cpp`
   returns `ret=279 "Check head tag failed"` (ioctl `0x8004430d`, outer `ret:270`).
   The head tag is the ASCII string at file offset `0x2004` (right after the `0x3a3aaa33` inner
   magic in the esbc/ascend_aicpu header). `drv_pkicms.ko` checks it against two compiled maps:
   - `g_img_id_tag_map_soc`: `tschfw`, `aicpu_kernels`, `fftsplusfw`, `tsch_patch`, `tschplugin910b`
   - `g_img_id_tag_map_cms`: `tee`, `dtimage`, `uimage`, `initrd`, `sdpekdtimg`, `sdimg`

    Brute-forcing all 11 tags in small probe containers (`9701`–`9710`): the extend path expects
    `tschfw` (img_id 0). Passing the tag, the payload then fails `rsa_verify_soc` with
    `0x88200203` — the same TEE RSA-PSS wall as type 0, reached under a different img_id. And
    because the head tag lives *inside* the signed code span, the stock `aicpu_kernels`-tagged
    container can never satisfy type 1 either: **extend is a dead channel for 310P3 products.**
3. **Stock control installs cleanly** (`9801`, pristine md5 `ccf7ed4c…`, type=0 →
   `Load package success`), confirming the channel and tooling are healthy; the failures above
   are the signature/head-tag gate, not delivery.

**Net:** the user's "mix old+new" idea is confirmed *structurally* — the extend path is a second,
newer-style worker — but it is gated by the same vendor trust (head tag ∈ fixed map + `soc_verify`
RSA-PSS), so it does not by itself yield a signature-free door. The genuinely signature-free
`CommonSinkPackageWorker` route remains reachable only by reviving the type-9 name (tsdaemon side)
or by writing `verifyFlag ≥ 8` on the device (§5 action 1 + `TYPE9-ROUTE-20261028.md`).

Probe artifacts: `9701`–`9710` (all 11 tags × small extend container) and `9801` (stock control)
on the rig at `/tmp/*_Ascend310P-aicpu_extend_syskernels.tar.gz`; the head-tag maps are in
`var/drv_pkicms.ko` (strings + `g_img_id_tag_map_*`). Device restored to whitelist md5
`c3342f9096309a8e6a5c865064084d3b`.

---

## 7. 2026-10-30 — Route B (explicit-type LoadPackage over HDC) started

**Framing: the goal is a Tier 1 shell on the Atlas cards (3× 310P3, 6 chips).** Routes A/B are
means to that end: every install channel we open is only useful insofar as it lands the
SandBoxSo payload (§3 route 1) without the CMS fight, because that payload opens the SSH door.

Observations so far:

1. **The host stack ships no tsdaemon at all** — the custom bringup skips it, so boot-time
   package-load never runs; the device lives entirely on rootfs-bundled kernels. The name→type
   routing table is therefore only live *after* a tsdaemon is (re)started on the host.
2. **HDC exposes full message sessions per service** (`service_TDT` / `service_TSD` are
   tsdaemon's own RPC endpoints), not just file transport. A small host tool can issue
   `LoadPackage` with an **explicit type**, bypassing the name→type table entirely — which
   sidesteps Route A's finding that only two names survive `inotify_watcher` routing.

**Recommended next step (single experiment, NPU 9 only):** boot the old rootfs variant — or
flash it to card NPU 9 — to revive the signature-free type-9 `CommonSinkPackageWorker`, and
drive it with a `verifyFlag = 8`-style no-CMS install of the SandBoxSo payload. That converts
the Oct-28 "validated but signed" finding (`TYPE9-ROUTE-20261028.md`) into a working shell on
the card: ctor writes `authorized_keys`, and Tier 1 is then reachable over the vNIC regardless
of what the package channel cost us.

**Alternative if old-rootfs revival fails:** build the small HDC message client and issue
`LoadPackage(type=9)` (or `type=0` with `verifyFlag ≥ 8` pre-set on the device FS) directly —
the explicit-type call removes tsdaemon's routing table from the critical path, leaving only
the kernel-side `soc_verify` gate to negotiate.

---

## 8. Ladder to the end state (custom firmware + secure-boot bypass)

End state defined precisely: the cards run *our* kernel + rootfs + userspace + ops, and no
Huawei-root signature is required for any component we change. Six boot-path negatives (§1) say
BL31 will not accept a customer-signed boot image today — so the bypass is realistically earned
*at runtime first*, then optionally pushed back into the boot path. Accepted-step ladder:

| Step | What we gain | Mechanism | Verification status |
|---|---|---|---|
| S1 — Tier 1 shell | Interactive `HwHiAiUser` on every chip | SandBoxSo ctor payload (§3 route 1) via old-rootfs type-9, `verifyFlag ≥ 8`, or explicit-type HDC `LoadPackage` (§7) | Oct-28 execution proof is hardware-validated; the boot-time install variant is the pending experiment |
| S2 — Read the ground truth | `/proc/cmdline` (`module.sig_enforce`?), `KEXEC_SIG(_FORCE)` via `bzImage`/`Config` on device, flag file contents, `sand_box/` layout | Just `cat` from S1's shell | Not started — trivially unblocked by S1 |
| S3 — Runtime custom firmware | Custom kernels/plugins/`.so` install freely every boot (no CMS fight) | `verifyFlag = 8` written from the shell + host-driver re-push (already own the driver) | Mechanism observed; not yet exercised as a steady state |
| S4 — Secure-boot bypass, runtime flavor | Arbitrary kernel+rootfs, unsigned, every boot | `kexec` from the stock boot; ctor payload auto-kexecs → stock signed loader becomes a shim for our OS | `CONFIG_KEXEC=y`/`KEXEC_SIG=y`, no FORCE *observed* in §4; must confirm S2's `/proc/cmdline` first |
| S5 — Boot-path bypass (stretch) | Flash-and-go: BL31 accepts our image directly | efuse/ATU options (jishuzhan evidence someone read them) or a BL31 verify hole per `SECUREBOOT-WEAKNESSES.md` | Card-time questions only answerable once S1's shell (or UART, §3 route 3) is on all chips |

Key structural point that makes this ladder safe: the OS boots from **host-side files** that we
already own (§1), and flash holds only HBOOT stubs. So from S4 onward the only Huawei-signed
thing in the boot chain is the loader/shim — the same posture as x86 "signed shim, anything
underneath." "Secure-boot bypass" is therefore *achieved in effect* at S4 even if S5 never
lands; S5 only removes the shim.

Sequencing note: S4's auto-kexec makes S1's SSH door moot for production (we'd have a real
shell/console anyway) but S1 remains the credential for everything that follows — no step above
S1 is attemptable without the ctor payload landing once.

---

## 9. 2026-10-30 evening — explicit type-9 door live-validated; container gates decoded

Executed the §7 experiment on NPU 9 with the existing `tsd_simple` client
(`SendAICPUPackageSimple(dev, path, name, sink=1)` from `libtsdclient`):

1. **Type-9 CommonSink works on the running image** — `insert package:... config` →
   V2 file send → `Begin to load package, type=9` → `common_sink_package_worker` →
   `Load common sink package success` for stock bytes (device hash recorded, rspCode 0,
   6/6 chips OK throughout). The name→type table is fully bypassed; no reboot needed.
   The morning's ":271 vs :279 two builds" note was actually two call sites (CommonSink
   vs extend `ResetPkgHashCode`) of one build — no image change ever occurred between
   Oct 28 and 30 (all `Ascend310P.cpio.gz` on the host are md5-identical to Oct 15).
2. **Sink sources are host-owned files**: `latest/conf/ascend_package_load.ini` (package
   list: legacy/hccl/mc2/hcomm/hixl-compat...), `latest/opp/Ascend310P/aicpu/…_legacy.tar.gz`,
   built-in `opp/built-in/op_impl/aicpu/kernel/aicpu_hcomm.tar.gz` (currently the Oct-29
   modified copy, 7,947,470 B; stock backup `*.stockbak` 7,991,266 B — hence its repeated
   verify-fail retries).
3. **Container layout decoded (legacy, 64,445,730 B)**: header `0x0..0x2100`
   (esbc magic `0x55aa55aa`@0x0; `ascend_aicpu` tag@`0x2000`; **sha256(gzip stream)@`0x2018`**;
   **code-len u32@`0x205c`**; **total-size u32@`0x588`**) | gzip(tar, 12 members,
   `aicpu_kernels_device/{bin_hash.cfg, *.so}`) `0x2100..0x3d7073e` | `0x12` zeros +
   `cms\0` + len u16 `0x23f1` | PKCS#7 `@0x3d70760` | post-CMS 12,753 B (opaque blob).
4. **flag=1 verification semantics (measured, 9 byte-flips + 5 rebuilds)**:
   - size gate (`GetCmsInfoOffSetFromPkg`) compares file length with header `0x588`;
   - `VerifyIniData: hash code cmp` re-hashes only a **<16 KB prefix of the gzip stream**
     (flips at stream-rel 0x4000/0x10000/0x40000/32 MB pass; 0x2100 fails — gzip magic);
   - `CheckCmsDataLenValid` derives CMS-internal positions from the length fields — any
     size shift lands its reads inside the DER and fails (`taglen` garbage);
   - body content is integrity-checked only by the transport checkcode (file sha256);
     failed installs are non-destructive (device hash unchanged, retries idempotent).
   - Net: **flag=1 content-frozen** — the code region has exactly zero slack
     (`code_len == stream size`), and Huawei's compressor is unreproducible beyond a
     ~22 KB prefix (python zlib level 6 matches stock stream bytes to `0x58f8`).
5. **verifyFlag**: still device-side, still 1; host mirror guess
   (`latest/device-sw-plugin/verify/flag`, incl. literal `Close`) had no effect on
   `SetVerifyFlag ... new:` — the Oct-28 conclusion holds: we need a device-side write.
6. Rig state after: opp legacy restored to stock (`.kbackup`), mirror dirs removed,
   device hash `d1715cdf…` (stock legacy installed), 6/6 OK. Artifacts:
   `~/exp-20261030/` (exports, strace).

**Next candidate (open tomorrow):** a host-reachable device-config write for the verify
flag (DMS/smrpc or `spi_mcu`), which converts the type-9 door into the signature-free
channel exactly as Oct-28 predicted — everything else (transport, routing, extraction
pipeline) is proven working tonight.

## 10. 2026-10-30 late session — verify-flag mechanism located; rootfs container is digest-pinned

1. **Verify-flag source (binary-verified)**: device `libtsdaemon.so` →
   `CmsCbbVerifyPkg::SetVerifyFlag` → `ParseVerifyFlagFromCfgFile` → opens
   `<base>/device-sw-plugin/verify/flag` (log `update verify flag old:1, new:1` =
   re-read per verify). `IsVerifyFlagValid`: value ≤7 required for checks; 8 ⇒
   `VerifyFlag is [close], skip package verify`. File lives on device tmpfs
   `/var` (host `/var/asdrv` absent); initrd `/var` entries are the only pre-shell
   seeder — blocked by (2).
2. **Rootfs esbc container is DIGEST-PINNED — first-ever rootfs modification attempt on
   this rig failed to boot**: ini slot (76 B, signed by cms) is literally
   `initrd, 39634b3e51…;` == sha256(`data[0x2000:0x2000+code_len]`) — verified on stock.
   Boot path enforces it: our surgical repack (gzip level-6 re-compress of full 364 MB
   cpio, boot-script append, header digest/code_len rebuilt, slots carried) ⇒ device
   stopped at PCIe handshake, 0/6 (bringup: `Expected six Ascend device nodes, found 0`).
   Note: the on-disk `Ascend310P.cpio.gz` "modified Oct 26" was byte-identical to
   `.orig` (touched only) — no modified rootfs had ever been boot-tested.
3. **Boot-pinning chain established**: type-9 package cmscbb (flag=1) pins package
   bytes; rootfs ini/cms pins payload bytes. Both need either the flag (≥8) or a
   re-signed CMS — same wall at two layers (custom-firmware plan S3/S4).
4. **Rig ops learnings (this session)**: chained reloads degrade the host HAL quickly
   (`npu-smi` instances get stuck in ioctl D-state and pin module refcounts; `systemctl
   start` on the `RemainAfterExit` service is a no-op — must `stop` first or unload
   fully); **host reboot is the reliable recovery** (6/6 every time). Full-clean unload
   needs `drv_davinci_intf_host`, `drv_vpc_host`, `ascend_logdrv`, `drv_seclib_host`
   in the list (earlier list omitted them) and npu-smi kills between passes.
5. **Flag-file seeding remains OPEN, unblocked candidates**: (a) whitelist traversal
   push into `/home/HwHiAiUser/**` if tsd's base there is honoured (path base of
   `GetHwCertCfgFilePath` not yet disambiguated between `/var/asdrv/dev<N>` and
   `/home/HwHiAiUser`); (b) sign a tiny custom ini/cms; (c) kexec-style boot (plan S4)
   where the payload itself writes the flag at constructor time.
6. Final state: 6/6 healthy; rootfs, opp legacy = stock (`.kbackup`,
   md5 `0c93f734…`); modified container kept at `~/exp-20261030/rootfs_x/surgical2.cpio.gz`;
   `hcomm` built-in still the Oct-29 modified copy (`.stockbak` available).

## 11. 2026-10-30 late session — type-0 gate decoded step by step; DMP re-confirmed stub

See `card_root_verify_flag_20261030.md` (§7a) for the full trail. Summary: a payload
-modified syskernels package pushed via the `tsd_load2` name-table route peeled four
gates (whitelist name → code_len = size − 0x2000 @`0x45c`/`0x145c` → digest =
sha256(code) @`0x2c`/`0x102c` → **RSA-PSS over the code region**, `0x88200203`).
The syskernels container uses the rootfs esbc geometry. `dsmi_cmd_get_custom_sign_flag`
returns `rc=8` (compiled-out stub, re-validated post-reboot). All host-pushable channels
now confirmed signature-pinned at flag=1; device left healthy on stock (6/6), opp slot
restored, modified package staged at `/tmp/Ascend310P-aicpu_syskernels.tar.gz`.
