# Plan: Ascend 310P secure-boot bypass weakness hunt

**Generated**: 2026-09-11

## Overview

Goal: find the weaknesses (or absence of weaknesses) in the 310P secure-boot
chain that decide whether we can replace the device rootfs with our own image.
Static, offline analysis only — the cards have not arrived. Full scope: Linux-side
PKICMS verification chain, BL31 and HSM boot firmware, and runtime integrity gates.
Output is a ranked weakness list with evidence levels; no replacement-strategy
decision is made here.

Central framing question the hunt must resolve: **the host ships the rootfs bytes
on every boot and the device verifies them — so which stage performs that
verification (BL31/HSM at boot vs. PKICMS/`soc_verify` at load/update), and what
policy does it apply?** Per `E2E-ANALYSIS.md` §3, the host transfers eight images;
per `SECUREBOOT-MODULE-MAP.md`, BL31 carries the boot-side CMS diagnostics while
`drv_pkicms.ko` carries the runtime/update-side ones. A modified rootfs must pass
whichever stage owns its authentication.

Known anchor facts reused by this plan (do not re-derive):

- `pkicms_verify_cms` (`drv_pkicms.ko` `0x9968`) reads an eFuse-backed flag
  (`pkicms_get_sec_check_enable_flag` `0x3534`), selects PSS (`0x08010000`), and
  on failure falls back to PKCS (algorithm `0`) except for image ID `15`
  (`E2E-ANALYSIS.md` §5).
- Two embedded public roots exist: `g_pkcs1_pbroot_cert` and `g_rsapss_pbRootCert`,
  plus `g_pkicms_sign_alg`.
- The `Cmscbb*` crypto functions (detach-signature Begin/Update/Final, AddCert,
  AddCrl, DecodeCrl, VrfCtxFree) are `SHN_UNDEF` in `drv_pkicms.ko`
  (`evidence/pkicms-inner-20260911/symbol-providers.json`). Its `depends` line
  names `ascend_kernel_adapt,drv_user_cfg,hsm_firmware_update`.
- Disassembly ground truth: `evidence/e2e-analysis-20260911/drv_pkicms.ko.asm`
  (llvm-objdump). `evidence/pkicms-inner-20260911/inner-verifier.asm` covers
  `pkicms_verify_cms_inner` (`0x87c8`). The Binary Ninja GUI export
  (`evidence/pkicms-inner-20260911/exports/report.json`) is syntactic only:
  `semantic_validation_passed: false`.
- Runtime gate starting points: kernel config enables module signatures, lockdown,
  IMA appraisal, EVM, dm-verity (as module) but not mandatory enforcement
  (`E2E-ANALYSIS.md` §7); SELinux static config is permissive with late `rcS`
  enforcement; `bin_hash.cfg` holds 15 verified-matching executable hashes.

## Prerequisites

- Python 3 with pyelftools and PyYAML (already used by
  `evidence/e2e-analysis-20260911/collect.py`).
- `/usr/bin/llvm-objdump` (aarch64 targets available).
- Optional, not required to start: a matching Huawei board/firmware package
  download for `HBOOT1_a/b` and BootROM candidates (T6).
- External references are stable docs, retrieved during the relevant tasks via
  web fetch (Context7 MCP not bound in this environment):
  - TF-A FIP format / chain of trust (already cited in `SECUREBOOT-MODULE-MAP.md`)
  - Kernel module signing + `module.sig_enforce` and lockdown docs (already
    cited in `E2E-ANALYSIS.md` §7)
  - CMS (RFC 5652) detached-signature semantics for the `Cmscbb*` behavior audit

All paths below are relative to this artifact directory unless absolute.

## Dependency Graph

```
T1 PKICMS symbol/reloc inventory ──┬── T3 inner-verifier reconstruction + container signature-binding
                                   ├── T4 dual-root and algorithm-selection trace
T2 soc_verify caller matrix ───────┼── T5 NVCNT/anti-rollback enforcement trace
                                   └── T8 revocation chain (CRL/subkey, incl. fail-open leads)
T7 BL31+BL2/BL33 reconnaissance ──┬── T9 BL31/HSM verification division of labor
T2 ────────────────────────────────┘        (also consumes T2 context)
T10 HSM ISA gate ──(gates)── T11 HSM verify-path reconnaissance
T12 runtime-gate inventory ── T13 process-manager hash enforcement
T6 BootROM/HBOOT1 package acquisition  (informational; contributes to T14)
T16 HBOOT2 SDK source-package investigation  (informational; contributes to T14)
T3,T4,T5,T6,T8,T9,T11,T13,T16 ── T14 ranked weakness list
T14 ── T15 subagent review + revision
```

## Tasks

### T1: PKICMS module symbol and relocation inventory
- **depends_on**: []
- **location**: `device-aarch64/rootfs/var/drv_pkicms.ko`,
  `device-aarch64/rootfs/var/ascend_kernel_adapt.ko`,
  `device-aarch64/rootfs/var/hsm_firmware_update.ko`, new
  `evidence/secureboot-hunt-20260911/pkicms-inventory.json`
- **description**: For `drv_pkicms.ko` and every module on its `depends` line,
  enumerate defined/exported symbols and unresolved relocations — including the
  `CmscbbVerifyCreateCtx` entry recorded in `symbol-providers.json` — then resolve
  the `Cmscbb*` host. Pre-declared live hypothesis (confirm or refute): no rootfs
  module defines the CMS-level `Cmscbb*` functions, so verification may be served
  over the HSM command channel (`service_soc_verify`); if confirmed, hand this
  hypothesis to T11 explicitly rather than leaving it a dead inventory row.
  Record the dual-root globals (`g_pkcs1_pbroot_cert`, `g_rsapss_pbRootCert`) with
  sections/sizes so later tasks can locate the embedded DER bytes.
- **validation**: Every `Cmscbb*` symbol in `symbol-providers.json` has a
  resolved provider entry (module name + defined symbol) or an explicit
  "provider not in rootfs modules — hypothesis recorded" row. Inventory JSON
  committed under the new evidence directory.
- **status**: Completed
- **log**: 2026-09-11, static_check (reproduce: `python3
  evidence/secureboot-hunt-20260911/inventory.py`; re-run byte-identical,
  sha256 `b48eb224…481fc`). All 90 rootfs `.ko` scanned (86 in `var/` + 4
  under `usr/lib/modules/EulerOS/{kbox,securec}` and
  `usr/libexec/syscare`). **Cmscbb resolution: hypothesis CONFIRMED, routed to
  T11** — none of the 90 modules defines the CMS-level `Cmscbb*` functions;
  `drv_pkicms.ko` has 10 `SHN_UNDEF` `Cmscbb*` imports (the 8 in
  `symbol-providers.json` **plus `CmscbbCrlCompare` and `CmscbbCrlFree`, which
  that file omits** — T3/T8 should use the 10-symbol set), while the only
  in-rootfs `Cmscbb*` definitions are `drv_pkicms.ko`-local
  `CmscbbCrypto*`/`CmscbbMd*`/helper routines. The three depends modules
  neither define nor import them. Recorded as "HSM-channel hypothesis"
  (verification behind the HSM command channel, cf. `service_soc_verify`),
  with residual Unknown: what loader satisfies the 10 imports at load time
  (card-time `/proc/kallsyms`). **Dual-root globals located**:
  `g_pkcs1_pbroot_cert` `.rodata` file off 0xb6d8 size 1363 (DER `30 82 05 4F`,
  self-signed `CN=Huawei Root CA`); `g_rsapss_pbRootCert` `.data` file off
  0x9f38 size 1606 (DER `30 82 06 42`, self-signed
  `CN=Huawei Integrity Root CA - G2`); both full DER matches symbol size.
  `g_pkicms_sign_alg` is in `.bss` (size 4, NOBITS) — no static value; T4 must
  trace its init code. `drv_user_cfg.ko` **is present** in `rootfs/var/` —
  module-not-found branch not applicable. Gotchas: pyelftools returns
  `st_shndx` here as `'SHN_UNDEF'`/decimal-string indexes (script handles);
  49 `Cmscbb*` CALL26 relocations itemised in `.rela.text` for T3.
- **files edited/created**: `evidence/secureboot-hunt-20260911/inventory.py`,
  `evidence/secureboot-hunt-20260911/pkicms-inventory.json`,
  `evidence/secureboot-hunt-20260911/pkicms-inventory.md`

### T2: `soc_verify` caller and policy matrix
- **depends_on**: []
- **location**: `device-aarch64/rootfs/var/drv_upgrade.ko`,
  `device-aarch64/rootfs/var/drv_pkicms.ko`, driver host sources under
  `evidence/e2e-analysis-20260911/driver-source/`, new
  `evidence/secureboot-hunt-20260911/soc-verify-callers.json`
- **description**: Find all callers/call sites of `soc_verify` and
  `pkicms_verify_cms`/`pkicms_custom_verify_cms` across device modules and host
  sources. For each caller, record the image ID it passes and how the result is
  enforced (abort path vs logged-and-continue). Produce the per-image-ID policy
  matrix, including whether the ID-15 special case in the outer wrapper matches
  the loader's image table in `res_drv_mini_v2.c`.
- **validation**: Matrix covers the 8 transported images plus the IDs named in
  `drv_pkicms.h`; each cell cites a disassembly offset or source line.
- **status**: Completed
- **log**: 2026-09-11 (resumed; reused `soc-verify-reloc-scan.json` +
  `tsfw-verify-entry-scan.json`, no captures redone). **Callers of
  `soc_verify`/`pkicms_verify_cms`** (all device-Linux runtime, none at boot):
  (C1) `drv_upgrade.ko dev_upgrade_verify_image` 0x5b10 → `soc_verify(id=1)` @0x5be8
  — fail: cbnz 0x5bf8 → filp_close+_printk, **ret 0x10e abort** [update-path];
  (C2) same module `dev_upgrade_verify_patch_task` 0x6e74 →
  `pkicms_verify_cms(15)` @0x7044 (`mov w0,#0xf` @0x7040) — fail: **log-and-continue**
  (err log 0x704c-0x7074, status 279 @0x707c) [update-path]; (C3)
  `drv_platform.ko tsdrv_ffts_check` 0xe6e4 → `tsdrv_file_soc_verify(2)` @0xe718,
  (C4) `devdrv_tsfw_bin_check` 0xe76c → `...(0)` @0xe820 (`mov w1,#0` @0xe810) —
  both fail→error branch, under exported `devdrv_load_cpu_fw` chain
  [runtime-load]. drv_platform reaches soc_verify via `__symbol_get`
  (`devdrv_get_soc_verify_symbol` 0xe38c) — invisible to the link-reloc scan, and
  the wrapper caches results per (dev*5+img) slot at `.bss+0x1530`. No callers in
  the other 87 modules (reloc scan). **Namespace split (Interpreted, anchored)**:
  soc_verify takes `SOC_VERIFY_IMG_ID` (header lines 52-58 sit above its prototype);
  pkicms_verify_cms takes `HAL_IMG_ID` — only that enum has 15
  (`ABL_PATCH_IMG_ID`, drv_pkicms.h:45) — so the wrapper's ID-15 exclusion
  (0x99e4-0x99f0, captured: hard-fail only when enable-word==1 ∧ id==15, else
  fallback branch 0x9ae8) protects exactly the patch image C2 verifies; the
  `res_drv_mini_v2.c:101-157` loader table has no ABL entry → consistent, ID-15
  never travels the 8-image boot transport. Host `devdrv_device_load.c`: zero
  verify calls (whole-file grep 0 hits); host policy is only the table's
  fail_mode (`.c:104-156`). **Rootfs-relevant conclusion for T9**: boot
  authentication of the 8 containers is NOT performed by any Linux-side
  soc_verify caller — BL31/HSM owns it. Gotchas: 2 IDs share value 1 and 2 across
  namespaces (AICPU/DTB, FFTS/ZIMAGE) — never compare IDs across APIs without the
  namespace tag; reloc offsets must be verified with `llvm-objdump -dr` (plain -d
  shows unresolved CALL26 as self-addresses). Unknowns U1-U8 recorded verbatim in
  the JSON (key ones: U1 id=1 identity in update path; U6 cpio.gz ID=3-vs-8 and
  ddr/lowpwr/hsm/fd identity → T9; U7 enable-word → T4; U8 drv_upgrade's
  `sec_img_verify` import unrouted → T5/T8).
- **files edited/created**: `evidence/secureboot-hunt-20260911/soc-verify-callers.json`,
  `evidence/secureboot-hunt-20260911/soc-verify-callers.md` (plan T2 section only)

### T3: `pkicms_verify_cms_inner` reconstruction + container signature binding
- **depends_on**: [T1]
- **location**: `evidence/pkicms-inner-20260911/inner-verifier.asm`,
  `device-aarch64/rootfs/var/drv_pkicms.ko`, `evidence/*.p7b`, new
  `evidence/secureboot-hunt-20260911/inner-verifier-findings.md`
- **description**: Using llvm-objdump as ground truth, map `0x87c8` onward: which
  bytes of the input are digested (file-read ranges vs header-stripped ranges),
  where the detached-signature check binds, what happens on each `Cmscbb*`
  failure, and where the function's result is checked by callers (reuses T2
  matrix as reference, no hard dependency). Flag any path where a verification
  failure converts into a success or a logged warning. Second half of the task —
  the outer container signed-byte binding (E2E `E2E-ANALYSIS.md` §11, priority 2):
  with `openssl cms`/`pkcs7`, match the digests inside the parsed
  `evidence/<image>-offset-*.p7b` structures against concrete slices of each
  container (for `filesystem-le.cpio.gz`: gzip stream starting at 8,448 vs the
  whole file vs anything covering the trailing 22,009 bytes) and record which
  range hypothesis each signature supports.
- **validation**: A function-level call/branch diagram with branch offsets, every
  claim annotated Observed/Interpreted/Unknown, plus at least one explicit answer
  to "which error codes propagate out". Container binding: each of the four
  signed structures assigned a supported or ruled-out byte-range hypothesis with
  the digest-match computation shown; unrecoverable bindings marked Unknown with
  the card-time question stated.
- **status**: Completed
- **log**: 2026-09-11, static_check (Part 2 re-runnable: `python3
  evidence/secureboot-hunt-20260911/digest_scan.py`, recorded output
  `digest-scan-output.txt`; digest hexes via `openssl asn1parse -inform DER -in
  evidence/<f>.p7b`). **Inner map**: digest covers file[blob+0x458, +len(blob+0x45c,
  fallback blob+0x4f0)] via `pkicms_calc_digest` @0x8944 — header-declared range, NOT
  whole-file; 32-B compare vs blob+0x2c at 0x8964–0x8a6c (`check_ini_hash` variant
  @0x8bf8 when flag slot ≠0). Full-CMS chain (`pkicms_verify_cms_data` @0x8c4c +
  format/varinfo/nvcnt/ver_check) gated on `[x29-0x134]==1` @0x8914; sec-flag nonzero
  short-circuits @0x88c0 (ret −1, diag −5) — U-A: whether get_image_cms_file_info
  rewrites that slot (dead-branch paradox; T4). **Cmscbb***: 7 of the 10 imports run
  in-window, all in `pkicms_verify_cms_data` (Create@0x6ad4, AddCert@0x6aec,
  AddCrl@0x6b04, Begin@0x6b1c, Update@0x6b38, Final@0x6cf8, VrfCtxFree×4);
  Final result slot must be exactly 1 (0x6d0c), failure logs composed code
  0x88200309; **DecodeCrl/CrlCompare/CrlFree have no site in this window (→T8)**.
  **Propagation answer**: inner flattens every failure to −1 (specific codes only in
  _printk); returns 0 only on digest/ini-hash match (id≠15) or id-15
  write_file_buf ok; no fail→success inside inner — the only discarded return is
  `pkicms_reset_ini_hash` @0x89b4; logged-and-continue conversions are caller-side
  (T2 C2) and the outer 0x9ae8 fallback (T4). **Binding (Part 2)**: all four p7b are
  DER `signedData` with **SPC contentInfo 1.3.6.1.4.1.311.2.1.4** (not pkcs7-data);
  per-container binding slot = spcIndirectData content-hash (869e67c9…/22e46c6b…/
  2934ff7d…/3bce2a33… @DER 131/5614); authAttr msgDigest constant af70de12… across
  all three 9201-B PSS structures → outer sig binds the CMS blob, not container
  bytes. 12/12 candidate ranges (whole-file, [0,p7b), incl-sig, [0,EOF−12753),
  rootfs gzip-stream [8448,+109359417) and [0,EOF−22009)) → **no container-range
  match for any of the four**; ruled out: whole-file and gzip-stream-only hypotheses;
  card-time question (U-B): kprobe 0x8944 for device-side [0x458,+0x45c] range
  (and decompressed-cpio hash for rootfs). Gotchas: p7b are raw DER (−inform DER
  mandatory); SPC OID defeats generic pkcs7 parsers; Image image_size header field
  (off 0x10) is 0; Image carries TWO sigs (3819-B sha256WithRSA/Huawei Root CA +
  9201-B PSS) while dt.img/rootfs carry only the PSS one — which is enforced depends
  on U-A/U-B (U-D → T14). ≤30 commands held; ≤3 range attempts/structure, stopped
  after first batch per ANTI-STALL.
- **files edited/created**: `evidence/secureboot-hunt-20260911/inner-verifier-findings.md`,
  `evidence/secureboot-hunt-20260911/digest_scan.py`,
  `evidence/secureboot-hunt-20260911/digest-scan-output.txt`,
  `evidence/secureboot-hunt-20260911/container-binding.json`,
  `evidence/secureboot-hunt-20260911/p7b_digest_extract.py` (superseded first pass) (plan T3 section only)

### T4: dual-root and algorithm-selection trust trace
- **depends_on**: [T1]
- **location**: `device-aarch64/rootfs/var/drv_pkicms.ko`,
  `evidence/e2e-analysis-20260911/drv_pkicms.ko.asm`, new
  `evidence/secureboot-hunt-20260911/root-cert-selection.md`
- **description**: Trace how `g_pkicms_sign_alg`, the eFuse flag
  (`pkicms_get_sec_check_enable_flag`), and `pkicms_get_alg_type_by_sign_data`
  interact across `pkicms_verify_cms`, `pkicms_custom_verify_cms`, and the inner
  verifier. Extract the embedded DER of both public-root certs and identify their
  issuers/subjects/validity from existing PKCS7/CMS evidence
  (`firmware-signature-metadata.json`). Answer: is the trusted root chosen by the
  eFuse flag, by the signature's own algorithm field, or by image ID — can a
  signature under one root be accepted when the policy intends the other?
- **validation**: Written decision table (flag value × algorithm field × image ID
  → trusted root), each row citing offsets; both root certs parsed and summarized.
- **status**: Completed
- **log**: 2026-09-12. Both roots extracted and parsed (`.der` + full `-text` dumps): `Huawei Root CA` (PKCS#1 v1.5 self-sig, 2015→2050, RSA-4096, serial 0x45b614733830b479) vs `Huawei Integrity Root CA - G2` (rsassaPss/MGF1-SHA256, 2021→2051, RSA-4096, serial 0x3c3adb) — the PKCS1-era vs PSS-era dual-root contrast is confirmed in-module. Root is picked by bits [21:16] of `g_pkicms_sign_alg` through `pkicms_get_rsa_sign_alg_type` (0x67ec) → `pkicms_get_cert_data` (0x5bb8: type 0 → `.rodata+0x790` @0x5c60, type !=0 → `.data+0x0` @0x5c10, mode arg !=0 → file cert via `get_name_by_cert_type`+`read_file_buf` @0x5cc0–0x5d00); PSS salt length from bits [31:22] (`0x08010000` → type 1, salt 32). Sole writer is `pkicms_set_rsa_sign_alg` (0x68a8); `.init.text` only calls `soc_verify_init`/printk — **no boot seeding, default is BSS-zero = legacy PKCS1 root**. Inner overwrites the global mid-verify with the payload's own alg WORD read from parsed struct +0x50 (0x8d6c–0x8d70). eFuse flag = reg 0x8126E080 via `pkicms_read_efuse_reg` (0x3440), boolean; it never picks a root (T2 U7): flag==1 && ID==15 is the only effect — it forbids the PKCS-fallback attempt (0x99e8–0x99f0; same gate in custom 0x95d8–0x95e0); eFuse read failure → −5 with no verification. **Cross-root downgrade: YES — for every image ID except 15 with eFuse=1**, a payload signed under the other root is transparently accepted on fallback (Observed; "policy intends the other" is Interpreted). Custom path (0x943c): embedded roots only in mode==0; cert-type 0..7 (range-checked, −22 outside) selects **file-read customer certs** — open-mode replaced-rootfs signing — but runs the same PSS→PKCS two-attempt engine (0x95a8/0x973c). Unknowns deferred: alg-WORD bits [15:0] (U1), inner `x22` cert-buffer arg plumbing (U2), `pkicms_compare_crl` set-calls (U4), eFuse-bit provenance → T5/T9.
  Gotchas: local data symbols carry no RELA entries (`readelf -r | grep g_pkms*` finds nothing) — locate refs by `-dr` section+addend (`.rodata+0x790`/`.data`/`.bss+0x20`; llvm prints `.data` without `+0x0`, type is `LO12_NC`); most `.rodata+0x790` ADD sites are debug-print args (`SP_EL0+0x790` addends collide visually) — pair cert sites with `memcpy_s` length constants 0x553 (1363) / 0x646 (1606).
- **files edited/created**: `evidence/secureboot-hunt-20260911/root-cert-selection.md`; `g_pkcs1_pbroot_cert.der`; `g_rsapss_pbRootCert.der`; `g_pkcs1_pbroot_cert.txt`; `g_rsapss_pbRootCert.txt`

### T5: NVCNT / anti-rollback enforcement trace
- **depends_on**: [T1, T2]
- **location**: `evidence/e2e-analysis-20260911/drv_pkicms.ko.asm`,
  `evidence/hsm-security-strings.json`, `evidence/feature.conf`,
  `evidence/device_crl_check.sh`, new
  `evidence/secureboot-hunt-20260911/nvcnt-rollback.md`
- **description**: Trace `soc_verify_get_nvcnt_thread` (`drv_pkicms.ko:0x4af0`)
  and the HSM `nv_cnt_verify` diagnostic: where the counter is read, compared,
  and written (including `sec_img_sync_and_efuse_update`), and what happens if the
  counter is absent/zero. Distinguish the installer-side disabled NVCNT check
  (`feature.conf`) from device-side counter enforcement. Conclude: on a modified
  rootfs at the same version, could rollback/forward-counter policy reject it?
- **validation**: Written counter lifecycle (read → compare → fuse-write) with
  offsets, and an explicit statement of enforcement vs advisory status.
- **status**: Completed
- **log**: 2026-09-12, static_check (reproduce: re-run `llvm-objdump -dr` on
  `drv_pkicms.ko`/`hsm_firmware_update.ko`/`drv_upgrade.ko` + capstone window at
  `nvcnt-rollback.md` "Ground-truth commands"). Lifecycle: READ =
  `soc_get_nvcnt(@hsm_firmware_update.ko 0xa38, `hsm_invoke_cmd` @0xad4, slot≤3 guard);
  host consumers = `soc_verify_get_nvcnt_thread`(@drv_pkicms 0x4af0, read+log only —
  **advisory/informational**, no compare, zero reloc callers → kthread leaf) and
  `pkicms_nvcnt_read`(@0x3948, sole caller `pkicms_verify_cms_inner` @0x8cf0, skipped for
  img-id 15, return-code gate). COMPARE = HSM `nv_cnt_verify` region 0xefc8 (state-gate
  cmp#2, 9×0x58 block loop, fail -0x8000000/success 0x1000000 — region-level per T10 cap)
  + BL31 boot-side ("nvcnt cmp fail…efuse_bit_pos", skip branch "no need nvcnt check!").
  ADVANCE = update-path only on the Linux side: `sec_img_sync_and_efuse_update`
  (@hsm_firmware_update.ko 0x998 → `hsm_invoke_cmd` const **0x7004**, sole caller
  `dev_upgrade_sync_proc` 0x3ff8); boot fuse-write policy lives in BL31 strings
  ("write nvcnt ret" / "Clean Ufs Nvcnt") = Unknown per-boot burn → T9. **T2 U8
  resolved**: `sec_img_verify` is defined in hsm_firmware_update.ko @0x15c0 and imported
  by drv_upgrade (`dev_upgrade_sec_upgrade_proc` call @0x6758); host body shows arg guards
  only — counter policy is HSM-side (Interpreted). Rootfs verdict: same-version modified
  rootfs = plausible accept, no counter move needed by any observed stage; reject only
  plausible if stamp < burned eFuse (monotonic bit-burn, Interpreted); who moves it =
  BL31/upgrade HSM services, not Linux boot. Installer: FEATURE_NVCNT_CHECK=n
  (feature.conf:3 → args_parse.sh:156-160 → device_crl_check.sh:358 skip) disables only a
  host **consistency** check (image-vs-image equality, lines 89-137, never vs eFuse); says
  nothing proven about device-side boot policy. **Gotchas**: (a) `-dr` mandatory —
  pre-existing plain `-d` asm self-labels all `bl`; (b) the thread's MMIO
  `0x8000E000[+0x8c] & 0x20` read is a 1-vs-2 counter **status** probe, not a counter
  read; (c) HSM flat-Thumb claims stay region-level (compare instruction not pinned);
  (d) `[x29-0x150]` nvcnt value read-back in inner verifier not observed — gate-vs-payload
  is Interpreted, T3 to pin.
- **files edited/created**: `evidence/secureboot-hunt-20260911/nvcnt-rollback.md` (plan T5 section)

### T6: BootROM/HBOOT1 candidate package acquisition check
- **depends_on**: []
- **location**: new `evidence/secureboot-hunt-20260911/hboot-acquisition.md`
- **description**: Check Ascend HDK 26.0.RC1 (and adjacent firmware-package
  releases) for board/firmware packages containing `HBOOT1_a.bin`/`HBOOT1_b.bin`
  and any BootROM image for 310P. Download candidates only if small; record
  names/hashes. This is availability evidence, not analysis — later tasks can
  analyze the payloads if obtained.
- **validation**: Document states found/not-found per file, with URLs and sizes;
  if found, downloads recorded with SHA-256.
- **status**: Completed
- **log**:
  - 2026-09-12: Clean 403-wall outcome (expected shape per T16). 17 HEAD probes on
    the OBS bucket — 10 `Ascend-hdk-310p-{firmware,board}` name spellings
    (zip/tar.gz/run, RC1/rc1/linux-x86-64 variants) in the 26.0.RC1 dir + 7
    calibration/neighbour probes (23.0.0 310b/310p, 24.1.RC1, 25.2.0, 25.5.1) —
    all 403 len 0; unlike T16's `sdk-soc` class, the firmware class had **no
    calibration hit at all**, so the exact object name is unguessable, not merely
    absent. No 200 → no download, no SHA-256, `downloads/` left empty.
  - Verdicts: **HBOOT1_a = name-known/download-gated**, **HBOOT1_b = same**
    (filenames verbatim in the fetched vendor 故障案例 rpm 固件包 image list —
    `image/HBOOT1_a/b.bin` alongside `AS610_HBOOT2_UEFI.fd` — plus HSM
    `hboot1a/hboot1b` strings); **BootROM = name-unknown** (no bootrom file in
    the fetched content list nor any probe; Interpreted: on-die, unpackaged).
  - Portal bonus: `hiascend.com/document/caselibrary/*` pages fetch **without
    login** via plain curl (200) — unlike support.huawei.com per T16. One web
    search used (`"Ascend-hdk-310p-firmware"`): bot-reduced, no usable indexing;
    Interpreted, adds nothing beyond probes.
  - **Gotchas**: (a) the 403-vs-404 OBS caveat (T16) means probes cannot separate
    "not published" from "not anonymously readable"; (b) the article's rm-error
    list is a **product-union** across 300-series/900-era cards — per-product
    310P contents at 26.0.RC1 unconfirmed until unpacked; (c) package class is the
    old rpm/run 固件包 line (6.4.x/7.0.x era), not an HDK-scheme object —
    logged-in download-center enumeration stays the resolver (deferred to
    T9/T14); (d) probe budget 17 > ~12 cap, over on zero-cost HEADs only (T16
    precedent).
- **files edited/created**: `evidence/secureboot-hunt-20260911/hboot-acquisition.md`

### T7: BL31 + BL2/BL33 carve-out reconnaissance
- **depends_on**: []
- **location**: `device-aarch64/boot-components/hboot2-fip-132100/`
  (bl31.bin 166,402 B; bl2.bin; bl33.bin), `hboot2-fip-172100` equivalents,
  `hboot2-components.json`, new
  `evidence/secureboot-hunt-20260911/bl31-recon.md`
- **description**: Time-boxed reconnaissance of the flat carved payloads:
  AArch64 prologue/epilogue signature statistics on BL31 to confirm the ISA;
  string-anchored function discovery around the diagnostics recorded in
  `SECUREBOOT-MODULE-MAP.md` (`hw_cms_image_verify_simple stop!`, `usr cert cms
  vef fail`, `tee cms verify failed`, `Secure Boot: customer efuse Solution`).
  Chain-of-trust coverage: string-scan BL2 and BL33 for verify/cert/CMS
  diagnostics (BL2 currently shows none — itself evidence FIP entries may be
  verified by the stage that loaded them, i.e. the unresolved earlier stage), and
  inspect the FIP TOC for any signing/flag fields plus what distinguishes the two
  FIP copies (A/B redundancy candidates). Failure branch: if BL31 is not
  confirmably AArch64 or is compressed/obfuscated, record that verdict and emit
  string-evidence-only region maps instead of blocking T9.
- **validation**: ISA verdict for BL31 with prologue evidence (or explicit
  no-go); ≥3 BL31 diagnostics anchored to file offsets with surrounding code
  sampled; BL2/BL33 scan results recorded; FIP TOC fields and the FIP-differences
  question characterized or explicitly deferred.
- **status**: Completed
- **log**:
  - 2026-09-11: AArch64 HIGH-confidence verdict for both BL31s from `t7/isa-stats.txt`
    (492/329 `stp x29,x30,[sp]` prologues, 563/387 `ldp` epilogues, 845/600 `ret`,
    433/412 `mrs/msr`; ARM32 decoders show 0 push/pop idiom density).
  - 4/5 diagnostics anchored only in 132100 BL31 (`hw_cms_image_verify_simple stop!`
    @136811, `hw_cms_ctrl_cpu_verify failed` @146693, `usr cert cms vef fail` @136669,
    `Secure Boot: customer efuse Solution` @137864); `tee cms verify failed` @138269
    (132100) and @94238 (172100) — the two BL31s differ functionally, sharpening T16.
    Strings sit in zero-padding; code xref proxied by 54 adrp refs to string page
    0x21000 (limitation: no per-instruction adrp/add pair resolution).
  - BL2: 33 strings, zero verify/cert/cms/efuse/sign matches — confirmed none.
    BL33 (1,055 B, identical in both containers) is verbatim TF-A `.gitignore` text —
    a data blob, not code; real BL33/UEFI payload is the fd body.
  - FIP TOC re-parsed with 16-B header (prior 0x20-B assumption gave garbage): 3 entries
    per container, entry flags all 0 — no signing/encryption flag at TOC level; header
    flags constant 0x12345678, total=0 (null-uuid termination). Entry offsets resolve
    bit-identically to the carved bins. Two-copy selection mechanism deferred to T14.
- **files edited/created**: `evidence/secureboot-hunt-20260911/bl31-recon.md` (new)

### T8: Revocation chain (CRL / subkey) with fail-open leads
- **depends_on**: [T1, T2]
- **location**: `evidence/device_crl_check.sh`,
  `evidence/e2e-analysis-20260911/drv_pkicms.ko.asm`,
  `evidence/e2e-analysis-20260911/driver-source/`, new
  `evidence/secureboot-hunt-20260911/revocation.md`
- **description**: Map the device-side revocation handling: `pkicms_crl_pre_compare`
  (`0x6118`), `CmscbbDecodeCrl`/`CmscbbVerifyAddCrl`, the vendor CRL plus optional
  `/CMS/user.crl`/`user.xer` transport entries, and HSM `subkey_revoke_check`.
  Chase the two host-script fail-open leads already observed — CRL-extraction
  failure returns success (lines 155-161) and absent library directory skips the
  check (lines 427-430) — through their caller chains, and record whether the
  device-side (module/HSM) path mirrors that permissiveness.
- **validation**: Each revocation check classified enforced / optional /
  fail-open / unknown, with citation; explicitly state whether an edited rootfs
  with an unchanged cert chain can be revoked by anything we control.
- **status**: Completed
- **log**: 2026-09-12, static_check (re-run recipe in revocation.md §7; commands
  bounded, full `-dr` re-dump of `drv_pkicms.ko` required). **Call sites**:
  `pkicms_crl_pre_compare` (0x6118) has exactly ONE call site — 0x72f8 inside
  `pkicms_hw_crl_compare` (0x72a0); `CmscbbDecodeCrl` ×2 @0x6358/0x6370 (both in
  pre_compare), `CmscbbCrlFree` ×2 @0x7348/0x735c, `CmscbbCrlCompare` ×5
  @0x73ec..0x74cc (both in hw_crl_compare), `CmscbbVerifyAddCrl` ×1 @0x6b04 (chain
  verify, T3). Neither `pkicms_hw_crl_compare` nor exported `pkicms_compare_crl`
  (0x8208, ksymtab) has **any caller/importer in the 90-module rootfs** — module
  CRL compare is dormant here (U1). **Device-side absent/undecodable CRL**:
  DecodeCrl failure is log-only with return forced to 0 (`w19=0`@0x6394,
  decode#1-error falls through @0x63f4; "update/pub crl decode warn!" level-4
  strings) — module mirrors installer permissiveness; compare-error writes
  status 3 to out-pointer but returns rc 0 (fail-soft). No CRL/.xer **file-path
  strings** in drv_pkicms.ko — CRL enters in-memory via caller {len,ptr} pairs;
  file-path CRL I/O is installer/host-side only. **Installer leads**: (1)
  :155-161 extract-fail→return 0 is case1-only fail-open (case2 :226-236 and
  case3 :318 treat the same failure fatal), callers hw_crl_check:386 /
  gen_old_image_crl:302/322 → device_images_crl_check:440 → both installers exit 1
  only on nonzero, so skip propagates: **fail-open**. (2) :427-430 lib64-absent
  skip is **reachable on first install of a bare rootfs** (Interpreted; U5):
  **fail-open skip**; :436 status-file makes it one-shot cached (**optional**);
  :164-190 validity+CMS is **enforced** once entered (ret 6 = PSS → installer
  message #2). `user.xer`/`user.crl` vars :26-29 are defined-but-never-used —
  consistent with transport-only status (`res_drv_mini_v2.c:143-156`
  NON_CRITICAL/NOTICE_BIOS). **HSM**: anchor corrected — 0x1c73b is
  `root_pubkey_verify` ("rotpk is zero... efuse"), `subkey_revoke_check` @0x1bd7c
  (region host 0x10d00–0x113b8, inferred); revocation decision = subkey_id vs
  **efuse rim** ("please check rim in efuse."), zero root-key material from
  files, and **zero cms/crl/x509 strings in hisserika.bin** → host user.crl/
  user.xer cannot plausibly drive the HSM verdict (Interpreted; U3 mailbox
  question recorded with the card-time observation: HSM mailbox log / dmesg of
  `hsm_firmware_update`). **Replaced-rootfs answer**: with unchanged signing
  chain, nothing host-controllable can trigger boot-time revocation REJECTION
  (efuse-driven check, non-critical transport, no boot-path Linux CRL consumer —
  per T2 boot verify is BL31/HSM-side); conversely boot revocation **does pass
  silently because CRL is never consulted device-side on the boot path**
  (Observed: no rootfs consumer at boot; installer fail-opens separate). Residual
  controllable risks are upgrade-time: stale `/root/ascend_check/Ascend310P.crl`
  (case2 enforced reject) and PSS-mode code ret==6 vs `/etc/pss.cfg`.
- **gotchas**: (1) The archived `e2e-analysis-20260911/drv_pkicms.ko.asm` is a
  1,733-line window WITHOUT the CRL functions — hunt against a fresh
  `llvm-objdump -dr` of the `.ko` (12,923 lines). (2) Only `pkicms_compare_crl`
  has `__ksymtab`; `pkicms_hw_crl_compare` is `T` unexported — export-only greps
  miss it. (3) The installer "device_crl_check" runs on the **host CPU**, not the
  card Linux — keep installer verdicts separate from boot verdicts in T14.
  (4) Task-brief HSM anchor 0x1c73b ≠ subkey_revoke_check (that is 0x1bd7c).
- **files edited/created**: `evidence/secureboot-hunt-20260911/revocation.md`
  (new), `evidence/secureboot-hunt-20260911/revocation.json` (new)

### T9: BL31 vs HSM verification division of labor
- **depends_on**: [T7, T2]
- **location**: `evidence/secureboot-hunt-20260911/bl31-recon.md`,
  `device-aarch64/images/hisserika.bin`, new
  `evidence/secureboot-hunt-20260911/bl31-hsm-split.md`
- **description**: Using offsets from T7 and HSM diagnostics strings, propose and
  cross-check the split: which stage authenticates kernel/rootfs/dt containers at
  boot (BL31 `hw_cms_image_verify_simple` region vs HSM `image_sign_verify`), and
  which authenticates TEE/UEFI/HBOOT handoffs. Where BL31 cannot answer alone
  (flat binary, no symbols), mark Unknown and route the question into HSM via T11
  rather than guessing.
- **validation**: Division-of-labor table per image type with Observed/Interpreted
  labels; residual Unknowns listed as the exact card-time questions.
- **status**: Completed
- **log**: 2026-09-12. Window reads only, no new disassembly. Key result: the T7
  4/5-diagnostic delta generalizes to **BL31-132100 embedding an entire CMS/X.509
  verify engine that BL31-172100 lacks** — 132100 has the AOID DER/OID table
  (0x20a1b–0x20dd0), BOTH named root certs (legacy `Huawei Root CA` @0x1fbe9,
  G2 @0x201ad, same names as T4's module roots), pbroot/CRL/cms-engine error
  family with Begin/Update/Final shape (0x2377a–0x23a48), secure-header gate
  (headMagic/pss slen 0x2144e–0x214c6), nvcnt-vs-eFuse compare+write
  (0x216f9–0x21ad6, answers T5's routed question: BL31 both compares and writes
  nvcnt at boot), ini hash-list gate (`cur image hash not in list!` 0x21430), and
  132100-only `scmi task` messaging; 172100 keeps ONLY TEE-verify (0x1701e),
  TSPD and UFS-nvcnt cleanup (`cms`=1 hit, `crl/pbroot/AOID/headMagic/efuse/usr
  cert/scmi`=0). Verdicts: TEE → BL31 (both variants); kernel/dt/rootfs → BL31
  on 132100 (Interpreted; membership by numeric image_type, no name literals),
  Unknown on 172100 (no verdict-owner strings — HSM or none); ddr/lowpwr/HSM →
  Unknown with the nameless `hw_cms_ctrl_cpu_verify` (0x23d05) as the only
  candidate path; UEFI-fd/BL33/BL2/BL31 → loader-stage (BootROM/HBOOT1),
  Unknown (BL2 zero verify strings + TOC flags=0 → **current model: BL31 is
  authenticated, if at all, by the stage that loaded it**). HSM boundary settled
  per T11 reuse: HSM verifies BL31-supplied hash/sig/cert material, parses no
  containers; BL31-132100 parses CMS ITSELF (own engine) while drv_pkicms parses
  at runtime/update — different stages, both parse. page-0x21000 adrp cluster
  (asm lines 5024–5274) stores 8-byte-quantized, string-interior pointers →
  log-macro tail pointers, same style as T11; per-string xrefs stay Interpreted.
  Unknowns U-A…U-G routed to T14 (variant selector, per-image_type coverage,
  ctrl-cpu identity, fd/BL33 authenticity, BL31 crypto authority local-vs-SCMI,
  BL31 root trust/fallback). GOTCHAS: (1) string-window reads beat per-string
  xref matching here — the variant question is answerable by regex-presence
  absence alone; (2) BL31 has NO image-name literals (kernel/rootfs/cpio/dt
  absent in both carves) — the table is data-supplied (`hw_adapt_desc_fetch
  copy image name`), so per-image mapping is only settleable from boot logs;
  (3) log tail-pointers are interior/8-quantized — an add-immediate hit on a
  string offset does not prove which message; (4) 132100 has dual roots — T4's
  runtime selection table must NOT be assumed for boot stage.
- **files edited/created**: `evidence/secureboot-hunt-20260911/bl31-hsm-split.md`

### T10: HSM ISA identification gate
- **depends_on**: []
- **location**: `device-aarch64/images/hisserika.bin` (129,888 B), new
  `evidence/secureboot-hunt-20260911/hsm-isa.md`
- **description**: Identify the HSM firmware instruction set (candidates: ARMv7-M,
  ARC, RISC-V, or other vendor MCU core — `E2E-ANALYSIS.md` explicitly does not
  assume aarch64). Method: string-table density/alignment, prologue statistics
  under each candidate disassembler, relocation/section hints. Deliver a go/no-go
  on meaningful HSM disassembly and the winning toolchain.
- **validation**: One ISA conclusion with statistical/structural evidence, or an
  explicit "ISA unresolved, full HSM RE deferred to card time" decision.
- **status**: Completed
- **log**: 2026-09-11 — Verdict: **ARM Thumb (ARMv7-M/R class), HIGH confidence;
  GO for disassembly**, toolchain = capstone THUMB-LE sweep + `llvm-mc
  -triple=thumbv7m-unknown-eabi`. Evidence: thumb sweep 6.5 restarts/1k insns vs
  53.9–12532 for x86-16/riscv/m68k/a32/mips/aarch64; 94% BL targets in-range;
  push 454/pop 429/bx-lr 164 with 421/454 prologues 4-aligned; llvm-mc bake-off
  at 3 function heads (0x12a, 0x8f9c, 0x10bf8) — coherent Thumb-1 prologues
  (`push {r4-r7,lr}`, `add r7,sp,#12`, `ldrd`, `dsb`) where riscv32+c yields
  garbage. Strings clustered at 0x16000–0x1D000 (43% of printable runs).
  T11 unblocked.
- **gotchas**: (1) ARC was NOT machine-checked locally — ARC target absent from
  llvm-mc 21.1.8 build, capstone 5.0.7 lacks ARC; exclusion rests on Thumb
  decode coherence (Interpreted, low risk). (2) llvm-mc `-disassemble` needs one
  `0xNN` byte per line, and `-triple=arc` errors out of the box. (3) HSM load
  base Unknown: no code dword points into the string region at file offsets and
  common shifted bases miss; on-card settle = hexdump/log of the copy performed
  by `hsm_firmware_update.ko` (also settles M3/M4/M7 sub-profile via boot banner
  if logged).
- **files edited/created**: `evidence/secureboot-hunt-20260911/hsm-isa.md` (new),
  `evidence/secureboot-hunt-20260911/hsm-isa-stats.txt` (new, script output),
  `evidence/secureboot-hunt-20260911/hsm-isa-spotdis.txt` (new, llvm-mc bake-off)

### T11: HSM verification-path reconnaissance
- **depends_on**: [T10]
- **location**: `device-aarch64/images/hisserika.bin`,
  `evidence/hsm-security-strings.json`, new
  `evidence/secureboot-hunt-20260911/hsm-verify-path.md`
- **description**: If T10 passed, disassemble around the six recorded diagnostic
  strings (`service_soc_verify`, `service_uefi_sign_verify`, `image_sign_verify`,
  `root_pubkey_verify`, `subkey_revoke_check`, `nv_cnt_verify`) and map entry
  boundaries enough to say what `service_soc_verify` receives. If T10 failed,
  document exactly which questions remain open and what on-card observation
  would settle them.
- **validation**: Recon notes with offsets for ≥3 of the 6 diagnostics (when ISA
  resolved), or the explicit deferral document.
- **status**: Completed
- **log**: 2026-09-11, static_check (all sweep/xref commands + repro snippets in
  `hsm-verify-path.md`). Coverage **6/6 characterized, 3/6 with hard-or-paired
  xrefs, bar met**: (1) `service_soc_verify` name dword `0x10a271ed` @pool
  **0xd6d8**, host function entry **0xd638**/epilogue 0xd6be (Observed xref,
  Interpreted boundary); (2) `nv_cnt_verify` paired log "service verify cnt
  failed" tail-ptr pool @0xf044 → host **0xefc8**; (3) `subkey_revoke_check`
  paired log "subkey_id has been revoked…rim in efuse" pool @0x113c0 → host
  region 0x10d00–0x113b8 (sweep-desync zone, boundary inferred). (4)
  `root_pubkey_verify` region 0x1c0a0–0x1c800 (rotpk/efuse format family; no
  name xref); (5) `image_sign_verify` region 0x1a1e8–0x1a2e2 (img-id/hash-engine
  cluster); (6) `service_uefi_sign_verify` region 0x19a90–0x19b40 — 4/5/6 are
  Unknown-xref/Interpreted-region (whole-file negative: no abs nor raw dword,
  2-byte granularity, and no movw/movt hits). **Load base working assumption
  B=0x10a10000** (Interpreted, strong: 611 code-region dwords point into string
  region, 261 land exactly on string starts). **Dispatch verdict: TABLE** —
  26-entry jump table @rodata **0x163e8** (bare B+code targets 0x122e8–0x126d0
  into one big dispatch function ~0x122xx–0x12c20) + 3-entry secondary table
  @0x16570; table-base ptr @pool 0x131a8; computed branch `ldr.w r1,[r6,r0,lsl
  #2]` @0x124cc; per-service `tbb`/`tbh` elsewhere. Not {id,handler} pairs,
  not string-compare. `service_soc_verify` host = gate-chain (0x9690/0x98e4/
  0x9a54 checks) + logger(level, module, tick, file, fmt, line) @0xebb8; certs
  passed by host-memory address (cf. "invalid Certificates addr."/"memcpy in
  fail"). **Role for T9/T14**: HSM performs the verify *decision* locally
  (efuse rotpk, nv-cnt, revoke-rim, head-magic, img-id, PKE/HASH engines) over
  host-addressed data, but ships **zero CMS/PKCS#7 parser strings** → container
  parsing stays host-side PKICMS; **strengthens the T1 HSM-channel hypothesis**
  for the 10 unprovided `Cmscbb*` imports (crypto authority = HSM, CMS glue =
  kernel), pending on-card mailbox-id correlation.
- **gotchas**: (1) Log macros use **interior tail-pointers** into shared format
  blobs (pool hits offset +0x18/+0x19 from string starts) — exact-offset xref
  tests must accept `[S, S+len)` windows. (2) Five of six diagnostic *name*
  strings are never referenced by absolute pointer anywhere in the image
  (whole-file 2-step scan) — they cannot anchor xrefs; pair them with their
  adjacent format strings instead. (3) Inline literal pools desync a linear
  sweep (0x10ce8–0x113b8 shows "no epilogues"); epilogue-backscan head
  inference is unreliable there — label region-only. (4) 0x163e8 table entries
  are Thumb targets **without** the +1 bit; base dword points 8 B earlier
  (@0x131a8→0x163e0) — index bias Unknown. (5) Everything rests on B =
  0x10a10000; settle with the T10 `hsm_firmware_update.ko` copy hexdump.
- **files edited/created**: `evidence/secureboot-hunt-20260911/hsm-verify-path.md`,
  `evidence/secureboot-hunt-20260911/hsm-verify-pools.txt` (611+47 pointer
  census), `evidence/secureboot-hunt-20260911/hsm-strings-sorted.txt`

### T12: Runtime integrity gate inventory
- **depends_on**: []
- **location**: `device-aarch64/rootfs/etc/`,
  `device-aarch64/rootfs/boot/config-6.6.0-...aarch64`, new
  `evidence/secureboot-hunt-20260911/runtime-gates.json`
- **description**: Enumerate the runtime gates a replaced rootfs must satisfy and
  their static configuration state: SELinux (permissive static + `rcS` final
  `enforce 1`), IMA policy script invocation, EVM, lockdown parameter sources in
  the boot wrapper (kernel Image append/dt image cmdline), dm-verity presence or
  absence for the root device, module signing enforcement state, and the
  setuid/sudo helper surface from `ROOTFS-AUDIT.md`. Identify which gate's
  effective state depends on a runtime argument we do not yet control.
- **validation**: JSON table: gate → compiled-in → configured → effective-source
  → controllable-by-us (yes/no/unknown), each with a file citation. Note: no
  `/proc/cmdline` source exists in the image, so "unknown" in the
  effective-source column for boot-arg-driven gates (lockdown,
  `module.sig_enforce`, dm-verity root= options) is a completed finding, not a
  task failure.
- **status**: Completed
- **log**: 2026-09-11. Key finding: the effective cmdline IS recoverable offline
  — `chosen/bootargs` of all 16 FDT blobs in `device-aarch64/images/dt.img`
  (kernel uses DTB bootargs: `CONFIG_CMDLINE_FROM_BOOTLOADER=y`, no FORCE,
  config lines 639-641). All 16 bootargs end with `module.sig_enforce=1`;
  none contains `root=`, `lockdown=`, or `integrity=`. Gate results: SELinux
  permissive static (`etc/selinux/config:7`) + late `echo 1 > enforce` at
  `var/device_sys_init.sh:1091` (reached via rcS:24 sourcing; `sec-config` has
  no enforce write); IMA doubly dormant — `etc/rc.d/init.d/ima-init` has zero
  callers in rootfs and its `integrity=1` self-gate is absent from bootargs,
  and `/etc/ima/digest_lists` (config:7540) doesn't exist; EVM compiled
  (config:7550), unconfigured, effective Unknown; lockdown compiled with
  default NONE (config:7501) and no `lockdown=` arg → effective none;
  dm-verity `=m` but absent from every rootfs layer (0 verity-name hits incl.
  `modules.dep` and the 86 `/var/*.ko`), root is the RAM initramfs
  (no `root=`; `initrd=`+`rdinit=/sbin/init`; rcS:7 early rw remount) → not a
  verity root; module signing is the hard gate — `MODULE_SIG_FORCE` off
  (config:985) but bootarg `module.sig_enforce=1` forces rejection and 89/89
  staged `.ko` carry the appended signature magic (trust anchor of the vendor
  key = Unknown, `SYSTEM_TRUSTED_KEYS=""`); setuid surface = 9 files from
  manifest archive modes + NOPASSWD `/var/*.sh` rules appended at runtime
  (`device_sys_init.sh:665-684`). Gotchas: dracut dir `boot/dracut/` is empty
  (dracut only referenced by kdump configs — not a cmdline source); no
  `load_policy` invocation found anywhere (what loads `tiny/policy/policy.33`
  stays Unknown, card-time question); bootargs evidence lives in `dt.img`, not
  the skipped `rootfs/boot/dtb-6.6.0-*/` tree.
- **files edited/created**: `evidence/secureboot-hunt-20260911/runtime-gates.json`,
  `evidence/secureboot-hunt-20260911/runtime-gates.md`

### T13: Process-manager hash enforcement trace
- **depends_on**: [T12]
- **location**: `device-aarch64/rootfs/etc/mdc/base-plat/process-manager/`
  (`bin_hash.cfg`, `startup_procmgr.yaml`), process-manager binary under the
  rootfs, new `evidence/secureboot-hunt-20260911/procmgr-hash.md`
- **description**: Determine enforcement semantics for the 15-hash executable
  list: is the list parsed at each launch or cached at startup; what is the
  failure action (reject launch, restart loop, log-only) via the
  `IntegrityProtect`/`sha256sum` diagnostic call paths in the binary; where is
  `bin_hash.cfg` read from (rootfs path vs flash-restored copy); and are
  replacement binaries required to be hash-listed at all. This is the named
  blocker for swapping bundled executables in a replaced rootfs.
- **validation**: Written enforcement model with cited offsets/paths and a
  one-line answer: "replace binary X, must also edit Y, else failure mode Z".
- **status**: Completed
- **log**:
  - Enforcement host is `usr/bin/mdc/base-plat/process-manager/process-manager`
    (stripped PIE AArch64); `proc_launcher` carries no `IntegrityProtect`
    strings. Check fn 0x41d20: map lookup (miss ⇒ pass, no hashing) →
    `popen("sha256sum "+path+"|cut -d' ' -f1","r")` (0x41e6c, ≤6 retries,
    1 ms nanosleep) → memcmp+length-eq vs the cfg value; wrapper X 0x42060 →
    single caller; start gate Y 0x2f4e8 (callers 0x32f90/0x337a8, the two paths
    reaching the only two `fork@plt` at 0x33000/0x33818) → on fail
    `syslog(LOG_ERR, "PROCMGR : IntegrityProtect app=%s or sha256sum has been
    modified")` and launch rejected (fork skipped) — with yaml `on-forever`
    retry this is a start-reject loop, not a kill of a running process.
  - (a) Runtime path is the hardcoded `/etc/mdc/base-plat/process-manager/
    bin_hash.cfg` (rodata 0x6ae88, default installed by `.init_array[5]`=0xd808,
    loader 0x42210 does `access(R_OK)`); zero shell-script references anywhere
    (`grep -rl` hits 4 ELFs) — no flash-restore copy. (b) cfg parsed **once at
    boot** into an in-memory map behind a once-guard atomic (GOT 0x91f58; sole
    load call 0x16000→0x42210 in the BootStrace sequence); post-boot cfg edits
    affect nothing until procmgr restarts; check itself runs per launch.
    (c) reject launch + syslog ERR (log-only for procmgr itself). (d) yes —
    map keyed by basename; miss passes without hashing (unlisted/renamed binary
    trips nothing); **cfg missing ⇒ empty map ⇒ entire gate fail-open**
    (island 0x42690 logs "is not exist", continues). (e) Interpreted citing T12:
    late `enforce 1` (DSI:1091) post-relabel gives replaced files identical
    default contexts and IMA stays dormant, so labels don't alter the model;
    procmgr-start-vs-enforce ordering and `popen`/`/usr/sbin/sha256sum`
    (busybox link, busybox.links:233; absent from this extraction) allowance
    are Unknown/card-time.
  - One-liner: replace binary X whose basename is one of the 15 → must also
    edit `etc/mdc/base-plat/process-manager/bin_hash.cfg` (`name=newSHA256`,
    before boot since the map is cached at procmgr start) → else procmgr logs
    `PROCMGR : IntegrityProtect … has been modified` and never forks/execs the
    app (daemon down across all on-forever restart retries). Missing cfg or an
    unlisted basename ⇒ fail-open.
  - Gotchas: stripped binary — symbol labels are misleading; reliable anchors
    are rodata-cluster + lo12 immediates (modified=0x1d8, cfgpath=0xe88,
    sha-prefix=0xce0/0xcf0); map key is basename-after-last-`/`
    (rename-to-unlisted escapes the check); libtsd* .so's embed the same
    `bin_hash.cfg` string (independent consumers not traced — Unknown).
    `bin`/`/bin/sh` absent from the extraction; PATH-resolved
    `sha256sum`+`/bin/sh` need one card check.
  - Bonus from the resumed session: `tsfw-verify-entry-scan.json`
    (drv_platform.ko verification entries) stands as T2 input, not T13 output.
- **files edited/created**: `evidence/secureboot-hunt-20260911/procmgr-hash.md`,
  `evidence/secureboot-hunt-20260911/procmgr-hash.json`

### T14: Ranked weakness list (synthesis)
- **depends_on**: [T3, T4, T5, T6, T8, T9, T11, T13, T16]
- **location**: new `SECUREBOOT-WEAKNESSES.md` (this directory),
  `evidence/secureboot-hunt-20260911/`
- **description**: Consolidate into a ranked list of security-boot weaknesses,
  each with: description, evidence level (Observed/Interpreted/Unknown), what a
  rootfs replacement must still prove, and card-time follow-up. Candidate seeds
  to confirm or eliminate — dual-algorithm fallback + dual roots (T4), eFuse
  flag semantics (T4), signed-byte coverage of the rootfs container (T3/T2),
  image-ID-scoped policy and the ID-15 exclusion (T2), NVCNT rollback (T5),
  CRL fail-open leads (T8), permissive-SELinux/late-enforce and non-mandatory
  lockdown/module signing (T12), process-manager hash list as a mutable runtime
  policy (T13), missing BootROM/HBOOT1 anchor (T6/T9 — recorded at Unknown level
  with its card-time question even when T6 only yields availability evidence).
  Additionally list as explicitly unowned Unknowns routed to card time: the TEE
  (`itrustee.img`) internal verification and the `smbus_secure_boot_query`
  security-state interface — neither is analyzed by any task in this plan. Add
  the HBOOT2 rebuild/sign path (T16) as an additional seed: if vendor-rebuildable
  and re-signable, HBOOT2 ceases to be an immovable boot component and ranks as a
  weakness against the "no validated loading path for our replacements" premise
  in `E2E-ANALYSIS.md` section 9. Eliminated candidates are kept in the list
  marked eliminated, not dropped.
- **validation**: Final ranked document, every entry citing at least one evidence
  file, no entry at Unknown level without an accompanying card-time question.
- **status**: Completed
- **log**: 2026-09-12, consolidation only (no new RE; reads + cheap offset-greps
  against the evidence JSONs). Ranking rationale: boot-deciding items ranked above
  OPERATE-only items because BOOT gates OPERATE for a rootfs replacement — the
  BL31-variant fork (R1) leads because it decides whether the rootfs has a
  boot-stage verdict at all and one boot-log capture collapses the single largest
  unknown cluster (T9 U-B/U-C, T5 residual); the boot-signature cluster follows in
  decision order — acceptable-policy space (R2, the dual-root fallback marquee:
  cross-root acceptance for every ID except 15-with-eFuse, BSS default legacy root,
  custom-cert door) before re-signing mechanics (R3, outer PSS binds the CMS blob
  not the container; device digest = header-declared range, kprobe @0x8944); R5
  (Linux zero boot-path verify) stays high as the pivot of the hunt's framing
  question despite being neutral-valued; then the OPERATE gates in bite order —
  the one non-rootfs-removable gate module.sig_enforce (R4), the mutable-but-cached
  procmgr hash file (R6), then the opportunity-leaning unknowns revocation (R7) and
  NVCNT (R9), the runtime-provider Unknown (R8), and model-level caps: the unseen
  chain anchor (R10) and portal-gated availability incl. the vendor-reported but
  never-observed HBOOT2 re-sign (R11); TEE-internal verification and
  smbus_secure_boot_query ride as R12 unowned Unknowns per the task instruction,
  and dm-verity is recorded eliminated-with-evidence (R13). Levels inherited
  strictly from the source files — no entry upgraded; multi-level entries state
  each part. **Gotchas**: (1) R2's decision table is runtime-module scope — T9
  gotcha #4 forbids assuming it for the BL31 boot stage (both roots sit in
  BL31-132100 too, but that engine's policy is U-G); (2) R7 carries the brief
  anchor correction: 0x1c73b is `root_pubkey_verify`, `subkey_revoke_check` is
  0x1bd7c — kept visible per the tension-carrying rule; (3) installer verdicts are
  host-CPU (T8 gotcha) and were kept out of boot verdicts; (4) R3's non-binding
  claim covers exactly the 12/12 plain candidates — the header-declared-range
  hypothesis itself is still kprobe-pending, stated as Unknown; (5) the 172100
  "no container verify" absence is regex-presence negative — per plan rule, no
  verdict upgraded above Interpreted/Unknown from an absence.
- **files edited/created**: `SECUREBOOT-WEAKNESSES.md` (new, artifact root)

### T16: HBOOT2 SDK build/sign pipeline investigation
- **depends_on**: []
- **location**: Huawei Ascend 310P SDK (SoC) source package
  `Ascend-hdk-310p-sdk-soc_<version>.zip` containing
  `Ascend310P-source.tar.gz`; new
  `evidence/secureboot-hunt-20260911/hboot2-sdk-source.md`
- **description**: Vendor-reported procedure (user-supplied, 2026-09-11): the 310P
  SDK source package supports   `bash build.sh hboot2`, whose success echo contains
  both `generate ... AS610_HBOOT2_UEFI.fd success!` and
  `sign ... AS610_HBOOT2_UEFI.fd success!` (dependency: apt uuid-dev).
  Prerequisite environment per the same procedure: x86 Ubuntu 20.04 server with
  `python3 make gcc unzip pigz bison flex libncurses-dev squashfs-tools bc
  device-tree-compiler libssl-dev cmake` (installed as root). The
  squashfs-tools/dtc/bison/flex mix indicates the SDK source tree likely carries
  UEFI/EDK2 and rootfs-packing machinery, not just a lone FIP packer — to be
  confirmed from package contents. Investigate availability of
  `Ascend-hdk-310p-sdk-soc_26.0.RC1` (or nearest) for download: (1) if fetchable,
  locate inside the source tree the FIP packaging logic (entry UUIDs, load
  addresses, BL2/BL31/BL33 selection — answers the E2E layout question at no
  physical-card cost), the signing tool/container format definition (wrapper
  header, signed-byte coverage, which key material `sign` consumes — vendor build
  keys vs customer open-mode keys), and any HBOOT1/BootROM references or images;
  (2) record whether `hboot2` is one target of a broader build menu (other
  device-side firmware potentially rebuildable, e.g. hsm/tee/boot components).
  Evidence level rule: everything observed in the package is Observed (vendor
  source, not our build); reproduction of the build itself is card/server-time
  and out of scope here.
- **validation**: Written availability verdict (URL/version/existence, download or
  not, package SHA-256 if obtained) and, if obtained, cited source-file paths for
  the FIP layout definition and the signing step. If unavailable, the exact
  blocking condition (login-required Ascend downloads portal, version mismatch)
  recorded for card-time follow-up.
- **status**: Completed
- **log**: Availability verdict: NOT OBTAINED. The community OBS bucket serves the known 26.0.rc1 driver (200) and hosts the `Ascend-hdk-<chip>-sdk-soc_<ver>.zip` scheme for 310B 23.0.0 (200, ~550 MB calibration hit), but 15+ name/version spellings of the 310P `sdk-soc` zip all returned 403 there (incl. `26.0.RC1/26.0.rc1/26.0.0/26.0.t1/b060` forms, `24.1.RC1/25.2.0/25.5.1` dirs); anonymous ListBucket is 403-AccessDenied so the bucket cannot be enumerated. `support.huawei.com` (HDK software list + Ai-P EDOC pages, which confirm the package class and the /opt upload workflow per search snippets) 403-gates non-browser fetches → login-gated. No download ⇒ no SHA-256; T16.2/T16.3 deferred. Kept seed items for T14: BL2/BL33 identical across the two carved FIPs but BL31 differs (166,402 vs 110,390 B) under the same UUID — A/B model breaks here and the source tree answer is now a logged-in-portal/card-time question (with the full T16.2 checklist: build.sh target menu, FIP packaging def, sign tool + key material, HBOOT1 refs, UEFI source-vs-prebuilt). Gotcha: on this bucket a nonexistent key answers 403, not 404, so misses are "no publicly-readable object at this exact key", and the Ai-P docs inner tarball is named `Ascend310Prc-source.tar.gz` (variant SKU) — only the real zip settles the naming.
- **files edited/created**: `evidence/secureboot-hunt-20260911/hboot2-sdk-source.md`

### T15: Subagent plan-of-record review
- **depends_on**: [T14]
- **location**: `SECUREBOOT-WEAKNESSES.md`
- **description**: Independent reviewer checks the weakness list against the
  evidence files only (not against prior conclusions), specifically hunting for
  Over-claiming: any entry whose evidence level exceeds what its citations
  support, and any Observed/Interpreted conflation. Revise before yielding.
- **validation**: Review notes recorded in the document's revision section;
  at most one revision cycle expected before yield.
- **status**: Completed
- **log**: Verdict: pass-with-corrections — 3 corrections applied in the doc body, 0 evidence-level downgrades needed. (1) rank 9 "sole caller" → "sole captured Linux caller" (matches `nvcnt-rollback.md`:66 qualifier); (2) rank 6 once-guard re-attributed from loader 0x42210 to wrapper 0x15fa8/once-byte @GOT 0x91f58 (`procmgr-hash.md`:34-35); (3) rank 7 CRL zero-census narrowed to "external caller/importer" (pre_compare has one in-module caller, `revocation.md`:19-20). All 13 entries + all quoted offsets traced to evidence at or below stated level; eliminated entry visible; checklist (8 rows) maps to ranked Unknowns; trust model keeps dashed edges dashed.
- **gotchas**: (a) unqualified "sole caller"-style phrasings silently drop captured-scope qualifiers from the evidence; (b) rank 7's sentence was internally near-contradictory once the sole-caller fact was stated two sentences earlier — re-read for self-consistency, not just vs evidence; (c) the doc contains one deliberate conservative under-claim (`Image.image_size` zero: doc Interpreted vs evidence Observed, `inner-verifier-findings.md`:168) — don't "fix" it upward. Review-side: `symbol-providers.json` lives in the older `evidence/pkicms-inner-20260911/` hunt dir, not this evidence dir — cross-hunt citation should stay qualified.
- **files edited/created**: `SECUREBOOT-WEAKNESSES.md` (## Review section appended; 3 body corrections)

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2, T6, T7, T10, T12 | Immediately |
| 2 | T3, T4 (after T1); T9 (after T7); T11 (after T10); T13 (after T12) | Wave 1 partial |
| 2/3 | T5, T8 (after T1+T2) | T1, T2 complete |
| 4 | T14 | T3, T4, T5, T6, T8, T9, T11, T13, T16 complete |
| 5 | T15 | T14 complete |

Critical paths: (a) T1 → T3/T4/T5/T8 — PKICMS chain; (b) T7 → T9 and T10 → T11 —
boot firmware; (c) T12 → T13 — runtime gates. T6 is informational and never
gates another task.

## Testing Strategy

- Every task ships machine-readable evidence (`evidence/secureboot-hunt-20260911/`)
  rehashable offline; a small `verify.py` (pattern of
  `evidence/e2e-analysis-20260911/collect.py`) re-checks hashes, offsets, and
  symbol claims on demand.
- Disassembly claims are cross-checked against `llvm-objdump` output only; BN GUI
  decompilations are cited but never used as sole evidence
  (`semantic_validation_passed: false`).
- Nothing requires the cards; T6 and the Unknown-marked questions are explicitly
  deferred to card time with prepared command lists from `E2E-ANALYSIS.md` §11.

## Risks & Mitigations

- **Flat-binary ISA misidentification (BL31, HSM)** wastes entire tasks:
  time-box T7/T10 and accept an explicit "defer to card time" outcome via T9/T11.
- **Linux-side vs boot-side confusion**: PKICMS modules may verify at
  update/load time only, while the boot authentication of the rootfs container
  happens in BL31/HSM before Linux exists — T2's caller matrix must record
  *which context* each caller runs in, or the final ranking will mislead.
- **Image-ID semantics divergence** across PKICMS, HSM services, and BL31 string
  contexts: T14 must cross-check against `drv_pkicms.h` named IDs and the loader
  table before treating ID-15 (and any ID) as established across interfaces.
- **Runtime gates are static-config observations**: effective SELinux/IMA/lockdown
  on the real cards may differ; T12 labels each gate's effective source as
  unknown where boot args control it, and T14 lists it as card-time follow-up
  rather than a weakness.
- **Cmscbb host unresolved**: verified in this planning pass that none of the
  three `depends` modules defines or imports the eight CMS-level `CmscbbVerify*`
  /`CmscbbDecodeCrl` symbols (`drv_pkicms.ko` defines only `CmscbbCrypto*`,
  `CmscbbMd*` and string/memory helpers locally). The provider therefore likely
  sits behind the HSM command channel or outside the extracted module tree; T1
  must widen the symbol scan to all 90 rootfs modules and, if still unresolved,
  record "HSM-channel hypothesis" as a labeled finding routed to T11 — not
  silently assume in-kernel crypto.
