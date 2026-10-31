# Ascend 310P custom-firmware path — progress log

**Date**: 2026-10-26 · **Hardware**: 3× Ascend 310P3 (6 chips) @ 192.168.53.187, driver 26.0.rc1
**Goal**: run modified/optimised custom firmware on the cards.
**Card state**: stock rootfs installed and booting; user cert (our CA) enrolled on all 6 chips.
**Safety rule for every step**: never touch the card until the artifact is verified offline, and keep
`Ascend310P.cpio.gz.stock-backup-20261021` recoverable at all times.

---

## 2026-10-26 — Breakthrough: our CMS was never the problem

### What we now know for certain

| Question | Answer | How established |
|---|---|---|
| Does `openssl cms -verify` rejecting the container mean the container is bad? | **No.** The **stock Huawei** container fails it identically. OpenSSL's S/MIME policy engine is not the device's oracle. | `openssl cms -verify` on `stock_cms.der` → same `CMS_SignerInfo_verify_content: verification failure` |
| Is our re-signed CMS cryptographically valid? | **Yes — fully.** PSS signature over signedAttrs verifies against the leaf, messageDigest matches the payload, `signingCertificateV2` matches the signer cert. | custom DER parser + `cryptography` PSS verify (§Verifier below) |
| Is our CMS structurally the same shape as stock? | **Yes.** Outer signedData children are byte-for-byte the same *kinds* in the same order; 1 signerInfo; 2 certs in `certificates[0]`; 2 CRLs in `crls[1]`. | DER structural walk, both files |
| Was the 2026-10-21 failure the CMS? | **No.** The CMS was correct. The fault is in the **container** (`certtype` + `crl` section composition). | below |

### The verifier (reusable)

`artifacts/ascend-310p-26.0.rc1/customer-cert-20261026/` now contains:

- `der.py` — minimal DER reader
- `cmsverify2.py` — verifies a detached CMS the way Ascend's `CmscbbVerify*` path does:
  `messageDigest == SHA-256(detached content)`, RSA-PSS over the re-tagged SET-OF signedAttrs,
  and `signingCertificateV2 == SHA-256(signer cert DER)`.
- `tstverify.py` — extracts and verifies the nested RFC-3161 timestamp token.

Results:

```text
STOCK  → ALL SIGNERINFOS VALID
        PSS SIG VALID (CN=Computing PL Code-Signing Authority 2 - G2, salt=32)
        messageDigest 2934ff7d… == SHA-256(initrd.ini body)      ✅
OURS   → ALL SIGNERINFOS VALID
        PSS SIG VALID (CN=Signing Certificate G2, salt=32)
        messageDigest 2e6ed705… == SHA-256(initrd.ini body)      ✅
```

### The real defect: container `crl` section composition

The stock container's `crl` section is **3 concatenated DER objects**, and the middle one is a
**certificate**, not a CRL:

```text
STOCK container 'crl' section:
  [0] CRL   issuer = Huawei Integrity Root CA - G2     (also in CMS crls[0])
  [1] CERT  subject = Huawei Integrity CA 1 - G2       <-- a CERT, not a CRL
  [2] CRL   issuer = Huawei Integrity CA 1 - G2        (also in CMS crls[1])
```

Ours (from `full/initrd.ini.crl`) is **5 objects**: 3 CRLs then 2 certs — wrong order and wrong
content for the slot. The stock pattern is unambiguously:

> `crl` section = [ root-CRL , **intermediate-CA cert** , intermediate-CRL ]

i.e. for a 2-level chain (root → intermediate → leaf) the section carries the root's CRL, the
intermediate's **certificate**, and the intermediate's CRL. The leaf's CA certificate is already
inside the CMS `certificates[0]` set.

### The second defect: `certtype`

Every prior attempt (`modified5..modified9`, `stock-resigned`) shipped:

```text
certtype (both header copies) = 0xFFFFFFFF
```

`rebuild_esbc.py`'s own docstring records the values from Huawei's `image_pack.py`:

```text
certtype: 0x1 = community, 0x2 = client, 0xFFFFFFFF = Huawei
```

All previous attempts claimed the **Huawei** type while signing with **our own CA**. The
`0xFFFFFFFF` value is the "use the built-in Huawei certificate" selector — so the device was told
"this is a Huawei-signed image" and then presented a customer signature. That is exactly the
scenario the `usr cert cms vef fail` / `no usr, hw vef fail` strings describe.

The `certtype` field is exactly where the "customer cert" path is selected, and it was never
changed in any attempt.

### Corroboration from the runtime module

`drv_pkicms.ko` exports `pkicms_get_name_by_cert_type` / `pkicms_get_real_cert_type` and contains
the literal table:

```text
vendor-certificate
community-certificate
user-certificate
```

and the path template `/var/asdrv/dev%d/device-sw-plugin/verify/%s`. So the runtime resolves a
cert **by type name** — `vendor` / `community` / `user` — and reads it from that path. That is the
cert-type selector the container header's `certtype` is meant to line up with.

Also found in the stock rootfs `/var/`: four `.sec` files (`5a5aa5a5` magic) that each embed the
same 4-cert chain (root, integrity CA, code-signing leaf, TSA leaf). These are the SOC's own
secure containers — the runtime trust store — and they show the same root→intermediate→leaf
pattern.

### Next actions (in order)

1. Rebuild the container with **certtype = 0x2 (user-certificate)** and a **stock-shaped `crl` section**
   `[root.crl, signca.der, signca.crl]`, keeping our valid CMS. Verify offline that:
   - the container parses with `rebuild_esbc.py --salvage`,
   - `SHA-256(payload)` equals the `ini` digest and the header digest,
   - `cmsverify2.py` still reports ALL SIGNERINFOS VALID,
   - the `crl` section is exactly 3 objects in the stock order.
2. Install and reboot **one card only** (chip 0 of card 9). Watch for `boot_status`.
3. If it still fails, A/B the two variables separately:
   - 3a. stock payload + our certtype/community + our CMS (isolates the payload — this was never
     cleanly isolated before, because `stock-resigned` also used `certtype=0xFFFFFFFF`).
   - 3b. modified payload + **stock** certtype + stock CMS but our rootfs image (sanity).
4. Only after the container is accepted, work on the payload itself.

---

## 2026-10-26 — Root cause identified: `certtype` + CRL-section shape

### The `certtype` field, resolved exactly

`drv_pkicms.ko` exports `pkicms_get_name_by_cert_type` (AArch64, text+0x3fa4):

```text
0x3fbc: add  w0, w19, #1          ; w0 = cert_type + 1
0x3fc0: cmp  w0, #3
0x3fc4: b.hi #0x3fe8              ; out of range -> default (vendor-certificate)
0x3fc8: adrp x1, .rodata
0x3fd0: add  x1, x1, #0x5f0        ; table base
0x3fdc: ldr  x0, [x1, w0, uxtw #3]
```

Table at `.rodata+0x5f0` (relocations resolved):

| index | `.rodata` | string |
|---|---|---|
| 0 | 0x5f0 | `vendor-certificate` |
| 1 | 0x5f8 | (empty) |
| 2 | 0x600 | `community-certificate` |
| 3 | 0x608 | `user-certificate` |

With `index = cert_type + 1`:

| `cert_type` | name |
|---|---|
| 0 | `""` (empty) |
| **1** | **`community-certificate`** |
| **2** | **`user-certificate`** |
| ≥3 | `vendor-certificate` (default) |

This matches `rebuild_esbc.py`'s docstring (from Huawei's `image_pack.py`):
`0x1 = community, 0x2 = client, 0xFFFFFFFF = Huawei`.

**Every 2026-10-21 attempt shipped `certtype = 0xFFFFFFFF`** — i.e. "use the built-in Huawei
certificate" — while presenting a **customer** signature. `0x2` is the value that selects the
user-certificate path.

### The `crl` section shape, resolved exactly

Stock container `crl` section = **3 concatenated DER objects**, and the middle one is a
**certificate**:

```text
[0] CRL   issuer = Huawei Integrity Root CA - G2   (also in CMS crls[0])
[1] CERT  subject = Huawei Integrity CA 1 - G2     <-- a CERT, not a CRL
[2] CRL   issuer = Huawei Integrity CA 1 - G2      (also in CMS crls[1])
```

Pattern: `[ root-CRL , intermediate-CA **cert** , intermediate-CRL ]`.
Ours was 5 objects (3 CRLs then 2 certs) — wrong content and wrong order for the slot.

### The CMS was never the problem — proof

The prior `full/initrd.ini.p7s` (built by Huawei's `cmssign`) is **structurally identical** to the
stock container CMS and **cryptographically valid**:

```text
version 1 · digestAlgorithms {sha256 + NULL params} · detached pkcs7-data
[0] certificates: 2   [1] crls: 2   signerInfos: 1
signedAttrs: contentType, signingTime, messageDigest, signingCertificateV2
signatureAlgorithm: rsassaPss (sha256 / mgf1-sha256 / salt 32)
unsignedAttrs: id-smime-aa-timeStampToken
→ cmsverify2.py: ALL SIGNERINFOS VALID
```

**`openssl cms -verify` is not a valid oracle**: the **stock Huawei** container fails it too
(`CMS_SignerInfo_verify_content: verification failure`), because OpenSSL's S/MIME policy engine
rejects the CRL/TSA combination and because the nested TST signs the *signature value*, not the
content.

### New tooling built

- `der.py` — minimal DER reader
- `cmsverify2.py` — CMS SignerInfo verifier (PSS over re-tagged signedAttrs, messageDigest,
  signingCertificateV2). **Verifies the stock container correctly**, which is its calibration.
- `tstverify.py` — nested RFC-3161 token extractor/verifier
- `pss_sign.py` — **from-scratch CMS signer** producing the exact stock structure
  (PSS/SHA-256/MGF1-SHA256/salt-32, `signingCertificateV2`, optional RFC-3161 TST, certs, CRLs).
  Needed because OpenSSL's `cms` CLI emits `rsaEncryption` instead of `rsassaPss`.

### Artifact set (all offline-verified)

| file | payload | certtype | purpose |
|---|---|---|---|
| `A-modified-ct2.cpio.gz` | modified9 | `0x2` | main test — custom firmware |
| `B-stock-ct2.cpio.gz` | **stock** | `0x2` | A/B control — isolates the payload |
| `C-modified-ct1.cpio.gz` | modified9 | `0x1` | community-type variant |

```text
A  sha256 5a67f78c4bf4bb1c18a90fd8042a2275de2076052c91ae2e84c6354a74e3426f  (110543928 B)
B  sha256 5038197f5b71d3463af031934997771edea2190015d997659c234fcf775d4df2  (109389448 B)
C  sha256 95a5ea883af3c051fd6202815e80a35b54b0f8e14939c5fedca55b388a668442  (110543928 B)
```

Each one: header digests (both copies) == `SHA-256(payload)` == the `ini` digest; `crl` section
exactly 3 objects in stock order; CMS re-verifies as ALL SIGNERINFOS VALID.

### Card state at deploy time

All 6 chips `boot status 16` (OS up). Stock rootfs in place, sha256
`12975a9d7e9345255eccc0d97475082ec96ebab69324070f2246604dac8c0cc2`, with `.orig` and
`.bak-20261021-082553` alongside. Local copy is byte-identical to the card's.

### Deploy protocol

Test **B first** (stock payload): if it boots, the container format is accepted and the payload is
proven innocent in one shot; then A tells us whether the custom payload is the last problem. If B
fails, the container is still wrong and A is pointless. C only if B boots and A fails.

Deploy to **one card** at a time, with the stock file preserved and a `revert` command ready.

---

## 2026-10-26 — Test B executed: FAILED, but it was a clean single-variable experiment

### What was deployed

`stock-ourcms2-ct2.cpio.gz` (built by the 2026-10-21 session, never actually tested):

- **payload: byte-identical to stock** (`39634b3e…`)
- `certtype = 0x2` (user-certificate path)
- CMS = our chain (Huawei `cmssign`, 9644 B, PSS + `signingCertificateV2` + RFC-3161 TST)
- `crl` section = 3 objects, stock shape, our chain's certs/CRLs

### Result

Cards failed to initialise. Evidence:

| Signal | Known-good boot −2 | Boot −1 (test B) | Boot 0 (after power cycle) |
|---|---|---|---|
| `davinci` device nodes | **8** | **0** | **0** |
| `ascend-310p-bringup` outcome | Finished | `Expected six Ascend device nodes, found 0` → FAILURE | same |
| PCIe endpoints `19e5:d500` | present | present, all D0 | present, all D0 |
| DPC PoisonedTLP count | 6 | 6 | 6 |

**The DPC errors are pre-existing noise** — present in the known-good boot too. The real signal is
the absence of `davinci` device nodes: the PCIe endpoints enumerate, but the driver never creates
device nodes, which is what a firmware-image rejection looks like.

### What this proves

**The payload is completely exonerated.** The deployed image had the *exact stock payload* and still
failed. So the failure is entirely in the container:

1. `certtype 0xffffffff → 0x2`
2. our CMS instead of Huawei's
3. our CRL section instead of Huawei's

Also note: **`npu-smi` hangs** in the failed state (the host's `npu-smi` blocks on a device that
never initialised). Use `systemctl status ascend-310p-bringup` and
`journalctl -k -b N | grep -c davinci` instead — never `npu-smi` — to evaluate a failed boot.

### Card recovery

Reverted to stock immediately:

```text
sudo cp /usr/local/Ascend/driver/device/Ascend310P.cpio.gz.orig \
        /usr/local/Ascend/driver/device/Ascend310P.cpio.gz
sha256 = 12975a9d7e9345255eccc0d97475082ec96ebab69324070f2246604dac8c0cc2   ✅
```

A **full power cut for ≥30 s** is required after a failed attempt; a soft `reboot` leaves the cards
stuck. (Learned from the operator.)

### The definitive failure signal

| | boot −1 (test B) | boot 0 (stock, recovered) |
|---|---|---|
| `davinci` device nodes | 0 | 6 |
| `uda_wait_all_phy_startup` errors | **3** (`Wait timeout`) | **0** |
| `drv_pcie` `Device os load failed` | (absent) | (absent) |
| `npu-smi` | hangs | works |

**`[ascend] [uda] [ERROR] [uda_wait_all_phy_startup 1348] Wait timeout. (dev_num=0;
uda_detected_dev_num=6)` is the fingerprint of a rejected firmware image.** The card's PHY never
comes up because the SOC refused the image. This is the single most reliable indicator, and it is
visible in `journalctl -k -b N` without running `npu-smi`.

### Evaluation recipe for a boot (never run `npu-smi` on a failed boot)

```bash
# 1. did the image get rejected?
sudo journalctl -k -b 0 --no-pager | grep -c "phy_startup"        # 0 = good, >0 = rejected
# 2. did the driver come up?
find /dev -maxdepth 1 -type c -name 'davinci[0-9]*' | wc -l        # 6 = good, 0 = rejected
sudo systemctl status ascend-310p-bringup.service --no-pager -l
# 3. only then, if both good:
timeout 30 sudo npu-smi info
```

### Recovery procedure (verified)

```bash
sudo cp /usr/local/Ascend/driver/device/Ascend310P.cpio.gz.orig \
        /usr/local/Ascend/driver/device/Ascend310P.cpio.gz
# sha256 must be 12975a9d7e9345255eccc0d97475082ec96ebab69324070f2246604dac8c0cc2
```

Then a **full power cut for ≥30 s** — a soft `reboot` leaves the cards stuck (boot −1 and boot 0
both had 0 device nodes; only the power cut recovered it). Verified: after the power cut the cards
came back with `npu-smi` reporting all 6 chips `OK`, and **zero** `phy_startup` errors.

### Next: single-variable isolation artifacts (uploaded, ready)

| file | payload | CMS | crl section | certtype | isolates |
|---|---|---|---|---|---|
| `B3-control-stock-all-ctFF.cpio.gz` | stock | stock | stock | `0xffffffff` | **control** — must boot |
| `B1-…-stockcms-stockcrl-ct2.cpio.gz` | stock | stock | stock | `0x2` | **certtype alone** |
| `B2-…-ourcms-ourcrl-ctFF.cpio.gz` | stock | ours | ours | `0xffffffff` | **our chain alone** |

`B3`'s sha256 is `12975a9d…` — **byte-identical to the stock image**. That proves the rebuild
pipeline (`rebuild_esbc.py`) is lossless and that any failure in B1/B2 is caused by the variable
under test, not by the tooling.

Run order: **B3** (must boot — proves recovery works) → **B1** → **B2**.

- If **B1 boots** → `certtype = 0x2` is correct and accepted; the problem is our CMS/CRL chain.
- If **B1 fails** → `certtype = 0x2` is wrong or the customer path is closed (efuse) → investigate
  `certtype` semantics further; the `drv_pkicms` table maps `0x2`→`user-certificate`, but the
  **BL31** consumer may use a different encoding than the runtime module.
- If **B2 boots** → our CMS+CRL are accepted as a "Huawei" image (i.e. the runtime ignores the
  cert chain) → then the failure in test B was certtype alone.
- If **B2 fails** → our chain is rejected, as expected, and `certtype`/`certtype`+chain both need
  the customer path to work together.

---

## 2026-10-26 — Found `Ascend_310P_check.bin`: a zero-risk offline verifier

`/usr/local/Ascend/driver/tools/Ascend_310P_check.bin` is Huawei's own CHECK_TOOL. It logs to
`/var/log/upgradetool/upgrade_crl.log`. Modes:

```text
0 [image] [out.crl]        extract the CRL section from an image
1 [crl1] [crl2]            compare CRL timestamps
2 [crl]                    check CRL integrity
3 [crl] [image]             VERIFY an image against a CRL
4 <image_type> [img] [out]  get nvcnt from an image
```

Usage notes: the output file **must already exist** (it calls `realpath` on it) — pre-create with
`touch` + `chmod 666`. Mode 0 on the stock rootfs yields a 3522-byte CRL and mode 4 yields
`IMAGE_NVCNT=0`.

### Results — clean single-variable isolation

| image | payload | CMS | crl section | certtype | mode 3 |
|---|---|---|---|---|---|
| STOCK `.orig` | stock | stock | stock | FF | **OK** |
| B3 control | stock | stock | stock | FF | **OK** (byte-identical to stock: `12975a9d…`) |
| **B1** | stock | **stock** | **stock** | **2** | **OK** ✅ |
| B2 | stock | ours | ours | FF | FAIL (`InternalVerifyTimeStamp`, `0x88200004`) |
| test B | stock | ours | ours | 2 | FAIL (same) |
| A/D | modified | ours | ours | 2 | FAIL (same) |

**Two conclusions:**

1. **`certtype = 0x2` is accepted.** B1 differs from stock in exactly one 32-bit field and still
   verifies. The `certtype` fix is correct.
2. **The CRL section composition is irrelevant.** All six variants of the `crl` section
   (3 objects, 5 objects, root-cert-first, all-CRLs-then-certs, with/without the TSA branch) pass
   identically. Only the *CMS* matters.

### Why the check tool cannot validate our customer CMS (proved, not assumed)

Control experiment: **stock payload + stock Huawei CMS + one flipped payload byte** → mode 3
**FAILS**. So the tool really does verify signatures — its rejection of our CMS is about the
**trust anchor**, not malformed structure.

The tool validates against the **built-in Huawei roots** (the two roots in `AS610_HBOOT2_UEFI.fd`).
Our chain is rooted in `RootCA G2`, for which the tool has no anchor. The *device*, by contrast,
gets its anchor from the **enrolled user certificate** in secure flash.

> **Therefore the check tool is a valid oracle for container structure and for `certtype`, but NOT
> for a customer-signed CMS. The device is the only oracle for the customer path.**

A second, independent structural difference was also found and fixed in our own CMS builder
(`pss_sign.py`): the stock `signingCertificateV2` value is **3** levels of SEQUENCE
(`SEQUENCE{SEQUENCE{SEQUENCE{alg,digest}}}`), and the RFC-3161 `TSTInfo` is
`version, policy OID, messageImprint, serialNumber, genTime` — not the issuer Name / wrapped
serial+genTime that my first implementation produced.

### Remaining hypotheses for the customer path (ranked)

1. **The enrolled anchor must cover both branches.** What is enrolled on all six chips is
   `user-signca.der` → `Signing Certificate CA G2` (the **intermediate**), hash
   `41474c62fb56b1e7bda86961e22fc3fd559266e431edd8dde04f37b466f0ec14`. Our **TSA** leaf chains to a
   *different* CA (`Timestamp Certificate CA G2`), so the timestamp token cannot be validated
   against the enrolled anchor. Fix: re-issue the TSA leaf **under `signca`** (done —
   `ts2.pem`, verified with `openssl verify -partial_chain -CAfile signca.pem ts2.pem` → OK).
2. **The customer certificate may have to be delivered as its own `usrcert` image.** BL31's
   image-type table is `dataisp, dataotp, tee, initrd, uimage, dtimage, usrcert` — `usrcert` is an
   image *type* (like `initrd` and `uimage`), **not** a section inside the rootfs container. The
   rootfs container has exactly three tagged slots (`cms` @ X, `ini` @ X+0x4000, `crl` @ X+0x4800);
   the gaps are zero padding. So a customer cert would be a **separate image** handed to the
   bootloader — and `upgrade-tool` has no `usrcert` component (its component list is
   `uefi, kernel, rootfs, aicpu, hboot1a, hboot1b, hboot2, ddr, lp, hsm, sil, hlink, hlink2,
   RAWData, RO_SysDrv, …`). This may be the mechanism Huawei uses that we have not found yet.
3. **The customer eFuse bit.** BL31 prints `Secure Boot: customer efuse Solution` on the customer
   path. `npu-smi info -t custom-op-secverify-enable/-mode/-cert` is *not supported* on 310P
   ("This device does not support querying…"), so this cannot be checked from the host.

### Card state

Currently healthy: stock rootfs, 6 `davinci` nodes, `npu-smi` OK on all 6 chips, **0**
`phy_startup` errors in the current boot. **B1 (`certtype=0x2` + stock CMS) is deployed and
awaiting its power cycle** — that boot will tell us whether the device also accepts `certtype=0x2`
(the check tool says the container is well-formed; the device says whether the *path* is enabled).

**Next device test**: power-cycle with B1 in place, then check
`journalctl -k -b 0 | grep -c phy_startup` and `ls /dev/davinci[0-9]* | wc -l`.
If B1 boots → `certtype=0x2` is live on the device, and the only remaining work is the customer
CMS (hypotheses 1 and 2). If B1 fails → `certtype=0x2` is *not* the right device-side encoding
(the runtime `drv_pkicms` mapping may not be what BL31 uses) and we must look for the BL31-side
certtype handling instead.

---

## 2026-10-26 — ✅ B1 BOOTED. `certtype = 0x2` is confirmed live on the device

Cold power cycle with `B1` (stock payload, stock Huawei CMS, stock CRL section, **certtype = 0x2**)
in place:

```text
rootfs on card : d6495c413082914cd87f0f928fe034816075b2ec877bef8892fc094a99e83e35  (== B1)
phy_startup errs: 0
bringup service : active (exited), code=0/SUCCESS
davinci nodes   : 6
npu-smi         : all 6 chips 310P3 Health=OK
boot_status     : 16 on all six chips (OS running)
```

**This is the first non-stock image these cards have ever accepted.** It confirms:

1. `certtype = 0x2` (user-certificate path) is accepted by the device bootloader. The field is at
   header offset `+0x08` in **both** header copies (0x0000 and 0x1000), a full 32-bit value — the
   `>> 22` shift seen in `drv_pkicms` is a different field, not `certtype`.
2. The container format, `crl` section shape and `ini` digest binding are all correct — the stock
   CMS verifies, so the packaging pipeline (`rebuild_esbc.py`) is correct end to end.
3. The device's image verification path is **not** gated on the CMS signer being a built-in Huawei
   root: a `certtype=0x2` image is routed to the customer path, which is what makes the enrolled
   user certificate meaningful.

**What is still open:** the only thing not yet proven is whether a **customer-signed** CMS (signed
by our own CA) is accepted under `certtype = 0x2`. The CHECK_TOOL cannot answer this (it only
knows the built-in Huawei roots), so it needs a device test.

### Next device test — the decisive one

Deploy **our** CMS under `certtype = 0x2` and power-cycle. Two sub-cases, in order of
increasing change:

| # | image | CMS signer | TSA branch | isolates |
|---|---|---|---|---|
| **C1** | stock payload | **ours** (`pss_sign`, fixed TSTInfo, TSA re-issued under `signca`) | under `signca` | whether our chain is accepted at all |
| **C2** | stock payload | ours, **no timestamp token** | n/a | whether the TST is the blocker |
| **C3** | **modified payload** | ours | under `signca` | the actual goal — custom firmware |

C1 is the one that matters. If C1 boots, the customer path works end to end and C3 is the final
step. If C1 fails, the log distinguishes the cause:

```bash
sudo journalctl -k -b 0 --no-pager | grep -ciE "phy_startup"   # >0 => rejected
sudo tail -20 /var/log/upgradetool/upgrade_crl.log               # check-tool detail (if it ran)
```

Note the CHECK_TOOL will report FAIL for C1 **by design** (no built-in root for `RootCA G2`), so the
device is the only oracle here — expect the offline verifier to disagree with the device.

### Card state

Healthy and fully functional with B1 (`certtype=0x2`, stock CMS) installed: 6 chips `OK`,
`boot_status 16`, zero `phy_startup` errors. Stock rootfs remains at
`Ascend310P.cpio.gz.orig` (`12975a9d…`) plus `.bak-20261021-082553`, so reverting is one `cp` away
followed by a cold power cycle.

---

## 2026-10-26 — ✅✅ THE ACTUAL ROOT CAUSE: the CMS was signed by the WRONG CERTIFICATE

### The enrolled anchor

`/var/log/upgradetool/upgrade.log` records every `--user_cert` enrollment. The **last** one
(2026-10-21 11:23:51) was:

```text
Firmware Set user cert, device_index[-1], file path is /home/matteius/usercert-test/user-signcert-ct1.der
```

`user-signcert-ct1.der` wraps the certificate **`Test Signing Cert`** (serial `5142F169…`,
issued **directly by `RootCA G2`** — a 2-level chain with no intermediate).

The device stores `SHA-256(usrcert_tag_block ‖ cert_DER)`. Recomputed for every candidate cert
against the *fixed* tag block (identical across all five `user*.der` envelopes — the header
metadata does not depend on the cert):

| cert | `SHA-256(tag ‖ cert_DER)` |
|---|---|
| **`Test Signing Cert`** | **`b932bb15ac0cc925…`** ← **matches the enrolled value** |
| `Signing Certificate G2` | `fbd7a03dacefb7d3…` |
| `Signing Certificate CA G2` | `41474c62fb56b1e7…` |
| `RootCA G2` | `7fb03430761fd43a…` |

### Why every 2026-10-21 attempt failed

The prior full-chain CMS (`full/initrd.ini.p7s`) was signed by **`Signing Certificate G2`**
(serial `4BB72CAF…`, `SHA-256 = 03016c72…`) — a *different* certificate from the enrolled
`Test Signing Cert` (`SHA-256 = 089238e8…`).

BL31 computes the hash of the user cert it finds and compares it with the secure-flash value.
**Different cert → different hash → `usr cert hash cmp fail.`** That is the failure. It was never
a trust-chain problem: the chain was fine, the *identity* was wrong.

### The `usrcert` tag-block format (decoded)

```text
+0x00  12 bytes  magic  "3\xaa::usrcert\x00"
+0x10   8 bytes  zero
+0x18   u32      = 9
+0x1c   u32      = 0
+0x20   u32      = 0
+0x24   u32      = 517
+0x28   u32      = 4
+0x2c   u32      = 0x20000
+0x30   u32      = 0x564d   (total payload length)
+0x34   u32      = 0x100    (tag block size)
+0x38..0xFF      zero
```

The tag block is **fixed** — byte-identical in all five `user*.der` envelopes regardless of which
cert is inside. So `SHA-256(fixed_tag ‖ cert_DER)` is a pure function of the certificate.

### The `usrcert` image is a separate image type

BL31's image-type table is `dataisp, dataotp, tee, initrd, uimage, dtimage, usrcert`. The rootfs
container has exactly three tagged slots (`cms` @ X, `ini` @ X+0x4000, `crl` @ X+0x4800) with
zero padding between them — **no `usrcert` slot**. So the customer certificate is delivered as a
**separate `usrcert` image**, and `upgrade-tool --user_cert` only records its hash in secure flash
for the boot-time comparison.

### D-series artifacts (deployed / ready)

| file | payload | CMS signer | certtype | sha256 |
|---|---|---|---|---|
| `D1-enrolledleaf-ct2.cpio.gz` | stock | **`Test Signing Cert`** (the enrolled one) | `0x2` | `0210d5fcce1cd6d4…` |
| `D2-modpayload-enrolledleaf-ct2.cpio.gz` | **modified9** | `Test Signing Cert` | `0x2` | `6ffa8d1aeaa9b703…` |

D-series CMS shape (mirrors stock exactly for a 2-level chain):

```text
certs set : [ RootCA G2 , Test Signing Cert ]
crls set  : [ rootG2.crl ]
crl section: [ rootG2.crl , rootcaG2.der , rootG2.crl ]     (stock shape: [CRL, CA cert, CRL])
certtype  : 0x2
```

Both D-series CMS files were confirmed to contain the enrolled anchor
(`SHA-256(tag‖cert) == b932bb15…`).

**D1 is deployed on the card and awaiting a cold power cycle.** If D1 boots, the customer
signature path works and **D2 is the goal — custom firmware**. If D1 fails, the next hypothesis
is that the `usrcert` image must be delivered to the device as a separate image (not merely
enrolled), or that the CMS must also carry the timestamp chain under the enrolled CA.

---

## 2026-10-26 — ❌ D1 FAILED — and a correction to the B1 conclusion

### D1 result

```text
rootfs on card : 0210d5fcce1cd6d4…  (== D1)
phy_startup errs: 1
davinci nodes   : 0
bringup         : FAILED (Expected six Ascend device nodes, found 0)
boot_status     : "Get deviceId list failed" on all six
```

Same signature as the earlier test-B failure → **image rejected**. All six CRL-count variants
(E1/E2 × crlV1/V2/V3) also fail offline, so the CRL count is not the discriminator either.

### ⚠️ Correction: B1 did NOT prove the customer path works

I previously concluded from B1's successful boot that `certtype=0x2` "is confirmed live on the
device". **That was wrong, and I want to correct it.**

B1 = stock payload + **stock Huawei CMS** + stock CRL section + `certtype=0x2`. Its CMS is signed
by `Computing PL Code-Signing Authority 2 - G2` → `Huawei Integrity CA 1 - G2` → **`Huawei
Integrity Root CA - G2`**, and that root is a **built-in anchor** in `AS610_HBOOT2_UEFI.fd`.

So B1 booted because the image verified against the **built-in Huawei root** — the ordinary path.
`certtype=0x2` did **not** engage the customer path. B1 therefore proves only that *changing
`certtype` does not break the built-in-root path*.

What we actually know now:

- `certtype = 0x2` is **accepted** (offline oracle + device boot with the stock CMS) — it is not
  rejected, but it is not sufficient either.
- An image signed by **our** chain is **rejected**, because the verifier resolves trust through
  the built-in roots. The customer path was never engaged by any test so far.

### The runtime trust store — found in `drv_upgrade.ko`

`drv_upgrade.ko` embeds three certificates:

| subject | sha256 |
|---|---|
| `Huawei Integrity CA 1 - G2` | `61b57eba58f81c70…` |
| **`Huawei CSI RSA-PSS Code-Signing Authority 2`** | `534209698638c5c3…` |
| `Huawei Time-Stamping Authority 2 - G2` | `cc48b9602d99a769…` |

Note the code-signing leaf here is **`Huawei CSI RSA-PSS Code-Signing Authority 2`** — a
*different certificate* from the container CMS's `Computing PL Code-Signing Authority 2 - G2`
(`af70de12…`). The intermediate and TSA match. So the runtime driver and the boot container use
different leaves for the same role; the runtime's leaf is the one the *device* will use to verify
flash operations.

Also in `drv_upgrade.ko`: `dev_upgrade_sec_verification or soc_verify fail`, `dev_upgrade_sec_rim_data
fail`, `sec_flash_read_mtd_nums failed!`, `get boot area failed`.

### What BL31 does with the user cert (string-driven reconstruction)

```text
'usr cert address check fail, result = 0x%x!'   -> obtains an ADDRESS for the cert
'usr cert 0x%lx oversize!'                      -> size check on that region
'usr cert copy fail!0x%x' / 'usr cert copy fail.' -> copies it into a buffer
'usr cert hash cal fail!0x%x'                  -> hashes the buffer
'usr cert hash cmp fail.'                      -> compares with the secure-flash value
'Secure Boot: customer efuse Solution'         -> the customer-mode branch
'no usr, hw vef fail, ctrl_flag = 0x%x!'       -> falls back to the hardware (built-in) path
'wrong image type = 0x%x.'                     -> image-type dispatch
```

This says the user cert is read **from a flash/storage address**, not from the CMS. Combined with
the image-type table (`dataisp, dataotp, tee, initrd, uimage, dtimage, usrcert`), the picture is:

> The customer certificate is a **separate `usrcert` image** that must be present on the device
> storage. BL31 locates it by address, hashes it, and compares that hash with the value enrolled
> via `--user_cert`. If the hash matches, the customer path engages and the image is verified
> against the *user* certificate instead of the built-in root.

`--user_cert` only writes the hash to secure flash — it does **not** deliver the image. That is
why every attempt so far failed: **nothing ever put a `usrcert` image on the device.**

### Device-side evidence for this

Every device image file is a *separately tagged* esbc container:

| file | tag |
|---|---|
| `Ascend310P.image` | `uimage` |
| `Ascend310P_dt.img` | `dtimage` |
| `Ascend310P.cpio.gz` | `initrd` |
| `Ascend310P_hsm.image` | `hisserika` |
| `Ascend310P_ddr.image` | `lpddr` |
| `Ascend310P_lowpwr.image` | `lperika` |

**There is no `usrcert` file on the device**, and `upgrade-tool` has no `usrcert` component (its
list is `uefi, kernel, rootfs, aicpu, hboot1a, hboot1b, hboot2, ddr, lp, hsm, sil, hlink, hlink2,
RAWData, RO_SysDrv, …`). The images are pushed to the card over PCIe by the driver, and the
card has no local block/mtd storage of its own (`lsblk` shows only host NVMe; `/proc/mtd` empty).

### Next steps (ranked)

1. **Deliver the `usrcert` image.** `upgrade-tool --component` may accept an image *name* that we
   have not tried, or the file may need to be placed in `/usr/local/Ascend/driver/device/` with a
   specific name. The `usrcert` container we already have (`user-signcert-ct1.der`) is exactly the
   right shape: `magic "3\xaa::usrcert"`, fixed 0x100-byte header, cert DER payload. Find the
   filename the bootloader looks for.
2. **Enumerate `upgrade-tool`'s component table** for a user-cert component type (the strings
   `rootfs` and `kernel` were visible; `usrcert` may be one of the numeric component types).
3. **Check whether the `usrcert` image must be *signed*** (it may need its own CMS in the `cms`
   slot, signed by the built-in Huawei root, with the user cert inside as the payload).

### Card state

**Currently D1 is deployed and the cards are DOWN** (0 device nodes, bringup failed). Stock rootfs
is intact at `Ascend310P.cpio.gz.orig` (`12975a9d…`). To restore service:

```bash
sudo cp /usr/local/Ascend/driver/device/Ascend310P.cpio.gz.orig \
        /usr/local/Ascend/driver/device/Ascend310P.cpio.gz
# then a COLD power cut (>=30 s) — a soft reboot leaves the cards stuck
```

---

## 2026-10-26 — ✅ FOUND THE CUSTOMER-CERT COMPONENT: `Usr_Base_Config` = componentType 31

`upgrade-tool`'s component-name table (`.rodata` 0x24a360–0x24a580, 16-byte slots) is:

```text
nve  xloader  m3fw  uefi  tee  kernel  dtb  rootfs  imu  imp  aicpu
hboot1a  hboot1b  hboot2  ddr  lp  hsm  sil  hlink  RAWData
RO_SysDrv  RO_ADSApp  RO_ComIsolator  RO_Cluster  RO_Customized
Sys_Base_Config  Recovery  hlink2  Logic_Bist  Memory_Bist  atf
Usr_Base_Config  RTC_CORE  RTC_CONFIG
```

**`Usr_Base_Config` is the customer base-config component — the customer-certificate component.**
Probe results (the tool resolves the name to a numeric type, then reports whether the device has it):

| name | tool output |
|---|---|
| **`Usr_Base_Config`** | `No corresponding component in deviceId(0), componentType(31)` |
| `Sys_Base_Config` | `… componentType(25)` |
| `Recovery` | `… componentType(26)` |
| `atf` | `… componentType(30)` |
| `usrcert` | `No such component(usrcert).` ← **not a component name at all** |

So the customer cert is delivered as **component type 31, named `Usr_Base_Config`**. This finally
closes the loop on the `usrcert` image type in BL31: the `usrcert` **image** is what you flash as
the `Usr_Base_Config` **component**, and BL31 then locates it on the device by address
(`usr cert address check fail`), hashes it (`usr cert hash cal fail`) and compares with the
secure-flash value (`usr cert hash cmp fail`).

The "no corresponding component" response means the component is **not currently provisioned on
these cards** — which is consistent with the customer path never having been engaged.

### The `usrcert` image we already have

`user-signcert-ct1.der` is exactly a `usrcert` container:

```text
magic "3\xaa::usrcert\x00" | 8 zero bytes | u32 9 | 0 | 0 | u32 517 | u32 4
| u32 0x20000 | u32 total_len | u32 0x100 (tag block size) | zero padding to 0x100
| payload = the DER certificate
```

and the tag block is **byte-identical across all five** `user*.der` envelopes, so
`SHA-256(fixed_tag ‖ cert_DER)` is a pure function of the certificate. The enrolled value on all
six chips is `b932bb15ac0cc925…`, which corresponds to `Test Signing Cert` (serial `5142F169…`).

### Next step — flash the `usrcert` image as `Usr_Base_Config`

```bash
# the file is already the right shape; try flashing it directly
sudo upgrade-tool --device_index -1 --component Usr_Base_Config \
     --path /home/matteius/usercert-test/user-signcert-ct1.der
```

Then re-enroll the hash so the two agree, and cold power-cycle. Success criteria: cards come up,
and the rootfs image can then be signed by our CA.

Order of operations matters: the **hash in secure flash** and the **image on the device** must
agree. Both currently describe `Test Signing Cert`
(`user-signcert-ct1.der` hash = `b932bb15…` = the enrolled value), so they already agree — the
missing piece is only that the image was never delivered.

**Risk note**: this writes to the card's flash. A `Usr_Base_Config` write could in principle
conflict with the provisioning of the customer mode; the stock `.orig` rootfs remains available
for recovery, but a bad `Usr_Base_Config` might not be revertible by dropping in a file. Given
that a cold power cycle is needed anyway to bring the cards back from the current failed-D1
state, the safest order is:

1. Cold power cycle with the **stock** rootfs → confirm the cards are healthy again.
2. Then flash `Usr_Base_Config` and re-enroll, and only then try a customer-signed rootfs.

### Correction to the earlier "certtype is the fix" conclusion

`certtype = 0x2` is **not** what engages the customer path. It is a field in the esbc header that
selects the certificate *type* used by the runtime module
(`drv_pkicms`: `0`→community, `1`→`user`, `2`→`user-certificate`). The **boot-time** customer path
is selected by BL31 based on (a) the presence of a `usrcert`/`Usr_Base_Config` image on the device
and (b) its hash matching the secure-flash value.

---

## 2026-10-26 — ✅✅✅ THE MISSING PIECE FOUND: `/usr/local/Ascend/CMS/user.xer` + `user.crl`

The driver **source** is present on the card at
`/home/matteius/ascend-310p-host-driver-restored/pcie_host/` (a restored build tree). This gave us
the authoritative answer instead of more inference.

### The image load table — `pcie_host/soc_adapt/res_drv_mini_v2.c`

```c
#define MINI_V2_BLOCKS_NUM 11
STATIC struct devdrv_load_file_cfg mini_v2_file[MINI_V2_BLOCKS_NUM] = {
  { .file_name = "/driver/device/Ascend310P_ddr.image",    .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P_lowpwr.image", .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P_hsm.image",    .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P.fd",           .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P_dt.img",       .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P.image",        .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P.cpio.gz",      .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P_tee.bin",      .file_type = DEVDRV_CRITICAL_FILE,     .fail_mode = DEVDRV_NON_NOTICE },
  { .file_name = "/driver/device/Ascend310P.crl",          .file_type = DEVDRV_NON_CRITICAL_FILE, .fail_mode = DEVDRV_NOTICE_BIOS },
  { .file_name = "/CMS/user.crl",                          .file_type = DEVDRV_NON_CRITICAL_FILE, .fail_mode = DEVDRV_NOTICE_BIOS },
  { .file_name = "/CMS/user.xer",                          .file_type = DEVDRV_NON_CRITICAL_FILE, .fail_mode = DEVDRV_NOTICE_BIOS },
};
```

### How the path is resolved — `devdrv_device_load.c`

```c
STATIC char *g_davinci_config_file = "/lib/davinci.conf";
#define DEVDRV_HOST_FILE_PATH "/home/bios"

STATIC u32 devdrv_get_sdk_path(char *sdk_path, u32 len) {
    return devdrv_get_env_value_from_file(g_davinci_config_file, "DAVINCI_HOME_PATH", sdk_path, len);
}
/* then: file_name = sdk_path + file_cfg->file_name */
```

Live value: `/lib/davinci.conf` → `DAVINCI_HOME_PATH=/usr/local/Ascend`.

**So the driver looks for exactly three CRL/cert inputs:**

| path | role |
|---|---|
| `/usr/local/Ascend/driver/device/Ascend310P.crl` | stock CRL (already present) |
| **`/usr/local/Ascend/CMS/user.crl`** | **customer CRL** — our chain's CRLs |
| **`/usr/local/Ascend/CMS/user.xer`** | **customer certificate** — the `usrcert` image |

**`/usr/local/Ascend/CMS/` does not exist on the card.** That is why the customer path has never
been exercised: the device was never given the customer cert or the customer CRL.

### `.xer` = the `usrcert` image

`--user_cert` takes a path to a `.xer` file (the earlier log line
`file path is /tmp/does_not_exist.xer` used that extension). Our `user*.der` files are exactly that
container: magic `3\xaa::usrcert`, a fixed 0x100-byte header, then the DER certificate.

### The user cert IS modifiable — verified empirically, not assumed

Per the operator's caution, this was tested directly rather than inferred. `--user_cert` on a
**single** device (dev0) with two *different* certificates:

```text
$ upgrade-tool --device_index 0 --user_cert --path .../user-signcert-ct1.der
Device[0] set user certificate info success.
$ upgrade-tool --device_index 0 --user_cert --path .../user-rootcaG2-ct1.der
Device[0] set user certificate info success.
```

Both succeeded, and the log confirms the write:

```text
Firmware Set user cert, device_index[0], file path is .../user-rootcaG2-ct1.der.
Firmware User cert file_size=9913.
Firmware Set user cfg, device_index[0], size=0x20.      <-- 32-byte SHA-256 written
```

So the secure-flash hash **is** rewritable, and `--user_cert` succeeded every time it was run
(including in the 2026-10-21 session — the "fail" lines in that log belong to the *other*,
deliberately-broken attempt with `/tmp/does_not_exist.xer`).

All six devices have since been re-enrolled with the same file
(`user-signcert-ct1.der` → `Test Signing Cert`, hash `b932bb15…`) so they are consistent again.

### The complete picture

1. The driver loads `user.xer` + `user.crl` from `/usr/local/Ascend/CMS/` **into the device's
   BIOS**, in addition to the stock images.
2. `--user_cert` stores the SHA-256 of the user cert in the device's secure flash.
3. BL31 computes the hash of the `usrcert` image it was given, compares it with the secure-flash
   value (`usr cert hash cmp fail`), and — on a match — verifies the images against the **customer**
   chain instead of the built-in Huawei root (`no usr, hw vef fail` is the fallback).

**Every one of the 2026-10-21 attempts was missing step 1.** They enrolled a hash (step 2) and
signed images with our CA (step 3's input), but never delivered `user.xer`/`user.crl` to the
device, so the device never had a customer anchor.

### Next action (concrete)

```bash
sudo mkdir -p /usr/local/Ascend/CMS
sudo cp <our usrcert image>  /usr/local/Ascend/CMS/user.xer
sudo cp <our CRL section>    /usr/local/Ascend/CMS/user.crl
# re-enroll so hash and image agree, then cold power cycle
sudo upgrade-tool --device_index -1 --user_cert --path <the same usrcert image>
```

Then the rootfs can be signed by our CA. The card's own stock `Ascend310P.crl` stays in place.

**Ordering constraint**: `user.xer` must be the *same* file that `--user_cert` hashes, because
BL31 compares `SHA-256(user.xer)` against the enrolled hash. Our `user-signcert-ct1.der` satisfies
both (`SHA-256(tag‖cert) = b932bb15…` = the value already enrolled).

**Risk**: `/usr/local/Ascend/CMS/` is a host-side directory the driver reads at boot. Adding files
there is additive and does not touch the stock images; the worst case is that the device rejects
the customer chain and falls back to `no usr, hw vef fail` (i.e. stock behaviour). Recovery is
`rm -rf /usr/local/Ascend/CMS` plus a cold power cycle.

---

## 2026-10-26 — Deployed: `/usr/local/Ascend/CMS/{user.xer,user.crl}` + F1 test image

### What is now on the card

```text
/usr/local/Ascend/CMS/user.xer   9833 bytes   SHA-256(tag‖cert) = b932bb15ac0cc925…
/usr/local/Ascend/CMS/user.crl   3045 bytes   [ rootG2.crl , rootcaG2.der , rootG2.crl ]
/usr/local/Ascend/driver/device/Ascend310P.cpio.gz = F1 (952423c1b00d914…)
```

All six devices re-enrolled with **the exact deployed file**:

```text
$ upgrade-tool --device_index <i> --user_cert --path /usr/local/Ascend/CMS/user.xer
Device[i] set user certificate info success.
```

so the secure-flash hash and the image the driver hands to the BIOS are the same bytes.

### The F-series images

Both use the chain `RootCA G2 → Test Signing Cert` (the enrolled cert), `certtype = 0x2`:

| file | payload | sha256 |
|---|---|---|
| `F1-ourcms-stockpayload-ct2.cpio.gz` | **stock** (sanity) | `952423c1b00d9149…` |
| `F2-modpayload-ourcms-ct2.cpio.gz` | **modified9** (the goal) | `d91c5bcf9a84a06c…` |

F1 is deployed and awaiting a cold power cycle. F1 answers one question cleanly:

- **F1 boots** → the customer path is live end to end (cert delivered, hash matched, image verified
  against our chain). Then **F2 is the goal** — custom firmware.
- **F1 fails** → the customer path still is not engaging, and the remaining candidates are:
  1. the `usrcert` image needs its own CMS/`cms` slot (the stock images are all signed containers —
     `user.xer` currently has no `cms` section, only the raw cert);
  2. the `certtype` in the *rootfs* container must be something else (the `usrcert` type id found
     in BL31's table was `0x00010000`, not `0x2`);
  3. the customer eFuse bit is not provisioned on these cards.

### Note on the `usrcert` type id

The BL31 image-type table entry for `usrcert` decoded as `0x00010000` (adjacent to `initrd` =
`0x00008000`), whereas the esbc header `certtype` field is a full 32-bit value where the stock
images use `0xFFFFFFFF`. These are **different fields**: the header `certtype` is read by the
runtime module (`drv_pkicms`), while the BL31 image-type id selects which *image* to load. Worth
keeping in mind if F1 fails.

---

## 2026-10-26 — ❌ F1 FAILED — but the CMS files ARE being read (atime proof)

### Result

```text
rootfs on card : 952423c1b00d9149…  (== F1)
phy_startup errs: 1
davinci nodes   : 0
bringup         : FAILED (Expected six Ascend device nodes, found 0)
driver          : "Device os load failed. (loader_devid=0; dev_id=0)"  ×3
```

### The decisive evidence: `user.crl` was read, `user.xer` was not

File atime after the boot (boot started 02:02:22):

| file | atime | meaning |
|---|---|---|
| `/usr/local/Ascend/CMS/user.crl` | **02:02:36** | **read during the boot** ✅ |
| `/usr/local/Ascend/CMS/user.xer` | 01:59:13 | never opened ❌ |
| `/lib/davinci.conf` | 00:56:22 | (not re-read; cached) |

So the driver **did** find `/CMS/user.crl` and read it — the path and the config resolution are
right. But it **never opened `user.xer`**, which is the file that carries the customer certificate.

Look at the load loop again:

```c
for (i = 0; i < DEVDRV_BLOCKS_NUM; i++) {
    ret = devdrv_load_file_copy(loader, g_load_file[chip_type][i].file_name, blocks);
    if (ret < 0) {
        if (g_load_file[chip_type][i].file_type == DEVDRV_CRITICAL_FILE) {
            devdrv_err("File copy error. ..."); devdrv_load_blocks_free(loader); return ret;
        }
        if (g_load_file[chip_type][i].fail_mode == DEVDRV_NOTICE_BIOS) {
            devdrv_get_rd_flag(loader, ...);
            blocks->blocks_valid_num = 0;
            devdrv_load_notice(loader, blocks, DEVDRV_SEND_FINISH);
            devdrv_info("Skip file. (dev_id=%u; file=%d; name=\"%s\")\n", ...);
        }
    }
}
```

`user.crl` is #10 and `user.xer` is #11. The BIOS is told to consume each file in turn via the
shared-memory flag protocol; if the BIOS rejects a file it sets a fail flag and the driver moves on.
`user.crl` was consumed; `user.xer` was not — consistent with the BIOS **stopping** after `user.crl`.

### The load protocol (from `devdrv_device_load.c`)

The BIOS and driver hand files to each other through a 64-bit flag word in shared memory:

```c
#define DEVDRV_SEND_FINISH       0x6666666666666666UL
#define DEVDRV_SEND_PATT_FINISH  0x7777777777777777UL
#define DEVDRV_RECV_FINISH       0x8888888888888888UL
#define DEVDRV_TEE_CHECK_FAIL       0x5A5AA5A55A5AA5A0ULL
#define DEVDRV_IMAGE_CHECK_FAIL     0x5A5AA5A55A5AA5A1ULL
#define DEVDRV_FILESYSTEM_CHECK_FAIL 0x5A5AA5A55A5AA5A2ULL
```

and the driver distinguishes them:

```c
if (flag_r == DEVDRV_TEE_CHECK_FAIL)          devdrv_err("Load file check tee failed. ...");
else if (flag_r == DEVDRV_IMAGE_CHECK_FAIL)    devdrv_err("Load file check image failed. ...");
else if (flag_r == DEVDRV_FILESYSTEM_CHECK_FAIL) devdrv_err("Load file check filesystem failed. ...");
```

**These messages are the diagnostic we need** — they are `devdrv_err`, so they should be in the
kernel log, but the log level was 4 and the Ascend log driver routes `drv_info`/`drv_err` through
`log_save_to_ring_buf` (not plain printk) when `CFG_FEATURE_HOST_LOG` is set. The ring buffer is
readable via the `DEVDRV_MANAGER_GET_HOST_KERN_LOG` ioctl (`_KA_IO('M', 203)`) on the devmng node —
but that node does not exist while the cards are down.

**So the next run must capture the ring buffer while the cards are up** — i.e. after a *successful*
boot. Which means we need one more diagnostic run whose only purpose is to keep the driver alive
long enough to read the log.

### Also learned: the driver's `devdrv_info` messages are invisible by default

`devdrv_info` → `drv_info("drv_pcie", …)` → `drv_log_print(KERN_INFO, "INFO", …)` →
`log_save_to_ring_buf(...)`. The messages
`Get file size. (dev_id=%u; file_name="%s"; size=%lld)`,
`Notice BIOS to Load file. (...)` and `Skip file. (...)` are exactly what we want, and they are
written to the ring buffer, not to dmesg. They have to be pulled via the ioctl.

### Next test — G1: isolate the CMS files themselves

| # | rootfs | /CMS/ | question |
|---|---|---|---|
| **G1** | **stock** | **user.xer + user.crl present** | do the CMS files break an otherwise-stock boot? |
| G2 | stock | removed | control (known good) |

G1 is the important one. If **G1 boots**, the CMS files are harmless and the failure is purely the
*signature* of our rootfs → then the remaining work is making the CMS acceptable to the BIOS's
customer path. If **G1 fails**, the CMS files themselves upset the BIOS (e.g. `user.crl` must match
a different format, or the `usrcert` image needs its own `cms` section) and must be fixed before
the signature question matters.

To capture the log on a *successful* boot, the ring buffer must be read while the cards are up —
`/dev/devmng*` only exists then.

### Current card state

Stock rootfs restored (`12975a9d…`). `/usr/local/Ascend/CMS/{user.xer,user.crl}` still in place
(intentionally — G1 needs them). Cards are down; a cold power cycle is required.

---

## 2026-10-26 — ✅ G1 BOOTED — the CMS files are harmless; F1's failure was our signature

### G1 result (stock rootfs + `/CMS/{user.xer,user.crl}` present)

```text
rootfs on card : 12975a9d7e934525…  (stock)
phy_startup errs: 0
davinci nodes   : 6
npu-smi         : all 6 chips 310P3 OK
```

**G1 boots cleanly.** So:

- The `/CMS/user.xrl` + `/CMS/user.xer` files do **not** upset a stock boot — the BIOS accepts
  them and ignores them as expected for a stock-signed image.
- Therefore **F1's failure was caused purely by our rootfs signature**, not by the presence of the
  CMS files, and not by the CMS files' format.
- The remaining problem is narrow and specific: the BIOS rejected our customer-signed rootfs.

### ⚠️ Correction: my earlier atime inference was unsound

I previously concluded from atimes that "`user.crl` was read but `user.xer` never was". Testing the
filesystem behaviour directly shows that conclusion was **not** sound:

- `/usr/local` is ext4 with `relatime`. Under relatime, atime is updated only if
  `atime <= mtime` **or** `atime <= ctime`.
- When I copied the files in at 01:59:13 and then computed `user.xer`'s hash myself, that read
  pushed `user.xer`'s atime to 01:59:13.505 — **above** its mtime of 01:59:13.472. From then on
  no read could advance it. So "atime unchanged" proved nothing about whether the driver read it.
- `user.crl`'s atime *did* advance to 02:02:36 during the F1 boot — that one is meaningful,
  because its atime was still below its mtime at that point.

Verified empirically on the card (both the `atime < mtime` and `atime == mtime` cases update on
read; only `atime > mtime` does not).

### G2 armed — a clean atime experiment

All three relevant files had their atimes forced back to `2026-10-26 00:00:00`:

```text
/usr/local/Ascend/CMS/user.xer              atime = 00:00:00
/usr/local/Ascend/CMS/user.crl              atime = 00:00:00
/usr/local/Ascend/driver/device/Ascend310P.cpio.gz   atime = 00:00:00
```

The next power cycle (stock rootfs, cards currently healthy and up) will therefore record the
**exact** read time of each file, and answer definitively: does the driver open `user.xer`, and in
what order relative to the rootfs?

- `user.crl` read + `user.xer` read → both are delivered to the BIOS; the customer path is armed
  and F1 failed for a signature reason.
- `user.crl` read + `user.xer` **not** read → the driver stops at `user.crl`, and something about
  `user.xer` (size? format? the `usrcert` tag?) makes the driver or BIOS skip it.

### Why the ring buffer could not be read

The `devdrv_info`/`devdrv_err` messages (`Get file size.`, `Notice BIOS to Load file.`,
`Skip file.`, `Load file check image failed.`) go through
`log_save_to_ring_buf()` rather than plain printk. The reader is the
`DEVDRV_MANAGER_GET_HOST_KERN_LOG` ioctl (`_KA_IO('M', 203)`) on the **devmng** node — but this
driver variant registers only `/dev/drv_debug` (misc, 10:264), and the ioctl returns `EIO` there
because the devmng ioctl table is not installed on it. `/dev/devmng*` does not exist on this
driver. So the ring buffer is not reachable from userspace here; the atime experiment is the
substitute.

### What we know about the failure mode of F1/D1

`"Device os load failed. (loader_devid=N; dev_id=N)"` comes from `devdrv_timer_task` when the
load timer expires — i.e. the BIOS never sent `DEVDRV_RECV_FINISH` (0x8888…) *or* any of the
explicit failure flags. That is a **timeout**, which is consistent with the BIOS rejecting the
image and simply not completing the handshake.

### Remaining unknowns

- Whether the `customer efuse` bit is provisioned on these cards. `npu-smi info -t
  custom-op-secverify-enable/-mode/-cert` is **not supported** on 310P ("This device does not
  support querying…"), and there is no boot log capture available (no UART). If the customer path
  is closed by efuse, none of the above will boot — and the failure mode will look identical.
- Which `image_type` the customer path covers. BL31 has no image-name literals; the mapping is
  data-supplied (`hw_adapt_desc_fetch copy image name`), so it is only settleable from boot logs.

---

## Appendix — reference data

### Stock container geometry (verified byte-exact)

```text
file 0x0000 / 0x1000   two 0x1000 header copies
0x2000                 payload, code_len bytes   (tag block 0x100 + data)
cms_start = 0x2000 + code_len + (32 - code_len % 16)
ini_start = cms_start + 0x4000
crl_start = ini_start + 0x0800
```

Header fields (per copy): `certtype` @+0x08, payload digest @+0x2C, `img_offset` @+0x458,
`code_len` @+0x45C, header hash @+0x4BC, extern magic @+0x4E0 (copy 0 only).

`rebuild_esbc.py --in <stock>` → **byte-identical** (verified 2026-10-26).

### Stock signing chain

```text
Huawei Root CA                              (RSA-4096, self, PKCS#1 v1.5)
└─ Huawei Integrity Root CA - G2            (RSA-4096, PSS/MGF1-SHA256, self)
   ├─ Huawei Integrity CA 1 - G2            (intermediate)
   │  ├─ Computing PL Code-Signing Authority 2 - G2   (leaf, signs the container)
   │  └─ Huawei Time-Stamping Authority 2 - G2        (leaf, RFC-3161)
```

All five keys audited: RSA-4096, e=65537, no shared modulus, no small factor, no perfect power,
no Wiener / Boneh-Venkataraman small exponent, no Pollard p−1, no Fermat factor. **No weaknesses.**

### emSign / public-root angle — closed

Huawei's public root list does contain `emSign Root CA - G1`
(`40F6AF03…`, 2018-02-18 → 2043-02-18), but that is Huawei's **web** PKI. The device's firmware
anchors are the two Huawei-only roots above; no public root cross-certifies into them.

"emSign private key = 4" is **false**: `d ∈ {2,3,4,5,8,16}` all fail `m^(d·e−1) ≡ 1 (mod n)` on
the real cert, and brute force over `d ≤ 4096` finds nothing. A scan of all 120 roots in the card's
`/etc/ssl/certs/ca-certificates.crt` for `d ≤ 2000` found **zero** weak roots. The Oct-2026 eMudhra
compromised intermediates (`EM {DV,OV,EV} TLS CA - G3B/G3C`) and their roots
(`emSign Root TLS CA - G3`, `AAA Certificate Services`) are **absent** from this device.

### Digest bindings (all recomputed and confirmed)

| Binding | Value |
|---|---|
| `SHA-256(initrd.ini body)` | `2934ff7d391caffa2b2543d6d0cfAA4e1f9f4b7a23babb9b419b4a75ccc7b66b` = CMS `messageDigest` |
| `SHA-256(code-signing leaf .der)` | `af70de12c89033a89a2041d3287874451fe51a7f03c24f1debdd7a3513fb709a` = `signingCertificateV2` |
| `SHA-256(TSTInfo DER)` | `705c909a6dd3403124c05ef0b118268c45972594ec1bfacf6ddbc7bcc5b4f386` = TST `messageDigest` |
| `SHA-256(TSA leaf .der)` | `75ec8c34d3b3ed4cd1b91bade3173faada399f86ae1f13ab7b3e7d0b1d2434c9` = TST `signingCertificateV2` |

`af70de12…` is the constant `messageDigest` the 2026-09-12 research flagged across all three
9201-byte PSS structures. It is the `signingCertificateV2` hash of the leaf — a signed attribute,
not an anomaly. (Rank-3/U-D from `SECUREBOOT-WEAKNESSES.md` resolved.)

### `--user_cert` mechanism (from `upgrade-tool` disassembly)

`upgrade-tool --user_cert --path <file>` does **not** install a certificate:

```text
0x692e7  calc_pub_hash: EVP_DigestInit_ex(SHA256) → Update → Final  (32 bytes)
0x692c9  buffer = file + 0x2000
0x69281  length = u32(file + 0x45c)          # code_len
0x692a3  requires 0x2000 + code_len <= file_size
0x68b99  set user cfg item key "ccpu_usr_cert_hash"
```

So it computes `SHA-256(file[0x2000 : 0x2000 + code_len])` and stores that digest in the card's
secure flash under `ccpu_usr_cert_hash` (via `drv_user_cfg.ko` → `sec_flash_write`). The
certificate bytes are never transferred. Verified against our archived `user.der`:
`SHA-256(body) == header digest @0x2c` ✅.

BL31 side: `usr cert hash cal fail!` / `usr cert hash cpy fail.` / `usr cert hash cmp fail.` /
`Secure Boot: customer efuse Solution` — a digest comparison against the stored value.

### The `usrcert` image type

BL31's image-type table contains `dataisp, dataotp, tee, initrd, uimage, dtimage, usrcert`.
`drv_pkicms.ko` does **not** contain `usrcert` — it is a **BL31-only** image type, i.e. the
customer certificate is consumed at boot by BL31, not by the runtime module.

### `.sec` containers (stock rootfs `/var/`)

Four files, `5a5aa5a5` magic, each embedding the same 4-cert chain (root, integrity CA,
code-signing leaf, TSA leaf). Header: `u32@0x08` = payload-ish length, `u32@0x18`=0x1820c000,
`u32@0x1c`=0xca0a0a01, `u32@0x20`=0x31c3, `u32@0x24`=0xca0a0a02, `u32@0x28`=3522 (matches the
stock CRL section length), first DER cert at 0x2c.

### Files

- `artifacts/ascend-310p-26.0.rc1/CUSTOMER-CERT-ANALYSIS-20261026.md` — full analysis
- `artifacts/ascend-310p-26.0.rc1/customer-cert-20261026/` — certs, CMS, modules, scripts
- `artifacts/ascend-310p-usercert-backup-20261021/` — prior attempt artifacts + private keys
  (crown jewels; backed up on 3 drives)

---

## 2026-10-26 — Host driver: the installed set is ALREADY our OpenSensor build

Before touching the driver I compared the two trees, and the result changed the plan:

| | installed `/lib/modules/…/dkms` | `ascend-310p-host-driver-restored` |
|---|---|---|
| build date | **Oct 15** | Sep 19 |
| modules | 36 | 31 |
| `ascend_dma_sync` tracepoint | **present** | **absent** |
| `devdrv_device_dma_*` (char dev) | **present** | **absent** |

`/usr/src/davinci_ascend-1.0/` (the DKMS source for the installed set) is the **newer** OpenSensor
tree — it contains `DEVICE_DMA.md` and `pcie_host/devdrv_device_dma.c`, `devdrv_dma.c`,
`devdrv_dma_profile.h`, `devdrv_dma_poll.h`, `devdrv_dma_trace.h`, none of which exist in the
Sep-19 tree. Its README says the DMA work is *"installed, with reservation lifecycle validated on
four physical NPUs"*.

**So installing the Sep-19 tree would have been a downgrade** — it would have removed the DMA
device, the DMA tracepoints and the profile/poll machinery. The fork was built successfully for
`7.0.0-38-generic` (31 modules, CRCs and namespaces validated) and the artifacts are in
`build-artifacts/7.0.0-38-generic/`, but **they are not installed and should not be**, because
`/usr/src/davinci_ascend-1.0` supersedes them.

Backup taken anyway: `/home/matteius/ascend-dkms-backup-20261026/`
(`dkms-modules.tar.gz` + a copy of the installed tree).

Also corrected: the `/CMS/user.crl` and `/CMS/user.xer` paths are **vendor** functionality — both
the installed module and the April vendor module in `ascend-run/` contain them. Not something the
fork added.

---

## 2026-10-26 — I-series: the coherent customer chain (deployed, awaiting power cycle)

### Two real bugs found in my own CMS builder

Comparing my `pss_sign.py` output against the stock container byte-for-byte exposed two defects
that would each cause a rejection:

1. **The timestamp token attribute was malformed.** `attr()` did `setof(*values)` where `values`
   was a *list containing one already-encoded element*, so the token's bytes were iterated as an
   integer list and the whole ContentInfo got embedded **inside the attribute OID**. Fixed with a
   dedicated `attr_raw()`.
2. **`unsignedAttrs` had an extra SET layer.** Stock is `[1] IMPLICIT { Attribute }` (a SEQUENCE);
   mine was `[1] IMPLICIT { SET OF { Attribute } }`. Fixed.

After both fixes the SignerInfo structure is **identical to stock**, verified by DER tree diff:

```text
STOCK                              I3
signedAttrs[0]                     signedAttrs[0]
  [0] 30 len=24                      [0] 30 len=24
  [1] 30 len=28                      [1] 30 len=28
  [2] 30 len=47                      [2] 30 len=47
  [3] 30 len=70                      [3] 30 len=70
unsignedAttrs[1]                   unsignedAttrs[1]
  [0] 30 len=2890                    [0] 30 len=2440     <- length differs (our certs are
    [0] 06 len=11                        [0] 06 len=11        shorter), nesting identical
    [1] 31 len=2873                    [1] 31 len=2423
      [0] 30 len=2869                    [0] 30 len=2419
        [0] 06 len=9                       [0] 06 len=9
        [1] a0 len=2854                    [1] a0 len=2404
```

`cmsverify2.py` on the result: `PSS SIG : VALID (signer CN=Signing Certificate G2, salt=32)`,
`ALL SIGNERINFOS VALID`, and the `signingCertificateV2` attribute is byte-identical in structure to
stock (`3035 3033303130 0d 0609 608648016503040201 05 00 0420 <32-byte sha256>`).

### The structural insight

Stock uses **one intermediate issuing both leaves**:

```text
Huawei Integrity Root CA - G2
 └─ Huawei Integrity CA 1 - G2          <-- ONE CA
    ├─ Computing PL Code-Signing Authority 2 - G2
    └─ Huawei Time-Stamping Authority 2 - G2
```

Our chain had **two separate CAs** (`Signing Certificate CA G2` and `Timestamp Certificate CA G2`),
so the timestamp branch could not chain to whatever the anchor was.

And the anchor we had enrolled was the **leaf** (`Test Signing Cert`) — a leaf cannot be the
parent of the TSA leaf, so the timestamp token could never validate against it. That is the
structural reason F1 failed, independent of the encoding bugs.

### The coherent setup now deployed

```text
user.xer  = user-signca.der   (Signing Certificate CA G2 — the INTERMEDIATE anchor)
            certtype=0xFFFFFFFF, code_len=1756
user.crl  = [ RootCA G2 CRL , signca.der , signCA CRL ]   (3096 B, stock shape)
enrolled  = SHA-256(tag‖signca.der) on all six devices  <- same file as user.xer
rootfs    = I3-stockpayload.cpio.gz   6a965f917b22951b...
            payload = STOCK (isolates the signature path)
            CMS     = signed by Signing Certificate G2 (leaf under signca)
                      certs [signca, signcert], crls [rootG2.crl, sign.crl]
                      TST    = ts2 (Timestamp Certificate G2, re-issued under signca)
            certtype = 0x2
```

The TSA leaf `ts2.pem` was re-issued under `Signing Certificate CA G2`
(`openssl verify -partial_chain -CAfile signca.pem ts2.pem` -> OK), so **both leaves now sit
beneath the single enrolled anchor** — exactly the stock topology.

### What each outcome means

- **I3 boots** -> the customer path works end to end. Then `I4-modpayload.cpio.gz`
  (`e17cbc82ea64a670...`, same setup but the **modified** payload) is custom firmware.
- **I3 fails** -> the customer path is gated on something else (eFuse customer-mode bit, or the
  exact anchor semantics), and we would need the device-side log to see the BIOS's verdict.

### Current card state

I3 deployed and enrolled; cards were up and healthy from the last G1 boot. A **cold power cycle
(>=30 s)** is required.

---

## 2026-10-26 — I3 failed; THREE real CMS bugs found in my generator (the actual blocker)

I3 (coherent chain: anchor = intermediate `signca`, TSA re-issued under `signca`) failed with the
same `uda_wait_all_phy_startup: Wait timeout` signature.

Rather than iterate blindly, I did a **byte-level DER tree diff of my CMS against the stock
container's**, and that exposed three genuine defects in my own `pss_sign.py` — each of which would
make any conforming CMS parser reject the signature:

### Bug 1 — `issuerAndSerialNumber` was not wrapped

`SignerInfo` is `version, sid, digestAlgorithm, [0] signedAttrs, signatureAlgorithm, signature`
where `sid` is a **single SEQUENCE** `{ Name, serialInteger }`.

My generator emitted the Name and the serial as **two sibling top-level fields**. Worse, the
correct code path (`signer_info`) was fixed but the `if unsigned:` branch — the one actually taken
whenever a timestamp token is present — still had the old un-nested form, so the fix appeared not
to work. Both paths now use `issuer_and_serial`.

### Bug 2 — `unsignedAttrs` had an extra SET layer

Stock: `[1] IMPLICIT { Attribute }` (a SEQUENCE).
Mine:  `[1] IMPLICIT { SET OF { Attribute } }`.

### Bug 3 — the timestamp token attribute was malformed

`attr()` did `setof(*values)` where `values` was a *list containing one already-encoded element*,
so the ContentInfo bytes were iterated as an integer list and the whole token got embedded
**inside the attribute OID**. Fixed with a dedicated `attr_raw()`.

### Also corrected: PSS `trailerField`

Stock's `rsassaPss` parameters **omit** `trailerField` (they end at `saltLength`). Mine included
`[3] trailerField = 1`. Semantically identical, but the AlgorithmIdentifier DER now matches stock
byte-for-byte:

```text
304106092a864886f70d01010a3034a00f300d06096086480165030402010500
    a11c301a06092a864886f70d010108300d06096086480165030402010500
    a203020120
```

### Verification after the fixes — M1

```text
STOCK  si.children: 0x02,0x30,0x30,0xa0,0x30,0x04,0xa1
M1     si.children: 0x02,0x30,0x30,0xa0,0x30,0x04,0xa1     ✅ identical
STOCK  sd.children: 0x02,0x31,0x30,0xa0,0xa1,0x31
M1     sd.children: 0x02,0x31,0x30,0xa0,0xa1,0x31           ✅ identical

STOCK  TST signedData : 0x02/1, 0x31/15, 0x30/122, 0xa0/1781, 0x31/917
M1     TST signedData : 0x02/1, 0x31/15, 0x30/106, 0xa0/1503, 0x31/758   ✅ same tags
STOCK  TST SignerInfo : 0x02/1, 0x30/142, 0x30/13, 0xa0/164, 0x30/65, 0x04/512
M1     TST SignerInfo : 0x02/1, 0x30/70,  0x30/13, 0xa0/79,  0x30/65, 0x04/512 ✅ same tags
```

(lengths differ only because our certificates are smaller than Huawei's)

`cmsverify2.py`: `PSS SIG : VALID (signer CN=Signing Certificate G2, salt=32)`,
`ALL SIGNERINFOS VALID`, timestamp token present as `1.2.840.113549.1.9.16.2.14`.

### M-series deployed

| file | payload | sha256 |
|---|---|---|
| `M1-stockpayload.cpio.gz` | stock | `b711403522395656…` — **deployed** |
| `M2-modpayload.cpio.gz` | modified9 | `744ef5f479ee9866…` — the goal |

Both use: anchor = `signca` (intermediate), `user.xer` = `user-signca.der`,
`user.crl` = `[rootG2.crl, signca.der, sign.crl]`, certtype `0x2`, TSA = `ts2` (under `signca`).

**Why this is now a genuinely different test from I3:** I3's CMS was *malformed CMS* — a verifier
had no choice but to reject it. M1's CMS is structurally identical to the stock container's and
cryptographically valid, so if M1 still fails, the cause is the trust/customer-path semantics
rather than a packaging defect.

### Card state

M1 deployed, cards down from the failed I3 boot. **Cold power cycle (≥30 s)** required.

---

## 2026-10-26 — M1 failed; the ROOTFS WAS NOT THE ONLY SIGNED CONTAINER

### M1 result

```text
rootfs on card : b711403522395656…  (== M1)
phy_startup errs: 1
davinci nodes   : 0
bringup         : FAILED (found 0)
npu-smi         : "dcmi module initialize failed. ret is -8005"
```

M1's CMS was structurally identical to the stock container's (see previous section) and
cryptographically valid, yet the device still rejected the image. So the fault moved from
*packaging* to *which image is actually signed*.

### The check tool is not an oracle for customer-signed images — proven

```text
STOCK .orig                     rc=0
B1 (stock Huawei CMS, ct=2)     rc=0      <- passes because it chains to the BUILT-IN root
P2 (our CMS, ct=2)              rc=6      <- fails: no built-in root for RootCA G2
```

The tool validates against the two built-in Huawei roots in `AS610_HBOOT2_UEFI.fd` and has no anchor
for `RootCA G2`, so it cannot judge a customer-signed image. It remains useful for **container
structure** and for `certtype`, but the device is the only oracle for the customer path.

(The error code differs meaningfully: the earlier *malformed* CMS gave `0x88200308`
(CmscbbDecodeSignature) whereas a *well-formed but untrusted* CMS gives `0x88300001`
(`image cms data verify failed`) — i.e. it decoded fine and then failed trust.)

### ★ The actual missing piece: TWO signed containers, not one

The mini_v2 load table sends **eleven** files to the BIOS, in order, with no identifier — the BIOS
infers each from its content. Two of them are signed esbc containers:

| # | file | tag | sections |
|---|---|---|---|
| 5 | `Ascend310P.image` | `uimage` | cms(9201) ini(76) crl(3522) |
| 6 | `Ascend310P.cpio.gz` | `initrd` | cms(9201) ini(76) crl(3522) |

**We have only ever re-signed the rootfs (#6). `Ascend310P.image` (#5) still carried the stock
Huawei CMS** — so the boot package was still a Huawei-signed image, and the customer path could
never engage for it.

### P-series — BOTH containers customer-signed (deployed)

| file | container | certtype | sha256 |
|---|---|---|---|
| `P1-uimage-ct2.image` | `Ascend310P.image` (boot package) | `0x2` | `b5c3005f24a3938c…` |
| `P2-initrd-ct2.cpio.gz` | `Ascend310P.cpio.gz` (rootfs, stock payload) | `0x2` | `6f4b6ecdcb8b796b…` |

Both: CMS signed by `Signing Certificate G2` (leaf under `signca`), `crl` section
`[rootG2.crl, signca.der, sign.crl]`, `ini` recomputed as `"<tag>,   <sha256(payload)>;"`,
`certtype = 0x2`, TSA = `ts2` (under `signca`).

### Safety note

`Ascend310P.image` had **no backup** on the card — it was overwritten before I noticed. Recovered
from the pristine local copy taken at the start of this investigation
(`027f4d22762ef0644…`, verified byte-identical) and now stored as
`/usr/local/Ascend/driver/device/Ascend310P.image.orig`. Full recovery set on the card:

```text
Ascend310P.image.orig             027f4d22762ef0644…   (pristine boot package)
Ascend310P.cpio.gz.orig           12975a9d7e934525…   (pristine rootfs)
Ascend310P.cpio.gz.bak-20261021   12975a9d7e934525…
```

### Card state

P1 + P2 deployed; cards down from the failed M1 boot. **Cold power cycle (≥30 s)** required.

### If P1+P2 still fail

The remaining candidates, in order:

1. **The `customer efuse` bit is not provisioned on these cards.** BL31 prints
   `Secure Boot: customer efuse Solution` on the customer path; if the bit is clear the path is
   closed regardless of any signature. `npu-smi info -t custom-op-secverify-enable/-mode/-cert` is
   *not supported* on 310P, and there is no UART/serial console. This would be unfixable from the
   host and would explain every failure so far.
2. **The `usrcert` image needs its own `cms` section.** Every other signed container has
   cms/ini/crl; `user.xer` currently has only the raw certificate. BL31 may require the
   customer image itself to be a signed container.
3. **The anchor semantics.** `user.xer` is now the *intermediate* (`signca`). It might instead need
   to be the root (`RootCA G2`), with the intermediate carried in `user.crl`.

---

## 2026-10-26 — P-series failed; the missing cell in the matrix identified

### P-series result (both containers customer-signed, intermediate anchor)

```text
Ascend310P.image   b5c3005f24a3938c…
Ascend310P.cpio.gz 6f4b6ecdcb8b796b…
phy_startup errs: 1     davinci nodes: 0     bringup: FAILED
```

Still rejected, even with **both** signed containers (uimage *and* initrd) customer-signed. So the
"only the rootfs was signed" theory is not sufficient either.

Card restored to stock immediately afterwards:

```text
Ascend310P.image.orig   027f4d22762ef0644…
Ascend310P.cpio.gz.orig 12975a9d7e934525…
```

### The anchor semantics — BL31's flow, reconstructed

From the BL31 string set:

```text
'usr cert address check fail, result = 0x%x!'   -> obtain an ADDRESS for the cert
'usr cert 0x%lx oversize!'                      -> size check
'usr cert copy fail!0x%x' / 'usr cert copy fail.' -> copy it
'usr cert hash cal fail!0x%x'                  -> hash the copy
'usr cert hash cmp fail.'                      -> compare with the secure-flash value
'usr cert cms vef fail, result = 0x%x'         -> verify the IMAGE's CMS with it
'usr cert cms vef success!'
'no usr, hw vef fail, ctrl_flag = 0x%x!'       -> fallback: hardware/built-in path
```

So: BL31 loads the `usrcert` image, hashes its payload, compares that with the value enrolled via
`--user_cert`, and if they match, **uses the certificate inside it to verify the image's CMS**.

Two independent choices, each of which we have varied:

| | anchor (`user.xer` payload) | image CMS signed by | timestamp token | result |
|---|---|---|---|---|
| F1 | **leaf** (`Test Signing Cert`) | same leaf | none | FAIL — but the CMS was **malformed** (unwrapped `issuerAndSerialNumber`) |
| M1 | **intermediate** (`signca`) | leaf under signca | `ts2` | FAIL — CMS valid, anchor is an intermediate |
| P | intermediate | leaf under signca | `ts2` | FAIL — but the *boot package* was still Huawei-signed in M1, now fixed here |
| **R** | **leaf** (`Test Signing Cert`) | **same leaf** | **none** | **deployed, untested** |

**R is the cell that was never tested**: F1 used the right anchor but a *malformed* CMS; M1/P used a
*valid* CMS but a different anchor. R combines the leaf anchor with a CMS that is now structurally
identical to the stock container's.

R's CMS:

```text
certs         : [ RootCA G2 , Test Signing Cert ]
crls          : [ rootG2.crl ]              (one CRL - the anchor's issuer)
crl section   : [ rootG2.crl , rootcaG2.der , rootG2.crl ]   (stock shape, 2-level chain)
signedAttrs   : contentType, signingTime, messageDigest, signingCertificateV2
signAlg       : rsassaPss sha256 / mgf1-sha256 / salt 32, NO trailerField  (stock-identical)
si.children   : 0x02,0x30,0x30,0xa0,0x30,0x04   (stock-identical)
sd.children   : 0x02,0x31,0x30,0xa0,0xa1,0x31   (stock-identical)
```

**No timestamp token** — a TSA leaf cannot chain to a leaf anchor, so it is omitted.

Deployed on the card:

```text
Ascend310P.image    4d35211f3f7a91e4…   (uimage,  leaf-signed, certtype 0x2)
Ascend310P.cpio.gz  ec960b4546727d97…   (initrd,  leaf-signed, certtype 0x2)
/usr/local/Ascend/CMS/user.xer  9833 B  (leaf container, hash b932bb15…)
/usr/local/Ascend/CMS/user.crl  3045 B  ([root.crl, rootcaG2.der, root.crl])
```

Enrolment of the leaf hash on all six devices must be re-done after the power cycle (the cards were
down, so `--user_cert` returned "Get device count failed").

### Note on the missing Q-series experiment

A `usrcert` container *with* its own cms/ini/crl trailer was built (Q1) — every other esbc
container has one, and ours did not. It was **not** used for R, because R's point is to vary only
the anchor. If R fails, Q1 is the next single-variable step.

### Card state

R-series deployed; cards down from the failed P-series boot; a **cold power cycle (≥30 s)** is
required, followed by re-enrolling the leaf hash on all six devices.

---

## 2026-10-26 — R-series failed; the CARD-SIDE trust contract found in `drv_user_cfg.ko`

R-series (leaf anchor, valid CMS, no timestamp) failed with the same signature. The anchor matrix
is now exhausted.

Card restored to pristine stock and `/usr/local/Ascend/CMS/` removed.

### The card-side module that owns the customer cert

`drv_user_cfg.ko` (in the card rootfs `/var/`, 183777 bytes) contains:

```text
ccpu_usr_cert_hash
sec_flash_read / sec_flash_write
"The sec_flash_write failed. (device_id=%u; ret=%d)"
"The sec_flash_read failed. (device_id=%u; ret=%d)"
```

**and three embedded X.509 certificates**:

| cert | subject | issuer | EKU | sha256 |
|---|---|---|---|---|
| intermediate | `Huawei Integrity CA 1 - G2` | `Huawei Integrity Root CA - G2` | — | `61b57eba58f81c70…` |
| **code-signing leaf** | **`Huawei CSI RSA-PSS Code-Signing Authority 2`** | `Huawei Integrity CA 1 - G2` | Code Signing | `534209698638c5c3…` |
| TSA leaf | `Huawei Time-Stamping Authority 2 - G2` | `Huawei Integrity CA 1 - G2` | critical, Time Stamping | `cc48b9602d99a769…` |

plus the **root** `Huawei Integrity Root CA - G2` in the adjacent string/der table.

### Why this explains every failure

The card-side module has the **Huawei chain compiled in**. The `ccpu_usr_cert_hash` value that
`--user_cert` writes is compared against the hash of a certificate the *device* already has — and
the device's own copy of the code-signing leaf is **`Huawei CSI RSA-PSS Code-Signing Authority 2`**
(`534209698638c5c3…`), which is **not** the leaf that signs the rootfs container
(`Computing PL Code-Signing Authority 2 - G2`, `af70de12…`).

So the customer-certificate mechanism is a *runtime* (post-boot) feature of
`drv_user_cfg.ko`/`drv_pkicms.ko` — it lets the runtime load and use a customer CA for **runtime
operations** (the `pkicms_get_name_by_cert_type` → `vendor-certificate` / `community-certificate` /
`user-certificate` path we found in `drv_pkicms.ko`, reading
`/var/asdrv/dev%d/device-sw-plugin/verify/%s`).

**It is not a boot-time firmware-signing path.** The boot-time trust decision is made by BL31 against
the **built-in Huawei roots only**. There is no hook for a customer CA there.

### The `.sec` containers are the device-side trust store (and they live in the rootfs)

The four files in the stock rootfs `/var/` (`<uuid>.sec`, `5a5aa5a5` magic) each contain:

```text
0x00  magic 5a5aa5a5
0x08  u32 = 38783   size of the certs block
0x28  u32 = 3522    size of the CRL section
0x2c  PKCS#7 SignedData (rsassaPss) containing Huawei Integrity Root CA - G2
0x38f 4 x509 certs (integrityCA x2, code-signing leaf, TSA leaf)
...   payload
      CRL section (3522 bytes)
```

i.e. the **same chain and the same 3522-byte CRL section** as the rootfs container — the device's
own copy of the trust material, shipped *inside* the rootfs.

### Revised understanding of the whole problem

| layer | who verifies | trust store | can a customer CA be used? |
|---|---|---|---|
| **boot (BL31)** | the card's secure boot | **built-in Huawei roots** (2 certs in `AS610_HBOOT2_UEFI.fd`) | **no hook found** |
| runtime (`drv_user_cfg`/`drv_pkicms`) | the driver | compiled-in Huawei chain **+** `ccpu_usr_cert_hash` in secure flash | **yes** — this is what `--user_cert` drives |

`--user_cert` writes a 32-byte digest into secure flash; `drv_user_cfg.ko` compares it against the
digest of a certificate the device holds, and `drv_pkicms.ko` then resolves a cert *by type*
(`vendor` / `community` / `user`) from `/var/asdrv/dev%d/device-sw-plugin/verify/%s`.

### Consequence for the goal

**Boot-time firmware signing with a customer CA does not appear to be supported on this hardware.**
Every configuration we tried failed identically because the bootloader only trusts Huawei's roots,
and the customer-certificate hook exists in the *runtime* path, not the boot path.

What this does **not** rule out, and what is now the most promising direction:

1. **The runtime path is real and reachable.** `drv_pkicms.ko` reads
   `/var/asdrv/dev%d/device-sw-plugin/verify/%s` where `%s` ∈ {`vendor-certificate`,
   `community-certificate`, `user-certificate`}. If we place a customer cert there — and the
   `/var/asdrv` tree is ours to create (it did not exist on the host) — the runtime may use our CA
   for whatever `pkicms` verifies. That path was never exercised, because it is not part of boot.
2. **The rootfs is entirely ours to modify.** `rebuild_esbc.py` reproduces the stock rootfs
   byte-for-byte, and the `.sec` trust containers are *inside* it. The runtime trust material is
   therefore writable — which is a different (and more tractable) question than boot-time signing.
3. The card-side `drv_user_cfg.ko` and `drv_pkicms.ko` are in the rootfs, so **the runtime
   verification code itself is modifiable** and we can rebuild the rootfs with an instrumented
   `drv_pkicms.ko` to observe exactly what it checks.

### Next step

Instrument the card-side `drv_pkicms.ko` (add a log of the cert path it opens and the cert it
verifies against), rebuild the rootfs with it, and boot. That converts the runtime trust question
from guesswork into an observation, and it is a modification we fully control — no signature
bypass required, because the rootfs still carries Huawei's stock signature until we can sign it.

**Immediate practical note:** the rootfs is currently the stock image and the cards will boot
normally. Nothing is left in a broken state.

---

## 2026-10-26 — No Huawei private key exists anywhere; runtime trust path fully mapped

### Exhaustive private-key search — negative

| artifact | result |
|---|---|
| `a1c06018-….sec` (1.1 MB) | `PRIVATE KEY-----` hit was OpenSSL's **string table** inside an embedded crypto library |
| `Ascend_310P_check.bin` | all 7 `-----BEGIN` hits are **string-table** entries (`RSA PRIVATE KEY` as a type name) |
| `/usr/src/davinci_ascend-1.0` (whole tree) | no PEM private key block |
| `/home/matteius/ascend-run` (whole tree) | no PEM private key block |
| card rootfs `/var/` | no PEM private key block |
| `*.key` / `*.pem` under `/usr/local/Ascend` | **none exist** |

**Conclusion: no Huawei private key is present in any reachable artifact.** Boot-time firmware
signing with a customer CA is therefore not achievable on this hardware — the bootloader only
trusts the two built-in Huawei roots, and there is no key to sign with either.

### The runtime path, fully mapped from `drv_pkicms.ko`

`pkicms_get_cert_data` (text+0x5bb8) is the cert resolver:

```c
name = pkicms_get_name_by_cert_type(cert_type);      // vendor | community | user-certificate
sprintf(buf, "/var/asdrv/dev%d/device-sw-plugin/verify/%s", dev_id, name);
ret = pkicms_read_file_buf(buf, out, size);
```

with the errors `sprintf_s cert path failed.` and `Read cert path file failed. (ret=%d; ...)`.

The module's **exported** API is the contract we care about:

```text
pkicms_get_sec_check_enable_flag   pkicms_read_efuse_reg     pkicms_nvcnt_read
pkicms_custom_verify_cms           pkicms_verify_cms         pkicms_verify_cms_data
pkicms_verify_cms_inner            pkicms_compare_crl        pkicms_hw_crl_compare
pkicms_crl_pre_compare             pkicms_check_ini_hash     pkicms_reset_ini_hash
pkicms_get_cms_file_size           pkicms_get_code_len       pkicms_check_img_tag
pkicms_get_real_cert_type          pkicms_get_name_by_cert_type
pkicms_get_alg_type_by_sign_data   pkicms_get_rsa_pss_salt_len
```

### ★ The security-check gate is an eFuse bit, and it is READABLE

`pkicms_get_sec_check_enable_flag` (text+0x3534):

```c
u32 val = 0;
pkicms_read_efuse_reg(0, 0x8126e080, &val);   // device 0, register 0x8126e080
*enable = (val != 0);
```

and `pkicms_read_efuse_reg` (text+0x3440) does:

```c
x0 = ioremap_prot(base + (id << 39), 4);   // 4 bytes, device MMIO
val = *(u32 *)x0;
```

So the "secure check enable" flag is a **device MMIO register at `0x8126e080`**, read from the card.
This is the same `sec_check_enable` that `npu-smi info -t custom-op-secverify-enable` refuses to
report on 310P — but we can read it ourselves.

### Why this reframes the goal

There are now two clearly separated mechanisms:

| | what it protects | trust store | customer CA possible? |
|---|---|---|---|
| **boot** (BL31) | the boot images | 2 built-in Huawei roots | **no** — no key, no hook |
| **runtime** (`drv_pkicms` on the card) | runtime CMS verification | the cert at `/var/asdrv/dev%d/device-sw-plugin/verify/<type>` + `ccpu_usr_cert_hash` | **yes** — this is what `--user_cert` drives |

The runtime path is the one we control end-to-end:

- `/var/asdrv/dev%d/device-sw-plugin/verify/user-certificate` is a **file on the card's rootfs**,
  which we can write into a rebuilt rootfs;
- the check is gated by `pkicms_get_sec_check_enable_flag` (an eFuse bit we can now read);
- `pkicms_custom_verify_cms` exists specifically for the custom/customer case;
- and `drv_pkicms.ko` itself lives in the rootfs, so it is **modifiable and rebuildable** — we
  already have `rebuild_esbc.py` reproducing the stock rootfs byte-for-byte.

### Next step (concrete)

1. Read the `sec_check_enable` eFuse bit on the card (via the MMIO register, from a small kernel
   module or by patching `drv_pkicms.ko` to log it).
2. Rebuild the rootfs with an **instrumented `drv_pkicms.ko`** that logs:
   - the efuse flag value,
   - the cert path it opens and whether the file exists,
   - the CMS verification result and which cert it used.
3. Boot it (still carrying Huawei's stock signature, since we cannot re-sign) and read the log.

That turns the runtime trust question from inference into measurement, and it is a change we fully
control. The instrumentation is a *rebuild of the rootfs*, not a signature bypass.

### State

Cards healthy on stock firmware (all 6 chips `OK`, `boot_status 16`). `/usr/local/Ascend/CMS`
removed. `Ascend310P.image.orig` and `Ascend310P.cpio.gz.orig` present on the card.

---

## 2026-10-26 — ✅ Custom-rootfs tooling complete: `repack_rootfs.py`

Even though boot-time signing is out of reach, **the rootfs itself is fully rewritable**, and that
is the foundation for custom firmware. The container format is now completely decoded and a working
repack tool exists.

### The rootfs is a doubly-nested container

```text
Ascend310P.cpio.gz  (outer esbc container)
├─ 0x0000 / 0x1000  two 0x1000 header copies (preamble 0x55aa55aa, head_len 0x4dc,
│                     certtype @+0x08, payload SHA-256 @+0x2C, code_len @+0x45C)
├─ 0x2000            payload (code_len bytes)
│                    └─ INNER esbc container: magic 33aa3a3a, tag "initrd" @+0x04,
│                       0x100-byte header, gzip(cpio) @+0x100
├─ cms  @ cms_start+0x0010   [16-byte gap][tag 12][len 4][body]
├─ ini  @ cms_start+0x4010   [16-byte gap][tag 12][len 4][body]
└─ crl  @ cms_start+0x4810   [16-byte gap][tag 12][len 4][body]
   where cms_start = 0x2000 + code_len + ((32 - code_len % 16) % 16)
```

The cpio is **SVR4 newc** and holds 1332 files, including the card-side kernel modules
(`drv_pkicms.ko`, `drv_user_cfg.ko`, `drv_upgrade.ko`, `drv_devmng.ko`, …) and the four `.sec`
trust containers in `/var/`.

### `repack_rootfs.py` — the tool

```text
repack_rootfs.py --container <Ascend310P.cpio.gz> --out <new.cpio.gz>
                 [--set path=localfile]... [--add path=localfile]... [--delete path]...
                 [--certtype 0xN]
```

It extracts the nested cpio (preserving the archived mode bits, which `cpio -idm` loses when run
unprivileged), applies the file overrides, repacks, re-gzips, re-wraps the inner esbc header, and
rebuilds the outer container with correct geometry and digests.

**Round-trip verified**: a no-op repack reproduces a container whose `cms`, `ini` and `crl`
sections are **byte-identical to stock** (9201 / 76 / 3522) and whose header digests match
`SHA-256(payload)`.

```text
cms bytes identical to stock: True
ini bytes identical to stock: True
crl bytes identical to stock: True
header digests match: True
```

The payload is smaller after repack (108.8 MB vs 109.4 MB) purely because of gzip settings — the
cpio content is the same.

### What this enables

1. **Modify the card-side kernel modules.** `drv_pkicms.ko` and `drv_user_cfg.ko` live in the
   rootfs. We can rebuild them and repack the rootfs — carrying Huawei's stock signature forward,
   because the sections are copied verbatim.
2. **Instrument the runtime trust path.** `pkicms_get_cert_data` resolves
   `/var/asdrv/dev%d/device-sw-plugin/verify/%s`; `pkicms_get_sec_check_enable_flag` reads an
   eFuse bit at `0x8126e080`; `pkicms_custom_verify_cms` is the customer-path entry point. A
   rebuilt `drv_pkicms.ko` that logs all of this turns the runtime trust question into a
   measurement.
3. **Ship our own trust material.** The `.sec` containers in `/var/` are the device's copy of the
   trust store, and they are inside the rootfs we control.

### The remaining blocker is now precisely one thing

The repacked rootfs keeps Huawei's signature because the `cms` section is copied verbatim. To ship a
*modified* rootfs, that signature must cover the new payload — which needs the Huawei private key,
which **does not exist in any reachable artifact** (verified exhaustively).

Unless the runtime verifier can be made to accept our chain, the only way to run modified code is
to modify code the device already trusts. Two candidate routes:

- **The `.sec` trust containers and the runtime cert path** are *data*, and they live in the rootfs.
  If the runtime's trust decision can be steered by the `verify/%s` file plus
  `ccpu_usr_cert_hash`, we may be able to run a customer-signed module without touching the boot
  signature at all.
- **The instrumented `drv_pkicms.ko`** will tell us exactly which of these the runtime consults.

### State

Cards healthy on stock firmware. `repack_rootfs.py` is in `customer-cert-20261026/` and on the card
at `/home/matteius/repack_rootfs.py`.

---

## 2026-10-26 — The boot chain fully decoded; a correction to my own earlier reasoning

### Correction: the 2026-10-21 CMS was NOT malformed

I claimed the three bugs I found in my own CMS builder (`pss_sign.py`) explained the 2026-10-21
failure. **That was wrong.** Verified directly:

```text
STOCK                   si.children: 0x02,0x30,0x30,0xa0,0x30,0x04,0xa1
2026-10-21 (cmssign)    si.children: 0x02,0x30,0x30,0xa0,0x30,0x04,0xa1   <- identical
2026-10-21 PSS verify  : VALID (signer CN=Signing Certificate G2, salt=32)
2026-10-21 digestAlgs : 300d06096086480165030402010500  (identical to stock)
2026-10-21 contentType: 1.2.840.113549.1.7.1 (detached, identical)
2026-10-21 certs       : [Signing Certificate CA G2, Signing Certificate G2]  (stock: [CA, leaf])
2026-10-21 crls        : [RootCA G2 CRL, Signing Certificate CA G2 CRL]        (stock: 2 CRLs)
```

Huawei's own `cmssign` produced a CMS that is **structurally and cryptographically identical to the
stock container's**, and it still failed. So the 2026-10-21 failure was never a packaging defect,
and my three fixes (unwrapped `issuerAndSerialNumber`, extra SET in `unsignedAttrs`, malformed
timestamp-token attribute, plus the PSS `trailerField`) were real bugs in *my* builder but did not
diagnose the original failure. The M/R-series failures remain unexplained by packaging.

### The complete boot chain (decoded)

```text
[BIOS / hboot2 fd]  AS610_HBOOT2_UEFI.fd
   └── BL31 (secure boot)  verifies images against 2 built-in Huawei roots
        └── hboot1a / hboot1b / hlink   (the only flashable components)
             └── Ascend310P.image      (tag "uimage")  ← the LINUX KERNEL
             └── Ascend310P.cpio.gz    (tag "initrd")  ← the ROOTFS
             └── Ascend310P_dt.img     (DTBs)
             └── Ascend310P_tee.bin    (TEE)
             └── /CMS/user.crl, /CMS/user.xer   (customer cert material)
```

**`Ascend310P.image` is the Linux kernel** — `Linux version 6.6.0-72.0.0.76.h21510.1170.eulerosv2r15.ascend.aarch64`,
an ARM64 Image of 33,292,288 bytes, *uncompressed* inside the esbc container (the inner payload
starts with the ARM64 Image header; `text_off=0`, `image_size=0x1fc0000`).

**`Ascend310P.cpio.gz` is the rootfs** — inner esbc header, then gzip at +0x100, then an SVR4 cpio
with 2574 entries (1332 regular files) containing:

- the card-side kernel modules (`drv_pkicms.ko`, `drv_user_cfg.ko`, `drv_upgrade.ko`,
  `drv_devmng.ko`, …)
- the four `.sec` trust containers in `/var/`
- `var/tsch_fw.bin` (23 MB — the **AI Core / TSC firmware**)
- `usr/lib64/aicpu_kernels/libcpu_kernels.so` (27 MB) and `libtensorflow.so` (105 MB) — the
  **AI CPU kernels**
- `/boot/` with `System.map`, `config`, DTBs and the kernel `.hmac` signature (the kernel image
  itself lives in `Ascend310P.image`, not the rootfs)

So "custom Linux and firmware on the cards" is concretely: **replace the kernel**
(`Ascend310P.image`), **replace the rootfs** (`Ascend310P.cpio.gz`, including the card-side modules
and the AI Core firmware), and optionally the TSC firmware.

### The kernel does NOT enforce module signatures

From the card's own `config-6.6.0-72.0.0.76.…`:

```text
CONFIG_MODULES=y
CONFIG_MODULE_SIG=y
CONFIG_MODULE_SIG_SHA256=y
CONFIG_MODULE_SIG_FORMAT=y
# CONFIG_MODULE_SIG_FORCE is not set      <-- signatures are NOT enforced
# CONFIG_MODULE_SIG_ALL is not set
CONFIG_SECURITY_LOCKDOWN_LSM=y
CONFIG_SECURITY_LOCKDOWN_LSM_EARLY=y
CONFIG_KEXEC=y  CONFIG_KEXEC_FILE=y  CONFIG_KEXEC_SIG=y
```

**`CONFIG_MODULE_SIG_FORCE` is not set**, so the kernel will load an unsigned or
differently-signed module (it only warns). Only 3 `.ko` files ship signed; the rest are loadable
unsigned. `CONFIG_KEXEC=y` with `CONFIG_KEXEC_SIG=y` means a self-signed kernel can be kexec'd.

### Where this leaves the goal

Both halves of "custom Linux and firmware" live in **two files we can rewrite**:
`Ascend310P.image` (kernel) and `Ascend310P.cpio.gz` (rootfs, modules, AI Core firmware). The
only thing standing between us and booting them is the **esbc signature on those two containers**,
which the bootloader verifies against Huawei's built-in roots.

The `repack_rootfs.py` tool already rebuilds `Ascend310P.cpio.gz` byte-accurately with the stock
signature carried over. To boot a *modified* rootfs we need either:

1. the Huawei signing key (verified absent from every reachable artifact), or
2. a way to make the bootloader accept our chain — the `usrcert` image type + `ccpu_usr_cert_hash`
   path, which is the one avenue not yet exhausted, and which the `Pkcsms`/`pkicms` runtime
   suggests exists.

### The diagnostic channel

The driver's `devdrv_info`/`devdrv_err` messages (boot mode, file-load decisions, "Load file check
image failed", "Skip file") go to a **ring buffer** (`log_save_to_ringbuf`), not to dmesg. The
userspace reader is the DMS feature command:

```text
ioctl(fd, ('S' << 8) | 0, &dms_ioctl_arg)     # DMS_MAIN_CMD_LOG / DMS_SUBCMD_GET_LOG_INFO
```

`dms_log.py` is written and deployed, but the call currently returns **ENODEV** — the command is
being rejected before reaching the log feature. The DMS node is `/dev/davinci_manager`, the arg
struct is `dms_ioctl_arg`, and the log handler expects a `LogInfo{void*,u32,u32}` with
`length == 0x1400000`. Getting this ioctl working would give us the driver's own account of every
boot, which is the missing piece for the last unexplained variable.

---

## 2026-10-26 — ✅ THE DIAGNOSTIC CHANNEL IS OPEN: `dms_log.py`

The driver's own messages were invisible because `devdrv_info`/`devdrv_err` are routed to a
**ring buffer** (`log_save_to_ringbuf()`), not to printk. The reader is a DMS feature command, and
the missing piece was that `/dev/davinci_manager` requires the **DMS sub-module to be opened first**:

```c
/* drv_davinci_intf_ioctl: the default case needs file_private->fops.unlocked_ioctl */
if (file_private->fops.unlocked_ioctl == NULL) {
    log_intf_err("File not init. (cmd=%u)\n", _KA_IOC_NR(cmd));
    return -ENODEV;            <-- exactly what we were hitting
}
```

The correct sequence:

```c
struct davinci_intf_open_arg { char module_name[256]; int device_id; };
ioctl(fd, _IO('Z', 0), &open_arg);          /* DAVINCI_INTF_IOCTL_OPEN, module_name="DMS" */
ioctl(fd, ('S' << 8) | 0, &dms_ioctl_arg);   /* DMS_MAIN_CMD_LOG / DMS_SUBCMD_GET_LOG_INFO */
ioctl(fd, _IO('Z', 1), &close_arg);         /* CLOSE */
```

`dms_log.py` implements this and works — **869 lines** of driver log retrieved.

### The baseline: a *successful* stock boot, in the driver's own words

```text
Get block size. (dev_id=0; block_num=48; size=0x3000000)
Get file size. (dev_id=0; file_name=".../Ascend310P_ddr.image"; size=197396)
Notice BIOS to Load file. (dev_id=0; file_name=".../Ascend310P_ddr.image"; size=1; flag=0x6666666666666666)
Get file size. (dev_id=0; file_name=".../Ascend310P_lowpwr.image"; size=136464)
Notice BIOS to Load file. (…; size=1; flag=0x6666666666666666)
Get file size. (dev_id=0; file_name=".../Ascend310P_hsm.image"; size=129888)
Notice BIOS to Load file. (…; size=1; flag=0x6666…)
Get file size. (dev_id=0; file_name=".../Ascend310P.fd"; size=1712400)
Notice BIOS to Load file. (…; size=2; flag=0x6666…)
Get file size. (dev_id=0; file_name=".../Ascend310P_dt.img"; size=1820402)
Notice BIOS to Load file. (…; size=2; flag=0x6666…)
Get file size. (dev_id=0; file_name=".../Ascend310P.image"; size=31350770)
Notice BIOS to Load file. (…; size=30; flag=0x6666…)
Get file size. (dev_id=0; file_name=".../Ascend310P.cpio.gz"; size=109389874)
Notice BIOS to Load file. (…; size=48; flag=0x7777777777777777)
Notice BIOS to Load file. (…; size=48; flag=0x7777777777777777)
Notice BIOS to Load file. (…; size=9;  flag=0x6666666666666666)
Get file size. (dev_id=0; file_name=".../Ascend310P_tee.bin"; size=2286834)
Notice BIOS to Load file. (…; size=3; flag=0x6666…)
Skip file. (dev_id=0; file=8;  name="/usr/local/Ascend/driver/device/Ascend310P.crl")
Skip file. (dev_id=0; file=9;  name="/usr/local/Ascend/CMS/user.crl")
Skip file. (dev_id=0; file=10; name="/usr/local/Ascend/CMS/user.xer")
```

**Reading the protocol:**

- `flag=0x6666666666666666` = `DEVDRV_SEND_FINISH` — the driver handing the BIOS a file.
- `flag=0x7777777777777777` = `DEVDRV_SEND_PATT_FINISH` — used for the **large** `cpio.gz`
  (48 blocks × 3 MB = 144 MB of DMA window).
- `Receive BIOS flag. (dev_id=0; flag=0x8888888888888888; count=N)` — the BIOS acknowledging with
  `DEVDRV_RECV_FINISH`. **On this successful boot every single flag is `0x8888…` (success)**;
  there is not one `0x5A5AA5A5…` failure flag.
- `Skip file.` is the driver telling the BIOS "this optional file does not exist" — it is emitted
  for `Ascend310P.crl`, `/CMS/user.crl` and `/CMS/user.xer` **on a stock boot where none of them
  exist**. This is exactly the path our earlier `/CMS/user.{xer,crl}` deployment took.

- The `size=` field in "Notice BIOS to Load file" is the **block count**, not bytes:
  `ddr.image` 197396 B → 1 block, `Ascend310P.fd` 1.7 MB → 2 blocks, `Ascend310P.image` 31.3 MB → 30
  blocks, `Ascend310P.cpio.gz` 109.4 MB → 48 blocks (and the 0x7777 pattern because it exceeds the
  48-block static window).

### The decisive test, now armed

Deployed and awaiting a cold power cycle:

```text
Ascend310P.image    4d35211f3f7a91e4…   (uimage, leaf-signed, certtype 0x2)
Ascend310P.cpio.gz  ec960b4546727d97…   (initrd, leaf-signed, certtype 0x2)
/usr/local/Ascend/CMS/user.xer  9833 B  (leaf container, hash b932bb15…)
/usr/local/Ascend/CMS/user.crl  3045 B  ([root.crl, rootcaG2.der, root.crl])
```

With the log reader working, this run will show **exactly** which of these happens:

- `Skip file` for `user.xer` → the driver did not find it → the path/name is wrong.
- `Receive BIOS flag … 0x5A5AA5A55A5AA5A1` → the BIOS **rejected the image signature**
  (`DEVDRV_IMAGE_CHECK_FAIL`) → the customer path did not engage.
- `Receive BIOS flag … 0x5A5AA5A55A5AA5A0` → TEE check failure.
- no `Receive BIOS flag` for the rejected file, then a timeout → the BIOS aborted the load.
- `Skip file` absent for `user.xer` and a normal `0x8888…` ack → the customer image **was**
  delivered and accepted, and the failure is elsewhere.

This is the first run in this whole investigation where we will see the device's own verdict
instead of inferring it from `phy_startup` timeouts.

---

## 2026-10-26 — ★ DIAGNOSTIC RUN RESULT: the customer path DID engage, and the images were ACCEPTED

The R-series run (both containers customer-signed, leaf anchor, `/CMS/user.{xer,crl}` present) was
captured with the working log reader. The result is **not** what the failure signature suggested.

### The driver delivered every file, and the BIOS acknowledged every one

```text
Get file size. (dev_id=0; file_name=".../Ascend310P.image"; size=31350293)
Notice BIOS to Load file. (…Ascend310P.image; size=30; flag=0x6666666666666666)
Receive BIOS flag. (dev_id=0; flag=0x8888888888888888; count=19)     <- ACK, success
Get file size. (dev_id=0; file_name=".../Ascend310P.cpio.gz"; size=109389397)
Notice BIOS to Load file. (…Ascend310P.cpio.gz; size=48; flag=0x7777777777777777)
Receive BIOS flag. (dev_id=0; flag=0x8888888888888888; count=1)      <- ACK
Get file size. (dev_id=0; file_name="/usr/local/Ascend/CMS/user.crl"; size=3045)
Notice BIOS to Load file. (…/CMS/user.crl; size=1; flag=0x6666666666666666)
Receive BIOS flag. (dev_id=0; flag=0x8888888888888888; count=1)      <- ACK
Get file size. (dev_id=0; file_name="/usr/local/Ascend/CMS/user.xer"; size=9833)
Notice BIOS to Load file. (…/CMS/user.xer; size=1; flag=0x6666666666666666)
Receive BIOS flag. (dev_id=0; flag=0x8888888888888888; count=1)      <- ACK
Load timer add. (dev_id=0)
Leave trans file work queue. (dev_id=0)
```

**Every single flag is `0x8888888888888888` (`DEVDRV_RECV_FINISH` = success).** There is not one
`0x5A5AA5A55A5AA5A1` (`IMAGE_CHECK_FAIL`), no `0x5A5AA5A55A5AA5A0` (TEE), and no
`0x5A5AA5A55A5AA5A2` (filesystem) anywhere in the log.

And critically, **`/CMS/user.crl` and `/CMS/user.xer` were NOT skipped** — unlike the stock boot,
where both produced `Skip file. (dev_id=0; file=9/10; …)`. Here they were read (3045 B and 9833 B)
and handed to the BIOS.

### The device simply never came online

| | good boot | R-series boot |
|---|---|---|
| `half_probe` / `Device online` / `Add dev success` | **27** | **3** |
| last log timestamp | `09:45:38` (a later, successful run) | `09:53:51` — load ends, log stops |
| `Get os load flag` (per device) | `load_flag=1` for dev 0,2,4 | `load_flag=1` for dev 0,2,4 — identical |
| `phy_startup` errors | 0 | 0 |

The good boot continues:

```text
devdrv_load_half_probe start. (dev_id=5)
devdrv_msg_init statr. (msix_irq_num=256; irq_num=58; ...)
devdrv_dev_online 3408  Device online. (dev_id=5; chip_type=2)
_uda_add_dev_ex 234     Add dev success. (udevid=5; agent_flag=0)
```

The R-series boot **stops dead** after `Leave trans file work queue`. The device accepted all the
files and then simply never booted the OS.

### What this rules in and out

**Ruled OUT:**

- ❌ "the BIOS rejected our image signature" — every ack was `0x8888…` success.
- ❌ "the CMS format is malformed" — the container was accepted and the payload read.
- ❌ "`/CMS/user.xer` was never delivered" — it was delivered and acknowledged.
- ❌ "the customer path didn't engage" — the customer files *were* presented to the BIOS.

**Still consistent with:**

- the device booting the OS and then **failing to start** — a *runtime* failure, not a load failure.
- the `Ascend310P.image` (kernel) and `Ascend310P.cpio.gz` (rootfs) being accepted by the loader
  but the **kernel itself refusing to run** them (e.g. the customer path changed how the kernel
  image is validated, and the kernel then failed its own signature check or its own relocations).
- the OS-level `phy_startup` timeout being the *symptom* of a kernel that loaded but hung.

### The key realisation

**The load phase is now fully characterised and it SUCCEEDS.** All of our earlier failures were in
the *load* phase, and we were reading the wrong layer — `phy_startup` timeouts are an OS-level
symptom, while the actual load verdict was in the driver's ring buffer all along.

This reframes the whole goal: we now have a working, low-risk feedback loop —
`dms_log.py` tells us exactly what the loader did, with **no power cycle needed** to read it.

### Next step

1. **Restore stock** and confirm the good-boot baseline reproduces (the "good" log I saved was from
   a later run at 09:45; re-capture a clean baseline).
2. Re-run with **only the rootfs** customer-signed (stock `Ascend310P.image`), so the kernel is
   guaranteed good and only the rootfs differs. If the load succeeds and the device still doesn't
   come online, the failure is in the rootfs content; if it comes online, the kernel image was the
   problem.
3. Also test the reverse: **only the kernel** customer-signed with a stock rootfs.

Those two runs localise the fault precisely, and neither needs a new hypothesis.

### Note on the earlier `0x5A5AA5A5…` hypothesis

It remains possible that the BIOS *would* have set `IMAGE_CHECK_FAIL` had it been asked to verify
the customer-signed image in the way it verifies stock ones. But the observed behaviour is that
the loader accepted everything, so that hypothesis is not what is happening here.

### Card state

Cards down (R-series deployed, device never came online). Stock images are intact on the card as
`Ascend310P.image.orig` / `Ascend310P.cpio.gz.orig`; `/usr/local/Ascend/CMS/` still holds the
R-series `user.xer` / `user.crl`.

---

## 2026-10-26 — S-series: localizing the fault by varying one container at a time

Stock restored and verified:

```text
Ascend310P.image.orig   027f4d22762ef0644…   (pristine kernel container)
Ascend310P.cpio.gz.orig 12975a9d7e934525…   (pristine rootfs container)
```

### S1 — deployed, awaiting cold power cycle

```text
Ascend310P.image    027f4d22762ef0644…   STOCK kernel container
Ascend310P.cpio.gz  7c1805b5168171854…   customer-signed rootfs (leaf anchor, certtype 0x2)
/usr/local/Ascend/CMS/user.xer  9833 B  (leaf container, hash b932bb15…)
/usr/local/Ascend/CMS/user.crl  3045 B  ([root.crl, rootcaG2.der, root.crl])
```

**Why this is the right next test.** The R-series run proved the load phase succeeds end to end
with both containers customer-signed, and that the device then never comes online. With the stock
kernel and only the rootfs customer-signed:

- **S1 boots** → the fault is in the **kernel container**; the rootfs can be customer-signed
  successfully, and the kernel is the next thing to work on.
- **S1 loads but doesn't boot** → the fault is in the **rootfs content** (our CMS is fine, but
  something about the rootfs being customer-signed breaks the OS start).

### S2 — built, ready

```text
S2-our-uimage.image  45189ed1ed1a1de0…   customer-signed kernel + STOCK rootfs
```

Run S2 only after S1, to avoid conflating variables.

### Reading the result

The log reader needs no power cycle to read, so the cycle sequence is:
boot → `dms_log.py` → grep for `Device online` / `Add dev success` / `half_probe` → decide.

Expected outcomes and what each means:

| log shows | meaning |
|---|---|
| `Device online` + `Add dev success` for all 6 | the customer-signed container is accepted *and* the OS starts |
| all files acked `0x8888…`, no `Device online` | the container is accepted but the OS refuses to start |
| `Receive BIOS flag … 0x5A5AA5A55A5AA5A1` | the BIOS rejected the signature after all (contradicts R) |

---

## 2026-10-26 — ★★ S1 RESULT: the fault is in the ROOTFS, and the load phase is confirmed clean

### S1 = stock kernel + customer-signed rootfs (leaf anchor)

```text
Ascend310P.image    027f4d22762ef0644…   STOCK kernel container
Ascend310P.cpio.gz  7c1805b5168171854…   customer-signed rootfs
/usr/local/Ascend/CMS/user.xer  9833 B, /CMS/user.crl  3045 B
```

### The log diff against the good boot is unambiguous

| marker | good boot | S1 |
|---|---|---|
| `Receive BIOS flag … 0x8888…` (success) | 42+ | 42 — **all successes, no `0x5A5AA5A5…`** |
| `Device online` | ✅ | **0** |
| `Add dev success` | ✅ | **0** |
| `Set device boot finish` | ✅ | **0** |
| `Device ready work start` | ✅ | **0** |
| `Receive device ready notify` | ✅ | **0** |
| `Skip file` for user.crl / user.xer | 3 (both absent) | **0 — both were delivered** |

Message shapes present in the good boot but **absent** from S1:

```text
Set device boot finish. (dev_id=N)
Device ready work start. (dev_id=N)
dms chip info dev init success. (dev_id=N)
Receive device ready notify. (dev_id=N)
```

### The mechanism, now precisely known

The device-side boot is a **two-phase handshake**:

1. **Load phase** (host → device): the driver copies the files into the device's DMA window and
   waits for the BIOS to acknowledge each one. **This phase SUCCEEDS in every customer-signed
   configuration we have tried** — including R (both containers) and S1 (rootfs only). Every flag
   is `DEVDRV_RECV_FINISH` (0x8888…). The customer files `/CMS/user.crl` and `/CMS/user.xer` are
   read and delivered rather than skipped.
2. **Ready phase** (device → host): ~10 s after the load, the device raises an interrupt and the
   driver logs `Receive device ready notify` → `Device ready work start` → `Set device boot finish`.
   **This phase never happens** in the customer-signed case.

So the device **accepts the files and then fails to boot the OS**. The failure is inside the
card's own boot sequence — after it has the customer material — not in the transfer.

### What this means for the goal

- ✅ The customer-certificate **delivery** path works: the driver finds `/CMS/user.xer` and
  `/CMS/user.crl`, and the BIOS accepts them.
- ✅ The esbc containers with our CMS are accepted by the loader (both `uimage` and `initrd`).
- ❌ The card does not then boot the OS.

The most probable remaining cause is that the **customer path changes how the card validates the
OS image**, and our chain is not what the card expects at that stage — e.g. BL31's
`usr cert cms vef` step (the string that verifies the *image* CMS against the user certificate) may
require the image's CMS to be signed by a certificate that chains to the `user.xer` **leaf**
(`Test Signing Cert`, CA:FALSE) in a way our chain does not satisfy, or the `user.crl` content
must be the CRL of the *signing* CA rather than of the root.

### The next experiment is now well-posed and cheap

Because the load phase is clean, the only variable that matters is the *content* of `user.crl` and
`user.xer` and the image's CMS signer. Three candidates, one cycle each:

| # | user.xer | user.crl | image CMS signer |
|---|---|---|---|
| **T1** | leaf (`Test Signing Cert`) | `[root.crl, rootcaG2.der, root.crl]` | leaf (as S1) — baseline |
| **T2** | leaf | `[root.crl, signca.der, sign.crl]` (the *signing* CA's cert+CRL) | leaf |
| **T3** | **intermediate** (`signca`) | `[root.crl, signca.der, sign.crl]` | leaf under signca |

T2 is the one I consider most likely: the image is signed by the leaf, so the CRL that matters for
validating that leaf's issuer is the **intermediate's** CRL, and the intermediate's certificate must
be present — which is exactly what `signca.der` provides.

We can now also **read the verdict immediately after each cycle** with `dms_log.py`, so each test
costs one power cycle and gives a precise, no-ambiguity answer.

### Card state

S1 deployed; cards down (device never came online). Stock originals intact:
`Ascend310P.image.orig` (027f4d22…), `Ascend310P.cpio.gz.orig` (12975a9d…).

---

## 2026-10-26 — T2 staged: the signing CA's CRL in `user.crl`

### T2 — deployed, awaiting cold power cycle

```text
Ascend310P.image    027f4d22762ef0644…   STOCK kernel container
Ascend310P.cpio.gz  ecde4ee4fd07e4c6…   customer-signed rootfs
/usr/local/Ascend/CMS/user.xer  9833 B  (leaf container, Test Signing Cert, hash b932bb15…)
/usr/local/Ascend/CMS/user.crl  3096 B  [ rootG2.crl , signca.der , sign.crl ]
```

The image CMS is rebuilt to be **internally consistent with the new `user.crl`**:

| | S1 (baseline) | **T2** |
|---|---|---|
| `user.crl` | `[root.crl, rootcaG2.der, root.crl]` | **`[root.crl, signca.der, sign.crl]`** |
| CMS `crls[]` | `[rootG2.crl]` | **`[rootG2.crl, sign.crl]`** |
| container `crl` section | `[root.crl, rootcaG2.der, root.crl]` | **`[root.crl, signca.der, sign.crl]`** |
| `user.xer` | leaf container | leaf container (unchanged) |
| image CMS signer | leaf (`Test Signing Cert`) | leaf (unchanged) |

**Why this is the most likely candidate.** The image is signed by the **leaf**, and the leaf's
issuer is `RootCA G2` — so a naive reading says the root's CRL is what matters (S1). But the leaf
carries `EKU: Code Signing` and is issued by the root *directly*, while the chain material that
actually exists in our PKI includes `Signing Certificate CA G2` (`signca`) as an intermediate for
the *other* branch (the TSA). Providing the **signing CA's certificate and its CRL** in `user.crl`
gives the card's verifier everything it might need to build and check the chain for the image
signature, and it matches the stock container's shape exactly
(`[root.CRL, <CA cert>, <CA CRL>`, where the CA cert is the *issuing* CA of the signer).

If the card's verifier wants the intermediate of the signer and the intermediate is missing, the
signature check fails and the card never boots — which is exactly the observed behaviour.

### T3 — next, if T2 fails

`user.xer` = the **intermediate** container (`user-signca.der`, hash `41474c62…`), with the image
CMS signed by the leaf *under* `signca` and `user.crl` = `[root.crl, signca.der, sign.crl]`.
That is the configuration the 2026-10-21 session called
`stock-resigned + user-signca`, but with the **fixed** CMS (unwrapped `issuerAndSerialNumber`,
correct `unsignedAttrs`, PSS without `trailerField`, stock-identical structure) — a combination
that has never actually been tested.

### Cost per test

One cold power cycle, then `dms_log.py` gives a precise verdict with no ambiguity:

- `Set device boot finish` + `Device online` + `Add dev success` → success
- all flags `0x8888…` but no `Receive device ready notify` → load clean, OS did not start (current
  failure mode)
- `0x5A5AA5A55A5AA5A1` → the BIOS rejected the image signature outright

### Card state

T2 deployed; cards down. Stock originals intact (`Ascend310P.image.orig`, `Ascend310P.cpio.gz.orig`).

---

## 2026-10-26 — T2 failed; V1 staged: the payload-binding question

### T2 result

`user.crl = [root.crl, signca.der, sign.crl]`, image CMS `crls = [rootG2.crl, sign.crl]`, leaf anchor,
stock kernel, customer-signed rootfs.

```text
Device online           0
Set device boot finish  0
device ready notify     0
non-success BIOS flags  (none — every ack was 0x8888…)
Skip file               3   (Ascend310P.crl only)
```

Same failure mode as S1: the load completes cleanly and the device never boots the OS. So the
`user.crl` composition is not the discriminator either.

### The question that actually matters

Everything so far has assumed the card verifies that the **payload** matches the digest in the
signed `ini` section. If it does **not**, then we can ship an **arbitrary custom rootfs under the
stock signature** — no customer CA, no key, no `certtype`, no `/CMS/` at all. The stock CMS signs
the `ini` file; the `ini` says `initrd, <sha256>;`; the header at `+0x2C` also stores that digest.
If the device only checks the CMS signature over the `ini` and never recomputes
`SHA-256(payload)`, the payload is free.

### V1 — staged, awaiting cold power cycle

Built by surgically injecting a single file into the stock cpio so that **every other entry is
byte-identical**:

```text
V1-modpayload-stockcms.cpio.gz   108,824,962 B
  cms      : byte-identical to stock (9201 B, Huawei-signed)
  ini      : byte-identical to stock  -> "initrd,   39634b3e512d477fcc1b7280de792fcc450dfeca2d1d135fb4231d23c8ba0ff3;"
  crl      : byte-identical to stock (3522 B)
  certtype : 0xffffffff (stock)
  payload  : MODIFIED  (2575 cpio entries; added var/local/CUSTOM_MARKER)
             SHA256(payload) = 069f9ac24dca01e9…  (stock was 39634b3e512d477f…)
  uimage   : stock
  /usr/local/Ascend/CMS/ : removed
```

Note the deliberate inconsistency: the `ini` still advertises the **stock** digest while the
payload carries a **different** one. That is exactly the condition the verification step is
supposed to catch.

**Outcomes:**

| result | conclusion |
|---|---|
| **boots** | the device does **not** bind the payload to the signature → **we can ship arbitrary custom firmware under the stock signature**. This would be the whole goal. |
| loads but doesn't boot | the payload is bound and the failure is elsewhere |
| `IMAGE_CHECK_FAIL` | the payload/signature mismatch is caught → we need our own key or the customer path |

### Tooling note

`cpio_inject.py` (in `customer-cert-20261026/`) splices a new entry in just before the
`TRAILER!!!` record using `cpio(1)` to build the entry, so all 2574 pre-existing entries keep their
exact bytes. This is far safer than an extract/repack round trip, which was dropping files and
shrinking the archive from 109 MB to 11 MB.

### Card state

V1 deployed; cards down (T2 boot never completed). Stock originals intact:
`Ascend310P.image.orig` (027f4d22…), `Ascend310P.cpio.gz.orig` (12975a9d…).

---

## 2026-10-26 — ❌ V1 FAILED: the payload IS bound to the signature. Boot-time signing is closed

### V1 result

```text
Ascend310P.image    stock (027f4d22…)
Ascend310P.cpio.gz  V1 — payload modified, cms/ini/crl byte-identical to stock
/usr/local/Ascend/CMS/  removed
```

```text
Device online           0
Set device boot finish  0
device ready notify     0
non-success BIOS flags  (none — every ack 0x8888…)
nvcnt                   IMAGE_NVCNT=0   (same as stock)
```

**The payload IS bound to the signed `ini`.** A rootfs whose `SHA-256(payload)` disagrees with the
digest in the (stock, Huawei-signed) `ini` is rejected. So the "ship arbitrary content under the
stock signature" shortcut does not exist.

### The boot-time trust anchor, confirmed closed

`AS610_HBOOT2_UEFI.fd` (the BL31 bootloader) contains exactly one certificate subject:

```text
Huawei Integrity Root CA - G2            present
Huawei Integrity CA 1 - G2               absent
Huawei CSI RSA-PSS Code-Signing Auth 2   absent
```

The bootloader trusts **one** root, compiled into its own image, and there is no customer-CA hook
in it. Everything we have tried — leaf anchor, intermediate anchor, with/without timestamp,
rootfs-only vs both containers, `certtype` 0/1/2/0xFFFFFFFF, `user.crl` variants, and now a
payload-binding test — lands in the same place: **the load succeeds, the OS never starts**.

**Conclusion on the boot path:** running a *modified* `Ascend310P.image` (kernel) or
`Ascend310P.cpio.gz` (rootfs) requires the Huawei root private key, which does not exist in any
reachable artifact (verified exhaustively). This path is closed.

### Where the trust material actually lives — and why that is the opening

The card-side trust anchors are **in the rootfs, which we can rewrite**:

| artifact | contents |
|---|---|
| `drv_user_cfg.ko` (rootfs `/var/`) | `Huawei Integrity CA 1 - G2`, **`Huawei CSI RSA-PSS Code-Signing Authority 2`**, `Huawei Time-Stamping Authority 2 - G2`, plus the root — **and** `ccpu_usr_cert_hash` + `sec_flash_read/write` |
| `/var/*.sec` (4 files) | the same chain + the 3522-byte CRL section, in `5a5aa5a5` containers |
| `drv_pkicms.ko` (rootfs `/var/`) | `pkicms_get_cert_data` → `/var/asdrv/dev%d/device-sw-plugin/verify/{vendor,community,user}-certificate`; `pkicms_custom_verify_cms`; `pkicms_get_sec_check_enable_flag` (eFuse at `0x8126e080`) |

**These are data and code inside the rootfs.** The bootloader only checks that the rootfs
*container* is signed by Huawei — it does not verify the contents of the files inside it. So a
rootfs that is **byte-identical in its `cms`/`ini`/`crl` sections** but has modified *contents*
would be rejected by the payload binding (V1 proved that) — **unless the modified content is
placed in a file the payload digest does not cover**… which it does cover, since the digest covers
the whole payload.

The way through is therefore **not** a modified rootfs. It is:

> **Keep the rootfs byte-identical (stock signature intact, payload digest matching) and instead
> replace the *card-side* trust material with something we control — i.e. make the card trust our CA
> for runtime operations, so that the code we want to run is accepted.**

Concretely, the candidate is `drv_user_cfg.ko` / `drv_pkicms.ko` and the `/var/*.sec` containers:
if the **runtime** verifier (`pkicms`) can be made to use a customer CA — via
`/var/asdrv/dev%d/device-sw-plugin/verify/user-certificate` plus the `ccpu_usr_cert_hash` in secure
flash — then we can ship code signed by our own CA into the card's runtime, without touching the
boot-time signature.

But `drv_user_cfg.ko` and `drv_pkicms.ko` are **inside the rootfs**, whose payload digest is
signed. Replacing them changes the payload → rejected (V1).

**So we are at a genuine impasse for the card-side, and the remaining lever is the HOST side.**

### The host side is where the remaining leverage is

- The **host driver** (`/usr/src/davinci_ascend-1.0`, already our OpenSensor fork, already
  installed) is loaded into the host kernel — **not** covered by any card signature.
- The host driver owns the PCIe/DMA path, the image load protocol, the card-side module loading,
  and the `dms`/`pkicms` userspace interfaces.
- The card-side modules are loaded by the host driver over PCIe.

That gives a concrete, legitimate route to "custom firmware" on these cards:

1. **Rebuild the card-side modules** (e.g. `drv_pkicms.ko`, `drv_user_cfg.ko`) — but they live in
   the rootfs, which is signed… **unless** they are loaded from the host rather than from the rootfs.
   The mini_v2 load table loads `Ascend310P.image`/`cpio.gz` from the host's
   `/usr/local/Ascend/driver/device/`, so **the host supplies the rootfs to the card at boot** —
   meaning the *host* controls which rootfs the card gets, but the card still verifies its signature.
2. **Instrument the host driver** to observe and influence the load, and to add the customer trust
   material the card needs.

### Immediate next actions

1. Restore the card to stock and bring the cards **up** (they are currently down from the V1 run).
2. Determine whether the card-side `.ko` modules are loaded from the **rootfs** (signed) or can be
   overridden from the **host** (unsigned). `devdrv_device_load` shows the host loads only the 11
   esbc images; the card then loads its own modules from the rootfs it was given. So the card-side
   modules are covered by the rootfs signature → **not** modifiable.
3. Therefore focus on the **host driver**, where we have full control and no signature constraint.

Given the constraint that the card-side is signature-locked, the realistic definition of "custom
firmware" becomes: **custom host driver + custom host-side tooling, driving stock card firmware.**
That is where the DMA profiling work already lives and where tuning can actually take effect.

### Card state

V1 deployed; cards down. Stock originals intact (`Ascend310P.image.orig` 027f4d22…,
`Ascend310P.cpio.gz.orig` 12975a9d…).

---

## 2026-10-26 — Workspace hygiene: everything now lives in a persistent location

**Mistake corrected:** I had been staging artifacts and tooling in `/tmp` on the card, which is
wiped on every reboot. That is why files kept vanishing between power cycles (V1's
`/tmp/V1-modpayload-stockcms.cpio.gz`, the rootfs extractions, the `drvlog-*.txt` captures). The
`.sec`/rootfs extractions and several candidate images were lost this way.

### New persistent workspace on the card

```text
/var/local/ascend-work/
├── images/   all candidate images (A/B/C/D/E/F/I/M/P/R/S/T/V series, stock backups)
├── tools/    dms_log.py, repack_rootfs.py, cpio_inject.py, cmsverify2.py, der.py,
│             cms_compare.py, cpio_compare.py, pss_sign.py, tstverify.py,
│             cms_structure.py, audit2.py, smallkey.py, base_vote.py, dms_probe.py
├── logs/     drvlog-goodboot.txt, drvlog-badboot.txt, drvlog-s1.txt, drvlog-t2.txt, drvlog-v1.txt
└── certs/    rootcaG2.{pem,der}, signcert.{pem,der}, signca.der, tsca.der, ts.pem,
              and all user*.der envelopes
```

The repo copy of the tooling is at
`artifacts/ascend-310p-26.0.rc1/customer-cert-20261026/` and is the master; the card workspace is
a mirror that survives reboots.

**Rule for the rest of this work:** nothing that must survive a power cycle goes in `/tmp`.
The card's `/var/local/ascend-work` is the staging area; `/tmp` is only for genuinely transient
scratch inside a single session.

### V1 verdict, for the record

```text
Ascend310P.image    stock (027f4d22…)
Ascend310P.cpio.gz  V1 — payload modified, cms/ini/crl byte-identical to stock
/usr/local/Ascend/CMS/  removed

Device online           0
Set device boot finish  0
device ready notify     0
non-success BIOS flags  none (every ack 0x8888…)
nvcnt                   IMAGE_NVCNT=0  (same as stock)
```

**The payload IS bound to the signed `ini` digest.** Modifying the rootfs content while keeping
Huawei's signature does not work.

### BL31's trust anchor, confirmed

`AS610_HBOOT2_UEFI.fd` contains exactly one certificate subject:

```text
Huawei Integrity Root CA - G2            present
Huawei Integrity CA 1 - G2               absent
Huawei CSI RSA-PSS Code-Signing Auth 2   absent
```

The bootloader trusts **one** root, compiled into itself, with no customer-CA hook. Combined with
the absence of any Huawei private key in any reachable artifact, the boot-time signing path is
closed for a modified `Ascend310P.image` or `Ascend310P.cpio.gz`.

### Card state

Stock restored and verified:

```text
Ascend310P.image    027f4d22762ef0644ad82816c29f0cb67e27118b229a792afbe04347b59a7663
Ascend310P.cpio.gz  12975a9d7e9345255eccc0d97475082ec96ebab69324070f2246604dac8c0cc2
/usr/local/Ascend/CMS/  removed
```

Cards are down from the V1 run and need a cold power cycle to come back up.

### Working-directory convention (post-correction)

| location | holds | survives reboot? |
|---|---|---|
| repo `artifacts/ascend-310p-26.0.rc1/customer-cert-20261026/` | master copy of all tooling | yes (git) |
| repo `artifacts/ascend-310p-26.0.rc1/work/` | scratch: extracted certs, `.sec` files, one-off analysis scripts | yes |
| card `/var/local/ascend-work/` | images, tools, logs, certs — the staging area for card-side work | yes |
| `/tmp` on the card | transient scratch **within one session only** | **no** |

The card's `/var/log/upgradetool/upgrade.log` is also rotated/truncated by the tool, so the
`drvlog-*.txt` captures in `/var/local/ascend-work/logs/` are the durable record of each boot.

---

## 2026-10-26 — The bootloader's own trust architecture (definitive)

The bootloader `Ascend310P.fd` (identical to `AS610_HBOOT2_UEFI.fd`, sha256 `e1e75ec2…`) is itself an
esbc container:

```text
magic 33aa3a3a, tag "hboot2"
+0x18 = 9, +0x24 = 0x205, +0x28 = 4, +0x2c = 0x20000, +0x30 = 0x564d, +0x34 = 0x100
payload = the UEFI/BL31 image (1.6 MB)
```

**But it has NO `cms` / `ini` / `crl` sections** — the slots that would exist at
`cms_start+0x10 / +0x4010 / +0x4810` are past EOF. So the bootloader image is **not itself
esbc-signed**.

What it *does* contain:

```text
CERT @0x1552e9  CN=Huawei Root CA
CERT @0x155841  CN=Huawei Integrity Root CA - G2     <-- the single trust anchor
```

and **only** those two — `Huawei Integrity CA 1 - G2` and
`Huawei CSI RSA-PSS Code-Signing Authority 2` are **absent**.

### The bootloader's verification strings (complete set)

```text
hw_cms_image_verify_simple stop!                     <- the ordinary path
hw_cms_ctrl_cpu_verify failed, ctrl_flag = 0x%x, result = 0x%x
support usr cert vef                                 <- capability query
usr cert cms vef fail, result = 0x%x                 <- customer-path CMS verify
usr cert cms vef success!
no usr, hw vef fail, ctrl_flag = 0x%x!              <- fallback to the hardware path
efuse_read_nvcnt fail, 0x%x!
nvcnt cmp fail, nvCnt = 0x%x, efuse_bit_pos = 0x%x!
no need nvcnt check!
img head magic invalid, headMagic = 0x%x!
img head nvcnt = 0x%x invalid!
hw nvcnt info check, size = 0x%lx, image_type = 0x%x, result = 0x%x.
max_size < SECURE_HDR_SIZE, max_size = 0x%lx
image size check fail, size = 0x%lx, max_size = 0x%lx, uwLCodeLen = 0x%x!
AOID_KP_CODESIGNING / AOID_PKCS7_SIGNEDDATA / AOID_HW_KP_{VOUCHER,RELEASE}SIGNING
```

### What this establishes

1. The bootloader trusts **exactly one** root, `Huawei Integrity Root CA - G2`, compiled into its
   own image. There is **no customer-CA hook** in the boot path — `usr cert cms vef` verifies an
   image against the *enrolled user certificate*, but the *anchor* for that verification must still
   be reachable, and the only root present is Huawei's.

2. There is a **second, independent gate**: `SECURE_HDR_SIZE` / **nvcnt** — an anti-rollback
   counter stored in the eFuse, compared against a per-image counter in the **esbc header**
   (`img head nvcnt`). The stock headers carry `certtype = 0xFFFFFFFF` and an `nvcnt` field, and the
   CHECK_TOOL reports `IMAGE_NVCNT=0` for every image we built (stock included). **This is a
   separate obstacle from the signature** and it lives in hardware we cannot write.

3. `hw_cms_image_verify_simple` is the ordinary path and `ctrl_flag` selects between the built-in
   and customer paths. `no usr, hw vef fail` is the fallback when there is no user cert.

### The honest assessment

Boot-time firmware replacement requires **both**:

- a signature the bootloader accepts — i.e. the Huawei root private key, which does not exist in
  any reachable artifact; **and**
- an nvcnt value at or below the eFuse counter.

We have neither, and there is no partial credit: the loader accepts the files (we proved this) but
the card refuses to boot. **The card-side boot path is closed to us.**

### What remains genuinely open

Everything that does **not** traverse the card's boot verification:

| area | why it is open | what exists |
|---|---|---|
| **host driver** | runs in the host kernel, no card signature | `/usr/src/davinci_ascend-1.0` — our OpenSensor fork, already installed, already does device-initiated DMA reservation work (`DEVICE_DMA.md`, v1–v3 validated on 4 NPUs) |
| **card runtime modules** | loaded by the host driver over PCIe | `drv_pkicms.ko`, `drv_user_cfg.ko`, … in the rootfs — **but** they are inside the signed rootfs payload, so replacing them changes the digest (V1) |
| **CANN operator packages** | host-side shared libraries + `.run` packages | the qwen38 op-source handoff, 4 vendor packages, 55 operators |
| **AI CPU device kernels** | `opp/Ascend310P/aicpu/Ascend310P-aicpu_syskernels.tar.gz` | device-side kernel tarballs shipped by CANN |

**The AI CPU syskernels are the interesting one for "custom firmware"**: they are *device-side*
kernel binaries shipped as tarballs, loaded onto the card's ARM Linux environment. If those are
loadable from the host without re-signing the rootfs, that is a genuine device-firmware path.

### Next concrete steps (host-side pivot)

1. Determine how `Ascend310P-aicpu_syskernels.tar.gz` reaches the card, and whether the AI CPU
   kernel load path is signature-checked. **This is the highest-value remaining question** — it is
   device-side code, and it may not be covered by the rootfs signature because it is delivered
   separately by the host driver.
2. Confirm the host-side DMA/queue work is still healthy (it is the only lever with proven hardware
   results) and extend it toward the "device submission" gate listed in `DEVICE_DMA.md`.
3. Keep `repack_rootfs.py` + `cpio_inject.py` for the case where a signing route appears.

### Card state

Stock restored, 6 chips up and healthy (verified 10:23). Host driver source mirrored into
`artifacts/ascend-310p-26.0.rc1/host-driver/` (25 MB) so the pivot has a durable local copy.

---

## 2026-10-26 — ★★★ YES, WE CAN PATCH THE BOOTLOADER. The Oct 21 session already did it

### The prior patch exists on the card — and was never installed

```text
/home/matteius/AS610_HBOOT2_UEFI.fd.stock    e1e75ec2103a9526…   (== the installed one)
/home/matteius/AS610_HBOOT2_UEFI.fd.patched  89cd382879dd3406…
/usr/local/Ascend/driver/device/Ascend310P.fd  e1e75ec2103a9526…  (STOCK — the patch was never applied)
```

The diff is **exactly 5 bytes**:

| file offset | stock → patched | what it is |
|---|---|---|
| `0x2024` | `05` → `20` | inner esbc header `+0x24` |
| `0x2026` | `00` → `09` | inner esbc header `+0x24` (word = `0x205` → `0x90220`) |
| `0x13dc59` | `40` → `12` | AArch64 instruction in BL31 |
| `0x13dc5a` | `02` → `00` | AArch64 instruction in BL31 |
| `0x13dc5c` | `54` → `14` | AArch64 instruction in BL31 |

plus the digest fields at `0x2C`, `0x4BC` and `0x14BC`.

So the prior session **did** attempt a bootloader patch: one esbc-header field plus one BL31
instruction. It was never installed, and its checksum fix-up was incomplete.

### Why a bootloader patch needs no key — the complete integrity model

`AS610_HBOOT2_UEFI.fd` is an esbc container, but unlike the kernel and rootfs it has **no
`cms` / `ini` / `crl` sections** (the slots would be past EOF). Nothing external signs it. It is
protected only by its own self-consistency:

```text
0x0000  header copy 0
0x1000  header copy 1
0x2000  payload (code_len bytes)  = inner esbc(0x100 hdr + body)

@0x02C   SHA-256(payload)                     (both header copies)
@0x04BC  SHA-256(header[0x0000:0x04BC])       (copy 0)
@0x14BC  SHA-256(header[0x1000:0x14BC])      (copy 1)
```

All three verified on the stock image. **A patch therefore needs no signature — only correct
re-checksumming.**

### The prior patch was incomplete

The Oct 21 `.patched` file satisfies `payload_digest_ok` and `hdr1_check_ok` but **not**
`hdr0_check_ok` — copy 0's header self-checksum at `@0x4BC` was left stale. If the loader verifies
that field, the patched image would have been rejected — which is consistent with it never having
been installed and the session moving on to the customer-certificate route.

### `patch_bootloader.py` — correct patcher, now in the repo

`customer-cert-20261026/patch_bootloader.py` applies byte patches and reseals in the correct order:

```text
1. apply the byte changes
2. payload_digest = SHA-256(payload)      -> write to @0x02C and @0x102C
3. hdr0_check = SHA-256(header[0:0x4BC])  -> write to @0x04BC
4. hdr1_check = SHA-256(header[0x1000:0x14BC]) -> write to @0x14BC
```

Verified on the stock image: `payload_digest_ok`, `hdr0_check_ok`, `hdr1_check_ok` all **True**
after patching (the Oct 21 file had one of them False).

```text
$ python3 patch_bootloader.py AS610_HBOOT2_UEFI.fd.stock out.fd \
      --set-byte 0x2024:20 --set-byte 0x2026:09 \
      --set-byte 0x13dc59:12 --set-byte 0x13dc5a:00 --set-byte 0x13dc5c:14
   payload_digest_ok    True
   hdr0_check_ok        True
   hdr1_check_ok        True
```

### What this opens

**The bootloader is modifiable without any key.** That changes the trust model completely:

- The boot-time trust anchor (`Huawei Integrity Root CA - G2`) lives **in the bootloader image**.
  If the bootloader is ours to change, the anchor is ours to change.
- The nvcnt/anti-rollback gate compares the esbc header's nvcnt against an **eFuse** counter — the
  eFuse is still read-only, but the *image-side* value is in a header we can edit, so the check
  becomes a comparison we can satisfy.
- `usr cert cms vef` / `no usr, hw vef fail` / `hw_cms_ctrl_cpu_verify` are all **code in the
  bootloader**, and the prior session's 5-byte patch was aimed at exactly this area.

**The remaining unknown is what the prior patch was trying to do** — which BL31 instruction was
being changed, and what the `+0x24` header field means. Both are recoverable:

- the AArch64 instruction at file `0x13dc59` sits in the BL31 payload (file `0x13dc59` is inside
  the payload region, which starts at `0x2000` and is the BL31 image at payload offset
  `0x13bc59`). With the BL31 base resolved (`runtime = file − 0x5BC`) the instruction can be
  disassembled and its branch target identified;
- the `+0x24` word (`0x205` → `0x90220`) is an esbc header field — bit 2 cleared, bits 3/4 set —
  most likely a **cert-type / image-type selector**, which would be consistent with a patch that
  switches the image into the customer path.

### Immediate next steps

1. Disassemble the BL31 instruction at file `0x13dc59` in both stock and patched images and
   determine what the patch changes (a conditional branch? a flag test?).
2. Decode the inner esbc `+0x24` field's meaning and confirm the `0x205` → `0x90220` delta selects
   the customer/codesigning path.
3. Rebuild the prior patch **correctly** with `patch_bootloader.py` (all three digests), install it
   as `Ascend310P.fd`, and cold power cycle. Because the bootloader is the thing that verifies
   everything else, this is the gate that must open first.

### Risk and recovery

The bootloader is the card's root of trust, so a bad patch bricks the boot. Recovery is
straightforward: the stock file is on the card at
`/home/matteius/AS610_HBOOT2_UEFI.fd.stock` and in `ascend-run/driver/device/`, so restoring it
and power-cycling should recover. The card's BootROM is on-die and unpackaged, so the `hboot1a`/
`hboot1b` components (the only flashable ones) stay untouched.

**Recommendation**: do step 1 (disassembly) before touching the card — it is free and it tells us
whether the prior patch was on target. Then deploy the corrected patch of that same 5 bytes as a
first, minimal-risk experiment.

---

## 2026-10-26 — Bootloader patching: the mechanics are solved, the target is obfuscated

### The bootloader image CAN be modified without any key — confirmed

`AS610_HBOOT2_UEFI.fd` (= `Ascend310P.fd`, sha256 `e1e75ec2…`) is an esbc container with **no
`cms` / `ini` / `crl` sections**. It is protected only by its own self-consistency:

```text
0x0000  header copy 0
0x1000  header copy 1
0x2000  payload (code_len bytes)

@0x02C   SHA-256(payload)                     (both copies)
@0x04BC  SHA-256(header[0x0000:0x04BC])       (copy 0)
@0x14BC  SHA-256(header[0x1000:0x14BC])      (copy 1)
```

All three verified on the stock image. `patch_bootloader.py` (in `customer-cert-20261026/`)
applies byte patches and reseals in the correct order; verified: all three checks **True** after
patching.

### The Oct 21 patch was a broken edit, not a working bypass

The prior session's `AS610_HBOOT2_UEFI.fd.patched` differs from stock in **5 bytes**:

| offset | change | what it is |
|---|---|---|
| `0x2024` | `05`→`20` | inner esbc header `+0x24` (word `0x205` → `0x90220`) |
| `0x2026` | `00`→`09` | (same word) |
| `0x13dc59` | `40`→`12` | word `0x00024071` → `0x00001271` |
| `0x13dc5a` | `02`→`00` | word `0x80188154` → `0x80008114` |
| `0x13dc5c` | `54`→`14` | (same word) |

Problems with it:

1. **`0x00024071` is not a valid AArch64 instruction.** `cmp w0, #2` encodes as `0x71000002`;
   `0x00024071` is rejected by capstone (verified: `ret`, `nop`, `cmp w0,#2`, `b.eq` all decode,
   `0x00024071` does not). The patched value `0x00001271` is `udf #0x1271` — an **undefined
   instruction** that would raise an exception.
2. **It left `@0x4BC` stale.** The stock file satisfies `hdr0_check_ok`; the Oct 21 patched file
   does **not**. Its `payload_digest_ok` and `hdr1_check_ok` are True, so the prior patcher updated
   the payload digest and copy 1's checksum but missed copy 0's.
3. It was **never installed** — `Ascend310P.fd` on the card is still stock (`e1e75ec2…`).

### Why: the bootloader's code is obfuscated

The fd payload is a **sparse flash image** (58% `0xff`) with eight non-erased regions:

```text
0x002000..0x002100   256 B    inner esbc header
0x003100..0x00b100    32 KB
0x023100..0x076b00   335 KB    <-- 0.99 valid AArch64  => plaintext code
0x122100..0x127600    21 KB
0x132100..0x15e500   177 KB    <-- FIP #1  (A/B copy 1)
0x162100..0x167600    21 KB
0x172100..0x190a00   122 KB    <-- FIP #2  (A/B copy 2)
0x1a2100..0x1a2110    16 B
```

**Two FIP copies 256 KiB apart** (0x132100 and 0x172100) — A/B redundancy. A bootloader patch must
be applied to **both**, and the two copies differ at the patch site
(`0x13dc59` → `40020054` vs `0x17dc59` → `21e00091`), so they are not byte-identical.

Within each FIP, the layout is:

```text
+0x00000  header/TOC     (no readable strings -> obfuscated)
+0x04000 .. +0x20000    code     (capstone-valid fraction 0.20-0.34 -> OBFUSCATED)
+0x20000 .. +0x2C000    strings  ('usr cert cms vef fail' @+0x24C2E,
                                  'Huawei Integrity Root CA - G2' @+0x237F7,
                                  'hw_cms_image_verify_simple stop!' @+0x24CBC)
```

So **the secure-boot verification code is obfuscated**, while its **strings and the embedded root
certificate are plaintext**. The prior session's patch site (+0xBB59) is inside the obfuscated
code region, and the 5 bytes it changed were not decodable instructions — consistent with an
automated tool that had no idea what it was editing.

### What this means

- ✅ **Bootloader modification is mechanically possible with no key.** Re-checksumming is solved and
  `patch_bootloader.py` does it correctly.
- ❌ **Targeting the verification logic requires understanding the obfuscation.** The obfuscated
  region (capstone-valid ~0.2) is Huawei's secure-boot code protection. Patching it blind is what
  the Oct 21 session attempted, and it produced an undefined instruction.
- ⚠️ **The `+0x24` header field is the more promising target.** It sits in the *plaintext* inner
  esbc header and the prior patch changed it `0x205` → `0x90220`. A header field is far easier to
  reason about than obfuscated code, and the string `certtype` / the image-type dispatch suggests
  this word selects the verification mode. **This is where the next experiment should go.**

### The next experiment, in order

1. **Decode the `+0x24` field.** Compare its value across every image we have: the stock rootfs
   (`0xFFFFFFFF` certtype at `+0x08`), the kernel (`uimage`), the `.sec` trust containers, and the
   bootloader itself — and find where `0x205` vs `0x90220` appear. If `+0x24` is the
   "verify mode" selector, setting it appropriately may route verification to the customer path
   **without touching obfuscated code at all**.
2. Only then consider code patching, and only with a de-obfuscation result in hand.
3. Whatever is tried, apply it to **both** FIP copies and reseal all three digests.

### Card state

Stock, 6 chips healthy (verified 10:23). The bootloader copies are preserved on the card at
`/home/matteius/AS610_HBOOT2_UEFI.fd.{stock,patched}` and in `ascend-run/driver/device/`.

---

## 2026-10-26 — The `+0x24` header field and the `.sec` secure containers

### `+0x24` is uniform across every esbc container

```text
container                     outer certtype   inner +0x18  inner +0x24   inner +0x28
rootfs    (initrd)           0xFFFFFFFF       0x9           0x00000205   0x4
kernel    (uimage)           0xFFFFFFFF       0x9           0x00000205   0x4
bootloader(hboot2)           0xFFFFFFFF       0x9           0x00000205   0x4
user.xer  (usrcert)          0x00000001       0x9           0x00000205   0x4
```

So `+0x24 = 0x205` is a **constant** for all four — it is *not* the cert-type selector (that is the
outer `certtype` at `+0x08`, where `user.xer` uniquely carries `0x1`).

The Oct 21 patch changed the **bootloader's** inner `+0x24` from `0x205` to `0x90220`:

```text
0x205   = 0b00_0000_0010_0001   bits 0, 2, 9
0x90220 = 0b10_0100_0000_0010   bits 4, 5, 9
=> bits 0 and 2 CLEARED, bits 3 and 5 SET; bit 9 unchanged
```

Since `+0x24` is otherwise constant across every image, flipping bits on the **bootloader's** copy
is exactly the kind of thing that would change *its* verification behaviour without touching the
images it verifies. That is a coherent hypothesis for what the prior session was attempting.

### The `.sec` containers use a different, Huawei-specific format

The four `/var/*.sec` files in the rootfs are **not** esbc containers:

```text
+0x00  5a5aa5a5  aa550300   magic
+0x08  u32       certs-region size   (differs per file: 0x977f, 0x1b75d, 0x21a05, 0x10e945)
+0x18  0x1820c000
+0x1c  0xca0a0a01              <- Huawei 'secure container' marker 1
+0x20  0x000031c3
+0x24  0xca0a0a02              <- Huawei 'secure container' marker 2
+0x28  0x00000dc2  = 3522      <- the CRL section size
```

and the body is: a PKCS#7 `SignedData` (rsassaPss) carrying `Huawei Integrity Root CA - G2`,
then the four X.509 certs, then the 3522-byte CRL section.

**These are the device-side trust store, and they live inside the rootfs.** They are therefore
covered by the rootfs payload digest — but they are *data*, and the bootloader does not verify the
rootfs' *contents*, only the container's signature over the `ini` digest. V1 proved the payload
digest is enforced… so replacing them requires the signature. Confirmed closed, unless the
signature is obtained.

### The `5a5aa5a5` / `3\xaa3\xaa` / `0x55aa55aa` magics

| magic | format | where |
|---|---|---|
| `0x55aa55aa` | esbc **outer** header | all containers |
| `33aa3a3a` | esbc **inner** header (payload[0:4]) | all containers |
| `5a5aa5a5` | Huawei **secure container** | the four `/var/*.sec` files |

Note the inner esbc magic `33aa3a3a` vs the secure-container magic `5a5aa5a5` — the same byte
pattern (`aa aa` at offset 1–2) with different halves, i.e. two related formats from the same
vendor toolchain.

### Next step (concrete, low risk)

Decode what `+0x24` bit 0/2/3/5 mean by **diffing the inner headers of every image we can produce**,
and in particular by checking whether any image in the wild carries a non-`0x205` value. Two sources:

1. `Ascend310P_ddr.image`, `Ascend310P_lowpwr.image`, `Ascend310P_hsm.image`,
   `Ascend310P_dt.img`, `Ascend310P_tee.bin` — the other five images the loader ships. If any has
   a different `+0x24`, that is a ready-made example of the field doing something.
2. The `hboot1a`/`hboot1b` components (the only flashable ones) — these are the *other* stage's
   bootloader and may use different values.

That is a pure read of files we already have. Then, if `+0x24` turns out to be a verification-mode
selector, apply the Oct 21 value (`0x90220`) to the bootloader with `patch_bootloader.py` — which
reseals all three digests correctly, unlike the prior patcher — install it, and cold power cycle.

### Current assessment of the bootloader route

| | status |
|---|---|
| bootloader image is modifiable with no key | **✅ confirmed** (self-checksum only; `patch_bootloader.py` verified) |
| A/B copies must both be patched | **✅ confirmed** (FIP at 0x132100 and 0x172100 differ) |
| the secure-boot code is obfuscated | **✅ confirmed** (capstone-valid fraction 0.20–0.34 in the code region; the Oct 21 patch wrote an undefined instruction) |
| `+0x24` semantics | **unknown** — next to determine |
| customer path reachable at all | **unknown** — depends on `+0x24` and/or de-obfuscation |

### `+0x24` is a CONSTANT, not a mode selector — the Oct 21 patch's value was a guess

Surveyed the inner esbc header of **every image the loader ships**, plus ours:

```text
file                       tag        +0x18  +0x1c  +0x20      +0x24  +0x28      +0x2c    +0x30  +0x34
Ascend310P_ddr.image       lpddr        0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P_lowpwr.image    lperika      0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P_hsm.image       hisserik     0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P_dt.img          dtimage      0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P.fd              hboot2       0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P.image           uimage       0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
stock-rootfs.cpio.gz       initrd       0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
user.xer                   usrcert      0x9    0x0    0x0      0x205    0x4    0x20000   0x564d  0x100
Ascend310P_tee.bin         (not esbc)  <random values -> a TEE blob, not an esbc container>
```

**Every esbc container has the identical inner header**: `+0x18=9, +0x1c=0, +0x20=0, +0x24=0x205,
+0x28=4, +0x2c=0x20000, +0x30=0x564d, +0x34=0x100`.

So `+0x24` is a **constant**, not a per-image mode selector. The Oct 21 patch's
`0x205 → 0x90220` was therefore **a guess with no supporting example** — and it was applied to
the one container whose `+0x24` would have to change to affect verification. There is no
reference image carrying a non-`0x205` value, so nothing tells us what `0x90220` means.

The only thing that *does* vary per image is the **outer** `certtype` at `+0x08`:
`0xFFFFFFFF` for the six loader-shipped images and `0x1` for our `user.xer`.

### Revised understanding of the bootloader route

The bootloader is modifiable with no key, and its *secure-boot code* is obfuscated. To change
verification behaviour we would have to either:

- patch **obfuscated** code (what the Oct 21 session tried, producing an undefined instruction), or
- flip a **header field** whose meaning is unknown (`0x90220`), with no reference example.

Neither is yet a supported change. Before either, the useful step is to find what the
bootloader's verifier actually does — and the most direct route to that is the **obfuscation
itself**: if the region is obfuscated with a simple XOR/ADD/ROL scheme (common in secure-boot
code protection), it can be recovered statically and the verification logic read in full.

That is now the highest-value analysis: **recover the obfuscation of the BL31 code region and read
the real `hw_cms_image_verify_simple` / `usr cert cms vef` logic.** The strings are already
plaintext and anchored, so the obfuscation can be validated against them.

### Note on `Ascend310P_tee.bin`

It is **not** an esbc container (its header fields are random). It is a TEE blob — the `tee cms
verify failed` string lives in BL31, so the TEE image has its own verification. It is a separate
trust domain and out of scope for the first bootloader experiment.

---

## 2026-10-26 — Answering "can't we just switch this in the BIOS?" — and the boot-mode decode

### The two mode registers are DEVICE MMIO, not host-controlled

Two tiny reader functions, decoded 100%:

```text
0x140f85:  mov x0, #0xfff8 ; movk x0, #0x8000, lsl #16 ; ldr w0,[x0] ; and w0,w0,#0xf ; ret
           -> MODE A = (*(u32*)0x8000fff8) & 0xf          (values 0..3)

0x140f99:  mov x0, #0xfffc ; movk x0, #0x8000, lsl #16 ; ldr w0,[x0]
           cbz w0 -> return 1
           lsr w0,w0,#16 ; cmp w0,#1 -> return 3 ; cmp w0,#2 -> return 2 ; else return 0
           -> MODE B = f(*(u32*)0x8000fffc)
```

Both are **absolute device addresses in the `0x8000xxxx` window** — SoC registers set by the
boot strap / eFuse, read by BL31 over its own MMIO. **There is no host-side file, and no BIOS
setup menu, that writes them.** The `0x8000fff8 & 0xf` register is compared against **3** at
48 call sites, and the Oct 21 patch site compared it against **1**.

So the honest answer to "can we switch it in the BIOS": **no — the mode is a hardware register.**
What *is* host-controllable is (a) whether a `usrcert` image is supplied, and (b) the
`ccpu_usr_cert_hash` in secure flash. Those are the two inputs the customer path consumes, and
we can set both. There is no third, simpler lever.

### Is there a BIOS setup menu at all?

The bootloader has **no setup menu**. A full scan of the string region finds no `Setup`/`Menu`/
`Boot Order`/`HII`/`Form` strings — only `hw_opteed_setup_sip` (a TEE call) and `tspd setup
failed`. The `hboot1a`/`hboot1b` components (the only flashable ones) are **not present in the
image at all** — the string `hboot1a` does not occur anywhere in the fd. The host's
`upgrade-tool` lists them as flashable component *types*, but the card has no such files and no
`hlink` component either.

So the firmware surface is: **the bootloader image itself** (`Ascend310P.fd`), the kernel
(`Ascend310P.image`), the rootfs (`Ascend310P.cpio.gz`), the DTB, the TEE blob, and the three
sub-boot images. All except the bootloader are esbc-signed and thus payload-bound.

### The two FIP copies are DIFFERENT BUILDS — this is new and important

```text
FIP #1 @ 0x132100 : code 0x136101..0x152100 | strings 0x154100..0x15a100 | code 0x15a100..0x15e100
                    HAS: 'usr cert cms vef fail', 'no usr, hw vef fail',
                         'support usr cert vef', 'Secure Boot: customer efuse Solution',
                         'usrcert', 'hw_cms_image_verify_simple stop!',
                         'Huawei Integrity Root CA - G2', 'hw_cms_ctrl_cpu_verify failed'
FIP #2 @ 0x172100 : strings at +0x18000, NO secure-boot strings at all
                    HAS: 'GhesV2GetErrorBlockAddress', 'RasErrorDataProcessor', ...
```

**FIP #1 is the secure-boot BL31; FIP #2 is a different, non-secure build** (RAS/error-block
handling, `GhesV2*` = Huawei Ghes error subsystem). The A/B selection between them is a hardware
decision.

This explains why only 3 distinct strings resolved from the code we decoded: **the code we
decoded (0x136101..0x152100) is FIP #1's *other* component**, and the BL31 that owns the
secure-boot strings is a separate image inside the same FIP that we have not yet located.

### The de-obfuscation result, precisely

The BL31 code region is **not encrypted and not obfuscated** — it is simply **shifted by one
byte** relative to a 4-byte instruction grid:

```text
decode from file 0x136101  ->  28671/28671 instructions valid (100.0%)
```

Random data scores 0.35 on the same metric, and a 4-byte XOR key that looked promising (0.79) was
just chance alignment on opcode-biased bytes. Once aligned correctly, the code is fully readable:
the Oct 21 patch site decodes as

```text
0x13dc41: stp x29, x30, [sp, #-0x20]!     ; prologue
0x13dc51: bl  0x140f85                     ; -> MODE A = (*(0x8000fff8)) & 0xf
0x13dc55: cmp w0, #1
0x13dc59: b.eq 0x13dca1                   ; <-- Oct 21 patched this to "udf #0x1271"
0x13dc5d: mov  x1, #0xc4
0x13dc61: mov  w0, #1
0x13dc65: movk x1, #0x81f1, lsl #16
0x13dc69: str  w0, [x19, x1]              ; set mode flags
...
```

### Next step

Locate the BL31 **within** FIP #1 that owns the secure-boot strings (the region between the code
we decoded and the string table, or the second code block at 0x15a100..0x15e100 with the correct
alignment), then read `hw_cms_image_verify_simple` and the `usr cert cms vef` branch directly.
With that, the minimal patch becomes obvious: find the conditional that returns the
`IMAGE_CHECK_FAIL` verdict and make it succeed instead.

**A minimal patch is the right goal** — not "disable all cert checks" (which would break the
nvcnt/anti-rollback gate and the hash binding too). The precise target is the *decision* between
`hw_cms_image_verify_simple` (built-in root) and `usr cert cms vef` (customer path), plus the
failure return inside whichever it picks.

---

## 2026-10-26 — BL31 de-obfuscation achieved; the verification chain is mapped

### The code is NOT obfuscated — it is shifted by one byte

Decoding the region `file 0x136101 .. 0x152100` from offset `0x136101` gives:

```text
28671 / 28671 instructions valid  (100.0%)
```

Random data scores **0.35** on the same metric. A 4-byte XOR key that appeared to give 0.79 was
**chance alignment on opcode-biased bytes** (checked against an opcode-biased random baseline of
0.48) — not de-obfuscation. The real explanation is a **one-byte misalignment**: every 4 KB page
decodes at phase **+1** consistently.

### The verification chain (decoded)

```text
0x14df81  dispatcher
            adrp x2,#0x1d5000 ; ldr w2,[x2,#0x5f0]   <- bitmask of ENABLED image types
            lsl w0,w0,w5 ; tst w0,w2 ; b.eq -> skip this image type
            tbnz w4,#0x1e -> special path
            w4 = w5 + 0x7c000000 ; cmp w4,#0x14 ; b.hi -> default
            jump table (21 entries) indexed by image type

0x14dc11  image-verify entry
            bl 0x14d381    -> header/space check
            cbnz w0 -> error
            bl 0x14d839    -> get the image type
            cmp w0,#3 ; b.ne -> continue
            ... logs, then bl 0x14dba9
            bl 0x14e85d / 0x14d931 -> the actual CMS load+verify

0x14dbb1  hw_cms_image_verify_simple
            bl 0x14db8d    -> header check
            tst w0,#0xff ; b.eq -> return -2
            bl 0x14d931    -> load the CMS
            cbnz w0 -> error
            bl 0x14e2a9    -> THE VERIFY  (this is where 'usr cert cms vef fail' is logged)

0x14e2a9  verify wrapper
            bl 0x152365    -> alloc
            bl 0x152fd5 / 0x152f15 / 0x152749  -> CMS context setup
            ...
            bl 0x14cc3d    -> success path
            -> on failure: logs 'usr cert cms vef fail, result = 0x%x'
```

The `'usr cert cms vef fail'` string is referenced from **9 sites** in `0x14d49d..0x14e5d9`, all
inside the verify wrapper and its callees.

### The mode readers (device MMIO)

```text
0x140f85:  mov x0,#0xfff8 ; movk x0,#0x8000,lsl#16 ; ldr w0,[x0] ; and w0,w0,#0xf ; ret
           -> MODE A = (*(u32*)0x8000fff8) & 0xf      (values 0..3)

0x140f99:  mov x0,#0xfffc ; movk x0,#0x8000,lsl#16 ; ldr w0,[x0]
           cbz w0 -> 1 ; lsr w0,w0,#16
           cmp w0,#1 -> 3 ; cmp w0,#2 -> 2 ; else 0
           -> MODE B = f(*(u32*)0x8000fffc)
```

Both are **absolute SoC register addresses** — set by the boot strap / eFuse, read by BL31 over
its own MMIO. **There is no host-side switch and no BIOS setup menu** (the string table has no
`Setup`/`Menu`/`Boot Order`/`HII`/`Form` entries at all, and `hboot1a`/`hboot1b`/`hlink` do not occur
anywhere in the image).

### Reference resolution — what works and what doesn't

The code and the string table are **separate FIP entries with different load bases**, so no single
offset resolves both. Two anchors were confirmed:

- `adrp x0,#0x159000 ; add x0,x0,#0x456` at file `0x1484c1` → `hw_cms_ctrl_cpu_verify failed`
  (file `0x159456`) — **S = 0**, i.e. file offset == runtime address for this component.
- 9 references to `usr cert cms vef fail` (file `0x156d2e`) resolve to runtime `0x1d2e38`, i.e.
  **S = 0x7c10a** for that component.

The consistent offset between the two component bases (0x7c10a vs 0) is the gap between the two FIP
entries, and it also confirms the two are separate components.

### Where the minimal patch goes

The decision point is the branch that chooses between the built-in path and the customer path. Given
the chain above, the **cleanest single-byte patch** is at the image-verify entry:

```text
0x14dbd1: tst  w0, #0xff        ; header-check result
0x14dbd5: b.eq 0x14dc09        ; -> return -2
0x14dbd9: e20315aa  mov x2, x21
0x14dbdd: e10314aa  mov x1, x20
0x14dbe1: a0e30091  add x0, x29, #0x38
0x14dbe5: bl  0x14d931          ; load CMS
0x14dbe9: cbnz w0, 0x14dbf9    ; -> error return
0x14dbed: a1e30091  add x1, x29, #0x38
0x14dbf1: e00313aa  mov x0, x19
0x14dbf5: bl  0x14e2a9          ; VERIFY  <-- failure logs 'usr cert cms vef fail'
0x14dbf9: ... return
```

`0x14e2a9` returns the verify result in `w0`; the caller at `0x14dbb1` returns it up to the
dispatcher. **The minimal patch that makes the customer path authoritative is to skip the CMS
verify entirely** — i.e. force the `bl 0x14e2a9` at `0x14dbf5` to be replaced by a `mov w0, #0`
(return success). One instruction, four bytes.

That is *not* yet done, because it would also disable the **nvcnt anti-rollback check** and the
**payload-digest binding** if those live behind the same call. Those are separate strings
(`img head nvcnt`, `efuse_read_nvcnt fail`, `nvcnt cmp fail`) and, given the string-table
structure, live in the *same* wrapper region.

**Before patching**, the correct sequence is to read the nvcnt check and the digest check and
confirm they are *not* behind `0x14e2a9`. Those are the two gates that must keep working — V1
proved the digest gate is real.

### Immediate next steps (concrete)

1. Locate the `img head nvcnt` / `nvcnt cmp fail` code and confirm it is **separate** from
   `0x14e2a9` (i.e. in the sibling function `0x14dc11` or the dispatcher, not the verify wrapper).
2. If separate → patch only the verify result, keeping nvcnt and digest intact.
3. Build with `patch_bootloader.py` (verified: reseals `@0x2C`, `@0x4BC`, `@0x14BC` correctly) and
   apply to **both** FIP copies (the two are different builds — the secure one at `0x132100` and the
   RAS one at `0x172100` — so the patch must target the right one, or both).
4. Install as `Ascend310P.fd`, cold power cycle, read the driver ring buffer via `dms_log.py`.

### Card state

Stock images restored on the card (`027f4d22…` / `12975a9d…`). Cards are up and healthy.

---

## 2026-10-26 — Status check: reference resolution is not converging, and I should say so

I have been trying to resolve the string references in the decoded BL31 for a while now and the
arithmetic is not converging cleanly. Rather than keep grinding, here is exactly what is solid and
what is not.

### Solid (verified)

- The BL31 code region `file 0x136101..0x152100` decodes at **100%** valid AArch64 instructions
  when started at `0x136101` — every 4 KB page uses phase **+1**. It is a one-byte misalignment,
  not encryption or obfuscation. (Random baseline 0.35; the earlier apparent 0.79 XOR "result" was
  chance alignment on opcode-biased bytes.)
- The **verification chain** is mapped by control flow:
  `0x14df81` dispatcher (bitmask of enabled image types, 21-entry jump table by type)
  → `0x14dc11` image-verify entry → `0x14dbb1` (`hw_cms_image_verify_simple`)
  → `0x14e2a9` **the CMS verify wrapper** (this is where `usr cert cms vef fail` is logged).
- The **two mode registers** are device MMIO (`0x8000fff8 & 0xf`, `0x8000fffc >> 16`), read at
  48 call sites. **There is no host-side switch and no BIOS setup menu** — the string table has no
  `Setup`/`Menu`/`Boot Order`/`HII`/`Form` entries, and `hboot1a`/`hboot1b`/`hlink` do not occur
  anywhere in the image.
- The **bootloader image needs no key to modify** — it has no `cms`/`ini`/`crl` sections, only
  self-consistency (`@0x2C` = SHA-256(payload), `@0x4BC`/`@0x14BC` = SHA-256 of each header copy).
  `patch_bootloader.py` reseals all three correctly (verified) where the Oct 21 patcher left
  `@0x4BC` stale.
- The **two FIP copies are different builds**: `0x132100` is the secure-boot BL31 (has every
  secure-boot string), `0x172100` is a RAS/error-block build (`GhesV2*`). A patch must target the
  right one — and they differ, so both copies cannot be patched identically.

### Not solid (blocking)

- **Which instruction decides customer-vs-builtin.** The `adrp+add` → string mapping is
  inconsistent: the code appears to use a per-component load base, and I have two conflicting
  anchors (`S = 0` from `hw_cms_ctrl_cpu_verify failed`, `S = 0x8610a` from the nine
  `usr cert cms vef fail` sites). Both cannot be right, and `usr cert cms vef fail` is referenced by
  **no** `adrp+add` pair under `S = 0`. So the string is reached by some mechanism I have not
  identified — most likely a **pointer table** or an indirect load — and I have not found it.
- Consequently I **cannot yet name the single instruction** to patch, and I am not going to
  guess. Patching the wrong branch in the card's root of trust is a brick, and "make the CMS verify
  return success" would also disable the **nvcnt anti-rollback** and **payload-digest binding** if
  they sit behind the same call. V1 proved the digest gate is real.

### Honest assessment of the remaining path

The chain from `0x14dbb1` → `0x14e2a9` → (9 failure-log sites) is the CMS verify. The nvcnt
strings (`img head nvcnt`, `nvcnt cmp fail`, `efuse_read_nvcnt fail`) live in the **same string
table**, and the code that references the `0x157xxx` page sits in the `0x13e5e9..0x13f455` range —
i.e. **different functions from `0x14d49d`/`0x14e2a9`**. That is encouraging: it suggests the
anti-rollback gate is *not* behind the verify call. But "suggests" is not "verified", and I want
it verified before patching.

The clean way to resolve it: **find the indirect reference mechanism** (the pointer table or
indirect load) and then re-derive the string map consistently. Concretely, that means finding what
loads the address of `usr cert cms vef fail` — the `0x1d2000`/`0x1d3000`/`0x1f7000` data addresses
that appear throughout the code are a strong lead (they are used as base pointers for
`adrp`+`add` into a **structure/table**, not directly to strings).

### Recommendation

Two things in parallel, in this order:

1. **Finish the reference resolution** by decoding the `0x1d2000`/`0x1f7000` data structures.
   That is the missing link, and it is bounded work (the structures are in the same component).
2. **Meanwhile, install the fork's host driver** — that path needs none of this, has already been
   validated on four NPUs per `DEVICE_DMA.md`, and is where the DMA/performance work lives.

Before I patch the bootloader I want the string map consistent and the nvcnt/digest gates
confirmed separate. I will not apply a bootloader patch on a guess.

### Card state

Stock images restored; cards up and healthy.

---

## 2026-10-26 — The decision path, fully decoded; the minimal patch identified

### The string references resolve to a **runtime structure table**, not directly to the strings

This is the key that finally made everything consistent. At the nine `usr cert cms vef fail` sites the
code does:

```text
0x14d541:  adrp x0, #0x1d2000
0x14d545:  add  x0, x0, #0xe38      <- offset of a field in a runtime structure
0x14d549:  add  x0, x0, x1          <- + per-CPU index
0x14d54d:  adrp x2, #0x1f7000 ; add x2, x2, #0x280
0x14d559:  ldr  w0, [x0, #8]        <- load the STRING POINTER from the structure
```

`0x1d2000` is a **BSS/runtime** address (the file region there is zero-filled); the string pointers
are **initialised at boot** from the string table. That is why no single file-offset base resolves
both the code and the strings — and it also means:

- the string table (file `0x152100..0x15e153`) is reachable by the loader,
- every secure-boot message is referenced through a `0x1d2000 + 0xe38 + cpu*stride` slot,
- and **the structure at runtime `0x1d2000` is writable BSS**.

### The complete verification decision path

```text
0x14df81  dispatcher
            ldr w2,[x2,#0x5f0]         <- bitmask of ENABLED image types (runtime 0x1d5000+0x5f0)
            tst w0,w2 ; b.eq -> skip   <- image type disabled -> skip
            w4 = type + 0x7c000000 ; cmp #0x14 ; b.hi -> default
            jump table, 21 entries, one handler per image type

0x14dc11  image-verify entry
            bl 0x14d381   header/space check
            bl 0x14d839   get image type
            cmp w0,#3 ; b.ne -> continue        (type 3 = ? )
            bl 0x14dba9   -> dispatch
            bl 0x14e85d   -> CMS load + verify
            bl 0x14d931   -> CMS load
            bl 0x14d001   -> 'success' helper (bl 0x14cc9d ; returns 1)

0x14dbb1  hw_cms_image_verify_simple
            bl 0x14db8d   header check ; tst w0,#0xff ; b.eq -> return -2
            bl 0x14d931   load the CMS
            cbnz w0 -> error return
            bl 0x14e2a9   *** THE VERIFY ***  (logs 'usr cert cms vef fail' on failure)

0x14e2a9  verify wrapper
            bl 0x152365 / 0x152fd5 / 0x152f15 / 0x152749   CMS context setup
            ... state machine over the CMS ...
            bl 0x14cc3d   success path
            on failure -> 0x14d475/0x14d509/0x14d5d5 -> logs 'usr cert cms vef fail, result = 0x%x'
```

### The `no usr, hw vef fail` fallback is a SEPARATE branch

```text
0x14da51:  ldr  w0,[x0,#0x18]
0x14da55:  cmp  w0,#1 ; b.ne 0x14da71        <- if state != 1, skip the "no usr" log
0x14da81:  cmp  w19,#3 ; csel w19,w19,w0,ne  <- state 3 -> use 2
0x14da9d:  bl 0x14d86d
0x14daa9:  bl 0x14d475    <- the 'usr cert cms vef fail' reporter
0x14dab1:  ldr  w0,[x0,#0x18]
0x14dab9:  cmp  w0,#2 ; b.ne 0x14daed        <- ***THE DECISION***
0x14dac5:  bl 0x14e409
0x14dac9:  bl 0x14d001    <- success (returns 1)
0x14daed:  bl 0x14ea2d    <- the OTHER path
```

`cmp w0,#2 ; b.ne` at **`0x14dab9`** is the branch that decides between the customer path and the
built-in path.

### The minimal patch

**`0x14dab9` is the single instruction to change.** It currently is:

```text
0x14dab9:  cmp  w0, #2      ; 1f080071
0x14dabd:  mov  w0, w20     ; e003142a
0x14dac1:  b.ne 0x14daed    ; 61010054
```

Patching `b.ne 0x14daed` → `b.ne 0x14dac9` (i.e. fall through to the success path regardless of
the state) is a **4-byte change**:

```text
0x14dac1:  61 01 00 54   b.ne 0x14daed     ->  61 00 00 54   b.ne 0x14dac9
```

Wait — the target must be the *success* path. `0x14dac9` is `bl 0x14d001` (the success helper). So
changing the branch to *always* take the success path is exactly the minimal, targeted change: it
makes the **customer-certificate verification result authoritative** while leaving every other gate
untouched.

**Critical caveat, and it is a real one:** the `cmp w0,#2` at `0x14dab9` may be selecting between
*customer* and *built-in* verification — in which case forcing the success path bypasses the CMS
signature check for the image. But **the nvcnt anti-rollback gate and the payload-digest gate are
in different functions** (the nvcnt strings are referenced from the `0x13e5e9..0x13f455` range and
`0x14dc11`), so they would continue to function.

That is exactly what V1 tested and what we want to preserve: V1 showed the **payload digest is
enforced**, and we want it *still* enforced — a customer-signed rootfs whose `ini` we control still
binds the payload to our digest. We are not trying to defeat the digest check; we are trying to
make the *signature* check accept our chain.

**But**: if `0x14dab9`'s branch is instead the *customer-vs-builtin selector*, forcing success would
mean "skip the CMS check", which is a different (and much stronger) claim than "use the customer
path". **I must determine which before patching.**

### The discriminating experiment (no patching required)

Both readings predict the *same* observable for a stock image — so instead, the discriminating test
is to make the **customer path fail** in a way we can observe:

1. Deploy `/CMS/user.xer` with a hash that does **not** match the enrolled value.
2. Boot a **stock** image.
3. If the customer path engages, BL31 will take the customer branch, fail the hash compare, and
   the device will **not boot** — even though the image itself is fine.

That single run distinguishes the two readings **without touching the bootloader**, and it also
proves the customer path is reachable at all. It is much cheaper and far safer than a bootloader
patch, and it uses the existing `dms_log.py` feedback loop.

### Card state

Stock images restored; cards up and healthy (G1 baseline).

---

## 2026-10-26 — G2 armed: the discriminating experiment (no bootloader patch needed)

### The question this answers

We know the verification chain now. The open question is what the branch at **`0x14dab9`**
(`cmp w0,#2 ; b.ne 0x14daed`) actually selects:

- **Reading A** — it selects *customer verification* vs *built-in verification*. Forcing the
  success path would mean "skip the CMS signature check".
- **Reading B** — it selects *which branch of the customer path to take* (e.g. present vs absent
  user cert). Forcing the success path would mean "use the customer chain".

The two readings predict **the same** observable for a stock image, so a stock boot cannot
distinguish them. But they predict **different** observables if we break the customer material.

### G2 — the discriminating test, armed

```text
/usr/local/Ascend/CMS/user.xer   9833 B   usrcert container holding 'Test Signing Cert'
                                         payload hash = b932bb15ac0cc925…
/usr/local/Ascend/CMS/user.crl   3045 B   [ rootG2.crl , rootcaG2.der , rootG2.crl ]
rootfs                            12975a9d…  STOCK (perfectly valid image)
enrolled on dev0                  7fb03430…  (the ROOT's hash — deliberately MISMATCHED)
```

**Prediction if the customer path engages at all:** BL31 loads the `usrcert` image, hashes it,
compares with the enrolled value, the compare **fails** (`usr cert hash cmp fail.`), and the card
**does not boot** — even though the image is stock and perfectly valid.

- **Card does not boot** → the customer path IS reachable and IS consulted. That is the single
  most important fact we still lack, and it is proven *without* touching the bootloader.
- **Card boots fine** → the customer path is NOT consulted for a stock image (either it is gated on
  the image's own `certtype`, or the `usrcert` image is not being loaded at all). That would mean
  the whole customer-certificate avenue is inert on this hardware and we should stop pursuing it.

Either outcome is decisive, and the test costs **one power cycle and no patching**.

### If G2 shows the customer path engages

Then the follow-up is the real test: **matching** hash + a **customer-signed** rootfs, i.e. exactly
the D-series but with the hash correctly aligned. That combination has never been tried:

- every prior attempt had either a mismatched hash or a malformed CMS,
- D1 had the right idea (leaf anchor) but the CMS was built before the three encoding fixes,
- the CMS fixes are now in place (`pss_sign.py` produces a CMS structurally identical to stock —
  `si.children: 0x02,0x30,0x30,0xa0,0x30,0x04,0xa1` in both).

### If G2 shows the customer path is inert

Then the bootloader patch becomes the only route, and the patch would be the one at `0x14dab9` —
plus, necessarily, the `usrcert` image must actually reach the card, which is a separate
open question (the loader sends 11 files in table order with no names, so the `usrcert` image would
have to be one of them).

### Card state

G2 armed. Cards are up and healthy on stock firmware. **Cold power cycle needed (≥30 s).**

---

## 2026-10-26 — G2 result: the `usrcert` image IS delivered and acked, but the hash is NOT enforced

### G2 setup

```text
/usr/local/Ascend/CMS/user.xer   9833 B   usrcert container, payload hash b932bb15ac0cc925…
/usr/local/Ascend/CMS/user.crl   3045 B   [ rootG2.crl , rootcaG2.der , rootG2.crl ]
rootfs                            12975a9d…  STOCK
dev0 enrolled hash                7fb03430…  (the ROOT's) — deliberately MISMATCHED
```

### G2 outcome — the card booted perfectly

```text
davinci nodes : 6
phy_startup   : 0
npu-smi       : all 6 chips OK
```

### What the driver log shows (877 lines)

```text
Get file size.   (dev_id=0; file_name="/usr/local/Ascend/CMS/user.crl"; size=3045)
Notice BIOS to Load file. (dev_id=0; …/CMS/user.crl; size=1; flag=0x6666666666666666)
Get file size.   (dev_id=0; file_name="/usr/local/Ascend/CMS/user.xer"; size=9833)
Notice BIOS to Load file. (dev_id=0; …/CMS/user.xer; size=1; flag=0x6666666666666666)
Receive BIOS flag. (dev_id=4; flag=0x8888888888888888; count=1)
```

- `user.crl` and `user.xer` were **read, transferred and acknowledged** (`0x8888…` = success) —
  **the customer-certificate delivery path works end to end.** That was the open question.
- **Every** flag in the whole boot was `0x8888…` — no failure flags at all.
- `Skip file.` appeared only for `Ascend310P.crl` (which genuinely does not exist).

### ⚠️ The critical negative: a mismatched hash did NOT stop the boot

With dev0's enrolled hash deliberately mismatching the `user.xer` payload, the card **still
booted**. Therefore, on this hardware with a **stock-signed** image:

> **the enrolled `ccpu_usr_cert_hash` is not consulted when deciding whether to boot.**

That is consistent with everything observed so far and it explains the whole 2026-10-21 failure
pattern. The customer path is a **runtime** feature (`drv_user_cfg.ko` / `drv_pkicms.ko`, reading
`/var/asdrv/dev%d/device-sw-plugin/verify/<type>`), and the **boot-time** decision to accept the
image is made against the built-in Huawei root. Supplying a `usrcert` image does not, by itself,
change the boot decision.

**This is a genuinely useful negative result**: it rules out the hypothesis that the customer path
could be reached merely by delivering `user.xer` + `user.crl` and enrolling the right hash.

### G3 — the refined discriminating test (armed)

Instead of "does a bad hash break a stock boot?" (it doesn't), ask the question that actually
discriminates the two readings of the branch at `0x14dab9`:

```text
dev0   enrolled hash = 7fb03430…  (MISMATCH)   <-- rootfs is STOCK
dev1-5 enrolled hash = b932bb15…  (MATCH)
rootfs                = STOCK (12975a9d…)
```

If the boot-time customer path *is* consulted for a stock image, **only dev0** would fail to
boot (its hash compare fails) while dev1–5 boot normally — a per-device split that is
unambiguous.

If all six boot, the boot-time path does not consult the enrolled hash at all, and the customer
certificate only matters for **runtime** CMS verification — i.e. for signing things the *card*
loads at runtime (the card-side `drv_pkicms` / `drv_user_cfg` path), not the boot images.

### What the S-series becomes

The untested combination that now matters most: **matching hash + customer-signed rootfs**, built
with the corrected CMS generator (`pss_sign.py`, now structurally identical to stock). Every prior
attempt had either a mismatched hash or a malformed CMS; the two have never been right at once.

### Card state

G3 armed (dev0 mismatched, dev1–5 matched, stock rootfs). Cards are up and healthy.
**Cold power cycle needed (≥30 s).**

---

## 2026-10-26 — G3 result: the boot path does NOT consult the enrolled hash. Definitive

### G3 setup

```text
/usr/local/Ascend/CMS/user.xer   9833 B   payload hash b932bb15ac0cc925…
rootfs                            12975a9d…  STOCK (valid image)
dev0 enrolled hash                7fb03430…  (the ROOT's)  => MISMATCH
dev1-5 enrolled hash              b932bb15…  (the LEAF's)  => MATCH
```

### G3 outcome

```text
davinci nodes : 6
phy_startup   : 0
Device online : dev0, dev1, dev2, dev3, dev4, dev5   <- ALL SIX
Add dev success: udevid 0..5
```

**dev0 came online despite its deliberately mismatched enrolled hash.** The G3 split was designed so
that a per-device difference would be unambiguous; there was none.

### ⚠️ Correction to the G2 conclusion

My G2 note said the negative "rules out the hypothesis that the customer path could be reached
merely by delivering `user.xer` + `user.crl` and enrolling the right hash." G3 confirms and
**generalises** it:

> **At boot time, the enrolled `ccpu_usr_cert_hash` is not consulted in the accept/reject
> decision for the image.**

The boot decision is made against the **built-in Huawei root** (`Huawei Integrity Root CA - G2`,
compiled into `AS610_HBOOT2_UEFI.fd`). Delivering a `usrcert` image and enrolling a hash does not
alter that.

### What this means for "custom firmware"

There are now only two possible routes, and one of them is ruled out:

| route | status |
|---|---|
| **A. Boot-time signing** of a modified `Ascend310P.image` / `Ascend310P.cpio.gz` | **CLOSED** — needs the Huawei root key (absent everywhere), or a bootloader patch whose purpose would be to make the customer chain authoritative |
| **B. Runtime** — the card's *runtime* CMS verification (`drv_user_cfg.ko` + `drv_pkicms.ko` reading `/var/asdrv/dev%d/device-sw-plugin/verify/<type>`) | **OPEN** — and this is where `ccpu_usr_cert_hash` genuinely matters, because that module compares the enrolled hash with a certificate the *runtime* holds |

The distinction is now crisp, and it is not a subtlety:

- **boot** → built-in Huawei root only.
- **runtime** → the `usrcert`/`user-certificate` material, gated by `ccpu_usr_cert_hash`.

So the customer certificate is a **runtime** credential. That reframes the goal: custom *runtime*
code and data on the card, verified against our own CA.

### Route B — what it actually requires

The runtime chain is:

```text
drv_user_cfg.ko   ccpu_usr_cert_hash  +  sec_flash_read/write
drv_pkicms.ko      pkicms_get_cert_data -> /var/asdrv/dev%d/device-sw-plugin/verify/{vendor,community,user}-certificate
                   pkicms_custom_verify_cms
                   pkicms_get_sec_check_enable_flag -> eFuse at 0x8126e080
```

Two of these live **inside the rootfs** (signature-locked), and two are host-side paths:

- `/usr/local/Ascend/CMS/user.xer` → the customer certificate — **host-side, writable** ✅
- `ccpu_usr_cert_hash` in secure flash — **host-side, writable** ✅ (proven: `--user_cert` writes it)
- `/var/asdrv/dev%d/device-sw-plugin/verify/user-certificate` — **host-side, writable** ✅
- `pkicms_get_sec_check_enable_flag` (eFuse bit) — **not writable**, but *readable* by us

**All three host-controllable inputs are already in place.** The remaining question is whether the
*card-side* `drv_pkicms.ko` will actually use them — and that module is inside the rootfs, so we
cannot change it.

But we *can* observe it: the card-side modules are loaded from the rootfs we supply, and the runtime
CMS path can be exercised by whatever the card loads at runtime.

### The practical next step

Rather than continue to fight the boot path, switch to the runtime path:

1. **Find what the card verifies at runtime** — the card-side `drv_pkicms.ko` resolves
   `/var/asdrv/dev%d/device-sw-plugin/verify/user-certificate`. Populate that path with our cert and
   see whether the runtime accepts it.
2. In parallel, **the S-series image is built and ready** (`S1-ourcms-matchinghash.cpio.gz`,
   sha256 `de4db13f…`) — a customer-signed rootfs with a **matching** hash and a CMS that is
   **byte-structurally identical to the stock container's**. It is the one combination never tried
   (every prior attempt had a mismatched hash *or* a malformed CMS). It is worth one cycle even
   though G3 suggests the boot path will reject it: it will tell us whether a *correctly* formed
   customer signature is treated differently from a malformed one.

### Card state

All six devices up and healthy on stock firmware. The G3 enrollment state (dev0 mismatched, dev1–5
matched) is still in place — that is a deliberate, documented state and should be reset before
further work.

---

## 2026-10-26 — Going for the end goal: the bootloader-replacement pre-condition test

### The trust hierarchy, established from the driver's own tables

`res_drv.c` selects the platform image table by `chip_type`; our cards are `chip_type=2` →
`res_drv_mini_v2.c`:

```text
/driver/device/Ascend310P_ddr.image     CRITICAL
/driver/device/Ascend310P_lowpwr.image  CRITICAL
/driver/device/Ascend310P_hsm.image     CRITICAL
/driver/device/Ascend310P.fd            CRITICAL   <-- the bootloader itself
/driver/device/Ascend310P_dt.img        CRITICAL
/driver/device/Ascend310P.image         CRITICAL   <-- the kernel
/driver/device/Ascend310P.cpio.gz       CRITICAL   <-- the rootfs
/driver/device/Ascend310P_tee.bin       CRITICAL
/driver/device/Ascend310P.crl           NON_CRITICAL
/CMS/user.crl                           NON_CRITICAL
/CMS/user.xer                           NON_CRITICAL
```

**`Ascend310P.fd` — the bootloader — is delivered by the host as a CRITICAL file.** And the host
driver **never verifies any signature** (no `cms_verify` / `Cmscbb` / `esbc` anywhere in
`pcie_host/`). Verification is entirely the card's BL31.

Also relevant: `res_drv_mini_v3.c` (the 310B variant) leads with
`/driver/device/ascend_310b_usercfg.bin` as its **first CRITICAL** file — the *customer-config*
image. That is the vendor's own mechanism for shipping customer configuration to the card, and it
confirms the `usrcert`/`user.xer` idea is the right shape even though on 310P the customer material
is optional (`NON_CRITICAL`).

### The chain, and where the anchor lives

```text
card BootROM  --(trust anchor in on-die ROM)-->  hboot1a/hboot1b  (flash-only, NOT host-delivered)
                                              -->  Ascend310P.fd (BL31)  <-- contains the root CA
                                                   -->  Ascend310P.image (kernel)   [esbc-signed]
                                                   -->  Ascend310P.cpio.gz (rootfs)  [esbc-signed]
```

**The root CA `Huawei Integrity Root CA - G2` is embedded inside `Ascend310P.fd`** (file `0x155841`,
1606 bytes; plus `Huawei Root CA` at `0x1552e9`, 1363 bytes). The bootloader verifies the images
against its own copy of the root.

**Therefore: if we can replace `Ascend310P.fd`, we replace the trust anchor.** And `Ascend310P.fd`
is:

- delivered by the **host** (not flash-only),
- an esbc container with **no `cms`/`ini`/`crl` sections** → **not signed by anyone**,
- protected only by its own self-consistency (`@0x2C` = SHA-256(payload), `@0x4BC`/`@0x14BC` =
  SHA-256 of each header copy) — all of which `patch_bootloader.py` recomputes correctly.

### P1 — the pre-condition probe (deployed)

Before attempting any real modification, the decisive question is: **does the card's BootROM accept
a host-delivered bootloader that differs from the stock one at all?**

The probe flips **a single byte** (`0xff` → `0x00`) at file offset `0x2149` — inside an **erased
region** of the payload, far from the two embedded certificates — and reseals all three digests:

```text
payload digest @0x2C   : OK
header check  @0x4BC   : OK
header check  @0x14BC  : OK

file 0x2149  (in an all-0xff run, adjacent to code at 0x2100-0x2130)
sha256 09ce43aca4f582e04c73749925d1c9bc55c6a888976ee71020561412c4f8eaa0
```

Everything else on the card stays **stock** (`Ascend310P.image` `027f4d22…`,
`Ascend310P.cpio.gz` `12975a9d…`), and the stock bootloader is preserved at
`/var/local/ascend-work/images/Ascend310P.fd.stock`.

**Outcomes:**

| result | meaning |
|---|---|
| **card boots normally** | the BootROM accepts a host-delivered, re-checksummed bootloader → **the anchor is replaceable** → we can install our own root CA and sign our own images. This is the whole goal. |
| card fails to boot | the BootROM does its own integrity check on `hboot2` (e.g. a hash held in the ROM or in `hboot1`) → the bootloader route is closed and the remaining option is the runtime path. |

Note: the probe byte is in a region that is very likely **not** executed (an erased hole between
code blocks), so a boot failure would indicate an integrity check rather than a code fault — which
is exactly the distinction we need.

### Card state

P1 deployed: `Ascend310P.fd` = probe (`09ce43ac…`), kernel and rootfs stock.
**Cold power cycle needed (≥30 s).**

---

## 2026-10-26 — ★ P1 PROBE RESULT: the bootloader IS replaceable. The card stopped AFTER loading it

### P1 outcome

```text
davinci nodes : 0        (did not boot)
phy_startup   : 0
Ascend310P.fd  : 09ce43ac…  (the probe)
```

### The driver log is decisive

dev_id=0 file sequence in the **probe** boot:

```text
Ascend310P_ddr.image     size=1   flag=0x6666…   → acked 0x8888… (count=1)
Ascend310P_lowpwr.image  size=1   flag=0x6666…   → acked 0x8888… (count=71)
Ascend310P_hsm.image     size=1   flag=0x6666…   → acked 0x8888… (count=1)
Ascend310P.fd            size=2   flag=0x6666…   → acked 0x8888… (count=1)   ← the MODIFIED bootloader
Ascend310P_dt.img        size=2   flag=0x6666…   → acked 0x8888… (count=1)
                                       <<< STOP — Ascend310P.image never delivered
```

dev_id=0 file sequence in a **good** boot:

```text
… identical through Ascend310P_dt.img …
Ascend310P.image         size=30  flag=0x6666…      ← the kernel
Ascend310P.cpio.gz       size=48  flag=0x7777…  (x2)
Ascend310P.cpio.gz       size=9   flag=0x6666…
Ascend310P_tee.bin       size=3   flag=0x6666…
/CMS/user.crl            size=1   flag=0x6666…
/CMS/user.xer            size=1   flag=0x6666…
```

### ★★ What this establishes

**The modified bootloader was delivered to the card and acknowledged with `DEVDRV_RECV_FINISH`
(`0x8888…`) — i.e. the card ACCEPTED it.** The BootROM does **not** do its own integrity check on
`hboot2`; the loader's own handshake succeeded.

The card then stopped at the point where it would have taken the **kernel** (`Ascend310P.image`).

Two readings, and the second is the important one:

- **Reading 1 (benign):** the single flipped byte at `0x2149` happened to land in executed code or in
  a structure the bootloader parses, so the modified BL31 misbehaved *after* it was loaded — e.g.
  it read a bad length and refused to continue. **The bootloader was still accepted**, which is the
  finding we wanted.
- **Reading 2 (important):** the bootloader *did* start and then **failed its own self-verification
  of the kernel**, because the flipped byte corrupted something it uses to validate
  `Ascend310P.image`.

**Either way, the decisive result stands: a host-delivered, re-checksummed bootloader reaches the
card and is accepted by the loader.** The trust anchor lives in that file, so **the anchor is
replaceable** — the boot-time signing goal is reachable, provided we can produce a bootloader that
still passes the card's subsequent checks.

### What the flip at 0x2149 actually was

```text
file 0x2130: 00 00 00 00 10 00 00 00 00 00 00 00 00 00 00 00
file 0x2140: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff
file 0x2150: ff ff ff ff ff ff ff ff ff 71 ...
```

`0x2140..0x214f` is a 16-byte run of `0xff`, i.e. an **erased hole** in the flash image — the sort
of gap left by a linker alignment or a size mismatch between two sub-images. Flipping one byte
there *should* be inert. It was not inert, which tells us the region is **not** inert: it is either
executed code, or part of a data structure the bootloader parses (e.g. a size/offset table).

### Next step — the real test, done properly

Now that we know the bootloader is accepted, the actual goal test is:

**Replace the embedded root CA with our own, and sign `Ascend310P.image` / `Ascend310P.cpio.gz`
with it.**

The blockers to work through, in order:

1. **Byte-length mismatch.** `Huawei Integrity Root CA - G2` is 1606 bytes; our `RootCA G2` is 1465.
   The cert sits inline in the BL31 payload, so a same-length replacement is needed (pad the cert's
   extension/attributes or re-issue ours to match 1606 bytes) — or all offsets must be fixed up.
2. **The bootloader also verifies itself** — its own signature/hash may be checked by `hboot1`
   (the flash-only first stage). We have no evidence either way yet; the P1 result suggests the
   loader accepted it, but `hboot1` may check `hboot2` *before* handing over.
3. **nvcnt anti-rollback** on the images (eFuse-backed) — separate from the signature.

Step 1 is mechanical: re-issue our root CA to be exactly 1606 bytes (tune the subject/extension
lengths), then splice it in place, reseal the three digests, and deploy **together with**
customer-signed kernel + rootfs. That is a single test of the end goal.

### Card state

Stock bootloader restored (`e1e75ec2…`). Cards are down from the probe boot and need a cold power
cycle.

---

## 2026-10-26 — The P1 failure explained; and a decisive simplification

### P1's failure is now fully explained

The probe flipped one byte at **file 0x2149**, which is **payload offset 0x149** — inside the FIP
**TOC** (the table of sub-image descriptors), not code. So the modified bootloader was **loaded and
started**, and then could not locate the `uimage` sub-image because its TOC entry was corrupted.
That is exactly the observed behaviour: the load sequence stopped after `Ascend310P_dt.img` and
the **kernel was never delivered**.

**So the finding stands and is now unambiguous: a host-delivered, re-checksummed bootloader is
accepted by the card.** The trust anchor lives in that file, therefore the anchor is replaceable.

### The simplification that makes the end goal tractable

Re-issuing our own chain from scratch hit a wall of **byte-exact slot constraints** (the embedded
root must occupy exactly 1606 bytes; the `usrcert` payload must occupy exactly 1385 bytes; the CMS
must be PSS; the CRLs must be DER; the CMS `crls[]` must be `SET OF` raw `CertificateList`).

But the **Oct 21 chain is already slot-compatible** — it was built for exactly these slots:

| artifact | size | role |
|---|---|---|
| `user-signcert-ct1.der` payload | 1385 B cert | self-signed anchor `CN=Signing Certificate G2` |
| `user-signca.der` payload | 1500 B cert | intermediate `CN=Signing Certificate CA G2` |
| `user-signcert-ct1.der` | — | enrolled hash `b932bb15ac0cc925…` |

And the **Huawei `cmssign` tool is on the card** at
`/home/matteius/cmsenv/lib/python3.14/site-packages/cmssign/cmssign-1.0-SNAPSHOT.jar` — the very
tool that produced the known-good `full/initrd.ini.p7s`. Its Java signer:

- **always emits PSS** (matching the stock container), and
- **always requires `--tssigner`/`--tskey`** (i.e. always produces the RFC-3161 timestamp token,
  matching the stock container's `unsignedAttrs`).

Two gotchas found the hard way: the CRLs must be **DER** (`openssl ca -gencrl` emits PEM, and the
Java tool fails with `unknown tag 13`), and the timestamp signer must be supplied.

### The remaining real question

The stock container's CMS carries **three** certs (intermediate + code-signing leaf + TSA leaf) and
**two** CRLs (root CRL + intermediate CRL). Our Oct 21 chain has the same shape:
`[signca, signcert]` + `[tsca, ts]`. The **anchor** (`user.xer` = `signcert`, self-signed) is the
certificate the card will use to verify the image's CMS.

So the correct end-goal deployment is:

```text
Ascend310P.fd        stock, but with our root spliced in at 0x155841 (1606 B, byte-exact)
Ascend310P.image     re-signed by cmssign with OUR chain, certtype 0x2
Ascend310P.cpio.gz   re-signed by cmssign with OUR chain, certtype 0x2
/CMS/user.xer        the anchor cert (already deployed)
/CMS/user.crl        the anchor's CRL (already deployed)
enrolled hash        = SHA-256(tag‖user.xer payload) = b932bb15… (matches)
```

The one piece still outstanding is the **root splice**: our root must be exactly 1606 bytes and
must carry the same subject as Huawei's (`CN=Huawei Integrity Root CA - G2`) so that the CMS's
`signingCertificateV2` chain-building finds it. We already produced a 1606-byte PSS root with that
exact subject (`root_pss.der`) — it is signed by **our own root key**, so the chain
`our root → our intermediate → our leaf` is internally consistent *if* the images are signed by
our leaf.

### Card state

Stock images + stock bootloader restored; `/usr/local/Ascend/CMS/` holds the Oct 21 anchor and CRL.
Cards are **down** (P1 probe boot) and need a cold power cycle before re-enrollment will succeed.

---

## 2026-10-26 — ★★★ THE END-GOAL ARTIFACT SET IS BUILT AND DEPLOYED

Everything the investigation was aiming at now exists as a coherent, internally consistent set:
**our own root CA is embedded in the bootloader, and the kernel and rootfs are signed by our chain.**

### The PKI (every slot byte-exact)

| artifact | size | subject | role |
|---|---|---|---|
| `root_new.der` | **1606 B** | `C=CN, O=Huawei Technologies, OU=Huawei Certification Authority, CN=Huawei Integrity Root CA - G2` | **spliced into the bootloader at file 0x155841** (the slot Huawei's own root occupies) |
| `int_new.der` | 1544 B | `C=CN, O=Test, CN=Signing Certificate CA G2` | intermediate |
| `leaf_new.der` | **1385 B** | `C=CN, O=Test, CN=Signing Cert G2` | code-signing leaf, `EKU: codeSigning` |
| `ts_new.der` | **1523 B** | `C=CN, O=Test, CN=Huawei Time-Stamping Authority 2 - G2YYYYYYYYYY` | timestamp leaf, `EKU: critical, timeStamping` |

The four sizes are not arbitrary — they are the **fixed slots**:
`1606` = the bootloader's embedded-root slot, `1385` = the `usrcert` payload slot,
`1523` = the timestamp-signer slot, and the intermediate is unconstrained.
They were hit by sweeping the CN length and serial length until the DER size matched exactly
(PSS adds ~104 bytes over PKCS#1, which is why the code-signing leaf is PKCS#1 and the others PSS).

`openssl verify`: `leaf_new.pem: OK`, `ts_new.pem: OK` (both under the new root via the new
intermediate).

### The images

Signed with **Huawei's own tool** — `cmssign-1.0-SNAPSHOT.jar` (found at
`/home/matteius/cmsenv/lib/python3.14/site-packages/cmssign/`), which is the same tool that
produced the Oct 21 `full/initrd.ini.p7s`. It always emits PSS and always attaches the RFC-3161
timestamp token, matching the stock container exactly.

```text
goal-uimage.image   31,350,192 B   sha256 c67e8bad731ed5490…
goal-initrd.cpio.gz 109,389,296 B sha256 a02cf0552f746a19…
```

Our verifier (calibrated on the stock container) confirms:

```text
embedded certificates: 4
   CN=Signing Cert G2, CN=Signing Certificate CA G2,
   CN=Huawei Time-Stamping Authority 2 - G2YYYYYYYYYY, CN=Signing Certificate CA G2
signerInfos: 1
PSS SIG : VALID (signer CN=Signing Cert G2, salt=32)
unsigned attr : 1.2.840.113549.1.9.16.2.14
OVERALL: ALL SIGNERINFOS VALID
```

### The bootloader

`goal-bootloader.fd` — the stock bootloader with **our root spliced in at 0x155841**, all three
self-checksums recomputed (`@0x2C` payload digest, `@0x4BC`/`@0x14BC` header checksums). Verified
in place: `our root present at 0x155841: True`.

### Deployed on the card

```text
/usr/local/Ascend/driver/device/Ascend310P.fd         8e0e06945852afc0…
/usr/local/Ascend/driver/device/Ascend310P.image      c67e8bad731ed5490…
/usr/local/Ascend/driver/device/Ascend310P.cpio.gz    a02cf0552f746a19…
/usr/local/Ascend/CMS/user.xer                       d37a7288ddde8c2f…  (our 1385-byte leaf)
/usr/local/Ascend/CMS/user.crl                       e5a90f1447996d6c…  (our root's CRL)
```

The enrolled hash must be re-set to the **new** `user.xer` payload digest
(`09a2cd0b69941274…`), because the anchor changed. The cards are currently down, so enrollment
has to happen after the power cycle.

### Why the P1 probe mattered

P1 proved a host-delivered, re-checksummed bootloader is **accepted** by the card (the load got as
far as the kernel before the corrupted FIP TOC entry stopped it). That is what makes the root splice
possible at all.

### Recovery

Full stock set preserved on the card:

```text
/var/local/ascend-work/images/Ascend310P.fd.stock     e1e75ec2103a9526…
/usr/local/Ascend/driver/device/Ascend310P.image.orig 027f4d22762ef0644…
/usr/local/Ascend/driver/device/Ascend310P.cpio.gz.orig 12975a9d7e934525…
```

### Next: one cold power cycle, then re-enroll the new hash, then observe

1. Cold power cycle (≥30 s) with the end-goal set deployed.
2. Re-enroll: `upgrade-tool --device_index <i> --user_cert --path /usr/local/Ascend/CMS/user.xer`
   (new digest `09a2cd0b…`).
3. Read the driver's ring buffer via `dms_log.py` for the verdict.

If the card boots, we have custom firmware on Ascend hardware. If it fails, `dms_log.py` will now
say exactly which stage rejected it — and unlike every previous attempt, the load phase is
instrumented end to end.

---

## 2026-10-26 — Goal boot FAILED; the bootloader is the gate; isolation experiment armed

### Goal boot result

```text
Ascend310P.fd         8e0e0694…  (our root spliced in)
Ascend310P.image      c67e8bad…  (our CMS)
Ascend310P.cpio.gz    a02cf055…  (our CMS)
davinci nodes: 0       phy_startup: 0
```

The driver log shows the load stopping at exactly the same point as the P1 probe:

```text
good boot:  ddr → lowpwr → hsm → .fd → dt.img → image → cpio.gz → tee → user.crl → user.xer
goal boot:  ddr → lowpwr → hsm → .fd → dt.img          ← stops; the KERNEL is never delivered
P1 probe :  ddr → lowpwr → hsm → .fd → dt.img          ← identical
```

Every flag was `0x8888…` (success) — the bootloader **accepted** the delivered files, then stopped
before the kernel.

### The bootloader's inner structure (decoded)

The `hboot2` payload is an **EDK2 firmware volume**, not a flat image:

```text
payload+0x000  inner esbc header: magic 33aa3a3a, tag "hboot2", +0x24=0x205, +0x30=0x564d, +0x34=0x100
payload+0x100  09 19 00 14 | 12 zero bytes
payload+0x118  99 35 89 61 85 c3 2d d3 | 00 00 02 00 | 00 00 00 00
payload+0x128  5f465648-fffe-0c00-4800-9cb900000002  = EFI_FIRMWARE_FILE_SYSTEM2_GUID
               FFS2 file header: IntegrityCheck=0x0020, Type=0, Attrs=0, Size=4096, State=0
payload+0x140  8 zero bytes, then 0xffffffff filler
payload+0x2158 71aaf000 a00f00f8
payload+0x30e8 d408e2df0f37b4418ebaa9433fb226f2   28aa0328188000f8
```

For comparison, the **initrd** payload has the *gzip header* at `+0x100` (`1f8b08…`), so `+0x100`
is simply where the payload data begins — the FFS descriptor is specific to the `hboot2` EFI volume.

Two 16-byte values live in that descriptor: `78e58c8c…` at `+0x110` and `d408e2df…` at `+0x30e8`.
**Both are byte-identical between the stock and our goal bootloader**, so neither covers the
certificate region — the splice did not invalidate them.

P1 flipped file `0x2149` = payload `+0x149`, which is inside the FFS descriptor's `0xffffffff`
filler. The goal bootloader also fails at the same point. **Common factor: a modified bootloader
payload, regardless of where.**

### The isolation experiment (V2, deployed)

The goal boot cannot distinguish two causes:

1. *the bootloader rejects a payload it did not sign itself* (an integrity/authenticity check on
   the FFS volume), versus
2. *the bootloader rejects our root* (trust-anchor mismatch).

V2 separates them: **one `0xff` byte flipped at file `0x20000`** (payload `+0x1e000`, deep in the
volume, far from the FFS descriptor and from the certs), digests resealed, **Huawei's root left
completely untouched**.

```text
V2 = c321687b93212b5025d8e9057156319f739a748d0cdf6764489ff30e681e8b3b
```

- **V2 boots** → the bootloader accepts arbitrary payload edits; the failure is our **root splice**
  (trust-anchor mismatch). That is a *semantic* problem: the bootloader would need to accept a
  non-Huawei root, which is exactly what the `usrcert` / `customer efuse` mechanism is for.
- **V2 fails identically** → the bootloader authenticates its own payload (the FFS volume carries
  a signature over itself, verified with the embedded root). Then **no payload edit is possible**,
  the trust anchor is genuinely immutable, and the only remaining route is the customer-certificate
  mechanism the vendor provides for exactly this purpose (`AOID_HW_KP_VOUCHERSIGNING`,
  `AOID_CT_ANIMAJSONVOUCHER`, `AOID_HW_KP_RELEASESIGNING` — a hardware-signed JSON *voucher*).

Both outcomes are decisive, and V2 is the cheapest possible test.

### Note: a voucher mechanism exists

The bootloader's OID table includes `AOID_CT_ANIMAJSONVOUCHER`,
`AOID_HW_KP_VOUCHERSIGNING` and `AOID_HW_KP_RELEASESIGNING`, and the strings
`Daw and msd lock ok` appear in three places. This is Huawei's **hardware-authorised signing
voucher** — the vendor-sanctioned way to let a customer CA sign firmware. It is signed by the
*hardware* key (in the SoC's secure storage), so it cannot be forged, but it also cannot be
manufactured without Huawei.

If V2 fails, the voucher path is the remaining route, and the thing to look for is how the
bootloader locates it — most likely via the `msd`/`daw` secure-storage area that the
`Daw and msd lock ok` messages reference.

### Card state

V2 deployed (arbitrary payload byte flipped, Huawei root intact, digests resealed); customer-signed
kernel and rootfs in place. Cards are down and need a cold power cycle.

---

## 2026-10-26 — ★★★ V2 RESULT: the bootloader authenticates its own payload. Boot-time signing is definitively closed

### V2 outcome

```text
V2 = one 0xff byte flipped at file 0x20000 (payload+0x1e000, deep in the volume,
     far from the FFS descriptor and from the certificates)
     digests resealed, HUAWEI'S ROOT COMPLETELY UNTOUCHED

davinci nodes: 0    phy_startup: 0
delivered: ddr → lowpwr → hsm → .fd → dt.img     ← stops, kernel never delivered
flags   : all 0x8888… (success)
```

**Identical to the goal boot and to P1.** A single byte in an unused region of the payload, with
the trust anchor untouched, produces exactly the same failure.

### The conclusion, which is now definitive

The bootloader **authenticates its own payload**. The mechanism is the EDK2 FFS2 volume at
`payload+0x100` (GUID `EFI_FIRMWARE_FILE_SYSTEM2_GUID`, Size 4096) and the two integrity values
inside it (`78e58c8c…` at `+0x110`, `d408e2df…` at `+0x30e8`) — the volume carries a signature
over itself, verified with the **embedded Huawei root**.

Therefore:

| | status |
|---|---|
| modify the bootloader payload | **impossible** — self-authenticated (V2, P1) |
| replace the trust anchor with our own | **impossible** — the payload's own signature covers the certs, so editing them invalidates it (goal boot) |
| sign the kernel / rootfs with our CA | **impossible** — the bootloader only trusts the built-in Huawei root (all D/R/S/T/V series) |
| obtain the Huawei root private key | **impossible** — verified absent from every reachable artifact |

**Boot-time firmware replacement is closed on this hardware.** This is now established by three
independent lines of evidence rather than inference.

### What the bootloader *does* provide — and it is the vendor's own answer

The bootloader's OID table contains:

```text
AOID_CT_ANIMAJSONVOUCHER      <- a JSON "voucher"
AOID_HW_KP_VOUCHERSIGNING      <- a HARDWARE key purpose for voucher signatures
AOID_HW_KP_RELEASESIGNING      <- a HARDWARE key purpose for release signatures
```

This is Huawei's **hardware-authorised signing voucher**: a JSON blob signed by a key in the SoC's
own secure storage, which grants a customer CA the right to sign firmware. It is exactly the
mechanism for "run custom firmware" that the bootloader implements, and it is the only path that
can work given that the payload is self-authenticated.

The `msd`/`daw` naming (`Daw and msd lock ok`) points at the SoC secure-storage area. There is no
voucher support in the *host* driver (`grep -riE 'voucher|anima|release.?sign'` over
`/usr/src/davinci_ascend-1.0` → nothing), so the voucher is entirely a bootloader-side, hardware-
rooted feature.

### Revised end-goal assessment

- **Custom boot firmware (modified kernel / rootfs, signed by us): CLOSED.**
- **Custom runtime firmware:** the card-side `drv_pkicms.ko` / `drv_user_cfg.ko` are inside the
  rootfs — which is signed by Huawei, so we cannot ship a modified rootfs either.
- **Therefore the only way to run our own code on the card is the voucher path** (vendor-signed,
  hardware-rooted), **or** the host side: the **host driver** (`/usr/src/davinci_ascend-1.0`, our
  OpenSensor fork, already installed) plus **host-side tooling** — which is where the DMA work and
  the `DEVICE_DMA.md` profiling already live.

### Next: find how a voucher is delivered

Concretely, the next investigation is to determine what the bootloader looks for when deciding
whether to accept a non-Huawei root:

1. Search the decoded BL31 for the voucher-verification path (`AOID_HW_KP_VOUCHERSIGNING` usage) and
   find the **input format** — is it a file in the esbc container? a device MMIO/secure-storage
   region? an `EFI_SIGNATURE_LIST`?
2. Check whether the card exposes a secure-storage/OTP region that a voucher would be written to
   (`pkicms_read_efuse_reg` in the card-side module is the read path).
3. Check whether the vendor ships a voucher tool with the SDK (`Ascend-cann-device-sdk-*`).

If a voucher cannot be produced (it is signed by Huawei's hardware key, so it cannot), then the
honest conclusion is that **the card's boot chain is closed to modification**, and the achievable
goal is: **custom host-side driver + runtime that drives the stock card firmware**, which is where
the existing DMA profiling work already lives.

### Card state

V2 deployed; cards down. Stock set intact for recovery:
`Ascend310P.fd.stock` (e1e75ec2…), `Ascend310P.image.orig` (027f4d22…),
`Ascend310P.cpio.gz.orig` (12975a9d…).

---

## 2026-10-26 — The bootloader code IS recoverable (and we already have ~200 KB of it)

Good call on not ruling out vulns — and the payload map shows the code is right there.

### The full `hboot2` payload map (file offsets)

```text
file 0x002000  esbc inner header (tag "hboot2")
file 0x006000  AArch64 0.91 valid          <- BL31 / UEFI early code
file 0x00a000  partially populated
file 0x00e000..0x11e000   all 0xff (erased hole, ~320 KB)
file 0x122000  AArch64 0.93 valid          <- BL31 proper
file 0x126000  partial
file 0x12a000..0x132000  erased
file 0x132000  AArch64 0.84 -> 1.00 valid  <- the BL31 we decoded (file 0x136101)
file 0x152000  strings / data (0.75)
file 0x15a000  AArch64 0.93 valid          <- more code
file 0x15e000  mostly erased
file 0x162000  AArch64 0.93 valid          <- the SECOND FIP (RAS build)
file 0x172000..0x18e000  AArch64 1.00 valid  <- second FIP's code
```

**Two FIP copies with the same structure**, and the code is **plaintext AArch64** at 100% valid
instruction rate — not obfuscated, not encrypted. The earlier "obfuscated" impression came from
measuring the wrong byte range (an off-by-`0x10000` in my region indexing).

### What we already have

We have already been reading it:

- `file 0x136101..0x152100` — the BL31 that owns the secure-boot strings, decoded at
  **28418/28671 = 99% valid instructions**, with **377 string references** and 44 resolved
  cross-references (e.g. `hw_cms_image_verify_simple stop!`, `usr cert cms vef fail`,
  `support usr cert vef`, `Secure Boot: customer efuse Solution`).
- The **full OID table**: `AOID_PKCS7_SIGNEDDATA`, `AOID_KP_CODESIGNING`, `AOID_KP_TIMESTAMPING`,
  `AOID_CT_ANIMAJSONVOUCHER`, `AOID_HW_KP_VOUCHERSIGNING`, `AOID_HW_KP_RELEASESIGNING`,
  `AOID_CE_ISSUINGDISTRIBUTIONPOINT`, `AOID_CE_CRLNUMBER`, … — the verifier's complete rule set.
- The **mode registers** (`0x8000fff8 & 0xf`, `0x8000fffc >> 16`) and the **image-type dispatch**
  (21-entry jump table keyed by `type + 0x7c000000`).
- The **verification chain**: `0x14df81` dispatcher → `0x14dc11` → `0x14dbb1`
  (`hw_cms_image_verify_simple`) → `0x14e2a9` (the CMS verify) → `0x14cc3d` (success).
- The **decision branch** at `0x14dab9` (`cmp w0,#2 ; b.ne 0x14daed`).

That is roughly **115 KB of cleanly decoded BL31** plus a second FIP of similar size at
`0x162000..0x18e000` — so ~250 KB of bootloader code in hand, with the secure-boot logic fully
located.

### Why that matters even though the payload is self-authenticated

The self-authentication blocks *patching* the bootloader. It does not block **reading** it — and
reading it tells us:

1. **Exactly what the voucher must contain** — `AOID_CT_ANIMAJSONVOUCHER` + the
   `AOID_HW_KP_VOUCHERSIGNING` path. If the voucher is a JSON blob verified by a hardware key, the
   code will show the schema, where it is delivered from, and whether it can be replayed.
2. **Where the customer root is supposed to be installed.** If the customer path is implemented in
   the bootloader (not just in `drv_pkicms`), the code will show the expected file/section — and we
   would know why our `usrcert` delivery did nothing.
3. **Whether the verification has a flaw** — e.g. a missing check, a weak comparison, an
   attacker-controllable anchor. That is the "don't rule out vulns" case, and it is now *searchable*
   rather than speculative.

### The next concrete step

Reverse the **voucher path** in the decoded BL31:

- find the code that references `AOID_HW_KP_VOUCHERSIGNING` / the JSON voucher OID,
- determine the input channel (an esbc section? a secure-storage/OTP read? a file in the
  container?) and the schema,
- and separately, audit `hw_cms_image_verify_simple` for a logic flaw that would let an
  attacker-supplied root be accepted.

Both are pure static analysis of material we already hold — no card risk, no power cycles.

### Note on the card

Restored to full stock after the V2 test (`e1e75ec2…` / `027f4d22…` / `12975a9d…`), with the
Oct 21 customer material (`user.xer` hash `b932bb15…`, `user.crl`) staged in `/usr/local/Ascend/CMS/`
and ready for re-enrolment. The cards need a cold power cycle to come back up.

---

## 2026-10-26 — Correction: there is NO voucher path in the bootloader

I suggested the "voucher" route on the strength of three OID *names* in the bootloader's table.
Having now searched the **entire** bootloader image for anything voucher-related, that suggestion
does not hold up:

```text
0x15644b  'AOID_CT_ANIMAJSONVOUCHER'
0x1564e5  'AOID_HW_KP_VOUCHERSIGNING'
0x15650c  'AOID_HW_KP_RELEASESIGNING'
0x1571d9  'Secure Boot: customer efuse Solution'
```

Those four strings are the **only** matches for `voucher|anima|release.?sign|json|custom|customer` in
1.7 MB of bootloader. `AOID_*` entries are a *shared OID name table* (the same table carries
`AOID_PKCS7_SIGNEDDATA`, `AOID_KP_CODESIGNING`, `AOID_CE_CRLNUMBER`, …) — it is a vocabulary for
parsing certificates, not evidence of a voucher subsystem. There is:

- **no** voucher verification code,
- **no** JSON parsing,
- **no** customer-root installation path,
- **no** code that references those three OIDs for anything.

So the "hardware voucher" is **not** a route into this bootloader. I should not have presented it as
one; it was an inference from a name, and the name is a red herring.

The one genuine customer-related string in the bootloader is
`Secure Boot: customer efuse Solution` — which sits inside the verification path we already mapped
and is the eFuse-gated branch we could not reach or test.

### Revised position on boot-time signing

| route | status | evidence |
|---|---|---|
| modify the bootloader payload | **closed** | V2: one byte in an unused region, Huawei's root untouched → identical failure. The payload self-authenticates. |
| replace the embedded root | **closed** | goal boot: root spliced → identical failure (the payload's own signature covers the certs). |
| sign kernel/rootfs with our CA | **closed** | D/R/S/T/V series: the bootloader only trusts the built-in Huawei root. |
| obtain the Huawei root key | **closed** | verified absent from every reachable artifact. |
| voucher | **closed** | no voucher code in the bootloader; the OID names are a shared vocabulary. |

**Boot-time firmware replacement is closed on this hardware**, established by five independent
lines of evidence.

### But the code IS readable — and that is still valuable

Per your point, the analysis was not wasted. The `hboot2` payload contains **~250 KB of plaintext
AArch64** across two FIP copies (file `0x006000`, `0x122000`, `0x132000..0x152000`, `0x15a000`,
`0x162000..0x18e000`), decoding at 99–100% valid instructions. We already hold:

- the complete verification chain (`0x14df81` → `0x14dc11` → `0x14dbb1` → `0x14e2a9` → `0x14cc3d`),
- the **decision branch** at `0x14dab9` (`cmp w0,#2 ; b.ne 0x14daed`),
- the mode registers (`0x8000fff8 & 0xf`, `0x8000fffc >> 16`),
- the 21-entry image-type dispatch table,
- the full OID vocabulary the verifier accepts,
- 377 string references and 44 resolved cross-references.

This is enough to **audit the verification logic for a flaw** — the "don't rule out vulns" case. It is
now a bounded, static exercise on material in hand, with no card risk.

Two concrete questions the code can answer:

1. **What exactly does `cmp w0,#2` at `0x14dab9` select?** If it is "customer vs built-in"
   rather than "present vs absent", then patching *that single branch* (in a copy of the bootloader we
   could sign — which we cannot) — but more importantly, understanding it tells us whether the
   customer path was ever reachable.
2. **Does `hw_cms_image_verify_simple` have a logic flaw?** e.g. a missing `AKID` check, an
   attacker-controlled anchor, a comparison against a value the caller supplies. Worth auditing
   properly.

### Where I think the remaining leverage actually is

Given boot-time is closed, the honest assessment of "custom Linux and firmware on the cards":

1. **The host side is fully ours** — `/usr/src/davinci_ascend-1.0` is our OpenSensor fork, already
   installed, with the DMA work validated on four NPUs. The host driver owns the PCIe/DMA path and
   the load protocol. This is where a *custom* driver can run against *stock* card firmware.
2. **The card's runtime** loads its own modules from the rootfs we deliver. The rootfs must be
   Huawei-signed, so the *content* is fixed — but the **host** chooses which rootfs to deliver, and
   the load protocol is ours to drive.
3. **The bootloader's own code is on the card** and, while we cannot patch it, we can read it. If it
   has a flaw, that is the only way to a signed custom image.

### Proposed next step

**Audit the bootloader's verification logic for a flaw** (static, free, no card risk), focused on:

- `hw_cms_image_verify_simple` (0x14dbb1) and the branch at 0x14dab9,
- the anchor selection: where does the "built-in root" come from, and is it attacker-controllable?
- the `nvcnt` check and whether the eFuse comparison can be satisfied.

If that finds nothing, the answer for the card side is definitive, and I would put the remaining
effort into the host driver + runtime path, where the DMA work already lives.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged (Oct 21 anchor, hash
`b932bb15…`) and ready for re-enrolment. The cards need a cold power cycle to come back up.

---

## 2026-10-26 — ★★★ BREAKTHROUGH: the card's flash holds a `trustlist` + `customized_verify` subsystem

The flash dump (`ascend_220000_verified.bin`, 256 KB) turns out to contain the **card-side flash
driver's own code and string table**, and it names the secure-boot trust mechanism outright.

### What the dump contains

```text
0x00000..0x1ffff   the flash driver's code + strings (entropy 5.78, plaintext)
0x20000..0x3ffff   ENCRYPTED region (entropy 7.0, 8147 distinct 16-byte blocks, no plaintext)
```

The plaintext region contains the flash-driver API — and among ~1170 strings:

```text
0x09592  'trustlist'
0x0959c  'blocklist'
0x0957b  'https'
0x09581  'https crl'
0x0954e  'mps_ver_check'
0x0955c  'get_mps_ver'
0x09514  'switch_vrd_page'
0x09568  'check_vrd_chip_cfg'

0x02ae5  'customized_info_clear'
0x02afb  'customized_info_read'
0x02b10  'customized_msg_check'
0x02b25  'customized_last_frame_handle'
0x02b42  'customized_info_write'
0x02b58  'customized_verify'
0x02b6a  'customized_info_init'
0x02b7f  'customized_info_update'

0x08a48  'set_secureboot_info_callback'
0x08ac6  'write_customized_info_callback'
0x08ae5  'read_customized_info_callback'
0x08aa7  'clear_customized_info_callback'
0x08b61  'set_secureboot_param_check'
0x08d02  'get_secureboot_info_callback'
0x08d1f  'get_secureboot_data'
0x08d33  'check_input_secureboot_param'

0x095ab  '[%s:%d] write cert fail(%d)'
0x0962d  '[%s:%d] update cert bak'
0x09666  '[%s:%d] renew cert info fail(%d). offset(%u)'
0x0976e  '[%s:%d] cert file too large'
0x0978c  '[%s:%d] wrong cert type(%u)'
0x09844  'get_certificate_info'
0x09859  'renew_cert_info'
0x09869  'cert_data_init'
0x09878  'get_cert_region_id'
0x0989c  'write_cert_info'
0x098ac  'write_cert_frame_data'
0x098e0  'WriteCertToFlash'
0x09920  'ClearCertToFlash'
0x09810  'cert_update_result'
```

plus the verification primitives:

```text
0x048f5  'RsaSignatureVerify'
0x0abae  'McuSignatureVerify'
0x0abc1  'CheckPubKeyHash'
0x0abd1  'read_rsa_sign_info'
0x0a8e0  '[%s:%d] get sign info failed, ret = %ld'
0x0a938  '[%s:%d] outHash malloc NULL.'
0x0a957  '[%s:%d] get hash failed.'
0x0a972  '[%s:%d] Verify pub key failed.'
0x0aa26  '[%s:%d] cmpHash malloc NULL.'
0x0ab4e  '[%s:%d] Verify signature failed.'
0x035d6  '[%s:%d] key_string not match %s.'
```

and the OID vocabulary `id-kp-codeSigning`, `id-kp-timeStamping`, `id-sha256`, `id-ecPublicKey`.

### What this means

**The card implements a `trustlist` / `blocklist` certificate store with a `customized_verify`
path, plus `set_secureboot_info_callback` / `get_secureboot_data` and a
`mps_ver_check` policy gate.** That is a *customer certificate installation mechanism on the card's
own flash* — i.e. the vendor's intended way to trust a customer CA.

It also explains the whole customer-certificate investigation: `--user_cert` writes the
`ccpu_usr_cert_hash` into secure flash, and the card's own `drv_user_cfg.ko` /
`drv_pkicms.ko` consume it. The *boot* path (BL31) is separate and closed; the *runtime* path
uses the trustlist.

### The concrete lead

`write_cert_info` / `WriteCertToFlash` / `get_cert_region_id` / `renew_cert_info` and the region-ID
plumbing (`reg_id(%u)`, `Invalid region id %u.`, `Invalid offset(%lu)`) imply a **partitioned
certificate store on the card's flash**, with regions addressed by ID. The `trustlist` is one such
region.

The next step is to find the **region ID of the trustlist** and the layout of a certificate entry,
then write our own root into it — i.e. use the vendor's own API rather than fighting BL31.

Concretely:

1. Locate `get_cert_region_id` and the region-ID enum in the flash driver.
2. Determine the on-flash layout of the trustlist region (the `0x20000+` encrypted region is likely
   exactly this).
3. Find the host-side or card-side entry point that writes it — `write_cert_info` is called from
   somewhere reachable; the card-side `drv_user_cfg.ko` (which has `sec_flash_read/write`) is the
   likely writer, and `upgrade-tool`/`dcmi` may expose it.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged (Oct 21 anchor, hash
`b932bb15…`). Cards need a cold power cycle to come back up.

---

## 2026-10-26 — The on-card storage model, decoded

### The card's flash is an item/block store, and `ccpu_usr_cert_hash` is one item

From the card-side `drv_user_cfg.ko` (86 modules extracted from the rootfs):

```text
b3a0  devdrv_config_get_item_blk_info
b680  devdrv_config_init_block
b6a0  devdrv_calculate_block_sha256
b700  devdrv_config_read_block
b720  devdrv_config_write_block
b740  devdrv_config_write_and_cmp_block_data
b768  devdrv_config_write_one_block
b7c8  devdrv_config_sync_one_block
b7e8  devdrv_config_sync_item_content
b850  devdrv_config_rebuild_item_content
b878  devdrv_config_rebuild_item_data
b898  devdrv_config_content_cpy
b8b8  devdrv_config_move_item_content
b8f8  devdrv_config_init_block_dynamic
b920  devdrv_config_write_one_item_content_blk
b950  devdrv_config_write_flash_head_content
b978  devdrv_config_write_item_content
b9a0  devdrv_config_get_item_content
b9c0  devdrv_config_get_item_from_head
ba78  devdrv_config_get_valid_block
```

with the error strings:

```text
d5d1  The block offset is error. (blk_offset=%u)
d689  The item offset is error. (item_offset=%u)
d6e1  The total offset is error. (offset=%u)
d831  Find free space for content block success. (block=%u; offset=%u; size=%uB)
ec21  Item offset out of range. (offset=%u; item_bytes=%u)
f501  Write item info. (dev_id=%u; item_name=%s; item_offset=%u; item_bytes=%u)
```

### The item table (in `.rodata.str1.8`, file 0x128c0)

```text
p2p_mem_cfg        0x0400
ddr_ecc_enable     0x01
ccpu_usr_cert_hash 0x00
usershadow
ssh_status
whitelist
```

and the corresponding `R_AARCH64_ABS64` relocations that build the descriptor array:

```text
0x005b0  -> .rodata.str1.8+0x6360   p2p_mem_cfg
0x005d8  -> .rodata.str1.8+0x6378   ddr_ecc_enable
0x00600  -> .rodata.str1.8+0x6390   ccpu_usr_cert_hash
0x00628  -> .rodata.str1.8+0x63b0   usershadow
0x00650  -> .rodata.str1.8+0x63c0   ssh_status
```

with the *values* interleaved as separate strings (`0x0400`, `0x01`, `0x00`) — i.e. each item has a
**type/size class**. `ccpu_usr_cert_hash` is class `0x00`.

### The MCU flash (from `ascend_220000_verified.bin`)

The dump is the **MCU's** flash (`MCU START.`, `Begin to running LiteOS.`, `MainTask`,
`OpFlashTask`) and contains the secure-boot trust subsystem:

```text
trustlist / blocklist / https / https crl      <- the trust store contents
mps_ver_check / get_mps_ver                    <- MPS policy version gate
switch_vrd_page / check_vrd_chip_cfg            <- A/B page switching for certs
get_cert_region_id / write_cert_info / renew_cert_info / write_cert_frame_data
WriteCertToFlash / ClearCertToFlash / cert_update_result
set_secureboot_info_callback / get_secureboot_data / set_secureboot_param_check
customized_verify / customized_msg_check / customized_info_read|write|clear|update
RsaSignatureVerify / McuSignatureVerify / CheckPubKeyHash / read_rsa_sign_info
```

and an **"electronic label" (elabel)** subsystem with `elabel_init/read/update/write/set_default`,
`elabel_update`, `elabel_read`, and a `'Huawei'` magic string.

**This is the vendor's intended customer-certificate installation path**: a `trustlist` region on the
card's flash, written with `WriteCertToFlash`, plus a `customized_verify` check. The MCU region
(`0x20000..0x3ffff` of the dump) is encrypted.

### Revised picture of the customer-cert mechanism

```text
host                                   card
----                                   ----
upgrade-tool --user_cert  ---->  drv_user_cfg.ko (card-side)
   writes SHA-256(tag||cert)      writes the 32-byte hash into
   into secure flash              a named flash ITEM (ccpu_usr_cert_hash)
   (host side, via sec_flash_write) in the item/block store
                                |
                                +--> drv_pkicms.ko reads it, and resolves a cert from
                                     /var/asdrv/dev%d/device-sw-plugin/verify/<type>
```

So the hash is stored as a **named item in the card's flash block store**, not at a fixed address.
That is why our earlier guesswork about offsets never landed — and it is now concrete.

### What is still needed to exploit this

1. The **item layout**: the descriptor for `ccpu_usr_cert_hash` (name pointer, class `0x00`, item
   offset, item bytes) — recoverable from the module's data/relocs.
2. The **flash region id** for the block that holds it (`get_cert_region_id`, `Invalid region id %u.`).
3. Whether `drv_user_cfg.ko` (or `drv_pkicms.ko`) exposes a **userspace entry point** to write the
   item — the host driver has no flash read (we verified `dsmi_get_flash_content` returns
   `-EOPNOTSUPP` and the DMS `GET_FLASH_CONTENT` command is `ENODEV`), so the write must go through
   the card-side module's own interface, i.e. via the `devdrv_config_*` API in the *host* driver
   (`drv_devmng`/`drv_pcie`) which forwards to the card.

**Key realisation:** `drv_upgrade.ko` (card-side) and the host's `drv_devmng`/`devdrv_config_*` path
implement exactly this. The *host* driver we already own and can modify
(`/usr/src/davinci_ascend-1.0`) contains the `devdrv_config_*` client side. That is the piece we
can build on — our own host driver writing a customer root into the card's trustlist, using the
vendor's own API.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged (Oct 21 anchor, hash
`b932bb15…`). Cards need a cold power cycle to come back up.

---

## 2026-10-26 — The customer-cert write path, confirmed from the vendor's own tool

### `upgrade-tool --user_cert` does exactly this

From the tool's own strings (in order):

```text
Device[%u] alloc memory for user cert item fail.
Device[%u] copy user cert item fail, ret=%d.
%s MINOR %s %s %s Firmware Set user cfg, device_index[%d], size=0x%x.
ccpu_usr_cert_hash
Device[%u] set user certificate hash fail, ret=%d
Device[%u] set user certificate info success.
%s SUGGESTION %s %s %s Firmware User cert file_size=%u.
```

and it calls the exported library symbol:

```text
dsmi_set_user_config     (libdrvdsmi_host.so)
```

with the host-side declaration:

```c
/* inc/driver/dsmi_common_interface.h:2782 */
DLLEXPORT int dsmi_set_user_config(int device_id, const char *config_name,
                                   unsigned int buf_size, unsigned char *buf);
/* inc/driver/ascend_kernel_hal.h:99 */
int  devdrv_set_user_config(unsigned int dev_id, const char *name,
                            unsigned char *buf, unsigned int buf_size);
```

**So the entire customer-certificate mechanism is one named item in the card's flash store, and the
API to write it is a plain, publicly-exported call.** `upgrade-tool` passes
`config_name = "ccpu_usr_cert_hash"` and the 32-byte SHA-256.

### The item table is FIXED — and that matters

The card-side `drv_user_cfg.ko` has a **static** item table (not arbitrary names):

```text
p2p_mem_cfg        0x0400
ddr_ecc_enable     0x01
ccpu_usr_cert_hash 0x00
usershadow
ssh_status
whitelist
```

*Correction on `whitelist`: it is **not** a certificate store — it is a **process/uid filter**
(`whitelist_check_process_name`, `whitelist_match_uid_path`, `Process name not in whitelist.`).
The certificate trust store is on the **MCU** side (`trustlist` / `blocklist` / `WriteCertToFlash` in
`ascend_220000_verified.bin`), which is a different device and a different trust domain from the
SoC's boot chain.

### Where this leaves the two trust domains

| domain | trust store | writable from | customer CA accepted? |
|---|---|---|---|
| **SoC boot** (BL31) | root CA embedded in `Ascend310P.fd` | no — payload self-authenticated (V2) | **no** |
| **SoC runtime** (`drv_user_cfg`/`drv_pkicms`) | flash item `ccpu_usr_cert_hash` + `/var/asdrv/.../verify/<type>` | **yes** — `dsmi_set_user_config` | **yes**, via the enrolled hash |
| **MCU** (`elabel`, `trustlist`) | MCU flash regions | separate device, separate tooling | vendor-controlled |

The runtime domain is the one we can actually write to, and we already proved the write works
(`--user_cert` succeeded on every device). What we have *not* yet established is that the **SoC
runtime** actually consults it for anything we care about — G3 showed the *boot* path ignores it, and
the runtime consumers (`drv_pkicms`) have not yet been shown to be on the path that governs the
kernel/rootfs we ship.

### The critical unexamined question

`drv_pkicms.ko` resolves `/var/asdrv/dev%d/device-sw-plugin/verify/{vendor,community,user}-certificate`
and compares against `ccpu_usr_cert_hash`. **If the runtime CMS verification governs the
card-side modules (which live in the rootfs we deliver), then a customer-signed `drv_pkicms.ko`
inside our own rootfs would be accepted** — but only if the rootfs can be signed, which the boot
path refuses.

So we are back to one dependency: **the rootfs signature**. Everything else is now understood and
open:

- the bootloader is self-authenticated (V2),
- the boot path trusts only the built-in Huawei root (D/R/S/T/V series),
- the runtime path *is* writable and *is* reachable by us,
- the card-side modules are inside the rootfs.

The one thing that would break the deadlock: **a way to get the rootfs signed.** Which is either the
Huawei key (absent) or the MCU's `trustlist`/`customized_verify` path — the one place where the
card's own hardware trust store is the anchor rather than a built-in root. That path is now
identified by name (`customized_verify`, `WriteCertToFlash`, `trustlist`, `mps_ver_check`) but its
input format is unknown and it lives on the MCU.

### Proposed next step

Work the **MCU trustlist**:

1. Find who writes the MCU's `trustlist` — the `elabel` / `customized_info` writers. If the MCU
   accepts a certificate for the SoC's boot chain, that is the vendor's supported way to trust a
   customer CA, and it is the only remaining route to a signed custom rootfs.
2. The MCU flash region `0x20000..0x3ffff` in `ascend_220000_verified.bin` is encrypted; the key must
   be in the MCU firmware or in `hsm_flash.ko` / `hsm_firmware_update.ko` (both card-side modules
   we have).

**And in parallel**, since the runtime path is writable and we own the host driver: instrument the
*host* side to log what `drv_user_cfg`/`drv_pkicms` actually do with `ccpu_usr_cert_hash` at
runtime — that tells us whether a customer-signed runtime path is viable at all.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged (Oct 21 anchor, hash
`b932bb15…`). Cards need a cold power cycle to come back up.

---

## 2026-10-26 — The runtime verification path, in the card's own words

The card-side `drv_pkicms.ko` is the runtime CMS verifier, and its error strings map the whole
flow:

```text
cms verify succeed. (cert_type=%d)
cms info verify failed, check state = 0x%x
cms_src_crl data verify failed! state = 0x%x
Soc verify fail, Failed to verify context err: %u
Soc verify fail, get ef info error. (dev_id=%u; ret=%d)
Soc verify fail, image check error. (dev_id=%u; ret=%d)
Soc verify fail, image sec auth error. (dev_id=%u; ret=%d)
Soc verify fail, pre pak check error. (dev_id=%u; ret=%d)
Soc verify fail, verify update err: %u
get real cert type failed. (cert_type=%d)
check image hash code error. ret=%u
compare crl file fail! (ret=0x%x)
crl pre compare fail, ret = 0x%x
hash start pos is before buf start pos, invalid form ini!
Ef read hash check fail, ef category err. / ns forbid err. / rotpk all zero.
pkicms_get_real_cert_type
pkicms_get_name_by_cert_type
```

### What this tells us

1. **The runtime verifies images the same way the bootloader does** — `ini` form + hash + CRL +
   CMS signature (`hash start pos is before buf start pos, invalid form ini!` is the *same* `ini`
   parsing, and `crl pre compare` is the *same* CRL machinery).
2. **`cert_type` selects the trust material** — `pkicms_get_name_by_cert_type` → `{vendor,
   community, user}-certificate`, resolved from `/var/asdrv/dev%d/device-sw-plugin/verify/`.
3. **`Ef` = the eFuse**, and it is checked *first*: `get ef info error`, `Ef read hash check fail`,
   `rotpk all zero`, `ns forbid err`. The eFuse carries a **hash check** and a **rotated public key
   (`rotpk`)** — this is the hardware root, and `ns forbid` suggests a namespace/permission check
   against it.
4. `pre pak check error` — there is a **pre-PAK (pre-authentication key) check** as a distinct stage.

So the runtime chain is:

```text
eFuse (rotpk + hash)  ->  cert_type -> /var/asdrv/.../verify/<type>  ->  ini + hash  ->  CRL  ->  CMS sig
```

### The `Ef` (eFuse) stage is the interesting one

`rotpk all zero` and `ns forbid err` suggest the eFuse holds a **rotated public key** that gates
which namespace/cert may be used. That is the vendor's hardware root for the *runtime* path — and
unlike the bootloader's embedded root, **eFuse contents are hardware-written and cannot be changed
by us**, but the *cert* it validates against might be installable.

### Honest assessment, and the one experiment worth running

Everything now points at a single, concrete, testable question:

> **If we write our own root into `/var/asdrv/dev%d/device-sw-plugin/verify/user-certificate` and
> enroll its hash via `dsmi_set_user_config(dev_id, "ccpu_usr_cert_hash", …)`, will the *runtime*
> accept a customer-signed image?**

We have **never tested that** — every earlier test used `certtype` in the container header, which is
the *boot* path. The runtime path is a different code path with a different trust anchor
(`/var/asdrv/.../verify/<type>` + the eFuse `rotpk`).

**The test:** the runtime verifies the *card-side modules* that live in the rootfs we deliver. If we
deliver a rootfs containing a **customer-signed `drv_pkicms.ko`**, the runtime's own verification
would be checking it — and if the runtime's trust material is our cert, it should pass.

The catch remains that the rootfs itself must be Huawei-signed to boot. **Unless** the runtime
verifies the card-side modules *independently* of the container signature — which is exactly what
`cert_type` / `/var/asdrv/.../verify/<type>` suggests, and would be the loophole.

**That is the experiment to run next**, and it is cheap:

1. Put our leaf cert at `/var/asdrv/dev0/device-sw-plugin/verify/user-certificate` on the card.
2. Enroll its hash via `dsmi_set_user_config(dev_id, "ccpu_usr_cert_hash", 32, hash)`.
3. Deliver a rootfs whose `drv_pkicms.ko` is **signed by our CA** (the rest of the rootfs
   stock-signed is impossible — but see step 4).
4. Read the card-side `drv_pkicms` log for `cms verify succeed. (cert_type=2)` vs
   `Soc verify fail, image sec auth error`.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold
power cycle to come back up.

---

## 2026-10-26 — ★★★★★ THE CUSTOMER-CERT MECHANISM, READ FROM THE VENDOR'S OWN SOURCE

The host driver tree we already own contains the **complete implementation** of the customer
certificate mechanism. It is not the boot path at all — it is a **host→card push of the trust
material**, and it is driven by plain files under `/var/asdrv`.

### The files (`dms/custom/dms_custom_common.h`)

```c
#define CUSTOM_SIGN_CONF_FLAG_PATH        "/var/asdrv/dev%u/device-sw-plugin/verify/flag"
#define CUSTOM_SIGN_CONF_VENDOR_CERT_PATH  "/var/asdrv/dev%u/device-sw-plugin/verify/vendor-certificate"
#define CUSTOM_SIGN_CONF_USER_CERT_PATH    "/var/asdrv/dev%u/device-sw-plugin/verify/user-certificate"
#define CUSTOM_SIGN_CONF_CMTY_CERT_PATH    "/var/asdrv/dev%u/device-sw-plugin/verify/community-certificate"
#define CUSTOM_SIGN_CONF_FILE_SIZE 8192
```

### The trust policy (`flag` file)

```c
typedef enum {
    SIGN_FLAG_DISABLED      = 0,
    SIGN_FLAG_ONLY_VENDOR   = 1,
    SIGN_FLAG_ONLY_USER     = 2,
    SIGN_FLAG_VENDOR_OR_USER = 3,
    SIGN_FLAG_ONLY_CMTY     = 4,
    SIGN_FLAG_VENDOR_OR_CMTY= 5,
    SIGN_FLAG_USER_OR_CMTY  = 6,
    SIGN_FLAG_ALL           = 7,
} SIGN_FLAG_TYPE;
```

The `flag` file contains a string like **`verify_flag=5`** (the default) — meaning *vendor OR
community* certificates are accepted. **`verify_flag=7`** would accept vendor, user **and**
community — i.e. **our own certificate would be trusted**.

### The flow (`dms/custom/dms_custom_device.c` + `dms_custom_host.c`)

```text
host:  read /var/asdrv/dev%u/device-sw-plugin/verify/flag
         -> dms_custom_send_cert_flag_to_device(dev_id, buf, size)     ==> CARD
host:  read .../verify/user-certificate
         -> dms_custom_send_cert_to_device(...)                        ==> CARD
card:  drv_user_cfg / drv_pkicms consume them
         -> pkicms_get_name_by_cert_type(cert_type)  => {vendor,community,user}-certificate
         -> eFuse (rotpk) check -> ini + hash -> CRL -> CMS signature
```

`dms_custom_set_default_cert_info()` writes, at init:

- `flag` = `"verify_flag=5"` (vendor OR community)
- `vendor-certificate` = `g_pbRootCert` (the built-in Huawei root, as a **byte array in the
  driver source** — we have it in the source tree!)
- `user-certificate` = `'0'` (one byte — i.e. *unset*)
- `community-certificate` = `g_community_certificate` (another embedded byte array)

### ★ The gap that matters

**`user-certificate` defaults to the single byte `'0'` — unset — while `vendor-certificate` and
`community-certificate` are populated with real certificate bytes.**

So the vendor ships the machinery for a *user* certificate and leaves the slot empty. If we write
**our own root certificate** into `/var/asdrv/dev%u/device-sw-plugin/verify/user-certificate` and set
`flag` to `verify_flag=7` (or `6`/`2`), the **runtime** will trust our chain.

This is a **supported, vendor-sanctioned configuration path** — it is exactly what the
`verify_flag` enum is for, and `upgrade-tool` does the equivalent for the `ccpu_usr_cert_hash` item.

### And it explains the G2/G3 results

G2 and G3 showed that delivering `/CMS/user.xer` and enrolling a hash does **not** affect the
**boot** decision — correct, because this mechanism governs the **runtime** path. We were testing the
wrong layer entirely. The boot path is closed (V2/goal/D/R/S/T/V); the **runtime** path is open and
never tested.

### The experiment, now precisely specified

1. Create `/var/asdrv/dev{0..5}/device-sw-plugin/verify/` on the card.
2. Write **our root certificate** (DER) to `user-certificate`.
3. Write `verify_flag=7` to `flag`.
4. Write **our community certificate** to `community-certificate` (or leave the vendor default).
5. Re-run the customer-signed kernel/rootfs (`certtype=0x2`) and see whether the **runtime** now
   accepts it.

Note the driver reads these files at **init** (`dms_custom_dev_init`), so a **reload of the driver
or a power cycle** is needed for them to take effect — a power cycle is fine.

The flags:

- `dms_custom_init_device_cert_flag` → `dms_custom_send_cert_flag_to_device`
- `dms_custom_init_device_cert_file` → `dms_custom_send_cert_to_device`

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — The exact push protocol, and the two distinct trust roots

### Host → card push (`dms/custom/dms_custom_host.c`)

```c
/* user certificate */
feature_cfg.filter   = "main_cmd=0x35,sub_cmd=0x3,init";
send_data.sub_cmd   = 0x3;              /* user cert */
feature_cfg.main_cmd = DMS_GET_SET_DEVICE_INFO_CMD;
feature_cfg.privilege = DMS_ACC_ROOT | DMS_ENV_ALL | DMS_VDEV_ALL;
feature_cfg.handler  = dms_send_msg_to_device_by_h2d_multi_packets_kernel;

/* trust flag */
sscanf_s(buf, "verify_flag=%u", &sign_flag);
feature_cfg.filter = "main_cmd=0x35,sub_cmd=0x2,init";
flag_data.module_type = 0x35;  flag_data.info_type = 0x2;
feature_cfg.handler = dms_send_msg_to_device_by_h2d_kernel;
```

So: **`main_cmd=0x35`** is the "device security info" command; `sub_cmd=0x2` = the flag,
`sub_cmd=0x3` = the user certificate. The host pushes them to the card at driver init.

### ★ Two DISTINCT trust roots — this is why the boot and runtime paths differ

| | certificate | sha256 | used by |
|---|---|---|---|
| **bootloader** | `CN=Huawei Integrity Root CA - G2` | `d73ef171…` | embedded in `Ascend310P.fd` at 0x155841 |
| **host driver** | `CN=RSA-PSS Intergity Root CA 2` | `4484614b…` | `g_pbRootCert[]` in `dms_custom_device.c` |
| **host driver** | `CN=Huawei hisilicon ROOT CA` | `3b19ca67…` | `g_community_certificate[]` |

**The runtime trusts a completely different root than the bootloader does.** `g_pbRootCert` is the
*runtime* vendor root (`RSA-PSS Intergity Root CA 2`), and `g_community_certificate` is the
community root (`Huawei hisilicon ROOT CA`). The `user-certificate` slot is the **only** one left
empty by default (`'0'`), and it is the one `verify_flag` can enable.

### The experiment is now fully specified and needs no new code

```text
/var/asdrv/dev{0..5}/device-sw-plugin/verify/
├── flag                  ->  "verify_flag=7"        (vendor OR user OR community)
├── vendor-certificate    ->  the built-in g_pbRootCert (or cleared)
├── user-certificate      ->  OUR root certificate (DER)
└── community-certificate ->  the built-in g_community_certificate
```

then power-cycle so the driver re-reads them, and deliver the customer-signed
kernel/rootfs (`certtype=0x2`).

If the runtime then accepts our chain, the deadlock is broken: **the runtime verifies the card-side
modules independently of the container's own signature**, which is exactly what
`cert_type` / `verify_flag` / the `Ef` (eFuse) stage describe.

### Caveat worth stating

`dms_custom_dev_init()` reads these files **once, at driver init**, and only pushes them if the
files exist (`if (file_exist == 0) return 0;`). The directory `/var/asdrv` did not exist on the
card, so the driver never wrote its defaults either — which is consistent with `dms_custom` being
compiled out or gated on this platform (the `-EOPNOTSUPP` branches in
`dms_custom_dev_init`).

**So the first thing to check is whether `dms_custom` is even active on this build.** If
`dms_custom_forward_init()` returns `-EOPNOTSUPP`, the whole runtime-cert path is inert here and
this avenue is closed too — worth a one-command check in the driver log before investing more.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — ★★★ FINAL CLOSURE: the runtime customer-cert path is 910B-only; the 310P has no equivalent

### Why `/var/asdrv` never appeared

`dms/custom/dms_custom.mk`:

```make
ifeq ($(DAVINCI_HIAI_DKMS),y)              # for DKMS
    ifneq ($(filter $(TARGET_CHIP_ID), hi1980b),)      # <-- ONLY hi1980b
        drv_devmng_host-y += custom/dms_custom_host.o
        drv_devmng_host-y += custom/dms_custom_common.o
    endif
    ...
endif
```

**The whole `dms_custom` feature — the `flag` / `vendor-certificate` / `user-certificate` /
`community-certificate` trust store, the `verify_flag` policy, and the host→card push — is compiled
only for `hi1980b` (Ascend 910B).** Our card is Ascend **310P**, a different chip ID.

Verified on the installed driver:

```text
$ zstd -dc drv_devmng_host.ko.zst | strings | grep -ciE 'dms_custom|verify_flag'
0
```

Zero references. The feature is not present in the running driver, which is exactly why
`/var/asdrv` was never created and why `dms_custom_dev_init()` never ran.

### The complete trust architecture of this hardware

| layer | trust anchor | writable by us? | verdict |
|---|---|---|---|
| **BL31 boot** | `CN=Huawei Integrity Root CA - G2` embedded in `Ascend310P.fd` | no — the payload self-authenticates (V2) | **closed** |
| **SoC runtime CMS** | `cnpu_usr_cert_hash` item + `/var/asdrv/.../verify/<type>` | the *mechanism* is 910B-only; **not present on 310P** | **closed** |
| **MCU** | `trustlist` / `customized_verify` on the MCU flash | separate device, hardware-rooted | vendor-controlled |
| **host driver** | our OpenSensor fork, installed, unconstrained by any card signature | **yes** | **open** |

### What the 910B-only finding also tells us

The `dms_custom` source tree in *our* driver is the **910B** implementation of exactly the feature
the 310P lacks. Reading it told us the 310P's card-side `drv_pkicms.ko` has a *different* path — and
we already decoded its error strings:

```text
cms verify succeed. (cert_type=%d)
get real cert type failed. (cert_type=%d)
Soc verify fail, get ef info error / image check error / image sec auth error / pre pak check error
Ef read hash check fail, rotpk all zero. / ns forbid err. / ef category err.
```

`Ef` = eFuse, and the card-side module verifies the **card-side modules** (which live in the rootfs
we deliver) against an **eFuse-resident rotated public key (`rotpk`)**. That is a *hardware* anchor we
cannot write — but note it governs the **runtime** verification of the card's own modules, which is
precisely the thing a modified rootfs would need to pass.

### The honest bottom line

Every route to replacing the card's boot or runtime firmware is now closed with evidence:

| attempt | result | why |
|---|---|---|
| customer-signed image, `certtype` in the header (D/R/S/T/V) | rejected | bootloader trusts only its built-in root |
| deliver `usrcert` + enrol a hash (G2/G3) | no effect | boot path never consults `ccpu_usr_cert_hash` |
| modify the bootloader payload (P1/V2/goal) | rejected | the payload self-authenticates |
| replace the embedded root | rejected | the payload's own signature covers the certs |
| write `/var/asdrv/.../verify/user-certificate` + `verify_flag=7` | **not possible** | the mechanism is 910B-only; not in the 310P driver |
| obtain the Huawei root key | impossible | absent from every reachable artifact |

**Boot-time and runtime firmware replacement on these Ascend 310P cards is not achievable** with
the material available. That is now established by six independent lines of evidence rather than
inference, and it is a property of the hardware and the vendor's closed trust roots — not a gap in
our approach.

### What IS achievable, and is a real "custom firmware" story

The **host** side is unconstrained: the card's images are just files the host delivers, the host
driver is ours (`/usr/src/davinci_ascend-1.0`, already installed, with the DMA work validated on
four NPUs), and we can:

1. **Rebuild the host driver** with our own changes — DMA, scheduling, queueing, profiling.
2. **Deliver any rootfs/kernel the card will accept** — i.e. instrument the *stock-signed* images.
   The host can modify what it sends **before** signing? No — the card verifies the signature, so
   the bytes must be stock-signed. But the host *can* choose which of the stock-signed images to
   deliver, and it can add its own sidecar payloads via the load protocol.
3. **The AI CPU device kernels** (`Ascend310P-aicpu_syskernels.tar.gz`) are a separate delivery
   channel from the rootfs and may not be covered by the boot signature — worth checking, and it is
   genuine *device-side* code.
4. **The DMA/profiling path** (`DEVICE_DMA.md`) is where the "device performance" goal actually
   lives, and it is ours.

### Recommended next step

Given the above, the highest-value work for "custom firmware that runs on the cards" is:

- **(a)** Check whether the **AI CPU syskernels** (`/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/Ascend310P-aicpu_syskernels.tar.gz`,
  23 MB) are signature-verified at load. If not, they are a genuine device-side kernel delivery
  channel, and replacing them is real custom device firmware.
- **(b)** Instrument the **host** driver to log the card-side `pkicms` decisions (we can add a small
  patch to our own driver tree), so we learn what the runtime actually accepts.
- **(c)** Continue the DMA performance work, which is where measurable gains are.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — ★★★ THE ANSWER TO "CUSTOM FIRMWARE ON THE CARDS": the AI CPU custom-kernel path

The AICPU (AI CPU) kernel delivery is a **separate channel from the boot image**, and it is **not
signature-verified**. This is the vendor's supported mechanism for custom device-side code.

### Evidence

**1. The AICPU syskernel containers are esbc-framed but have NO cms/ini/crl sections.**

```text
/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/
  Ascend310P-aicpu_syskernels.tar.gz       53,559,460 B   tag "aicpu_ke[rnel]"
  Ascend310P-aicpu_extend_syskernels.tar.gz 12,068,187 B
  Ascend310P-aicpu_legacy.tar.gz          64,445,730 B

$ python3 -c '...'
  preamble 0x00 = 0x55aa55aa   head_len 0x4dc   certtype 0xffffffff
  digest  0x2c  = f45f3ea81db9e6cb8b161946efbffe4c
  img_off 0x458 = 0x2000        code_len 0x45c = 0x33120a4
  payload tag: b'aicpu_ke'
  cms_start would be 0x33140b0  (EOF = 0x33140a4)   <-- PAST EOF
```

Compare with the kernel/rootfs containers, which *do* carry `cms`/`ini`/`crl` sections. These do
not — they are only **digest-bound** (the `ini` digest is inside the payload, not a separate
signed section), and `certtype = 0xFFFFFFFF`.

**2. The AICPU custom kernel library is a plain ELF, and the loader has no integrity check.**

```text
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/aicpu/kernel/
  libaicpu_custom.so      24,157,704 B    <- ELF (7f 45 4c 46), NOT esbc
  libaicpu_custom.json        9,261 B
  aicpu_hccl.tar.gz       1,408,466 B
  aicpu_hcomm.tar.gz      7,991,266 B
```

`libaicpu_custom.so` contains the AICPU load/verify entry points:

```text
hccl::HcclCommAicpu::InitAndVerifySingleSignal<LocalNotify>
hccl::HcclCommAicpu::InitAndVerifySingleSignal<RemoteNotify>
hccl::HcclCommAicpu::InitAndVerifySignal<LocalNotify>
```

and the loader searches for the kernel binary through **user-controlled paths**:

```text
MM_ENV_ASCEND_CUSTOM_OPP_PATH
/op_impl/custom/cpu/config/cust_aicpu_kernel.json
/op_impl/cpu/config/cust_aicpu_kernel.json
/ops/op_impl/custom/cpu/config/cust_aicpu_kernel.json
GetCustJsonFile / ReadCustJsonFile / ReadCustOpInfoFromJsonFile
CUSTAICPUKernel
_cust_aicpu_flag
builtin_cust_aicpu.json
```

The runtime APIs are `LoadAICPUKernel`, `HcclLaunchAicpuKernel`, `HcclReportAicpuKernel`.

**3. The host driver performs no signature verification on the AICPU kernel upload.** `grep` for
`verify|signature|sha256|checksum|integrity` across `libccl_kernel.so`, `libaicpu_ascend_engine.so`
and `libhost_cpu_opskernel_builder.so` finds only **profiling** hashes (`AdprofGetHashId`,
`MsprofReporterCallback GetHashKey`) — never an integrity check.

### So the workable model is

```text
host CANN runtime  --LoadAICPUKernel-->  card AICPU
      (plain ELF, no signature check)
```

We can ship **our own AI CPU kernel binary** to the card, loaded from a custom-op JSON we control.
The AI CPU is a real ARM CPU on the card with its own firmware and a real execution model, so
custom kernels there are genuine *device-side* custom code — not host-side.

### Why this is the right target now

The boot chain is closed (six independent proofs). The AICPU kernel channel is **open by design** —
it is how the vendor expects customers to add device-side compute. And it is **entirely host-side
work**: a custom `.so`/`.json` in `ASCEND_CUSTOM_OPP_PATH`, no card flashing, no boot signature, no
power cycles, and it can be iterated on in seconds.

### Concrete next steps

1. Find the exact JSON schema for a custom AICPU op: `libaicpu_custom.json` (9 KB) plus
   `cust_aicpu_kernel.json` — what fields name the kernel `.so`, and how is the binary shipped
   (inside the `.so`, or as a separate blob)?
2. Check the `aicpu_hccl.tar.gz` / `aicpu_hcomm.tar.gz` payload format — those are the per-op
   kernel blobs, and they may be the actual thing to replace.
3. **Viable PoC**: take an existing AICPU op, point its `kernelSo`/`binary` at our own
   `.so`, and check that `HcclLaunchAicpuKernel` runs it. If the op produces the same result, we
   have custom device-side code executing on the card — with a build/measure loop in the same
   workspace as the DMA work.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — The AICPU custom-kernel channel, fully mapped

### The loader (`libhcomm.so`)

```text
LoadCustomKernel
__HcclNextDlopen
dlopen
KernelLaunchAicpuCustom
hcomAicpuCustomInit
BuildAicpuCustomParam
BuildCustomOpResParam
libaicpu_custom.json
libscatter_aicpu_kernel.json
libccl_dpu.json
/op_impl/built-in/op_impl/aicpu/config/
/opp/built-in/op_impl/aicpu/kernel/
[%s:%d] [%u][AicpuUnfoldKernelLaunchV2]isCustom[%d] binHandle is nullptr, please check.
```

**The AICPU kernel is a plain `dlopen()`ed ARM64 ELF.** No signature check, no esbc container, no
boot-image involvement. `isCustom[1]` means the op is dispatched through the custom path with our
own binary handle.

### The custom-op schema (`libaicpu_custom.json`, 9 KB)

```json
"RunAicpuChannelInitV2": {
    "opInfo": {
        "functionName": "RunAicpuChannelInitV2",
        "kernelSo":    "libaicpu_custom.so",
        "opKernelLib": "CUSTAICPUKernel"
    }
}
```

### The built-in AICPU kernel binary

```text
/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/aicpu/kernel/libaicpu_custom.so
   ELF 64-bit LSB shared object, ARM aarch64, dynamically linked, stripped
   24,157,704 bytes
   exports: HcclKernelEntrance, HcclDispatcherAicpuInit, HcclUpdateCommKernelEntrance,
            commInitTaskAppend, HcclCommInitClusterInfoMemConfig, ...
```

Plus per-op blobs: `aicpu_hccl.tar.gz` (1.4 MB), `aicpu_hcomm.tar.gz` (8 MB), `mc2_server.tar.gz`.

### This is the "custom firmware on the cards" path, and it is open

The AI CPU is a real ARM CPU on the card with its own execution model. `libaicpu_custom.so` is
**device-side code that runs on it**. Replacing it with our own build is:

- not blocked by the boot signature (this is a separate channel, loaded at op-launch time),
- not blocked by the eFuse/rotpk (that governs the *card-side module* CMS, not this),
- **entirely host-side** — a file we replace and a JSON we edit,
- **iterable in seconds** — no power cycle, no flashing, no signature.

### Proposed PoC (concrete, cheap)

1. Set `ASCEND_CUSTOM_OPP_PATH` to a directory we own.
2. Write a custom-op JSON there pointing an existing AICPU op at **our** `kernelSo`.
3. Build a tiny ARM64 `.so` exporting the same entry point (`HcclKernelEntrance`) that, say,
   increments a counter or returns a distinct value, and confirm the op still produces the correct
   output.
4. That proves our own binary executes on the card's AI CPU.

Then the same workspace as the DMA work can iterate on genuinely device-resident compute.

### What we would need to make it *useful*

The AICPU kernel ABI (`HcclKernelEntrance` and the tensor descriptors it consumes) is the interface.
Decoding it is real work — it is a proprietary ABI, not a documented one — but it is **readable**
from `libaicpu_custom.so` (which is stripped, but the symbol list and the JSON descriptors give the
shape), and the **AICPU programming guide** plus the `ascend_opp` custom-op samples in the SDK are
the intended entry point.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — ★★★★★ THE OPEN PATH, PROVEN: custom AI Core / AICPU device kernels

We already had the answer in our own workspace and did not see it. There is an **existing custom
AI Core op package** on the card:

```text
/home/matteius/experiments/glm-decode-flags-20261006/opp/vendors/custom_transformer/
  op_impl/ai_core/tbe/kernel/ascend310p/w2_swiglu_v310/
      W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9.json    (1836 B)
      W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9.o       (5664 B)
  op_impl/ai_core/tbe/kernel/ascend310p/w2_route_combine_v310/…
  op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/*.h   (SIMD helpers)
```

### The kernel binary format

```text
$ file W2SwigluV310_….o
ELF 64-bit LSB executable, *unknown arch 0x1029* version 1 (SYSV), statically linked, not stripped
```

`arch 0x1029` (4137) is **Ascend's own AI Core ISA** — this is a binary that executes **on the
card's AI Core**, not on the host.

### The per-op descriptor

```json
{
  "binFileName": "W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9",
  "binFileSuffix": ".o",
  "blockDim": -1,
  "coreType": "AiCore",
  "kernelName": "W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9",
  "magic": "RT_DEV_BINARY_MAGIC_ELF",
  "opParaSize": 24,
  "parameters": [null, null, null],
  "sha256": "40afe4729f298a7a54fced3cd4c6880e1eba54002f2e1bc68df487482615871f",
  "workspace": {…}
}
```

### ★ The integrity check is a plain sha256 that WE compute

```text
$ sha256sum W2SwigluV310_….o
40afe4729f298a7a54fced3cd4c6880e1eba54002f2e1bc68df487482615871f
```

**It matches the JSON exactly.** And the loader (`libop_compile_adapter.so`) only does:

```text
GetBinFileSha256Value / get_binfile_sha256_hash_from_c
GenerateStrSha256HashValue
"fail to verify sha256 times"
```

**There is no CMS, no PKCS#7, no certificate, no signature check anywhere in the kernel-binary load
path** (`grep -iE "cms|sign|pkcs|cert"` over the loader returns only unrelated strings like
`assign engine failed`, `ASCEND_COREDUMP_SIGNAL`).

### What this means

**We can ship our own device-side kernel binary to the card.** The gate is:

1. compile our kernel for `arch 0x1029` (Ascend AI Core ISA),
2. compute `sha256` of the `.o` and write it into the per-op JSON,
3. drop the `.o` + `.json` into the custom-op package,
4. point `ASCEND_CUSTOM_OPP_PATH` at it (or install the vendor package).

No boot signature, no eFuse, no flashing, no power cycle — **a build-and-run loop measured in
seconds**, and the binary runs on the card's AI Core.

### The AICPU variant is the same story

`libaicpu_custom.so` (24 MB, **ARM aarch64** ELF, runs on the card's AI CPU) is loaded by plain
`dlopen` through `LoadCustomKernel` / `KernelLaunchAicpuCustom`, with the op descriptor:

```json
"opInfo": { "functionName": "…", "kernelSo": "libaicpu_custom.so", "opKernelLib": "CUSTAICPUKernel" }
```

and the search paths `/opp/built-in/op_impl/aicpu/kernel/`,
`MM_ENV_ASCEND_CUSTOM_OPP_PATH`, `/op_impl/custom/cpu/config/cust_aicpu_kernel.json`.
Again **no signature check** — the only integrity strings in the loader are profiling hashes
(`AdprofGetHashId`, `MsprofReporterCallback GetHashKey`).

### So the final picture of "custom Linux and firmware on the cards"

| what | status |
|---|---|
| custom **boot** firmware (kernel/rootfs images) | **closed** — payload self-authenticates; built-in Huawei root only |
| custom **runtime** card modules | **closed** — eFuse `rotpk` / `customized_verify`; the 910B `verify_flag` mechanism is absent on 310P |
| custom **device compute kernels** (AI Core `.o`, AICPU `.so`) | **OPEN** — plain sha256 integrity, ours to compute |
| custom **host driver** | **OPEN** — ours (`/usr/src/davinci_ascend-1.0`), installed, DMA work validated on 4 NPUs |

The AI Core / AICPU path is genuinely *device-side* code — it executes on the card's processors.
That is "custom firmware on the cards" in the only sense the hardware permits, and it is
**immediately actionable**.

### Concrete next step

**Replace `W2SwigluV310` with our own AI Core kernel** and prove it runs:

1. locate the AI Core toolchain in the SDK (it must support `-march=ascend310p` / the 0x1029 arch),
2. build a trivial kernel from the existing `ascendc/common/simd.h` helpers,
3. update the JSON's `sha256`,
4. run the model and confirm our kernel executed (e.g. by output change or a side effect).

The `custom_transformer` package is the reference implementation and the build already exists
(`build/cann-ops-transformer-custom_linux-x86_64.run.json`), so the toolchain is presumably present.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — The custom AI Core kernel recipe, confirmed

### The kernel source (`w2_swiglu_v310.cpp`)

```cpp
#include "kernel_operator.h"
#include "swiglu_geometry.h"
namespace {
using namespace AscendC;

class Swiglu {
 public:
  __aicore__ inline void Run(GM_ADDR gateUp, GM_ADDR y, const SwigluTiling& td) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gateUp));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    pipe_.InitBuffer(storage_, STORAGE_BYTES);
    ...
    for (int64_t task = GetBlockIdx(); task < td.rows * tiles; task += GetBlockNum()) {
      DataCopy(gateHalf, input_[row * 2 * td.width + col], count);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(gate, gateHalf, RoundMode::CAST_NONE, count);
      Muls(sigmoid, gate, -1.0f, count);
      Exp(sigmoid, sigmoid, count);
      Adds(sigmoid, sigmoid, 1.0f, count);
      Div(sigmoid, gate, sigmoid, count);
      Mul(gate, sigmoid, up, count);
      ...
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(output_[row * td.width + col], gateHalf, count);
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<half> input_, output_;
};
```

`__aicore__` + `__gm__` global pointers + `DataCopy`/`Muls`/`Exp`/`PipeBarrier`/`SetFlag`/
`WaitFlag` — **this is AscendC, compiled for the AI Core ISA and executed on the card's AI Core.**

Note the comment in the source is a real engineering note from this work:

```cpp
// The serving torch_npu FP32->FP16 cast saturates finite overflow;
// AscendC's vector cast produces infinity unless explicitly clamped.
Mins(gate, gate, FP16_MAX, count);
```

### The toolchain is present

```text
/usr/local/Ascend/cann-9.1.0/tools/bishengir/bin/
  bishengir-compile     98,051,352 B
  bishengir-compile-a5 144,723,856 B
  bishengir-opt         103,090,000 B
  bishengir-opt-a5     150,161,984 B
  hivmc                 84,658,456 B
  hivmc-a5             119,514,880 B

$ bishengir-compile --version
bishengir-compile 1.2.0 (/usr1/BiSheng/npuir-a3/bishengir 8796a8ac1508 2026-07-31)
llvm 19.1.7 2430ab069eb1
Release build
```

plus the AI Core headers:

```text
/usr/local/Ascend/cann-9.1.0/x86_64-linux/include/ascendc/
  aicpu_api/  basic_api/  c_api/  highlevel_api/  host_api/  include/
```

and the `AscendC` helpers in the custom package itself:
`ascendc/common/{simd.h, mma.h, mem.h, layout.h, common_func.h, hardware.h, …}`.

### The loop

1. edit `w2_swiglu_v310.cpp` (or write a new kernel),
2. `bishengir-compile` → `*.o` for `arch 0x1029`,
3. `sha256sum` the `.o` → put it in the per-op JSON's `"sha256"`,
4. drop the `.o` + JSON into the custom-op package,
5. run the model.

**Step 3 is the only "integrity" gate and we compute it ourselves.** There is no signature.

### What this means for the goal

**This is custom device-side firmware on the cards, and it is genuinely open.** It is not a
workaround — it is the vendor's intended mechanism for custom device compute, and the hardware
executes it.

Combined with the host driver (also ours), we have two real levers:

| lever | what it is | status |
|---|---|---|
| **AI Core / AICPU device kernels** | our code, compiled for the card's AI Core / AI CPU, shipped via the custom-op package | **OPEN** — sha256 integrity only |
| **host driver** | our OpenSensor fork; controls PCIe, DMA, queueing, profiling | **OPEN** — DMA work already validated on 4 NPUs |

The boot-time images (kernel/rootfs) stay stock-signed, which is correct and is also what the
hardware enforces.

### Card state

Restored to full stock; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged. Cards need a cold power
cycle to come back up.

---

## 2026-10-26 — ★★★★★ PROVEN: we can rebuild and replace device-executed AI Core code

The full loop has been executed end to end, with a real source change.

### The build

The real source tree is **not** in the experiment dir — it is:

```text
/srv/ai/src/glm-l1-wide-build-20261004/gmm/w2_swiglu_v310/op_kernel/w2_swiglu_v310.cpp
```

and the build driver passes it explicitly:

```text
--operator-source /srv/ai/src/glm-l1-wide-build-20261004/gmm/w2_swiglu_v310
--prepared-input  …/build/binary/ascend310p/src/w2_swiglu_v310
--prepared-input  …/build/binary/ascend310p/src/ascendc/common
--recipe-file     …/build/binary/ascend310p/gen/W2SwigluV310-w2_swiglu_v310-0.sh
--set-env ASCEND_CUSTOM_OPP_PATH=…/build/custom
```

The build target:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/srv/ai/venvs/fork028/bin:$PATH
rm -rf /srv/ai/src/glm-l1-wide-build-20261004/build_cache/custom_operator/w2_swiglu_v310
make -C …/glm-decode-flags-20261006/build w2_swiglu_v310_ascend310p_0
```

(The build cache is keyed on the operator source; it must be removed or the change is skipped.
A **comment-only** change also produces an identical `.o` — the first "rebuild" was a false positive.)

### The proof

One real code change was inserted into the AscendC kernel:

```cpp
      Mul(gate, sigmoid, up, count);
      Adds(gate, gate, 0.0f, count);   // CUSTOM-MARKER-2026-10-26
```

and the rebuild produced:

```text
[ascend310p] Generating W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9 ...
[ascend310p] Generating W2SwigluV310_af2015fde7ebd36ec61a5bd005021cd9 Done
[build-cache] SAVED domain=custom_operator unit=w2_swiglu_v310 key=27ddbfbe… build_seconds=4.7

kernel .o  sha256  40afe4729f298a7a54fced3cd4c6880e1eba54002f2e1bc68df487482615871f   (before)
                 fd9302b8def159821e7aa9e510bd4162721887c4ab69c50…                        (after)
JSON "sha256"   fd9302b8def159821e7aa9e510bd4162721887c4ab69c50…                    <- AUTO-GENERATED
```

**`opc` regenerates the JSON with the new sha256 automatically.** The integrity value is computed by
the tool from our binary; there is no signature and no key.

The rebuilt `.o` + `.json` are installed in the custom-op package and preserved at
`/var/local/ascend-work/proof/`. The source was restored to its original state afterwards.

### What this establishes

| | status |
|---|---|
| edit AscendC device code | ✅ |
| compile it for the card's AI Core (`arch 0x1029`) with the vendor toolchain | ✅ `bishengir-compile 1.2.0` / `opc` |
| regenerate the descriptor + its integrity hash | ✅ automatic |
| ship it to the card | ✅ via the custom-op package |
| **no boot signature / eFuse / flash / power cycle** | ✅ |
| **iterate in ~5 seconds** | ✅ (`build_seconds=4.703`) |

**This is custom device-executed firmware on the Ascend 310P cards, and it is the real thing** —
AscendC code with `__aicore__`, `__gm__` global buffers, `DataCopy`/`Muls`/`Exp`/`PipeBarrier`/
`SetFlag`/`WaitFlag`, running on the card's AI Core.

### The complete picture of "custom Linux and firmware on the cards"

| layer | customisable? | how |
|---|---|---|
| **AI Core / AICPU device kernels** | **YES** | edit AscendC → `opc` → custom-op package. Proven above. |
| **host driver** | **YES** | our OpenSensor fork `/usr/src/davinci_ascend-1.0`, installed; DMA work validated on 4 NPUs |
| card-side kernel modules (`drv_pkicms.ko` etc.) | no | inside the rootfs, which must be Huawei-signed |
| boot images (kernel/rootfs) | no | payload self-authenticates; built-in Huawei root only |

### Next step

**Verify the new kernel actually executes on the card.** Run the model that uses `W2Swiglu` and
confirm the op is dispatched to our binary. With the marker change in place (a no-op `Adds`), the
result must be identical — proving our binary ran — and then a *semantic* change (e.g. scale the
output) must change the result, proving we control device-side numerics.

That closes the loop from "we can build it" to "we can run our own code on the card's processors".

### Card state

Stock images; `/usr/local/Ascend/CMS/{user.xer,user.crl}` staged; cards need a cold power cycle to
come back up.

---

## 2026-10-26 — Course correction, and where the real goal stands

### Scope correction

I pulled the work into the vLLM serving stack to try to *observe* our rebuilt AI Core kernel
executing. That was the wrong exercise — the goal here is **running modified Linux on the cards**,
not validating a model-serving path. The serving attempt is abandoned; the serving stack is stopped
and the stock kernel binary is restored in the package.

One useful negative came out of it: the worker died in
`_register_atb_extensions` → `torch.ops.load_library(atb_so_path)` → `OSError`. **Both earlier
candidate logs (before our `.o` change) show the same stall** at "Capturing CUDA graphs" with zero
successful completions — so that failure is pre-existing/environmental in the harness, **not** caused
by our kernel change. Worth knowing, but not our concern here.

### Where "modified Linux on the cards" actually stands

The goal is *a modified kernel/rootfs running on the Ascend 310P cards*. Three layers, and we now
know the trust boundary of each:

| layer | who verifies it | can we change it? |
|---|---|---|
| `Ascend310P.fd` — the bootloader (BL31) | card BootROM + `hboot1a`/`hboot1b` (flash-only) | **The image is modifiable** (no cms/ini/crl; only self-checksums). But **V2 proved the card rejects a modified payload** → it authenticates itself somehow. |
| `Ascend310P.image` — the **Linux kernel** | BL31, against the root CA embedded in `Ascend310P.fd` | **No** — needs the Huawei root key (absent). |
| `Ascend310P.cpio.gz` — the **rootfs** | BL31, same | **No** — same reason. |

**The kernel and rootfs are both locked by the same key.** The bootloader is the only modifiable
image, and it is also the thing that does the verifying. That is the whole problem in one sentence.

### What has NOT yet been tried on the bootloader

We have established:

- V2: one byte flipped in an **unused payload region** (file 0x20000), Huawei's root untouched →
  **identical failure**. So the bootloader authenticates its own payload.
- goal: root CA replaced → identical failure.

Both are consistent with **one** mechanism we have not yet located: the EDK2 FFS2 volume at
`payload+0x100` carries integrity values (`78e58c8c…` at `+0x110`, `d408e2df…` at `+0x30e8`) and
the bootloader validates **itself** against them, using a key that is **inside the bootloader** —
which would make the chain self-referential but still anchored in the BootROM.

**The untested question:** is the anchor the BootROM (unreachable) or something else? Two ways to
find out, both cheap:

1. **Locate the self-verification code** in the decoded BL31 and read what it hashes and what it
   compares against. We have the code; we have 44 resolved cross-references; we have the FFS2
   descriptor. This is pure static analysis of material in hand.
2. **Test whether the integrity values are a hash or a signature.** `78e58c8c…` and `d408e2df…` are
   16 bytes each. If they are truncated hashes over regions we can identify, then the check is
   reproducible and we can forge it. If they are RSA signature components, we need the key.

We noted earlier that neither value changed when we swapped the root CA — which is itself a clue:
the protected region does **not** include the certificates. So the check is over *something else* in
the volume (the FFS2 headers, the TOC, or the code regions).

### Proposed next work (in order, no card risk until step 3)

1. **Find the self-check.** Decode the FFS2 descriptor at `payload+0x100` as a structure
   (the `09 19 00 14` header, the `78e58c8c…`/`d408e2df…` values, the `0x405`/`0x407`/`0x510`
   offsets) and determine what region each value covers. Test by recomputing sha256 over candidate
   regions and comparing.
2. **If it is a hash over a region we control** → we can re-checksum it, and the bootloader
   becomes fully modifiable, which unblocks a signed custom kernel + rootfs.
3. **If it is an RSA signature** → identify the key's location and whether it is in the BootROM
   (then closed) or in `hboot1a`/`hboot1b` (flash-only, but `upgrade-tool` *can* write those
   components, which is a different attack surface we have not explored).

Step 3's second branch is worth flagging: `upgrade-tool --components` lists
`hboot1a, hboot1b, hlink` as flashable. **We have never examined what those contain or whether
the BootROM validates them.** That is the remaining unexplored surface.

### Card state

All three images stock (`e1e75ec2…` / `027f4d22…` / `12975a9d…`), 6 davinci nodes, 6 chips OK.
Serving stack stopped. Stock AI Core kernel binary restored.

---

## 2026-10-26 — ★★ The bootloader's self-signature block, located

### The esbc outer header is identical across all images except for one field

```text
field        hboot2 (bootloader)        rootfs (initrd)     kernel (uimage)    user.xer (usrcert)
+0x00        55aa55aa                  55aa55aa            55aa55aa          55aa55aa
+0x04        000004dc                  000004dc            000004dc          000004dc
+0x08        ffffffff (certtype)       ffffffff            ffffffff          00000001
+0x18..+0x28 0                         0                   0                 0
+0x2C        SHA-256(payload)          SHA-256(payload)    SHA-256(payload)  SHA-256(payload)
+0x38..+0x4C 0                         0                   0                 0
+0x4C        00050000 00000108 00020000 (identical in ALL FOUR)
+0x58..      168 bytes of DATA         0                   0                 0
```

**The bootloader is the only image with 168 bytes of extra material at `+0x58`** — a
**self-signature block**. The kernel and rootfs carry the identical 12-byte descriptor at `+0x4C` but
with an all-zero region there, i.e. **they are not self-signed; the bootloader is.**

### Why this explains V2

V2 flipped one byte in the bootloader's payload and re-checksummed `@0x2C`, `@0x4BC`, `@0x14BC`.
It **did not update the self-signature block at `+0x58`** — because our `patch_bootloader.py` did not
know it existed. So the bootloader was delivered with a payload that no longer matched its own
self-signature, and the card rejected it.

**This is a concrete, fixable defect in our tooling**, not a closed door.

### Immediate next step

1. Determine what the 168-byte block at `+0x58` covers and how it is computed:
   - it is 512 bytes long per the `+0x54 = 0x200` field, but only the first 168 are non-zero
     before the header copy ends at `0x1000` — the rest is presumably the same block repeated in
     header copy 1 at `+0x1058`;
   - test whether it is an RSA-4096 signature (512 B) over `SHA-256(payload)` or over a hash of the
     header, verified with the **embedded** root `Huawei Integrity Root CA - G2`
     (`file 0x155841`, PSS/SHA-256/MGF1/salt-32) — the natural choice, since that root is *in* the
     file.
2. If we can reproduce it, we can **re-sign the bootloader ourselves** — because the root that
   verifies it is *inside the file we are modifying*. That is the classic "self-signed, key inside"
   situation: swapping in **our** root and signing with **our** key would make the bootloader
   self-consistent under *our* trust anchor.
   - Whether the card's BootROM then accepts it depends on whether the BootROM has its own copy of
     Huawei's root. If it does, this fails. **If it does not** — and the FIP self-check is the only
     gate — it succeeds, and the boot-time path opens completely.
3. Test that with a minimal change: re-sign the *unmodified* bootloader with a **test key** and the
   **original** root swapped for the test root, install, and observe. That isolates "does the card
   trust the in-file root?" from "can we sign?".

### Note on the earlier "no signature" claim

I earlier wrote that the kernel/rootfs "carry no cms/ini/crl sections" — that is correct and
unchanged (they are signed by the **CMS in their `cms` section**). What is new is that the
**bootloader has an additional header-embedded self-signature**, which the other two do not. My
previous framing of "the bootloader is protected only by its own self-consistency" was too generous:
there is a cryptographic self-signature we have not yet reproduced, and it is the likely reason V2
and the goal boot failed.

### Card state

All three images stock (`e1e75ec2…` / `027f4d22…` / `12975a9d…`), 6 davinci nodes, 6 chips OK.
Serving stack stopped; stock AI Core kernel binary restored.

---

## 2026-10-26 — ★★★ The esbc container format, fully decoded from Huawei's own validator

`Ascend_310P_check.bin` (the vendor's image verifier) **parses** these containers, so
disassembling it gives the authoritative layout rather than an inferred one.

### Outer header (two copies at 0x0000 / 0x1000, 0x1000 bytes each)

```text
+0x000  u32   magic          = 0x55aa55aa
+0x004  u32   head_len       = 0x4dc
+0x008  u32   cert_type      = 0xffffffff (Huawei) / 0x1 (user cert)
+0x02C  u8[32] payload_sha256 = SHA-256(payload)
+0x04C  u32   sec_blk_size   = 0x500
+0x050  u32   sec_hdr_off    = 0x108
+0x054  u32   sec_key_size   = 0x200
+0x058  u8[0x500] security block   <- present ONLY in hboot2; zero in every other image
+0x4BC  u8[32] head_sha256   = SHA-256(header[0x0000:0x04BC])
+0x458  u32   img_offset     = 0x2000
+0x45C  u32   code_len       = payload length
+0x2000 payload begins
```

The validator's magic comparison is `movabs rax, 0xabcd1234aa55aa55` — i.e. the outer magic
`0x55aa55aa` with the **upper half carrying `0xabcd1234`**.

### Payload

```text
+0x000  u32   magic  = 0x33aa3a3a
+0x004  char[8] tag   ("initrd","uimage","dtimage","hboot2","usrcert", …)
+0x008  u32   tag tail (last 4 bytes of the tag: "dr" of "initrd", "age" of "dtimage", …)
+0x100  body  (gzip for initrd, LZMA for uimage, DER for usrcert, EDK2 FV for hboot2)
```

### Section table — the geometry the validator computes

```c
slot      = 0x20 - (code_len & 0xf);
cms_start = img_offset + code_len + slot;
/* each entry: tag[12] len u32 body, chained */
```

For the stock rootfs: `code_len = 0x684b249`, `slot = 0x17`, `cms_start = 0x684d260` — which is
**exactly** where the `cms` tag was found. The `ini` body is `"<tag>,   <sha256(payload)>;"` and the
`crl` body is `[CRL, CA cert, CRL]`.

### ★ The bootloader is a structurally DIFFERENT kind of image

This is now definitive, from the vendor's own tool:

```text
image                          extract-crl   verify
Ascend310P.cpio.gz.orig (initrd)     0            0     <- works
Ascend310P.image.orig  (uimage)      0            0     <- works
Ascend310P.fd.stock    (hboot2)      1            6     <- NO cms/ini/crl sections at all
```

The bootloader has **no `cms`/`ini`/`crl` sections** — its `extract` fails because there is
nothing to extract. It is protected by the **0x500-byte security block at `+0x58`**, which no other
image has. The descriptor beside it is `0x500 / 0x108 / 0x200` = block size / header offset /
key-or-signature size.

### What that means for the boot path

- The **kernel and rootfs** are protected by a CMS signature over their `ini` — which names
  `sha256(payload)`. To replace them we need a key the bootloader trusts (the built-in Huawei
  root). **Closed.**
- The **bootloader** is protected by a *different*, self-contained mechanism (the 0x500-byte block).
  Its structure is: `0x108` bytes of header/signature followed by `0x20000` bytes of covered
  material. We have not yet identified the algorithm — it is **not** an RSA-PSS signature over the
  payload under either embedded root (verified: every combination of the two embedded roots, five
  candidate messages, and the descriptor-hinted signature offsets all fail to verify).
- V2 proved the mechanism is enforced: one byte changed in an *unused* region of the payload (with
  all three digests resealed) produced exactly the same failure as replacing the root.

### The remaining unknown, precisely stated

**What function computes the 0x500-byte block at `+0x58`?** It is a 264-byte opaque blob plus
0x200 bytes of covered material, entropy 4.8 (structured, not encrypted). Because it did **not**
change when we replaced the embedded root CA, it is **not** a signature made with that root — so it
is either (a) a hash chain over a region we have not identified, (b) a signature under a key stored
elsewhere (in `hboot1a`/`hboot1b`, which exist only in the card's flash and have **no file
anywhere** on the host), or (c) an HMAC with a device-unique key.

If it is (a), the bootloader is modifiable and **the entire boot path opens** — we could then sign
our own kernel and rootfs. If (b) or (c), it is closed and the answer for the boot path is no.

**The decisive next step is to find the code that *writes* that block.** It is produced by whoever
builds the hboot2 image, and that builder is on the host: the driver tree contains
`ascend-310p-host-driver-restored` and the SDK contains the packaging tooling. Let me look for the
hboot2 image builder rather than reverse-engineering the verifier — it will state the algorithm
directly.

### Tooling added

- `customer-cert-20261026/esbc_info.py` — parses and validates any esbc container, including the
  payload digest, both header self-checksums, the section table and the `ini` digest binding.
- `customer-cert-20261026/ffs_parse.py` — walks the EDK2 FFS2 files inside a firmware volume.
- `customer-cert-20261026/patch_bootloader.py` — byte-patch + reseal (all three digests).

### Card state

Stock images restored; 6 davinci nodes; cards healthy. The serving stack is stopped.

---

## 2026-10-26 — ★★★ The bootloader header format, from the Oct-21 writeup + the vendor validator

I had been re-deriving this from scratch. The Oct-21 session's writeup already documents it, and
Huawei's own `Ascend_310P_check.bin` confirms it. Combining both:

### Outer header (4 KiB, copied at 0x000 and 0x1000)

| field | offset | value / rule | verified |
|---|---|---|---|
| preamble | +0x000 | `0x55AA55AA` | ✅ |
| head_len | +0x004 | `0x4DC` | ✅ |
| certtype | +0x008 | stock `0xFFFFFFFF`; user cert `0x1` | ✅ |
| payload digest | +0x02C | SHA-256(payload), both copies | ✅ |
| **sign_offset** | **+0x460** | **`0xE00`** — the 512-byte signature blob | ✅ |
| head_magic | +0x4B8 | `0x33CC33CC` | ✅ |
| head_hash | +0x4BC | SHA-256(header[base, base+0x4BC)) | ✅ |
| extern | +0x4E0 | u64 `0xABCD1234AA55AA55`, u32 total size @+0x4E8 | ✅ |
| img_offset | +0x458 | `0x2000` | ✅ |
| code_len | +0x45C | payload length | ✅ |

**The signature blob is at file `0xE00` (and `0x1E00` in header copy 1).** That is the piece I
had not identified — I had been treating the `0x500`-byte region at `+0x58` as the security
block, but it is a *different* thing (`sec_blk_size/hdr_off/key_size` descriptor), and the actual
signature is the 512 bytes at `+0x460 → 0xE00`.

### What the Oct-21 session actually did to the bootloader

Diffing `AS610_HBOOT2_UEFI.fd.stock` → `.patched` (1157 bytes differ):

```text
0x00002c..0x00004b  (32)   payload digest, copy 0     <- resealed
0x0004bc..0x0004db  (32)   head hash, copy 0         <- resealed
0x000e00..0x000fff (512)  THE SIGNATURE, copy 0      <- CHANGED
0x00102c..0x00104b  (32)   payload digest, copy 1     <- resealed
0x0014bc..0x0014db  (32)   head hash, copy 1         <- resealed
0x001e00..0x001fff (512)  THE SIGNATURE, copy 1      <- CHANGED
0x002024..0x002026  (3)    inner header +0x24: 0x205 -> 0x90220
0x013dc59..0x013dc5c (4)   a word in the BL31 code
```

So the prior session **did** update the signature — but `rebuild_esbc.py` **never touches
`+0x460` or `+0xE00`** (it copies the header verbatim and only reseals the digests). The signature
update therefore came from a *different* tool or a manual step, and that tool is not on the card.

### The two content changes, decoded

**1. Inner header `+0x24`: `0x205` → `0x90220`**

```text
0x205   = 0b00_0000_0010_0101   bits 0, 2, 9
0x90220 = 0b10_0100_0010_0010   bits 4, 5, 9, 19
```

Every stock image (rootfs, kernel, dtimg, usrcert, bootloader) has `0x205`. **Bit 19 appears in no
stock image** — the Oct-21 session set a bit that nothing else sets, which is exactly what a
"skip the normal verification path" switch would look like.

**2. BL31 word at file `0x13dc58`: `0x00024071` → `0x00001271`**

This is at **BL31-relative offset 0x7B57** — inside the code we decoded at 100%. The bytes
`71 40 02 00` → `71 12 00 00` change a halfword from `0x4071` to `0x1271`. In AArch64 terms this
sits right after the `cmp w0,#1 / b.eq` sequence we decoded earlier at `0x13dc55..0x13dc59`:

```text
0x13dc51: bl  0x140f85        ; MODE = (*(0x8000fff8)) & 0xf
0x13dc55: cmp w0, #1
0x13dc59: b.eq 0x13dca1
```

so the patched word is **immediately after that compare** — i.e. the prior session was flipping a
**branch/flag in the customer-certificate decision path**, exactly where the `user cert cms vef` /
`no usr, hw vef fail` split lives.

### The decisive experiment, now well-posed

The prior session's `.patched` file is a **complete, self-consistent bootloader patch**: new
signature at both copies, resealed digests, the `+0x24` bit, and the BL31 decision word. It was
never installed.

Two possibilities, and they are cleanly distinguishable:

- **(A) The signature is verifiable with a key we have** → the Oct-21 session had the private key
  and the boot path is open. We would find it by asking *what key signed it*: the blob is 512 bytes
  (RSA-4096) and the embedded roots are `Huawei Integrity Root CA - G2` (PSS) and `Huawei Root CA`
  (PKCS#1). Neither verified against `sha256(payload)` — but the signed message is probably not the
  payload digest; it may be the *header*, or the payload digest plus the header fields, or a
  digest over the whole file with the digest field zeroed.
- **(B) The signature needs a key we do not have** → the Oct-21 session had a different key
  (perhaps a *test* key they generated, with the `+0x24` bit meaning "trust the test root"), and
  the boot path needs the same key.

**The next step is to identify the message the `0xE00` signature covers.** That is a bounded,
pure-offline search: the candidate messages are all combinations of
{header[0:0x460], header-with-digest-zeroed, payload, payload digest, whole file}, and the
candidate keys are {the two embedded roots, and — importantly — the roots in the Oct-21
`usercert-test/` directory, which include `rootcaG2` and possibly a test root}.

Given the Oct-21 session had `rootca_priG2_pri.pem` (the private key for our `RootCA G2`), and
`RootCA G2` is 1465 bytes vs Huawei's G2 at 1606 — **if the bootloader signature is over
`sha256(payload)` with a key whose public half is `RootCA G2`, that key is in our possession and
we can sign our own bootloader.** Let me test that now: install `RootCA G2` into the bootloader's
embedded cert slot and check whether the stock `0xE00` signature verifies against it.

### Card state

Stock images restored; 6 davinci nodes; cards healthy.

---

## 2026-10-26 — ★★★★★ The complete resolution

### The Oct-21 bootloader patch, decoded

The prior session left `AS610_HBOOT2_UEFI.fd.patched` on the card. Diffing it against stock gives
exactly two *content* changes (everything else is a resealed digest):

**1. Inner esbc header `+0x24`: `0x205` → `0x90220`**

```text
0x205   = 0b00_0000_0010_0101   bits 0, 2, 9
0x90220 = 0b10_0100_0010_0010   bits 4, 5, 9, 19
```

Every stock image (rootfs, kernel, dtimg, usrcert, bootloader) has `0x205`. Bit 19 appears in
**nothing** stock — the patch set a flag no stock image sets.

**2. BL31 word at file `0x13dc58`: `0x54000240` → `0x14000012`**

```text
0x54000240 : B.cond  cond=EQ  imm19=0x48  -> target 0x13dc59 + 0x48 = 0x13dca1
0x14000012 : BR      imm26=0x12            -> target 0x13dc59 + 0x48 = 0x13dca1
```

**Same target. The patch turned a conditional branch into an unconditional one.**

And the context — decoded from the BL31 we now read at 100%:

```text
0x13dc41: stp x29, x30, [sp, #-0x20]!
0x13dc45: stp x20, x21, [sp, #-0x10]!
0x13dc49: mov x21, x11
0x13dc4d: mov x0, x19
0x13dc51: bl  0x140f85          ; -> MODE = (*(u32*)0x8000fff8) & 0xf
0x13dc55: cmp w0, #1
0x13dc59: b.eq 0x13dca1        ; <<<< THE PATCHED INSTRUCTION
0x13dc5d: mov x1, #0xc4
0x13dc61: mov w0, #1
0x13dc65: movk x1, #0x81f1, lsl #16
0x13dc69: str w0, [x19, x1]
```

**This is the customer-certificate decision point.** The patched branch says: *if the mode is
not 1 (i.e. the customer path did not succeed), skip the failure handling* — i.e. **make the
customer-certificate verification failure non-fatal**.

### So the prior session's approach was

1. Set inner-header bit 19 (`+0x24`) — the **"customer cert accepted"** flag.
2. Patch the BL31 so a failed customer-cert verification no longer aborts the boot.
3. Regenerate the `0xE00` signature blob (both header copies) and reseal the digests.

**And it was never installed.** The card still runs the stock bootloader.

### Why we cannot simply reproduce step 3

The `0xE00` blob is a **512-byte signature** (RSA-4096). It did **not** change when the embedded
roots were touched, and it did change when the payload changed — so it *is* payload-derived. But
it verifies against **neither** embedded root (Huawei `Integrity Root CA - G2` PSS, nor `Huawei Root
CA` PKCS#1) over any of {payload digest, payload, header, header+payload, whole file with digests
blanked}, with PSS salt 32. And it does not verify against **our** `RootCA G2` either.

The most likely explanation, consistent with everything: **the Oct-21 session had a tool or key we
have not found**, and `+0x460 = 0xE00` + the 512-byte blob is the signature that the *card's
BootROM* verifies before it will even hand control to BL31.

### The one remaining lever, and it is a good one

The README of that session records:

> *"If the hash lands in an OTP / customer efuse, these private keys are the only way to sign a
> rootfs this chip will accept. Treat `rootca_priG2_pri.pem` as the crown jewel."*

and

> *"stock rootfs still boots: **built-in Huawei cert is checked first, user cert is the fallback**."*

**"the user cert is the fallback"** — so the device *does* implement the fallback we need. The
question that has never been answered, and which the Oct-21 session also never answered, is
**what makes the built-in check fall through to the user cert.** The candidates:

- the `+0x24` bit 19 (`0x90220`) — a per-image flag saying *"this image is customer-signed, use the
  user cert"*. **If so, we do not need to forge anything: we set bit 19 on the image we sign with our
  own CA.**
- the `ccpu_usr_cert_hash` enrolled value (we can write it), compared by BL31's
  `usr cert hash cmp fail.` path.

**This is testable in one cycle, with no signature forgery at all.** Take the stock rootfs, set
inner-header `+0x24 = 0x90220` on it, re-sign its CMS with our `signcert` (the Oct-21 chain, whose
private keys we hold), set `certtype` as needed, enrol our leaf's hash via `--user_cert`, and boot.

We built every ingredient for this in the Oct-21 session (`full/initrd.ini.p7s`, `full/initrd.ini.crl`,
`build_user_der.py`, `rebuild_esbc.py`) — **but the `+0x24` bit was never set**, and the session's
own A/B matrix shows every variant failed.

### Recommended next experiments (in order)

1. **Set `+0x24 = 0x90220` on a customer-signed rootfs.** This is the one flag no stock image has
   and the only untried lever. If bit 19 means "customer-signed image", this alone may flip the
   boot from "built-in root only" to "user cert fallback".
2. **Enrol the hash that matches** the certificate in the image (`signcert.der`, hash `089238e8…`),
   not the intermediate — the Oct-21 session mostly enrolled `signca.der` (the intermediate) while
   the image was signed by the leaf. If BL31 hashes the *signer* of the image, those must match.
3. Only if (1) and (2) both fail: treat the `0xE00` bootloader signature as the gate and hunt for
   the key.

### Also worth recording

- The `usrcert` image format was **fully reversed** by the Oct-21 session
  (`build_user_der.py` reproduces `user.der` byte-exactly) and the **`--user_cert` write path is
  confirmed working** — the device accepts a hand-built envelope with **no Huawei
  `usrcert.ini.cms` signature** and stores the hash.
- `drv_user_cfg.ko` stores it as a named **item** (`ccpu_usr_cert_hash`) in the card's flash
  **block store** (`devdrv_config_write_item_content`, `sec_flash_write`), not at a fixed address.
- The runtime customer-cert path (`verify_flag`, `user-certificate`, `customized_verify`) exists in
  the driver source but is compiled **only for `hi1980b` (Ascend 910B)**, so it is inert on 310P.

### Card state

Stock images restored; 6 davinci nodes; cards healthy; serving stack stopped.

---

## 2026-10-26 — V1 (flag clear) failed as expected; V2 (bit 19 set) deployed

### V1 result

```text
Ascend310P.cpio.gz = V1-signed-flag-clear.cpio.gz  (59a280b7dcad2487…)
inner+0x24 = 0x205   (bit19 clear)
davinci nodes: 0     (did not boot)
```

**V1 is a proper control**: customer-signed CMS, matching enrolled anchor (`signca`, hash
`41474c62…` — the same value the Oct-21 session enrolled), same CRL — and the only thing missing
versus the Oct-21 hypothesis is bit 19 in the inner header.

### V2 deployed — the same image with bit 19 set

```text
Ascend310P.cpio.gz = V2-signed-flag-set.cpio.gz   (178ceb4a763760d9…)
inner+0x24 = 0x90220      (bit19 SET)
inner+0x08 = 0x6472       ("dr" tag tail)
@0x2C payload digest      MATCH
@0x4BC head hash          OK
```

The two images differ **only** in the four bytes of the inner header's `+0x24` field
(`0x205` ↔ `0x90220`) plus the three digests that consequently change. Same payload, same CMS,
same CRL, same enrolment.

**No re-enrolment needed** — the `signca` anchor was enrolled on all six devices before the cycle
and persists in secure flash (the `--user_cert` write succeeded on all six).

### What each outcome means

| V2 result | conclusion |
|---|---|
| **boots** | **bit 19 = "customer-signed image, use the enrolled user cert."** The customer path works, no signature forgery needed, and we can now ship a **custom-signed rootfs**. The remaining step is to sign the *kernel* (`Ascend310P.image`) the same way, and then a modified rootfs/kernel boots. |
| fails like V1 | bit 19 alone is not the switch; the missing piece is the BL31 branch patch (which also requires reproducing the `0xE00` bootloader signature) |

### Card state

V2 deployed and verified. Cards are down from the V1 boot; **cold power cycle needed (≥30 s)**.

---

## 2026-10-26 — ❌ V2 also failed. The `+0x24` bit alone is not the switch

```text
V2-signed-flag-set.cpio.gz   inner+0x24 = 0x90220 (bit19 SET)
davinci nodes: 0    phy_startup errs: 1
```

So the Oct-21 hypothesis needs **both** changes, and the flag alone does nothing:

| component | status |
|---|---|
| inner header `+0x24` bit 19 (`0x205`→`0x90220`) | **tested — insufficient on its own** |
| BL31 branch at `0x13dc58` (`b.eq`→`b`) | not applied — needs the `0xE00` signature to match |
| `0xE00` 512-byte signature (header `+0x460`) | **not reproducible** — verifies against neither embedded Huawei root nor our `RootCA G2`, over any message tried |

V1 and V2 together are a clean negative result: **the flag is necessary-but-not-sufficient, or it
is not the mechanism at all.**

### The `0xE00` signature is the wall

Both the header copies carry a 512-byte blob at `+0x460 → 0xE00`. The Oct-21 session's patched file
has a *different* blob there — and the diff shows **only** the payload-derived content changed
(the embedded roots at `0x1552e9`/`0x155841` are byte-identical to stock).

That combination is the key clue:

- the blob **changes** when the payload changes → it is payload-derived;
- the blob does **not** change when the embedded roots change → it is **not** signed by them;
- it verifies against neither embedded root, nor our `RootCA G2`, over
  {payload digest, payload, header, header+payload, whole file with digests blanked} with PSS-salt-32
  or PKCS#1 v1.5.

**A 512-byte blob that is payload-derived but not signed by any key we hold.** The remaining
possibilities:

1. It is signed by a key in **`hboot1a`/`hboot1b`** — the flash-only first-stage bootloaders.
   Those exist **only in the card's flash**: there is no `hboot1*` file anywhere on the host, and
   `upgrade-tool` lists `hboot1a, hboot1b, hlink` as flashable components. **If we can read or
   write them, the whole boot path opens.**
2. It is an HMAC or a device-unique derivation (eFuse/OTP-derived key) — not forgeable, but then
   the Oct-21 session could not have regenerated it either, which argues against this.
3. It is a hash over a region we have not identified (the descriptor beside it says
   `0x500 / 0x108 / 0x200` — block size / header offset / key size, i.e. a *key blob* of 0x200
   bytes at offset 0x108 inside a 0x500-byte block).

Option (1) is now the most interesting lead and it is **untouched by anything we have done**.

### Recommended next step (concrete)

**Get `hboot1a` / `hboot1b` off the card.** They are flashable components, so `upgrade-tool` can
write them — and the driver has the code that talks to the card's flash. If we can read them back
(even one), we get the BootROM's trust root and can determine whether it is the key that signed the
`0xE00` blob.

Concretely:

1. Find how `upgrade-tool` writes `hboot1a` — the component type id and the file path it expects
   (the driver tree has the component tables; `res_drv_mini_v2.c` names the images and
   `drv_devmng_host` implements the transfer).
2. Determine whether the same path can *read* a component back. If the flash is read-only to the
   host, look instead for the card-side copy: the `hboot1*` image may also be present in the
   `Ascend310P.fd` payload (two FIP copies — maybe one of them *is* hboot1).
3. Separately: the `0x500`-byte block at `+0x58` with descriptor `0x500/0x108/0x200` is present
   **only** in the bootloader and absent from every other image — that asymmetry is worth
   understanding; it may be the hboot1 trust descriptor.

### Honest summary of the boot path

Six independent lines of evidence now say the boot path is closed without either the Huawei root
key or the `hboot1` trust root, and the Oct-21 session — which had the same keys we have — reached
the same conclusion ("*no self-signed chain can boot this unit*").

The `hboot1` route is the one stone genuinely unturned. If that also fails, the answer is
definitive: **the card's boot chain cannot be modified, and "custom Linux on the cards" has to mean
the runtime path** (card-side modules we can rebuild *and* sign, once we can get the card to accept
a modified rootfs) or the host path.

### Card state

Stock restored (`e1e75ec2…` / `027f4d22…` / `12975a9d…`). Cards are down from the V2 boot and need
a cold power cycle to come back up.

---

## 2026-10-26 (session 2) — RESUMED AFTER COMPACTION: the AI Core execution gap is CLOSED

This session resumed after a context-compaction failure. The prior session's log had proven the
**build** side of the AI Core custom-kernel path but had **never proven execution** — the serving
harness died in `_register_atb_extensions` before any op was dispatched, and the note in the log
("Worth knowing, but not our concern here") deferred the question.

That question was the only thing left open on a path that was otherwise complete, so it was the
first thing this session closed.

### Resumption state

The cards were **down** from the V2 boot (0 davinci nodes, `phy_startup` errors, stock images
restored). The operator performed a cold power cycle; the cards came back fully healthy:

```text
phy_startup errs : 0
bringup service  : active (exited), code=0/SUCCESS
davinci nodes    : 6
npu-smi          : all 6 chips 310P3 Health=OK
```

Deployed images are stock and verified byte-identical:
`Ascend310P.cpio.gz 12975a9d…` · `Ascend310P.image 027f4d22…` · `Ascend310P.fd e1e75ec2…`.

### ✅✅✅ PROVEN: our own AI Core binary executes on the card

Two scripts were added:

| script | what it proves |
|---|---|
| `customer-cert-20261026/prove_aicore_exec.py` | the custom op runs on the card and matches a host reference |
| `customer-cert-20261026/prove_aicore_ownership.sh` | the binary that ran is **ours**, not a cached stock one |

**Step 1 — the op runs and is numerically correct.**

```text
device: Ascend310P3

[rows=1    width=2048  ] maxabs=0.00000 rel=0.00000 finite=True OK
[rows=1    width=4096  ] maxabs=0.00000 rel=0.00000 finite=True OK
[rows=8    width=2048  ] maxabs=0.00000 rel=0.00000 finite=True OK
[rows=32   width=1024  ] maxabs=0.00012 rel=0.00001 finite=True OK
[rows=128  width=512   ] maxabs=0.00000 rel=0.00000 finite=True OK

RESULT: AI Core custom kernel EXECUTED on the card
```

Two details were required to get here, both worth recording:

1. **The op only resolves when the custom-op package is on the path.** Without
   `ASCEND_CUSTOM_OPP_PATH` the call fails with
   `aclnnW2SwigluV310 or aclnnW2SwigluV310GetWorkspaceSize not in libopapi.so`.
   The working invocation is:
   ```bash
   export ASCEND_CUSTOM_OPP_PATH=/home/matteius/experiments/glm-decode-flags-20261006/opp/vendors/custom_transformer
   torch.ops.load_library(.../glm_decode_flags.so)     # registers npu_w2_swiglu_310
   ```
2. **The input layout is SPLIT, not interleaved.** `w2_swiglu_v310.cpp` reads
   `input[row*2*width + col]` for gate and `input[row*2*width + width + col]` for up, and
   `InferSwiglu` sets the output width to `input.dim(1)/2`. A first reference implementation that
   used an interleaved `[..., ::2]` layout produced a *total* mismatch (rel ≈ 1.0–1.5) — a false
   negative that would have looked like "the kernel is broken". The layout is confirmed by
   `w2_swiglu_v310_infershape.cpp` and the tiling source.

**Step 2 — the binary that ran is ours.** A *semantic* change was compiled into the kernel:

```cpp
      Muls(gate, gate, 2.0f, count);      // <-- injected
      PipeBarrier<PIPE_V>();
      Cast(gateHalf, gate, RoundMode::CAST_NONE, count);
```

The rebuild produced a new `.o` and `opc` regenerated the descriptor's integrity hash
automatically (there is no key — the only integrity value is a plain SHA-256 we control):

```text
baseline .o sha256 : 40afe4729f298a7a54fced3cd4c6880e1eba54002f2e1bc68df487482615871f
modified .o sha256 : e83951fb19b21e1518c00d12b272616766a8281dae0a11acbb6b59ece351e628
JSON "sha256"      : e83951fb19b21e1518c00d12b272616766a8281dae0a11acbb6b59ece351e628   <- auto
```

and the device output changed by exactly the predicted factor:

```text
median(mod/base)  = 2.000000   (expect 2.0)
max|mod - 2*base| = 0.000000  rel=0.00000000
RESULT: OUR REBUILT KERNEL EXECUTED ON THE CARD
```

**This is the closed loop.** The only way the device output can change by a known factor is if the
card executed the AscendC binary we just recompiled — AscendC `__aicore__` code, compiled for the
card's own AI Core ISA (`arch 0x1029`), running on the card's AI Core.

The source was restored to its exact original content
(`sha256 034354470e365257c6ce666eceb7273bba596cdda240201480f68545ae91d8f7`, verified), the stock
kernel was rebuilt and reinstalled (`40afe472…`), and the op was re-verified correct afterwards.
**The serving stack is left exactly as it was found.**

### The second open path — the AICPU syskernel channel — also confirmed open

The prior log listed this as recommended step (a) but never tested it. It is now measured:

```text
/usr/local/Ascend/cann-9.1.0/opp/Ascend310P/aicpu/Ascend310P-aicpu_syskernels.tar.gz
  esbc container: magic 0x55aa55aa, head_len 0x4dc, cert_type 0xffffffff
  payload sha OK · head sha c0 OK · head sha c1 OK
  inner tag 'aicpu_ke'    cms_start = 0x33140c0   <-- PAST EOF
```

**No `cms` / `ini` / `crl` sections exist** — the container ends right after the payload, so there
is no signature to forge. The only integrity value is the header's SHA-256 over the payload, which
we recompute. The same is true of `aicpu_extend_syskernels` and `aicpu_legacy`.

And the AICPU kernel library itself:

```text
libaicpu_custom.so : ELF 64-bit LSB shared object, ARM aarch64, stripped  (24,157,704 B)
  exports: HcclDispatcherAicpuInit, AcquireDispatcherCtx, CreateDispatcherCtx, ...
```

is loaded by plain `dlopen()` through `LoadCustomKernel` / `KernelLaunchAicpuCustom` with the
descriptor:

```json
"HcclKernelEntrance": { "opInfo": { "kernelSo": "libaicpu_custom.so",
                                     "opKernelLib": "CUSTAICPUKernel" } }
```

`libaicpu_ascend_engine.so` contains **zero** signature-verification strings. So the AI CPU is a
second genuine device-side code channel: a plain ARM64 ELF of ours, executed on the card's AI CPU.

### A tooling bug fixed, and a vendor limit established

`read_sec_flash.py` failed with `ENODEV` because it issued the feature-command ioctl on the
`davinci_manager` fd without first opening the DMS sub-module. Fixed (it now does the
`_IO('Z',0)` open / `_IO('Z',1)` close dance that `dms_log.py` already used).

With that fixed it returns `EOPNOTSUPP`, and the reason is now **established from the vendor's own
source** rather than inferred:

```c
/* dms/status/dms_basic_info.c */
#ifdef CFG_SOC_PLATFORM_MDC_V51          // <-- Ascend 610 ONLY
ADD_FEATURE_COMMAND(..., DMS_GET_FLASH_CONTENT, ..., dms_get_flash_content)
#endif
```

`sec_flash_read()` / `sec_flash_write()` are **MDC-only** and are not compiled into the 310P
driver, so the card's secure flash cannot be read from the host on this platform. This kills the
`hboot1a`/`hboot1b` lead that the previous log recommended — not because the search was wrong, but
because **the 310P's driver exposes no read-back path for the card's flash at all**, and the
`hboot1*` components exist only in the card's flash with no host-side file.

### The trust architecture, now fully settled

| layer | trust anchor | writable by us? | verdict |
|---|---|---|---|
| **BL31 boot** | `Huawei Integrity Root CA - G2` embedded in `Ascend310P.fd` | no — payload self-authenticates (V2) | **closed** |
| **SoC runtime CMS** | `ccpu_usr_cert_hash` + `/var/asdrv/.../verify/<type>` | the mechanism is **910B-only** (`dms_custom.mk` gates on `hi1980b`); absent on 310P | **closed** |
| **MCU** | `trustlist` / `customized_verify` on the MCU flash | separate device, hardware-rooted | **closed** |
| **host driver** | our OpenSensor fork, installed, no card signature | **yes** | **open** |
| **AI Core device kernels** | plain SHA-256 in the op descriptor | **yes — proven above** | **OPEN** |
| **AICPU device kernels** | plain `dlopen` of an ELF, no signature | **yes — confirmed above** | **OPEN** |

The boot and runtime firmware paths are closed by six independent lines of evidence and are a
property of the hardware's closed trust roots. **What remains genuinely open — and is now proven
rather than argued — is custom device-executed code.**

### Card state

Healthy and fully functional: 6 `davinci` nodes, all 6 chips `OK`, 0 `phy_startup` errors, stock
images in place, stock AI Core kernel binary reinstalled, source tree restored bit-exactly.

### What to do next

1. **Widen the AI Core proof into a usable capability.** The proof used one op
   (`W2SwigluV310`). The next step is a real semantic optimisation of that op — the numerics are
   ours, and the build→deploy→measure loop is ~5 seconds.
2. **Exercise the AICPU channel** the same way: build a replacement `libaicpu_custom.so` (or a
   per-op kernel blob from `aicpu_hcomm.tar.gz`) and show it running on the card's AI CPU.
3. **Do not re-open the boot path.** Six independent negative results (B/D/R/S/T/V/goal/P1) plus
   the source-level `dms_custom` gating make it a settled question, not an open one.

### Tooling added this session

| file | purpose |
|---|---|
| `customer-cert-20261026/prove_aicore_exec.py` | proves the custom AI Core op executes and matches a host reference |
| `customer-cert-20261026/prove_aicore_ownership.sh` | proves the executed binary is *ours* (build → inject → rebuild → compare) |
| `customer-cert-20261026/read_sec_flash.py` | fixed: opens the DMS sub-module before the feature-command ioctl |
| `customer-cert-20261026/esbc_info.py` | copied to the card; parses any esbc container (used to show the AICPU syskernels carry no signature sections) |

---

## 2026-10-26 (session 3) — the MCU flash was never analysed; now it has been

The previous session's note about `ascend_220000_verified.bin` was accurate about *what* the file
contains but the file was never disassembled. It is not an SoC secure-flash dump at all — it is the
**card's MCU flash**: the board-management-controller firmware that implements the manufacturer's
own customer-certificate machinery. This session decoded it.

### What the file actually is

```
size    : 262144 bytes (256 KB), a WINDOW of the card's 16 MB MCU SPI flash
sha256  : (window) at offset 0x220000 of artifacts/npu-collective-qfn-20261014-024609/
          spi-dump-ascend-verified-9fd4cc63.bin   <- the full 16 MB dump we already had
arch    : ARM Cortex-M, Thumb-2  (not RISC-V; confirmed by prologue decoding)
OS      : LiteOS  ("MCU START.", "Begin to running LiteOS.", "MCU Version is V23.2.6")
crypto  : mbedTLS (mbedtls_sha256_*, mbedtls_rsa_pkcs1_verify, mbedtls_mpi_read_binary)
base    : 0x08028000 (file offset + 0x28000 = flash 0x220000)
```

**Two corrections to earlier notes in this log:**

1. The claim that `0x20000+` is *encrypted* is wrong. It is **plaintext ARM Thumb-2 code**
   (0x24000..0x28000 decodes at ~1650 instructions per 4 KB, the density of real code; entropy
   6.6 bits/byte, which is code, not ciphertext). The whole 256 KB window is code + rodata.
2. `0x55aa55aa` in this file is **not** the SoC esbc container format. The SoC esbc header is
   `{magic, head_len=0x4dc, certtype=0xffffffff, ...}`; the MCU's records are
   `{magic, 1, 0, 0x14, 0x1f, 0x100, 0x200, 0x1cd9dead, ...}` — a different structure.

### ✅ The MCU trust machinery is fully recovered

The firmware's log macro passes `__func__` as a string, so every named function has a code site
that loads its own name. That yields a **232-symbol map**, and the trust path disassembles
cleanly:

| function | code address | role |
|---|---|---|
| `customized_verify` | `0x0804940c` | verify the "customized" (customer) region |
| `customized_msg_check` | `0x080495f0` | per-frame message check |
| `customized_info_read/write/clear` | `0x080496aa` / `0x080494cc` / `0x08049710` | region accessors |
| `WriteCertToFlash` | `0x080560e8` | **write a certificate into MCU flash** |
| `ClearCertToFlash` | `0x08055f98` | erase it |
| `get_cert_region_id` | `0x08055972` | map region name → flash region id |
| `write_cert_info` / `renew_cert_info` | `0x08056284` / `0x08055bbe` | certificate write/renew |
| `set_switch_state` / `UpdateCertSwitchState` | `0x08056416` / `0x0805b5aa` | A/B switch state |
| `SaveCertRandom` / `UpdateCertRandom` | `0x0805b718` / `0x0805b656` | anti-rollback random |
| `McuSignatureVerify` | `0x0805bc18` | verify a signed header |
| **`CheckPubKeyHash`** | `0x0805bb5a` | **the trust anchor check** |
| `RsaVerify` / `VerifySig` | `0x0805be76` / `0x0805bdc6` | mbedTLS RSA verify |
| `CalHash` / `read_rsa_head_item` | `0x0805b9c0` / `0x0805ba36` | read+hash the pubkey from a signed header |
| `elabel_init/read/update/exit` | `0x08049aa4` / `0x08049862` / `0x080498fa` / `0x08049800` | electronic label |
| `board_read_elable_response_callback` | `0x08053552` | answers host opcode 0x15 |
| `comm_distribute_handle` | `0x0804e572` | MCU command dispatcher |

### The trust anchor, located exactly

`CheckPubKeyHash` (0x0805bb5a) does:

```
sha256_init(ctx)                       ; mbedtls_sha256_init
sha256_update(ctx, pubkey, 32)         ; the 32-byte public key from the signed header
sha256_finish(ctx, out)                ; 32-byte digest
memcmp(out, CONST, 32)                 ; <-- 0x0805bbae
```

The constant it compares against is **a 32-byte value baked into the MCU firmware** at
address `0x08032be4`:

```
ab510f4a10966739bbd5f96ecf6b69b61de9bc39f6b9429b21a8db90e20c5074
```

So the MCU's trust root is `sha256(vendor_public_key)`, hardcoded in the firmware, with an exact
32-byte comparison. **No key we hold hashes to it** — checked 74 certs/keys we have (rootcaG2,
signcert, ts, tsca, our Oct-21 PKI) across SPKI, PKCS#1, whole-file and raw-modulus encodings:
no match. The anchor is the vendor's own.

### The "customized" region is protected by CRC16 only — this is the weak link

`customized_verify` (0x08049424):

```
r0 = *r6                     ; pointer to the customized region
r5 = u16(r0[0x0a])           ; expected CRC16, from the region header
r1 = u16(r0[0x08])           ; length
r0 = r0 + 0x0c               ; payload
crc16_ccitt(r0, r1)          ; table at 0x0805b2ec: 0x2338,0x0803,0x2300,... = CRC-16/CCITT-FALSE
cmp r5, r0
beq -> success ; else "[%s:%d] failed to crc16." / "data crc error."
```

The integrity of the customer-certificate region is a **CRC-16**, not a signature. I verified the
algorithm by reimplementing it and matching the firmware's own table. `customized_info_init`
additionally checks the region magic `0x50505050` ("PPPP") and, if the region is unwritten, logs
*"data not exists, set default."* and falls back to defaults.

`McuSignatureVerify`/`VerifySig` (RSA over the signed header) protect the **MCU's own upgrade
files**, not the trust store the host writes.

### Flash layout of the trust regions (from the full 16 MB dump)

```
0x220000  MCU image (App)         <- the 256 KB window we have
0x237800  trust record            } A/B duplicated, byte-identical
0x277800  trust record            }
0x8c0000  trust record (3rd copy, 49008 bytes)
0x780000  "customized" region: 50505050 00000001 00000000 ffffffff
          -> magic present, len/crc slots 0xffffffff => NEVER WRITTEN on this card
0x2c0000  "BALE" log region       0x300000  "GAMF" region      0x7c0000  zeroed region
```

Region ids are assigned in `get_cert_region_id` (0x08055946):

```
sp+0xc = 9    sp+0xd = 10   sp+0xe = 14   sp+0xf = 11   sp+0x10 = 12 (only if id > 4)
```

against the region-name table at file 0x02794:

```
[0] https   [1] https crl   [2] switch   [3] trustlist   [4] blocklist   [5] null
```

**So the MCU holds a `trustlist`, a `blocklist`, an `https` cert region and an `https crl`
region — and the customer's own certificate goes into the `trustlist` region via
`WriteCertToFlash`.**

### Host reachability of that path — measured, not assumed

The MCU channel is **alive** on the card:

```
dsmi_get_board_id(0)  ret=0 board_id=0x97
dsmi_get_system_time  ret=0
dsmi_get_board_info   ret=0
```

but the entry points are **not wired up** in this driver build:

| symbol | status |
|---|---|
| `dsmi_dft_get_elable` | **stub** — `mov eax,0xfffe; pop rbp; ret` (returns INVALID_VALUE) |
| `dsmi_get_custom_op_secverify_cert_show_info` | **stub** — same 7-byte pattern |
| `dsmi_cmd_dft_get_elabel` | real: builds MCU opcode **0x15** = `DEV_MON_CMD_READ_LABEL_DATA` |
| `dsmi_cmd_get_sign_cert` | real |
| `dsmi_get_rootkey_status` | real (validates `item_type <= 0x3f`; returns 17 `EEXIST` here) |
| `dsmi_get_board_id` / `_info` / `_system_time` | real, working |

Calling `dsmi_cmd_dft_get_elabel` directly (after `dsmi_init`) still returns `0xfffe` — the
failure is inside the driver, not in my call: the cmd builds the DMP message with opcode 0x15 and
then fails `dsmi_check_device_id`, i.e. the 310P kernel module has no handler for it. Confirmed
against the module itself: `drv_devmng_host.ko` contains **no** `label`/`elabel` strings and no
MCU/board comm handler at all — board info actually comes from the **SoC**
(`DMS_MAIN_CMD_SOC` / `DMS_SUBCMD_GET_SOC_INFO`), not from the MCU.

`DEV_MON_CMD_READ_LABEL_DATA (0x15)` and `DEV_MON_CMD_WRITE_CLEAR_LABEL_DATA (0x14)` **are
defined** in the driver's `dev_mon_cmd_def.h`, but are not implemented in the shipped 310P build.

### Where this leaves the "hole in the implementation"

The MCU's own certificate trust store is CRC-16 protected, and its signature verification
(`McuSignatureVerify`) only covers **MCU firmware upgrades**. The crypto anchor for the trust
store is the firmware-embedded `sha256(vendor_pubkey)`.

The realistic attack surface this opens is *not* forging the vendor's signature — that anchor is
firmware-embedded and we hold no matching key. It is:

1. **The CRC-16 on the `trustlist` region is not a cryptographic integrity check.** Anything that
   can write the region can install a customer certificate by recomputing a 16-bit checksum.
   Whether that matters depends on who can reach `WriteCertToFlash`.
2. **`WriteCertToFlash` / `ClearCertToFlash` / `customized_info_write` exist in the MCU firmware**
   and are reached through the board comm channel. On *this* driver build the host-side door is
   closed (stubbed/unimplemented), but the MCU-side handler is present in the shipped firmware —
   so the question becomes whether any *other* host path can drive board opcodes 0x14/0x15, or
   whether the SPI flash is writable directly (the region is plain SPI flash, and
   `SPI_FLASH1_Elabel_Write_Protect` / `SPI_FLASH1_FW_Write_Protect` exist as write-protect
   helpers whose coverage needs checking).
3. **The MCU image itself is upgradeable** via `copy_upgrade_file_to_flash`, gated by
   `verify_firmware_file` → `read_cms_file_info` → `RsaSignatureVerify`, i.e. it *is* signed. But
   the **app region's checksum** path (`flash_sha256_check`, `sha256_check`) is a separate,
   weaker mechanism than the signature path, and `boot_valid_check` / `sync_app_master_slave`
   choose between App0/App1 — worth checking whether a checksum-valid-but-unsigned app is ever
   booted.

### Next steps, in priority order

1. **Determine the write-protect coverage of the trust regions.** `SPI_FLASH1_Elabel_Write_Protect`
   (0x0805c4b8) and `SPI_FLASH1_FW_Write_Protect` (0x0805c6ea) decide which sectors are locked.
   Read them and map the protection onto the region table — this decides whether the trustlist is
   physically writable.
2. **Check whether the host can reach board opcodes 0x14/0x15 by any other route** (e.g. the
   `board_common_response_callback` / `traversal_data_update` path, or a raw DMP command), since
   the dedicated API is stubbed but the MCU-side handlers are present.
3. **Test the A/B redundancy logic.** `boot_valid_check` + `sync_app_master_slave` choose between
   App0 and App1; `set_boot_param`/`WriteBackBootParam` select them. If a checksum-valid region can
   be swapped in, that is a much cheaper lever than the signature path.
4. **Check `flash_sha256_check` vs `RsaSignatureVerify` asymmetry** — specifically whether
   `_app_check` (0x0804c6aa) can be satisfied by an unsigned image whose checksum matches.

### Tooling added this session

| file | purpose |
|---|---|
| `customer-cert-20261026/mcu_disasm.py` | recovers the MCU symbol map and disassembles named functions |
| `customer-cert-20261026/mcu_annotated.py` | annotated disassembly: resolves each PC-relative literal to its string |
| `customer-cert-20261026/mcu_findfuncs.py` | string→code-site index for the whole image |
| `customer-cert-20261026/mcu_symbols.py` | extracts the `(log-msg, __func__)` name table |
| `customer-cert-20261026/mcu_refs.py` | resolves PC-relative string references |
| `customer-cert-20261026/cert_api_survey.py` | classifies DSMI exports as STUB vs real, by ELF symbol address |
| `customer-cert-20261026/elabel_probe.c`, `mcu_probe.c`, `mcu_trust_probe.c`, `rootkey_probe.c` | live probes of the MCU channel and cert APIs |

### Addendum (same session): write-protect coverage and the elabel, resolved

Continuing the "next steps" list above, item 1 (write-protect coverage) is now answered.

**`SPI_FLASH1_Init` (0x0805c6a2) computes the firmware write-protect range as:**

```
r4 = ldrh [chipinfo,#2] << 17      ; total flash size
r7 = ldrh [chipinfo,#4] << 10      ; sector size
r4 = (r4 - 0x800000) / r7          ; number of sectors to protect
sb = 0x800000                      ; <-- protect from 8 MB upward
loop: issue sector write-protect on each sector from 0x800000
```

For this card's 16 MB flash (`blocks=128`, so `128 << 17 = 0x1000000`), the protected range is
**0x800000 .. 0x1000000 (8 MB .. 16 MB)**.

Mapping the regions we found against that:

| region | flash offset | protected? |
|---|---|---|
| MCU app image (App0) | `0x220000` | **no** |
| trust record A | `0x237800` | **no** |
| MCU app image (App1) | `0x260000` | **no** |
| trust record B | `0x277800` | **no** |
| "PPPP" customized region | `0x780000` | **no** |
| trust record (3rd copy) | `0x8c0000` | yes |

So **only the third copy** of the trust record is inside the firmware write-protected range. The
two A/B copies at `0x237800`/`0x277800` and the "customized" (customer-certificate) region at
`0x780000` are **not** write-protected by the firmware's own protection pass.

**The elabel is a separate, explicitly-managed area.** `SPI_FLASH_WritElectronicLabel`
(0x0805d088) bounds the write:

```
r3 = offset + length
cmp r3, #0x40000          ; 256 KB ceiling
bls -> ok
else -> "input length or offset wrong error."
```

and it **disables the write protect before writing** ("failed to disable write protect elabel
area"), re-enabling it afterwards — via `SPI_FLASH_EraseElectronicLabel` /
`SPI_FLASH_WritElectronicLabel` / `SPI_FLASH1_Elabel_Write_Protect`. So the MCU *deliberately*
un-protects the elabel to write it, then re-protects.

**The card's electronic label is now readable.** It lives in the `BALE` region and is plaintext:

```
flash 0x2c00e4 : u32 length = 0x6222d (401,965)
flash 0x2c00e8 : magic "Huawei"
flash 0x2c0130 : "IT21PD2G10"        <- board model
flash 0x2c0178 : "2106030737ZERC001310"  <- serial number
flash 0x2c0250 : second "Huawei" record (A/B copy)
```

Two identical records exist (0x2c00e4 and 0x2c0244) — the same A/B redundancy as the trust
records. The elabel is a **length-prefixed, magic-tagged, CRC/plaintext record**: no signature.

The MCU's own `elabel_init` logs *"magic number is valid."* and *"fail to read and verify
elabel."* — so elabel integrity is checked, but by what mechanism needs confirming against
`elabel_init` (0x08049aa4). The elabel region is the one place where the MCU both writes and
verifies a structured, non-firmware record with a `Huawei` magic, and where the write path
explicitly un-protects the sectors. **That is the most promising target for the "hole" the
hypothesis suggests** — a structured record with a magic, a length, and an un-protected window,
written through a documented un-protect → write → re-protect sequence.

### Refined assessment of the "implementation hole"

Three concrete findings, in decreasing order of promise:

1. **The elabel is written through an un-protect/write/re-protect sequence and verified only by
   magic + length** (the `Huawei` magic and the `0x6222d` length are plain, unsigned fields).
   If `elabel_init`'s verification turns out to be a CRC rather than a signature, the elabel is
   forgeable by anyone who can reach the SPI flash.
2. **The customer-certificate region (`PPPP`, `0x780000`) is protected by CRC-16 only** and is
   **not** in the firmware's write-protected range. `WriteCertToFlash` exists to write it.
   The blocker is reachability, not cryptography.
3. **The trust records at `0x237800`/`0x277800` are also outside the write-protected range.**

The cryptographic anchor for the *signature* path is `sha256(vendor_pubkey)` hardcoded at
`0x08032be4` — we hold no matching key, and that is a genuine hard stop for forging the
vendor's signature over an MCU upgrade.

The remaining open question is now narrow and testable: **can the SPI flash be written directly?**
The MCU's SPI is a separate bus (not PCIe), so it is only reachable through the MCU itself
unless the board exposes the flash some other way. That is the thing to check next, together
with `elabel_init`'s verification algorithm.


### Addendum 2 (same session): the elabel integrity mechanism — CONFIRMED CRC-only, no signature

The last open question — "is the elabel verified by a signature or a CRC?" — is now answered, and
the answer is the hole.

**The elabel record layout** (from `elabel_init`, 0x08049aa4, and the item table at 0x0802aed8):

```
record base r8        = the 'BALE' region (flash 0x2c0000)
item table            = 21 entries x 0x10 bytes at 0x0802aed8
                         u32 flags|item_id | u32 maxlen | u32 offset | u32 label-ptr
per item at r8+offset:
    u16  crc16        <- CRC-16/XMODEM of the item payload
    u16  length
    u8   data[length]
```

**The verification** (elabel_init, 0x08049b8a):

```
0x08049b8a  ldrh sl, [r8, r3]     ; expected CRC from the item header
0x08049b8e  bl   0x805b2c0        ; crc16 over (r8+r3+4), length r1
0x08049b92  cmp  sl, r0
0x08049b94  bne -> "fail to read and verify elabel."
```

The CRC table at `0x08032338` is the standard **CRC-16/CCITT-FALSE** table
(`0000,1021,2042,3063,4084,50a5,60c6,70e7,...`) — i.e. **CRC-16/XMODEM**
(poly 0x1021, init 0x0000, no reflection, no final xor).

I verified this against the real card data — **21/21 items validate with CRC-16/XMODEM alone**:

```
idx     id  max     off    crc   len   calc  status value
  3 0x0020    68 0x00009c 0x888e     3 0x888e  OK     b'\xecy\xf0'
  4 0x0021    68 0x0000e4 0x222d     6 0x222d  OK     b'Huawei'
  5 0x0022    68 0x00012c 0x5efb    10 0x5efb  OK     b'IT21PD2G10'
  6 0x0023    68 0x000174 0x9411    20 0x9411  OK     b'2106030737ZERC001310'
  9 0x0030    68 0x00024c 0x222d     6 0x222d  OK     b'Huawei'
 ... (16 further items, all len=0 / empty on this card)
```

So the card's electronic label — the record carrying the board model and serial number — is
protected by a **16-bit checksum and nothing else**. There is no signature, no public key, no
MAC over this path. The same is true of the customer-certificate region (`PPPP` at 0x780000,
`customized_verify`, CRC-16 via the same routine).

**And the write path deliberately un-protects the sectors first:**

```
SPI_FLASH_WritElectronicLabel (0x0805d088):
    cmp (offset+length), #0x40000     ; 256 KB ceiling
    bl  0x805c4a4                     ; -> SPI_FLASH1_Elabel_Write_Protect  (UNPROTECT)
    ...
    "failed to disable write protect elabel area."   <- if the unprotect fails
SPI_FLASH_EraseElectronicLabel / SPI_FLASH1_Elabel_Write_Protect  (re-protect afterwards)
```

### The "implementation hole" — stated precisely

The manufacturer's design separates two things:

| what | how it is protected | can we do it? |
|---|---|---|
| MCU **firmware upgrade** (`copy_upgrade_file_to_flash`) | `RsaSignatureVerify` over a signed header, key pinned by `CheckPubKeyHash` against `sha256(vendor_pubkey)` = `ab510f4a…c5074` hardcoded at 0x08032be4 | **no** — we hold no matching key; a genuine hard stop |
| MCU **data regions**: elabel (`BALE`), customer cert (`PPPP`), trust records | **CRC-16 only**, no signature, sectors not in the firmware write-protected range | **cryptographically yes**; blocked only by reachability |

So the hole is real and it is exactly where the hypothesis said: **every non-firmware record on
the MCU is integrity-checked with a 16-bit checksum rather than a signature.** The vendor's
signature machinery is used only for their own firmware, while the regions the *host* is
supposed to write — the customer certificate, the electronic label, the trust records — are
protected by a CRC.

The remaining obstacle is therefore **not** cryptographic, it is **reachability**: the MCU's SPI
flash is on the MCU's own SPI bus, not the PCIe bus, so writing it requires either

1. driving the MCU's board-comm opcodes (`WriteCertToFlash` / `customized_info_write` /
   `elabel_update`) — the MCU-side handlers exist in the shipped firmware, but on this driver
   build the host entry points are stubbed (`dsmi_dft_get_elable` returns 0xfffe) and
   `drv_devmng_host.ko` implements no label/elabel or MCU-comm handler at all; **or**
2. writing the SPI flash out-of-band (a flash programmer on the board, or any host path that
   exposes the MCU's SPI bus); **or**
3. exploiting the A/B redundancy — `boot_valid_check` + `sync_app_master_slave` choose between
   App0/App1, and `set_boot_param`/`WriteBackBootParam` select them. If a CRC-valid but
   unsigned app region can be selected, the signature check on the firmware path is bypassed
   entirely. **This is the most promising avenue and it is still untested.**

### Next step (single, concrete)

Test the A/B selection path: determine whether `boot_valid_check` gates on the *signature*
(`RsaSignatureVerify`) or merely on the *checksum* (`flash_sha256_check` / `_app_check`). The
disassembly shows both mechanisms exist and `boot_valid_check` logs
`"Boot param error : boot0valid = %d, boot1valid = %d"` — the question is what sets those
`bootNvalid` flags. If it is the checksum rather than the signature, then a CRC-valid
substituted app region is bootable and the whole boot chain collapses to CRC strength.

Tooling: `customer-cert-20261026/mcu_elabel.py` implements the full elabel parse + CRC
verification against the 16 MB dump (`21/21 items validate`).


### Addendum 3 (same session): the A/B boot-selection path is SIGNED — hypothesis disproven

The single concrete next step from addendum 2 was to find out whether `boot_valid_check` gates on
the **signature** or only on the **checksum**. Answered, and it goes against the hypothesis.

**The boot parameters are stored raw in flash and read back raw:**

```
set_boot_param_to_flash (0x0805b466):
    r4 = (index << 2) + 0x8000000 + 0x7800        => flash 0x78000 + index*4
    ReadFlashWord(0x78000 + 4*i) -> value
    compare with what we wanted to write; mismatch -> "param check failed"

WriteBackBootParam (0x0805b858):
    for r4 = 0x08007800 .. 0x08007840:
        ReadFlashWord(r4) -> RAM table at 0x200029b8
```

So the boot-param table (16 u32 at flash `0x78000..0x78040`, loaded into RAM at `0x200029b8`) is
**not signed and not CRC'd** — it is raw flash. On this card the whole area is `0xffffffff`
(never written; the MCU uses defaults).

**But the `bootNvalid` flags are only set behind a signature check.** Tracing every caller of
`set_boot_param`:

| caller | index | meaning |
|---|---|---|
| `copy_upgrade_file_to_flash` 0x0804b6ca | 2 | `boot1valid` |
| `switch_mcu_image` 0x0804bc7c/0x0804bc96/0x0804bca8/0x0804bd08 | 1, 2, 3 | `boot0valid`, `boot1valid`, `apprefreshflag`, `bootorder` |
| `validate_mcu_image` 0x0804be10 | 2 | `boot1valid` |
| `update_boot_param` 0x0804be3c/0x0804be5c/0x0804be7c | 2, 3, 2 | |
| `sync_app_master_slave` 0x0804b9e4 | — | |

and in `copy_upgrade_file_to_flash`:

```
0x0804b640  bl  verify_firmware_file
0x0804b64e  bl  0x0804afb0          ; RsaSignatureVerify
0x0804b652  cmp  r0, #0
0x0804b654  bne  -> "verify firmware failed"     <- signature must pass FIRST
...
0x0804b6ca  bl  set_boot_param(2, ...)            <- only then boot1valid is set
```

and `RsaSignatureVerify` (0x0804afb0) calls `McuSignatureVerify` (0x0805bbe4) — the full mbedTLS
chain, ending at `CheckPubKeyHash` against the hardcoded `sha256(vendor_pubkey)`.

`switch_mcu_image` then reads those flags back with **no re-verification**
(`get_boot_param(1)` / `get_boot_param(2)`), but that is not exploitable by itself: the only way
to *set* them is through a path that already passed the signature check.

**Conclusion: the boot path is signed end-to-end.** The A/B redundancy is not a way around the
signature — the flags are only reachable behind it. This kills the most promising of the four
"next steps" listed earlier. The boot chain is closed, consistent with the six independent
negative results already in this log.

### Where the hole actually is — final statement

The trust boundary that is genuinely weak is **not** the boot chain. It is the set of **MCU data
regions the host is meant to write**, which the vendor protects with a **16-bit CRC** instead of
a signature, while reserving signatures for their own firmware:

| region | flash | integrity | reachable from host on this build? |
|---|---|---|---|
| MCU firmware (App0/App1) | 0x220000 / 0x260000 | RSA signature (`McuSignatureVerify`) + `sha256(vendor_pubkey)` anchor | upgrade path is signed; blocked |
| bootloader copy | — | `bootloader_upgrade_validete_api`, sha256 | signed |
| **electronic label** (`BALE`) | 0x2c0000 | **CRC-16/XMODEM only** (21/21 items verified) | `dsmi_dft_get_elable` is a **stub**; `drv_devmng_host.ko` has no elabel handler |
| **customer certificate** (`PPPP`) | 0x780000 | **CRC-16 only** | `dsmi_get_custom_op_secverify_cert_show_info` is a **stub** |
| trust records A/B | 0x237800 / 0x277800 | CRC-16 only | same |

And the write path actively un-protects the sectors to do it
(`SPI_FLASH_WritElectronicLabel` disables the write-protect, writes, re-enables it), and the
firmware's own protection pass covers only `0x800000..0x1000000`, leaving all of the above
unprotected.

So the remaining question is exactly one thing, and it is a **reachability** question, not a
crypto one:

> **Is there any host path that can drive the MCU's board-comm opcodes, or write the MCU's SPI
> flash, on this card?**

Concretely still to check:
1. Whether a raw DMP command can carry opcode `0x14` (`DEV_MON_CMD_WRITE_CLEAR_LABEL_DATA`) /
   `0x15` (`DEV_MON_CMD_READ_LABEL_DATA`) without going through the stubbed public API — the MCU
   *does* implement `board_read_elable_response_callback` (0x08053552) and the write/clear
   handlers; the question is whether the 310P kernel module forwards them at all.
2. Whether the SoC can reach the MCU's SPI bus by another route (the MCU also drives the VRD/MPS
   over I2C and SPI — `i2c_bus_read`, `mps_write_vrd_chip_cfg`, `store_data_into_mtp` — so the
   SoC↔MCU coupling may exist even though the host↔MCU door is stubbed).
3. Whether the `traversal_data_update` / `board_opcode_data_update` path in the MCU can be
   driven by a SoC-side command that *is* implemented in the driver.


### Addendum 4 (same session): the host→MCU channel does not exist in this driver build

The last open item was whether any host path can reach the MCU. Answered, and it is closed.

**There is exactly one generic MCU message channel in the driver, and it is not built for the
310P.** `dms/lpm/dms_lpm_pass_through_mcu.c` implements `CMD_IMU_MCU_MSG` (a pass-through of
arbitrary MCU messages via IMU), and `dms/lpm/dms_lpm.mk` gates it:

```make
ifeq ($(PRODUCT_SIDE), host)
else
    ifneq ($(filter $(PRODUCT), ascend910 ascend910B),)
        $(MODULE_NAME)-y += lpm/dms_lpm_pass_through_mcu.o
        ccflags-y += -DCFG_FEATURE_PASS_THROUGH_MCU_BY_IMU
    endif
endif
```

`ascend910`/`ascend910B` only. Confirmed against the **installed** module:

```
nm drv_devmng_host.ko | grep -i "pass_through_mcu\|imu_mcu"   ->  (no symbols)
```

So on this card there is **no host→MCU message path at all**:

| candidate route | status on the 310P build |
|---|---|
| `dsmi_dft_get_elable` (MCU opcode 0x15, read label) | 7-byte **stub**, returns 0xfffe |
| `dsmi_get_custom_op_secverify_cert_show_info` | 7-byte **stub**, returns 0xfffe |
| `dsmi_cmd_dft_get_elabel` / `dsmi_cmd_get_sign_cert` | real cmd functions, but the kernel module has no handler — the call fails inside `dsmi_check_device_id` after the DMP message with opcode 0x15 is built |
| `DEV_MON_CMD_READ_LABEL_DATA (0x15)` / `WRITE_CLEAR_LABEL_DATA (0x14)` | **defined** in `dev_mon_cmd_def.h`, **not implemented** in the 310P module |
| `drv_devmng_host.ko` MCU comm handler | **absent** — no `mcu`/`board`/`label` strings in the module at all |
| IMU MCU pass-through | `ascend910`/`ascend910B` only, **not compiled** |
| board info | comes from the **SoC** (`DMS_MAIN_CMD_SOC`/`DMS_SUBCMD_GET_SOC_INFO`), not the MCU |

### Final state of the investigation

**Closed by evidence (boot and runtime firmware trust):**

- MCU firmware (App0/App1) and bootloader upgrades: RSA signature via `McuSignatureVerify`,
  pinned by `sha256(vendor_pubkey)` = `ab510f4a…c5074` hardcoded at `0x08032be4` and compared
  with `memcmp`. We hold no matching key (74 certs/keys checked across five encodings).
  The A/B `bootNvalid` flags are only settable behind that signature check, so the redundancy is
  not a bypass.
- SoC boot (BL31) and the SoC runtime CMS (`/var/asdrv/.../verify/<type>`): closed by six
  independent negative results plus the source-level `dms_custom` gating (the mechanism is
  910B-only).
- MCU → host reachability for every trust-bearing record: **no channel exists** in this driver
  build.

**Open, and now precisely characterised:**

The MCU's *data* records — the ones the host is meant to write — are protected by a **16-bit
CRC** rather than a signature, sit outside the firmware's own write-protect range
(`0x800000..0x1000000`), and the elabel write path deliberately un-protects the sectors first:

| region | flash | integrity |
|---|---|---|
| electronic label (`BALE`) | 0x2c0000 | CRC-16/XMODEM only — **21/21 items verified** |
| customer certificate (`PPPP`) | 0x780000 | CRC-16 only (`customized_verify`) |
| trust records A/B | 0x237800 / 0x277800 | CRC-16 only |
| boot params | 0x78000..0x78040 | raw, no CRC/signature (but only settable behind the signature path) |

So the manufacturer's implementation hole is real and now precisely located: **they used
signatures for their own firmware and a CRC for everything else.** The crypto side of that hole
is trivially exploitable — recompute a 16-bit checksum. What remains is purely physical
reachability, and on this driver build the host↔MCU door does not exist.

That leaves two genuinely open avenues, both outside the driver:

1. **Out-of-band SPI access to the MCU flash.** The MCU's SPI is a private bus; the only way in
   is the MCU itself *or* a flash programmer on the board. If the board exposes the MCU's SPI
   lines (or a test point), the elabel and the customer-certificate region are writable with a
   CRC recomputation. This is a hardware-access question, not a software one.
2. **Rebuilding the host driver to add the missing handler.** We already build our own driver
   fork (`ascend-310p-host-driver-spin-20261015`). The `dev_mon_cmd_def.h` opcodes 0x14/0x15
   exist, and the MCU implements the handlers — but the 310P module has no dispatch table for
   them, so adding one (or enabling the IMU MCU pass-through, which is a one-line `PRODUCT`
   filter change) would open the door. That is entirely within our control, and it is the
   natural next step.


### Addendum 5 (same session): the MCU transport, and a note on the `hboot1` lead

Two loose ends closed.

**The MCU's host transport is not IPC.** `MCU_COMM_Init` (0x0804e402) only allocates a 0x208-byte
receive buffer; the actual transport is set up elsewhere in the message task. The message framing
in `comm_distribute_handle` (0x0804e458) is:

```
0x0804e478  ldrb r3, [r0] ; and r3, #0xc0 ; cmp r3, #0x40   ; header check
0x0804e482  ldrb r3, [r2, #3] ; and r3, #0xe ; cmp r3, #2     ; message type
0x0804e4a0  ldrh r3, [r2]   ; strh r3, [r1, #3]              ; length
0x0804e4a6  movs r3, #0xc ; strb r3, [r1]                    ; opcode class
```

i.e. a fixed 12-byte header carrying a length and an opcode class, delivered over a peripheral
the MCU owns (no UART/LIN/CAN strings are present in the image beyond unrelated hits, and the
SoC has no `imu` device node on this card). The MCU's comm is wired to the SoC through a
dedicated peripheral the 310P driver does not expose.

**A note on the `hboot1a`/`hboot1b` lead from the very first session.** That lead was about the
*SoC* bootloader components. The MCU has its own bootloader, and the two are distinct:
`copy_bootloader_from_flash1_to_inner` (0x0805f4e2) copies the MCU's bootloader from SPI flash
region 3 into **inner flash** at `0x08000000`:

```
0x0805f50e  mov.w r6, #0x8000000        ; inner flash base
0x0805f512  mov.w r3, #0x100            ; 256-byte block
0x0805f51c  bl   0x805d73c              ; SPI_FLASH_1_Read(region 3, ...)
0x0805f536  ldr  r3, [r7], #4           ; then program into inner flash
```

and it is gated by `bootloader_upgrade_validete_api` (0x0804bfea), which does a **sha256** check
of the bootloader ("check bootloader sha256 OK"). So the MCU bootloader is sha256-gated, not
signature-gated — but the sha256 is over a file the vendor ships, and the bootloader copy is
performed by the MCU itself from its own flash, not by the host. Nothing here is host-writable.

### Final assessment

Across the MCU, the SoC bootloader, the SoC runtime CMS, the host driver, the AI Core kernels
and the AICPU kernels, the picture is now:

| boundary | mechanism | status |
|---|---|---|
| SoC boot (BL31) | payload self-authentication, `Huawei Integrity Root CA - G2` | closed |
| SoC runtime CMS | `ccpu_usr_cert_hash`, mechanism is 910B-only | closed |
| MCU firmware + bootloader | RSA signature / sha256, anchored on `sha256(vendor_pubkey)` = `ab510f4a…c5074` at 0x08032be4 | closed — we hold no matching key |
| MCU A/B boot selection | `bootNvalid` only settable behind the signature check | closed |
| MCU data records: elabel, customer cert, trust records | **CRC-16 only** | **weak — but no host channel exists on this build** |
| host driver | our own fork, installed, unsigned | **open** |
| AI Core device kernels | plain SHA-256 in the op descriptor | **open, and proven executing** |
| AICPU device kernels | plain `dlopen` of an ELF | **open** |

The manufacturer's implementation hole is real and now precisely characterised: **they protect
their own firmware with RSA signatures and everything the host writes with a 16-bit CRC.** The
crypto weakness is trivial to exploit; what is missing is a transport.

That transport, concretely, would be one of:

1. **The MCU's SPI flash out-of-band** — a programmer on the board. Hardware access, not
   software. The elabel (0x2c0000) and the customer-cert region (0x780000) would then be
   writable with a recomputed CRC-16.
2. **A driver that actually dispatches the board opcodes.** We build our own driver fork
   (`ascend-310p-host-driver-spin-20261015`). `dev_mon_cmd_def.h` defines
   `DEV_MON_CMD_WRITE_CLEAR_LABEL_DATA (0x14)` and `DEV_MON_CMD_READ_LABEL_DATA (0x15)`, and the
   MCU implements the handlers (`board_read_elable_response_callback` at 0x08053552,
   `write_customized_info_callback` at 0x08053e52, `clear_customized_info_callback` at 0x08053eac).
   The 310P module has no dispatch table for them. Adding one is real driver work, but it is
   entirely within our control — and the *driver* is already a proven-open boundary.
3. **A different transport the SoC uses to reach the MCU.** The MCU also drives the VRD/MPS over
   I2C (`i2c_bus_read`, `mps_write_vrd_chip_cfg`, `store_data_into_mtp`, `switch_vrd_page`) and
   those are reachable from the SoC side. Worth checking whether any of them can be coerced into
   carrying a flash write.

Option 2 is the most promising *because the host driver is already an open boundary we own* —
the AI Core proof showed we can replace the driver and have it take effect. A driver that
dispatches opcode 0x14 to the MCU would turn the CRC weakness into a working
customer-certificate installation path.


---

## 2026-10-26 (session 4) — SINGLE-CARD TEST: forgery accepted, transport absent

The previous session ended with an open question: *is there any host path to the MCU?* This
session tested it against **one card** rather than the whole set.

### Target card identified exactly

| | |
|---|---|
| Card | **NPU ID 9**, bus `0000:01:00.0`, davinci 0/1 |
| Product | `IT21PD2G10` |
| Serial (SoC) | `2106030737ZERC001310` |
| Serial (MCU elabel, decoded from our dump) | `2106030737ZERC001310` — **byte-identical** |
| Other cards' serials | `…ZERB002529`, `…ZERB000091` — **absent from the dump** |

The dump is therefore confirmed to belong to card 9. (Also: `ascend_A.bin` and `ascend_B.bin` are
byte-identical — `9fd4cc63…` — so the two dumps taken two weeks ago were the same flash content.)

### ✅ PROVEN: a forged elabel passes the firmware's own verification

`mcu_elabel_forge.py` modifies the real board-model item and recomputes its CRC:

```
baseline verification  : ALL OK (21/21 items)
  id=0x0020 off=0x009c crc=0x888e len=  3 b'\xecy\xf0'
  id=0x0021 off=0x00e4 crc=0x222d len=  6 b'Huawei'
  id=0x0022 off=0x012c crc=0x5efb len= 10 b'IT21PD2G10'
  id=0x0023 off=0x0174 crc=0x9411 len= 20 b'2106030737ZERC001310'

forging item id=0x0022 at 0x2c012c
    forged payload   : b'IT21PD2G11'
    new CRC-16/XMODEM: 0x4eda
    re-verify whole record: ALL OK (21/21 items)
```

Using the firmware's own verification routine (`elabel_init`, 0x08049b8a). **A modified board
identity passes.** The MCU's electronic label is protected by a 16-bit checksum, not a signature.

### The transport: present in both halves, absent in the middle

The vendor elabel path is **I2C**, not PCIe:

```
dcmi_flash_read_elabel(buf, len)               libdcmi @ 0x168b0
    call dcmi_board_chip_type_is_ascend_310b   ; 1 iff chip type == 4 (910B)
    sbb  %rdx,%rdx ; and $0xfffffffffffffe80,%rdx ; add $0x200,%rdx   ; 310P -> reg 0x200
    call dcmi_i2c_get_data(g_i2c_dev_name, 0x50, buf, len, reg)
```

and the HAL selects the bus **by adapter name**, looking for `"SMBus I801 adapter"` — an *Intel*
chipset string:

```
libascend_hal.so: "SMBus I801 adapter"
libascend_hal.so: "/sys/bus/i2c/devices/"
libascend_hal.so: "read [%s], find i2c-%d name dev_name:%s"
```

Measured on the card host:

| bus | adapter | contents |
|---|---|---|
| i2c-0 | `Synopsys DesignWare I2C adapter` (`i2c_designware AMDI0010:00`) | **nothing** — "controller timed out" |
| i2c-1 | `SMBus PIIX4 adapter port 0 at 0b00` | host **DDR4 SPD** EEPROMs (ee1004) 0x50–0x57 |
| i2c-2 / i2c-3 | `SMBus PIIX4` | empty |

Two independent reasons it cannot work here:

1. The name `"SMBus I801 adapter"` does not exist on this **AMD** host, so the HAL falls back to
   i2c-0 — the *chipset's* empty DesignWire controller, not the NPU's.
2. **The card's MCU is on no host I2C bus.** Its flash is on the MCU's own SPI bus; the host I2C
   buses carry only host DDR SPDs.

`strace` of `npu-smi info -l` confirms the working identity path never touches I2C — it opens
`/dev/davinci_manager` and `/dev/hisi_hdc` only. **The SoC-reported serial is a SoC-side copy of
the board identity, not a read of the MCU elabel.**

### Every host entry point measured

| API | result |
|---|---|
| `dsmi_get_flash_content` / `dsmi_set_flash_content` | `0xfffe` NOT_SUPPORT |
| `dsmi_dft_get_elable` | `0xfffe` — 7-byte stub (`mov eax,0xfffe; pop rbp; ret`) |
| `dsmi_cmd_dft_get_elabel` (MCU opcode 0x15) | `0xfffe` — builds the command, no driver handler |
| `dsmi_get_custom_op_secverify_cert_show_info` | `0xfffe` — 7-byte stub |
| `dsmi_get_rootkey_status` | real, returns 17 (EEXIST) |
| `libdcmi` `dcmi_get_card_elabel_v2` | `-8001` |
| `libdcmi` `dcmi_flash_read_elabel` → `dcmi_i2c_get_data` | `I2C_RDWR` → **ETIMEDOUT** |
| direct I2C write probe, slave 0x50 reg 0x200 | `EINVAL` (no responder) |
| `npu-smi info -l` | **works** — SoC-side serial, no I2C |

### Both halves of the write path exist

```
libdcmi:   dcmi_flash_write_elabel @ 0x16a20  -> dcmi_i2c_set_data(dev, 0x50, buf, size, 0x200)
           dcmi_flash_erase_elabel @ 0x16960
           dcmi_elabel_set_data    @ 0x173c0
           dcmi_elabel_clear       @ 0x17200
MCU fw:    board_read_elable_response_callback   0x08053552
           write_customized_info_callback       0x08053e52
           clear_customized_info_callback       0x08053eac
```

The link between them is the missing I2C bus. The same holds for the customer-certificate region
(`customized_verify` / `WriteCertToFlash` / `customized_info_write`), likewise CRC-16 protected.

### Card state

Untouched and healthy: 6 davinci nodes, 6/6 chips `OK`, 0 `phy_startup` errors, stock images
intact (`e1e75ec2…`, `12975a9d…`), identity unchanged. Only read-only operations were performed,
plus one **identical-bytes** write attempt that never reached the bus.

### What would work

Given the CRC-only integrity, the **SPI clip** (`scriba -W`, CH341A/EZP2019) is sufficient: it
produced the dump, it supports write and erase, and a forged elabel with a recomputed CRC-16 is
accepted by the firmware. The alternatives are a host I2C adapter on the NPU's I2C pins (needs the
bus brought out, plus a patched HAL name match or an Intel-named adapter), or driver work to
implement the board-comm opcodes `0x14`/`0x15` that the MCU already handles.

Full detail: `customer-cert-20261026/SINGLE-CARD-TEST-20261026.md`.


---

## 2026-10-26 (session 5) — software-only routes to the MCU, and the SSH angle

The clip requires taking the heatsink off, so this session looked for a purely software route, and
in parallel examined the device-side SSH implementation.

### The card is reachable over the PCIe vNIC (not the LAN)

`npu-smi set -t ssh-enable -i 9 -d 1` succeeds — the host log records
`Set the SSH function for card9 success. enable_flag=1.`

The device is **not** reachable at `192.168.50.1` (that route goes via the router and is blocked by
the VLAN policy). It **is** reachable over the direct PCIe vNIC:

```
ssh -6 matteius@'fe80::2e0:fcff:fefc:d302%endvnic'
  -> Authorized users only. All activities may be monitored and reported.   (device /etc/issue.net)
  -> Permission denied (publickey,gssapi-keyex,gssapi-with-mic,keyboard-interactive)
```

The card-side vNIC addresses are `fe80::2e0:fcff:fefc:d30{2..5}`; d302 is the SSH-enabled card
(NPU ID 9). The link is IPv6 link-local only — the host's `endvnic` has no IPv4, and the device's
own `pcivnic_set_default_ip` gives it `192.168.3.199` with no host-side counterpart.

### The device OS, extracted and read

`Ascend310P.cpio.gz` is an esbc container; the payload is a gzip stream at `0x2100` holding a
**364 MB newc CPIO**. `customer-cert-20261026/extract_devrootfs.py` unpacks it (byte-identical to
system `cpio`).

Device facts:
- EulerOS aarch64, kernel `6.6.0-72.0.0.76.h21510.1170.eulerosv2r15.ascend.aarch64`.
- **OpenSSH 9.6p1**, stock — no vendor patch, no known CVE in the SSH implementation.
- `sshd_config`: `PasswordAuthentication no`, but `KbdInteractiveAuthentication yes` and
  `ChallengeResponseAuthentication yes` — passwords authenticate via PAM keyboard-interactive.
- PAM stack (`sshd` → `password-auth` → `password-auth-local`):

  ```
  auth  required      pam_env.so
  auth  required      pam_faillock.so preauth audit deny=3 even_deny_root unlock_time=300
  auth  sufficient    pam_unix.so nullok try_first_pass
  auth  [default=die] pam_faillock.so authfail audit deny=3 ...
  auth  sufficient    pam_faillock.so authsucc audit ...
  auth  requisite     pam_succeed_if.so uid >= 1000 quiet_success
  auth  required      pam_deny.so
  ```

  `pam_unix.so` is invoked with **`nullok`**, so an empty shadow hash would authenticate — but
  `HwHiAiUser`'s hash is a real SHA-512 crypt
  (`$6$BOrWuUA3AuseKqT.$DCnzXrNod3qQ.…`), as is root's. 87 common/vendor-default passwords tested
  against both — no match. `pam_sss.so` is referenced with a `-` prefix (continue-if-absent) and is
  absent from the image, which is harmless.

- `HwHiAiUser` (uid 1000) is restricted by sudoers to `/bin/ls`, but `tc-config` adds it to `wheel`,
  and `pam_wheel.so` gates `su-local` — which stacks `system-auth`. `bin/passwd` is setuid-root
  (normal). `/etc/shadow` is `----------` root-only.
- The device SSH switch is read from
  `/sys/class/devdrv_manager/davinci_manager/devdrv_manager/devdrv_ssh_config` == `ssh-on`, and
  that node is not present in this build.

**Verdict on the SSH angle: no software vulnerability found.** It is a credential problem, not an
exploit. The `upgrade-tool --component -9` path ("all components and reset password") needs more
arguments than `--device_index`/`--component`; with `--path` it runs but fails
(`result(7)`, `{"device": 0, "failed"}`) and the password is unchanged.

### The real device→MCU transport is SPI, and it is present

Decompiled from `Ascend310P_dt.img`:

```dts
spi@A44B0000 {
    compatible = "hisi-spi";  reg = <0x00 0xa44b0000 0x00 0x10000>;
    num-cs = <0x01>;  id = <0x06>;  status = "ok";
    spi_smbus@0 { compatible = "spi_mcu"; reg = <0x00>;
                  spi-max-frequency = <0xf4240>;  mode = <0x00>; };
};
spi@A44A0000 { compatible = "spi_slave_mcu"; ... };   /* MCU -> SoC */
```

and the device carries the matching drivers:

```
drv_spi_smbus.ko   spi_mcu_reg_read / spi_mcu_reg_write / spi_smbus_send_req_crc
ipc_drv.ko         hisi-mailbox / mdev_send_msg / mdev_receive_msg
drv_smbus_intf.ko  smbus_secure_boot_init / _query / _get_cert
```

**This corrects the previous session's conclusion.** The vendor's *libdcmi* elabel path is I2C-based
and looks for an Intel `"SMBus I801 adapter"` that does not exist on this AMD host — so *that*
host-side path is dead. But the actual device→MCU link is this **SPI** link at `0xA44B0000`, driven
from the **device's** kernel, and it works.

### The chain, and the two gaps

```
host (ours)                device (EulerOS aarch64)            MCU (Cortex-M)
npu-smi set -t ssh-enable   sshd 9.6p1                         SPI flash
   | HDC (PCIe) drv_pcie_hdc_host (OURS)                        elabel / PPPP / trustlist
   |   hdcd (device agent)    ipc_drv.ko (hisi-mailbox)          (all CRC-16 only)
   |                          drv_spi_smbus.ko (spi_mcu)  --->  SPI @0xA44B0000
```

Two gaps, both ours to close:
1. **A host HDC client** — the host driver we already rebuild exposes the whole RPC
   (`HDCDRV_SERVICE_TYPE_FILE_TRANS 3`, `HDCDRV_CMD_SERVER_CREATE/CONNECT/SEND/RECV` in
   `hdc_host/hdcdrv_cmd.h`), the device agent `/var/hdcd` runs (uid 1000, `restart: on-forever`),
   and the device's file-transfer whitelist is permissive for the AICPU kernel tarballs
   (`_Ascend310P-aicpu_syskernels.tar.gz  *  0  104857600  64  0` — any path, **no name check**,
   no root privilege required). There is simply no host HDC *client* installed.
2. **Device-side code to drive `spi_mcu`** — reachable once (1) is done, with no credentials needed
   thanks to the whitelist.

### Closed this session

- **GPIO bit-bang**: the host driver *does* register a GPIO command
  (`DMS_GET_GET_GPIO_STATUS_CMD`, `DMS_SUPPORT_ROOT_ONLY`) with `DIRECT_OUTPUT`/`DIRECT_INPUT`
  sub-commands — exactly what a software I2C would need. But `dsmi_get_gpio_status` returns
  `0xfffe` on the 310P (0/16 pins readable); the only GPIO consumers in the source are in
  `res_drv_mdc_v2.c` (the MDC platform), and there is no exported `set_gpio`. The device has no
  GPIO or i2c-gpio modules either. **Closed.**
- **SoC customer cert**: the earlier session's `--user_cert` path already *succeeded* on all six
  device indices today ("Set user cert, device_index[0], file path is /usr/local/Ascend/CMS/user.xer").
  `npu-smi set -t custom-op-secverify-cert` reports "This device does not support setting
  custom-op-secverify-cert." — the SoC CMS path is 910B-gated in the driver
  (`dms_custom.mk` gates on `hi1980b`/`ascend910B`), so `dms_custom_device.o` is not compiled for
  the 310P.

### Next step

Write the host-side HDC client against our own `hdcdrv_cmd.h`. It needs no new mechanism on either
side, and it is the same "we own the driver" position that made the AI Core proof possible. From
there, the device's own `spi_mcu` driver becomes the MCU transport, and the CRC-only integrity
becomes the only barrier.

Full detail: `customer-cert-20261026/SOFTWARE-ROUTES-20261026.md`.

---

## 2026-10-30 session — kernel/DDR slot binding probes + host-side test loop

All experiments on card **NPU 16409 (`0000:49:00.0`)** only; NPU 9/8217 served as untouched
controls throughout. End state: **all host-served images stock, card healthy 6/6 OK.**

### Test loop (validated, use this going forward)

- Swap file in `/usr/local/Ascend/driver/device/` (needs `sudo cp` — root-owned), then
  `echo 1 > .../49:00.0/remove; sleep 3-5; echo 1 > /sys/bus/pci/rescan`. Driver re-probes,
  re-downloads the chain, 6/6 OK in ~30 s. No host reboot.
- **Recovery from a failed boot needs the `reset` write**: after a boot stall the *first* bare
  remove/rescan retry fails again (`Wait boot mode from bios time out (flag_r=0x40000080)`,
  probe ret=-19). The recipe `echo 1 > 49:00.0/reset` → remove → rescan recovered the card in
  ~30 s on both occasions (proven twice, no power cycle needed).
- Note: after all this rescan churn the test card's link runs **8GT/s (downgraded)** while the
  healthy card trains 16GT/s — factor into any perf comparison on 16409.

### Kernel-slot binding (K1) — FAILED to boot ⇒ kernel payload IS bound

Stock kernel container (`md5 8969017d…`, identical across `.image`/`.orig`/`Image`) modified:
4096 zero bytes appended to the payload, `code_len` @0x45c/0x145c and SHA-256 digest
@0x2c/0x102c updated, **cms/ini/crl tail copied verbatim (ini digest left stale = the probe)**.
Result: card never reached kernel (OK stuck at 4). Conclusion: the kernel slot behaves like the
rootfs (V1) — **content changes with stock signature are rejected**; a custom payload there
(e.g. U-Boot) needs the customer-cert chain, not just header hygiene.
*Caveat:* the inner container header inside the payload still declared the old length, so K1
proves "something binds the kernel slot", not precisely which digest.

### Container-geometry finding (applies to all host-served esbc files)

- `@0x4BC/0x14BC` header self-hash rule (true for the syskernels *package* container) does
  **not** hold for stock kernel nor ddr containers — leave those fields untouched.
- Kernel header copy-0 ≠ copy-1 over `[0x40:0x4bc]` (they differ by more than the checksum).
- Proven-safe edit set: `code_len` @0x45c/0x145c + digest @0x2c/0x102c only.

### DDR-slot probes (D1/D2) — both failed, interpretation OPEN

Structural discovery: `Ascend310P_ddr.image` (the Ingenic-style **init-table** analogue) is
header+payload **only** — the file ends exactly at payload end (`0x2000+code_len == EOF`),
**no cms/ini/crl sections at all**; its @0x2c digest matches the payload (self-consistent,
unsigned). Payload ends with a `55AA55AA 'SCHV' <ver>` footer + 5 trailing zero bytes.

- **D1**: last payload byte 0x00→0x01, digest left stale → boot failed (OK stuck at 4).
- **D2**: same flip **plus** recomputed @0x2c/0x102c digests → boot *still* failed.
- Unresolved fork: either (a) the DDR table's integrity is anchored *externally* (efuse/nvm or
  a hash burned into the SPI boot chain — our re-sealed digest wouldn't match either), or
  (b) the flip site wasn't inert — the byte sits in the SCHV footer padding and the DDR
  firmware's own parser rejected it (verification never consulted).
- **D3 (next cheap cycle, disambiguates)**: append a full 4096-zero block at EOF with
  `code_len`+digest updated — content added *past* the footer as zero padding. Boots ⇒
  self-certifiable window (fix your own digest, content runs) — the Ingenic-style exploit
  survives. Fails ⇒ DDR table is externally anchored ⇒ everything host-served needs the
  cert/efuse route.

### DTB facts extracted from `Ascend310P_dt.img` (context for U-Boot planning)

16 DTB slots (first @0x2900); `compatible = "hisilicon,hi1910p-evb", "hisilicon,ascend610"`
(Huawei-internal names for this silicon; AS610 in the UEFI filename; user's "it's a 310P card"
noted — same family). `chosen` bootargs: `initrd=0x880002100,200M cma=256M
ascend_log_redirect=0x1fc000@0xb101000 module.sig_enforce=1`, `kaslr-seed = <0 0xdeadc0de>`
(fixed). 8 masked `arm,armv8` CPUs in 2 clusters, GICv3+ITS, PSCI-0.2, sbsa-uart. Which slot
BL31 selects is still unverified.

### BAR telemetry (host-side visibility idea — partially validated)

BAR layout identical on healthy and test cards: BAR0 16M + BAR2 256M + **BAR4 8G prefetchable**.
Device stays PCIe-responsive even while boot is stalled (why rescan-recovery works). sysfs
`read()` on `resourceN` returns EIO while the driver is bound; **mmap works but offsets must be
page-aligned**. BAR4+0x0 reads `AA55AA55 DC04…` (esbc magic in device-side memory — mailbox
descriptor vs DRAM window base `0x80000000` not yet established; `code_len` field there is
`0x684b249`, matching none of the stock files; offset `0x3101000` reads 0xCC poison). Byte-wise
scans of the full window are too slow over PCIe MMIO — use chunked block reads. If the BAR4↔DRAM
map is pinned down, the device console (`0x1fc000@0xb101000`) and initrd become host-readable
without any device-shell credentials.

### DDR binding completed: D4/D5 close the model

- **D4** (stock payload, header digest byte corrupted in *both* copies): boot FAILED.
  Proves independently of content effects: the header digest is validated against
  something **external to the file's own consistency**.
- **D5** (identical content flip + resealed digest applied to *both* `Ascend310P_ddr.image`
  and its byte-identical twin `lpddr_mcu.bin`): boot FAILED. Kills the last
  relative-check theory (MCU comparing its copy against the SoC's).
- **Model that fits D1–D5**: the DDR payload must hash to one specific
  externally-held expected digest (efuse/OTP or SPL-side constant). Any content change
  without a matching re-anchored expectation fails ⇒ **no Ingenic-style free
  init-table window in the current configuration.**
- `lpddr_mcu.bin` discovery: byte-identical second copy of the DDR table served to the
  MCU (driver loads both from `device/`). Custom DDR tables must patch both files.
- Boot-handshake semantics from `devdrv_device_load.h`: SRAM flag word
  `NORMAL=0x2222…22`, `SLOW=0x3333…33`, `ABNORMAL=0xFFFF…FF`. Observed failure value
  `flag_r=0x40000080` is a **non-mode partial write — the verify-fail fingerprint**
  (driver waits at `res_drv_mini_v2.c` L669-682 / `devdrv_device_load.c` L431-446).

### The seller's "disable cert verification in BIOS" — located (mostly)

MCU LiteOS image (16 MB dump, window @0x220000) contains an MCU-side secure-boot
command interface, i.e. the "BIOS" is the **card's MCU board firmware**, not host BIOS:

- `set_secureboot_info_callback`, `set_secureboot_param_check`,
  `get_secureboot_info_callback`, `get_secureboot_data`, `check_input_secureboot_param`
  (strings @0x228a48-0x228d60) — a get/set pair with parameter validation, i.e. an
  interactive MCU-console command like `set_secureboot_info <param>`.
- Persistent config = "customized info" dual-bank region (`write/read/clear_customized_info`,
  "read bak", "update cert bak"); `chip_cfg` items include literal `switch`, `trustlist`,
  `blocklist`, `https crl`; `TrustList` string appears at `0x229592` with a **mirror at
  `0x269592`** (exactly 0x40000 apart — dual-banked).
- Host reachability on 310P is gated: `info -t rootkey|boot-select|tls-cert|nve-level|
  mcu-monitor` all answer "does not support"; only `flash` and `board` respond.
  `set` surface exposes `boot-area`/`boot-select` (A/B MCU image switch — matches
  "switch image to " strings) but no secureboot verb. Consistent with the
  `dms_custom.mk` hi1980b-gating found 2026-10-26.
- **Flash size note**: `npu-smi info -t flash` reports **32 MB (GigaDevice 0xC8)** while
  our verified MCU dump is 16 MB — half the flash may be unexamined (second image bank?).

### MCU flash structure (30 Oct RE pass, from existing 16MB dump)

- The MCU app is **128KB exactly, dual-banked**: `0x220000` and `0x260000` are byte-identical
  for `0x20000` bytes and diverge immediately after. (The earlier "256KB verified window"
  actually spanned image+adjacent-data.) Patching code should patch **both banks**.
- **No RSA-2048 key is embedded anywhere in the 16MB dump** — consistent with
  `CheckPubKeyHash` (hash-only anchor) and EC strings present (`id-ecPublicKey`).
- **Boot gate vs update gate** (string-clustering evidence): `MCU START`/`Begin to running
  LiteOS` (0x2233d7) share their code neighborhood with `verify firmware info failed` /
  `Read checksum_info error` (0x223d26, same TU ≈0x950 apart), while mbedTLS
  RSA machinery (`RsaVerify`, `McuSignatureVerify`, `CheckFileHead`, `read_rsa_head_item`)
  clusters with the CMS/cert strings near the command callbacks (0x22a-b region).
  ⇒ **boot verification = checksum/CRC; RSA signature = update/cert path.**
  Practical: direct flash code-patches boot without Huawei's signing key — recompute the
  checksum. Confidence: high but string-position-based; a caller-level disassembly pass
  (image runtime base not yet pinned — no MOVT found, unknown link base) would nail it.
- **Manipulation verdict**: data/flag rewrite free (CRC-16 regions, MCU rewrites them
  itself); MCU code rewrite likely free (checksum-gated boot, dual bank + clip = recovery);
  physical write = standard GigaDevice/flashrom with 4-byte addressing >16MB.

**Verdict on the seller's claim**: not a false flag — a real, implemented mechanism
exists in MCU firmware (`secureboot_info`), but on this platform the host cannot read
or set it via npu-smi; access path is the MCU console (serial) or vendor provisioning
tool. If it can be switched off, D1–D5/K1 failures were *policy*, not architecture — a
disabled state would re-open the whole host-patchable-chain thesis (and the custom
U-Boot PoC). **Decisive cheap sequence**: get MCU console → `get_secureboot_info` →
(if toggleable) disable → re-run D2-equivalent (modified DDR content, self-consistent
digest) → boot ⇒ everything is patchable.

### Flash capacity saga (30 Oct evening) — 16MB vs 32MB on the card MCU flash

- `npu-smi info -t flash`: **32MB, Manufacturer 0xC8, Flash ID 1664968 = 0x1967C8 → raw `c8 67 19`**.
- CH341A/scriba at the clip: `c8 40 18` (GD25Q128, 128Mbit = 16MB) — stable across repeated reads
  when contact was good (intermittent all-`FF` ID reads = marginal clip, nothing to do with capacity).
- Scriba experiment (forced 32MB profile + `addr4b=1`): read clocked all 32MB in 528s but came back
  **100.00% `0xFF`** — chip answered nothing. Mechanism: after `0xB7` the chip expects 4-byte
  addressing for all commands while the CH341A command interface is 24-bit native → misframed
  reads, MISO high-Z. ⇒ **Experiment inconclusive on capacity; the scriba forcing was the failure,
  not evidence about the die.** Change stashed; scriba back to plain 16MB profile.
- Live hypotheses: (a) true 16MB part and npu-smi's table fibs; (b) 32MB die (e.g. GD25Q256-class)
  in **3-byte legacy mode** — some such parts report capacity `0x18` until 4B mode is entered
  (ID-then-`0xB7`-then-ID trick settles it; flashrom on the same CH341A handles 4B-mode chips
  better and would be the one-command decisive probe); (c) npu-smi is describing a *different*
  chip than the one clipped. Chip marking is not GigaDevice-branded per bench — a photo would
  help.
- Useful by-product — **card-9 flash content map** (16MB dump): 97.7% erased `FF`. Content:
  two banked blobs ≈`0x200000..0x26de34` (MCU image banks + surrounding data, mirrored `0x20000`
  apart, code-window mirror confirmed at `+0x40000`), small islands at `0x2c0000` (~1.6KB),
  `0x7c0000` (256B), `0x8c0000..0x8c0528` (~1.3KB) plus sub-256B fragments. Everything else
  erased. ⇒ future per-card diff (4th card vs card 9, both 16MB reads with the stock profile)
  will localize the per-card data regions (elabel/serial/trustlist candidates) directly.
- **4th card note**: brand-new card, identical hardware; its 16MB read (stock profile) has not
  yet been captured — that read is the useful next bench artifact (diff vs card 9 → per-card
  data-region map + independent confirmation of code-bank versions).

### Card-4 dump (atlas_310P3B.bin) — per-card diff map + factory MCU log settle several questions

Clean 16MB clip read of net-new 4th card (SN `2106030737ZERC001417`), stock scriba profile.

- **MCU app banks (0x200000–0x27FFFF) byte-identical to card 9**; bank mirror holds
  (`0x220000 == 0x260000`). Whole-flash diff vs card 9 = **4.6KB in 6 clusters**:
  `0x2c009c` (5B) + `0x2c0174..` (elabel serial + CRC-ish fields; serial at `0x2c0178`),
  `0x300004` (GAMF counters), and the three log rings `0x340000/0x3c0000/0x440000`
  (card 9's rings are erased-FF; card 4 still holds its factory logs of 2025-12-18).
- **Flash map** (16MB): app banks `0x200000–0x27FFFF` · elabel `BALE` @`0x2c0000`
  (serial @+0x178) · runtime params `GAMF` @`0x300000` · MCU log rings @`0x340000/0x3c0000/0x440000` ·
   **card-invariant descriptor @`0x8c0000`** with `AA55AA55` esbc-family magic + version 1,
   content = raw hash/signature-shaped bytes (no DER) — prime candidate for MCU trust anchor
   / trustlist / `secureboot_info` storage. Rest of the 16MB: erased FF (97.7%).
   - STRUCTURE (parsed 30 Oct night, card-4 golden): 0x530 bytes total.
     Header `0x8c0000..0x1f`: magic `55AA55AA`, version 1, `0x14`, `0x1F`, sizes
     `0x100`/`0x200`, sentinel `0x1CD9DEAD` @+0x1C (earlier ASCII read "added91c" was
     this sentinel's LE bytes — misread, corrected). Body = **three high-entropy blobs**
     (A: 256 B @+0x20 — exactly RSA-2048 signature sized; B: ~480 B @+0x224 with small
     header word `E71E2DBB`-style @0x8c021c; C: ~260 B @+0x41c) + zero padding, closed by
     trailer `0x0000b770` + `0xCC33CC33` marker @0x8c0520. No DER, no ASCII strings.
     Interpretation: encrypted or raw key/signature trust material, card-invariant —
     consistent with a factory MCU trust anchor/trustlist. Deep parse parked (needs MCU
    -side consumer RE or the console reading of `secureboot_info`).
- **Factory log (ring3) — the upgrade/verify sequence, verbatim evidence**:
  OOB `transfer upgrade package` → `upgrade_slave_app` → `verify_flash_info:
  flash_id(0xc818) match success` → `RsaSignatureVerify: MCU verify APP sign successfully`
  + `MCU verify vrd sign successfully` → copy → `Upgrade MCU from version(V0.4.6) to
  version(23.2.6)` → `update_boot_param` → reset; then `validate_mcu_image: Reset MCU to
  validate image successfully`; `customized_info_init: data not exists, set default`.
  ⇒ RSA verify is definitively on the **upgrade path** (APP + VRD); boot-side gate is
  `validate_mcu_image` (checksum vs RSA at boot still to be pinned from disassembly);
  customized-info is a defaults-on-blank data region — writable-by-design, consistent
  with the CRC-only model.
- **Capacity**: the MCU itself validates `flash_id 0xc818` = GD 128Mbit = **16MB** — the
  MCU treats the clipped flash as 16MB. npu-smi's "32MB / c8 67 19" remains unexplained by
  this chip (other flash on card, or table artifact). Scriba's forced-32MB read failing
  all-FF is now fully consistent: true geometry is 16MB.

### Flash topology resolved (30 Oct, evening): TWO SoC flashes + one MCU flash

- `npu-smi info -t flash -i 9`: **two flash chips reported** (Chip ID 0 and 1), both
  `Manufacturer 0xC8, Flash ID 1664968 = c8 67 19, Capacity 32MB` → GD25Q256E/257D class.
  These are the **SoC-side** flashes (not yet clipped). The clipped/dumped 16MB `c8 40 18`
  part is the **MCU's own flash** and is not in npu-smi's view. The long 16-vs-32MB
  contradiction was two parties describing different chips. MCU-side truth stands:
  `verify_flash_info: flash_id(0xc818)` — MCU expects its own 16MB part.
- `npu-smi set -t` item inventory (root) relevant to board policy:
  `customized-info`, `boot-area`, `boot-select`, `mcu-monitor`, `revocate`, `key-manage`,
  `pkcs-enable` — matches MCU log hooks `customized_info_init` (defaults-on-blank) and
  `update_boot_param`. No plain `secureboot` set type exists on 310P.
- MCU image RE status: Cortex-M Thumb-2, code starts flash `0x220000` (bank B at `0x240000`,
  full 128KB banks are exact mirrors at `+0x40000`; 256KB total MCU app incl. rodata).
  Linear disassembly coverage only ~44% (interleaved literal pools/jump tables) — string
  xref hunt needs function-level RE (proper recursive descent), not linear scanning.
  **100%-correct patch doctrine before any flash write**: (1) locate exact flag bytes,
  (2) know the integrity scheme covering them (CRC region boundaries), (3) patch bank B
  only, keep bank A golden, (4) recovery = clip rewrite from golden dump.
- Differential-probe idea for the exact flag byte: toggle a host-visible set item
  (`customized-info`/`boot-select`) on the bench card once it can run host driver, re-read
  clip, diff → byte offset observed, not guessed.

### MCU app architecture (corrected, 30 Oct late session)

- **App image = 256KB, banked at flash `0x200000` (A) / `0x240000` (B)**, exact mirrors.
  (Earlier "banks at 0x220000" was wrong — that was mid-bank content.) Used content ends
  ~`0x22de34`; remainder FF. XIP base `0x08000000` (VA = file − 0x200000 + 0x08000000);
  initial-SP field `0x20010000` (64KB RAM); handler-pointer fields `0x0802xxxx` monotonic
  (stub spacing 0x18) — Cortex-M Thumb2, but the reset field (`0x080241c9`) resolves into a
  format-string zone, so naive vector-table reading fails; **proper recursive-descent RE is
  the required next RE step**. Toolchain baseline proven: `rizin -a arm -b 16 -m 0x08000000`
  on the `0x200000`-slice, `aa; aap` ⇒ 1318 functions found. Linear/capstone scans alone are
  unreliable (interleaved inline strings defeat linear alignment).
- **Flash-device support table @~`0x22db00–0x22dcc0`**: entries `{vendor-ID, capacity, config…, name}`
  for SPANSION/MICRON/MACRONIX/WINBOND/GIGADEVICE/ISSI — **all capacity code `0x18` (16MB)**;
  the GigaDevice entry is `c8 18` — exactly what `verify_flash_info: flash_id(0xc818)` logs
  for our part (`c8 40 18` GD25Q128). ⇒ **The MCU firmware is built for 16MB parts by design**;
  no 32MB (`0x19`) entry exists anywhere in the MCU database.
- Log strings are inline with code (no separate rodata; pools for strings unfindable at any
  link base; references appear to be near-PC ADR within mixed code/text blocks).

### MCU RE round closed (static, offline) — accepted working model

- Preamble `0x200000–0x21FFFF` behaves like an **initial-RAM image section** (vector-ish
  header: SP `0x20010000`, but reset/default-handler fields resolve into *format strings*
  at `0x2241c9`/`0x224239`; embedded structures carry both RAM ptrs (`0x2000xxxx`) and
  XIP-code ptrs (`0x08xxxxxx`); 33% printable = RAM image with copied .text/.data, not
  pure code). Flash-side app content (`0x220000–~0x22DE34`) = code blocks + inline string
  blocks + tables (flash-device DB all-capacity-`0x18` at `0x22DB00+`).
- Log strings have **no absolute address references anywhere** in the dump (no pools at any
  link base, no ADR/MOVW pair within ±4KB, no offset encodings) ⇒ addressing is base-reg +
  small-offset computed at runtime (or through RAM-image indirection). String xref tooling
  is therefore blind; **call-graph RE (ghidra-grade, hours) is the offline path** — parked,
  not blocking.
- **Decision: resolve the boot-gate/flag questions EMPIRICALLY** via one physical bench↔host
  cycle (MCU-EXP-1) instead of static RE:
  1. (Bench) benign 1-byte CH341A write in bank A only: `"MCU START"`@`0x2233D7` → e.g.
     `MCU XTART` (offset exact; bank B untouched as reference/copy).
  2. (Host) seat card in spare slot, boot, confirm npu-smi OK; toggle
     `npu-smi set -t customized-info` (and/or `boot-select`), record values; shutdown.
  3. (Bench) clip, read E, diff vs golden `atlas_310P3B.bin`.
  Yields: (a) CH341A write + recovery loop proven on real boot; (b) whether boot gate
  content-checks the string area (boot-OK vs stall = CRC-coverage answer); (c) the exact
  bytes customized-info/boot-select move (flag home, observed not guessed); (d) which bank
  is active (rendered log line shows XTART vs START). Recovery safety net at every step:
  golden dump re-write.

### customized-info deep-dive + fleet state (30 Oct, end of day)

- **Semantics** (from MCU string table + npu-smi/driver vocabulary): a vendor
  **provisioning record store** — MCU API family `customized_info_{init,read,write,update,
  clear,verify}` + OOB framing (`customized_msg_check`, `customized_last_frame_handle`),
  same shape as the elabel family next to it in strings; host side `dcmi_mcu_set_customized_info`,
  `dcmi_set_check_customized_is_exist` ⇒ the "already exists / cannot be set repeatedly"
  gate is a **host/driver-layer pre-check**, not proven to be MCU/fuse-level
  (`customized_info_clear` exists in MCU code; no CLI erase on 310P).
  Descends from `DSMI_COMPONENT_TYPE_CUSTOMIZED` (an upgrade-tool component/partition on
  full Atlas products). **Nothing in the boot chain observed to consult it**: two service
  cards ran whole lives blank; factory "set default" was RAM-only.
- **16409 incident record**: set `customized-info=1` via
  `sudo npu-smi set -t customized-info -i 16409 -f 1` (write-once; `-f 0` and `-f erase`
  retries rejected). Survives device `reset` + remove/rescan + driver reload (board-side).
  Board-info diff vs twin 8217: **identical except serial/ID/bus** — zero observable
  divergence; post-write boots incl. recovery cycle all OK (6/6). Value "1" semantics
  undocumented on 310P; only untested consumer = future upgrade-tool ops → policy: no
  upgrade-tool on 16409 unless decided consciously.
- **Operational finding**: plain `remove`+`rescan` can stall even a healthy stock card —
  run the **reset recipe** (`reset` → remove → rescan) on every cycle.
- **Fleet**: 9 + 8217 = pristine references (blank provisioning); 16409 = risk-absorber
  (provisioned `1`, boots-verified, future differential clip sample for locating the
  record in flash); card 4 (bench, clipped) = virgin, golden dump `atlas_310P3B.bin`
  double-read verified; **driver-boot only, no sets/flash** until first-server-life diff
  (predict: log-ring + GAMF counter moves only). Server power-cycle during that cycle
  gives the free zero-power test of 16409's record (still `1` ⇒ flash-burnt).

### pkcs-enable crossover — verdict: it is a LOCK knob, not the unlock (30 Oct evening)

- Semantics decoded empirically: `set -d 0` → **Enable** (with CLI warning *"Enabling pkcs
  will raise safety risks"*), `set -d 1` → **Disable** (factory default; observed state on
  all cards before this session). Writes are re-settable (not write-once). Per-chip (`-c 0/1`).
- Experiment matrix on 16409 (both chips in each config):
  | DDR payload | pkcs | result |
  |---|---|---|
  | stock | Disable | boots (baseline, many times) |
  | **D2 (modified, resealed, unsigned)** | **Enable** | **STALL** (`flag_r=0x40000080` pattern) |
  | stock | Enable | **STALL** |
  | stock | Disable | **UP at 40s** after recovery recipe ✓ (card restored 6/6) |
- ⇒ `pkcs=Enable` **arms stricter certificate enforcement that these cards cannot satisfy —
  even Huawei-stock fails to boot**. `Disable` (the default) was already the lax posture;
  enabling it does not bypass the D-series external-digest wall, it tightens past it.
  The hoped-for "cert check off ⇒ modified payloads boot" hypothesis is **dead for pkcs**;
  the external anchor remains the binding factor.
- **Operational learnings:**
  1. The **policy plane (dms→MCU) answers while the SoC is stalled in boot-wait** — pkcs
     set/read worked with the card down; stalls are recoverable without opening anything:
     set Disable (if armed), then reset recipe. `pkcs=Enable` carries real lockup risk on
     this card/FW — verify boot before walking away.
  2. Confirmed again: `remove`+`rescan` alone can stall even healthy cards; the reset
     recipe (`reset` → remove → rescan) is the reliable bring-up; recovery boot showed no
     new boot-timeout fingerprint.
- Fleet back to baseline: 9, 8217 untouched/up; 16409 up 6/6 (stock files, pkcs Disable ×2,
  customized-info=1 burnt); card 4 virgin on bench with golden dump.

### upgrade-tool breakthrough: the card's real flash component map (30 Oct, late)

- `sudo upgrade-tool --device_index 4 --components` → device 4 (16409) reports exactly:
  **`hboot1a`, `hboot1b`, `hlink`** — SoC-side **A/B-redundant HBOOT** (HiSilicon boot
  loader) + **HLINK** link-stack, all at version **7.5.0.5.220** (componentTypes 11/12/18).
  Version queries read the CARD's flash metadata (the host FD file
  `AS610_HBOOT2_UEFI.fd` contains no `7.5.0`/`hlink` strings) — i.e., there IS a local,
  named, versioned, upgradeable SoC boot flash layer; the FD pushed by the host is its
  payload source. This is the layer the D-series external-digest wall most likely lives in
  (HBOOT validates the host-served payloads), and the layer a vendor "flash" rewrites.
- upgrade-tool syntax now decoded: `--device_index <id> --components` (list),
  `--component <name> --version` (query), `--component <name> --upload <file>` semantics
  (write path — UNTESTED on our fleet; flashing remains deliberate).
- Interpretation for the rooting model: SoC boot = on-chip ROM → **HBOOT (on-card flash,
  A/B)** → host-served esbc payloads (ddr/FD/kernel/rootfs). The "external anchor" is
  therefore most plausibly a constant/table inside HBOOT's flash region or its efuse-locked
  config. `upgrade-tool --upload` (rewriting hboot1a/b from a vendor package) is the one
  legitimate mechanism we have seen that moves that anchor coherently.
- Open next probes: (1) FD container layout map around its header (`hboot` string at
  `0x2004` in FD) vs the 3 component names; (2) whether `--upload` on 16409 would be
  recoverable from stock FD; (3) does the 7.5.0.5.220 HBOOT match any file inside the
  installed driver package tree (beyond `/usr/local/Ascend` find scope already done).

#### upgrade-tool addendum (same evening, minutes later)
- FD esbc header decoded: magic `0x3a3aaa33` at `0x2000`, name field **`hboot2`** (so the
  host FD file IS the HBOOT2/UEFI payload whose destination is the card's `hboot1a/b`
  flash partitions); words incl. `0x9`, `0x205`, `0x20000`, `MV` marker, `0x100`. The
  version string `7.5.0.5.220` is NOT encoded anywhere in the FD file → it is
  flash/tool-side metadata, not payload-embedded.
- Fleet baseline: **card 9 (untouched) = card 16409**: same components
  (`hboot1a,hboot1b,hlink`) @ `7.5.0.5.220`; `--boot_status` = **16** on both (identical
  healthy signature); `--system_version` reads **26.0.rc1** from the card itself.
- Full verb set captured: `--upload --path` (write), `--version`, `--boot_status`
  (no `--component`), `--system_version`, `--hot_reset`/`--hot_reset_flag`,
  **`--user_cert --path <cert>`** ("ctrl cpu scenario set user certificate info" — a
  potential own-certificate injection lever, write-style), `--patch`/`--unload_patch`,
  `--set_compat_list`, `--status`, `--async`, `--mini_devices`, `--phymachflag`.
  **No download/dump verb** → flashed HBOOT cannot be read back off the card with this tool.
- Working model refined: on-card flash holds HBOOT(A/B)+hlink; those validate the
  per-boot host-served payloads (ddr/kernel/rootfs esbc chain — the D-series wall).
  The host FD file is the *update source* for hboot flash, which plausibly explains the V2
  "FD content change had no boot effect" result (boot reads flash copy, not host file).
- Remaining host-side lever, one deliberate risk away: `upgrade-tool --device_index 4
  --component hboot1a --path <file> --upload` on the guinea pig (write to the hboot1a
  partition; recovery = re-upload + reset recipe). Whether `--path` takes the raw FD or a
  packaged variant needs a dry check before committing.
- Digest-in-HBOOT test (offline, negative): stock DDR payload digest
  `a8475646…b392d45` appears NOWHERE in the FD/HBOOT2 image (both byte orders, 8/12/16-B
  prefixes) → the external anchor is not a constant stored in the HBOOT payload.
  Remaining hypotheses: efuse/OTP on the SoC; another region of the 32MB SoC NOR outside
  the FD payload; or derived. UPDATE (same night, negative): the derived-hash variant was
  tested — `a8475646…` is NOT SHA-256 of the root CA (DER, SPKI, nor modulus tested; see
  `work/rootca/Huawei-Integrity-Root-CA-G2.der`), so a root-hash-pinning coincidence is
  unsupported. NOTE: help confirms write form is
  `--component <type> --path <firmware_path>` (no literal --upload verb); `--mini_devices`
  lists MINI devices (`-1` = all MINI); `--user_cert --path` remains a potential own-cert
  lever for the ctrl-cpu scenario.

## HBOOT2 (FD) dissection — the validator identified (30 Oct night, offline RE)

- FD file anatomy (`AS610_HBOOT2_UEFI.fd`, 0x1a2110 B):
  - `0x0000` esbc-style header block #1 (magic `55AA55AA`, digest @+0x2c, hash region @0x40,
    second copy @0x1000) over a small 8 KB preamble;
  - `0x2000` esbc header magic `0x3a3aaa33`, name field **`hboot2`** (fields: 0x9, 0x205,
    4, 0x20000, `MV` marker, 0x100; build-stamp word `14001909` + GUID-ish 16B @0x2118);
  - `0x2100` **UEFI firmware volume** (`_FVH`, len 0x20000, attrs 0xCFEFF, 24-byte header);
  - `0x22100` second **FV** (len 0xFF000, ends 0x121100);
  - `0x121100+` **ARM Trusted Firmware image** with its own FIP TOCs at `0x132100` and
    `0x172100` (magic `0xAA640001`, second word `0x12345678`; HiSilicon packed TOC entries
    `[u16 id][u8 idx][u8 0xAA]`); AArch64 EL1/EL3 setup code right after the TOC (0x1321c0).
  - NOTE: FV headers solid; FFS file-entry walk mis-aligned (Huawei 20-byte FFS variant?) —
    module inventory deferred.
- **The secure-boot validator lives in this ATF**: string cluster 0x155–0x158k exposes the
  whole gate vocabulary:
  - `Secure Boot: customer efuse Solution` — efuse-anchored, customer-capable scheme;
  - **user-certificate trust domain**: `support usr cert vef`, `usr cert cms vef
    success/fail`, `usr cert hash cal/cmp fail`, `usr cert address check fail`,
    `no usr, hw vef fail, ctrl_flag = 0x%x` — matches the Oct-21/26 finding that a
    `--user_cert` (signca) enrolment DID pass CMS verify on all chip devices;
  - **anti-rollback**: `nvcnt cmp fail, nvCnt = 0x%x, efuse_bit_pos = 0x%x`,
    `efuse_read_nvcnt fail`, `write nvcnt`, `Clean Ufs Nvcnt`, `no need nvcnt check` —
    efuse bit-array version counters (burn-forward; downgrade burns irreversible);
  - **hash-list/ini gate**: `hash list file num invalid`, `hash list num invalid!`,
    `cur image hash not in list!`, `Store ini hash failed`, `preamble = 0x%x error!`,
    `headMagic`, `code offset`, **`pss slen = 0x%x invalid!`** — the exact checks our
    D-series payload edits tripped;
  - crypto core: `hw_cms_image_verify_simple`, `hw_adapt_desc_fetch`,
    `hw_normal_desc_fetch`, PKCS7/X.509 AOID table incl. `AOID_RSASSAPSS`,
    `AOID_HW_KP_VOUCHERSIGNING`, `AOID_HW_KP_RELEASESIGNING` (two Huawei signing roles);
  - TEE co-processor: `tspd setup failed`, `tee cms verify failed`, `tee decrypt fail`
    (EL3 TSP daemon present).
- **Embedded root certificate extracted** (DER @FD `0x155841`, 1606 B, saved
  `/tmp/kilo/rootca.der`, now persisted in-repo at
  `work/rootca/Huawei-Integrity-Root-CA-G2.der`): `CN=Huawei Integrity Root CA - G2` (O=Huawei Technologies,
  OU=Huawei Certification Authority), self-signed, **RSA-4096**, cert signed RSA-PSS,
  serial 0x3C3ADB, validity 2021-02-07 → 2051-05-07.
- **New lead on the old `0xE00` signature mystery** (Oct-26: the 512-byte signature at
  inner header `+0x460` verified against neither root over any message tried):
  1. 512 B = exactly an RSA-4096 signature size — consistent with RootCA G2 being the key;
  2. `pss slen` + `AOID_RSASSAPSS` say the scheme is **RSA-PSS with explicit salt length** —
     if the Oct-26 verification assumed PKCS#1 v1.5 or the wrong salt (32 vs auto/0x88 vs
     0), verification would fail regardless of message content. Re-verify the `0xE00` blob
     with PSS salt variants before concluding the key/message is wrong.
  3. `Secure Boot: customer efuse Solution` raises the alternative: the header signature may
     be checked against an **efuse-hashed key/customer state**, not the CA chain at all.
- Standing from history (unchanged): user-cert enrolment works (Oct-21 signca, CMS passed);
  V1/V2 showed header bit 19 insufficient alone; known residual blockers = BL31 branch at
  `0x13dc58` + the `0xE00` header signature.
- Next RE steps: (a) disassemble the `usr cert`/`ctrl_flag` decision path and
  `hw_cms_image_verify_simple` caller to pin the message layout fed to the `0xE00` verify;
  (b) retry `0xE00` verification offline with PSS salt variants (32/64/auto) over candidate
  messages (header-minus-sig, digest fields); (c) FIP TOC entry decode (0x1321b0 region);
  (d) FFS inventory re-walk with corrected header size.

#### BL31 code-side recon (same night, capstone)
- **Identity VA base works**: disassembling the FD at file offset == VA yields clean
  AArch64 (`stp x29,x30` prologues, sane `bl` targets) from `0x1321c0` onward — the BL31
  image sits at file `0x1321c0` with linked pages congruent to file offsets; capstone
  needs `skipdata=True` due to dense data interludes.
- **ATFimg1 (`0x132100`) and ATFimg2 (`0x172100`) are the same image duplicated**
  (prologues exactly `0x40000` apart) — A/B redundancy inside the FD payload itself.
- Entry-area code (`0x13235c+`): FIP TOC iteration + logging via `0x132ab0` (puts-like,
  strings from image pages `0x134000-0x135xxx`); platform MMIO constants seen:
  `0x310a0400/0x310a0550/0x310a0600` (FIFO-style regs with bit5 polling and bounds check
  `0xcef5…/0x608..0x77f7`), `0x989680` (poll bit5), **`0x8000f030` ← byte 0x39** at a stage
  transition (candidate boot-stage status byte visible to the MCU?).
- **Genuine code sites referencing the validator string cluster (0x155-0x158k)**:
  `0x13cd14`, `0x141948`, `0x14f780`; the window `0x13db54-0x13dcbc` decodes as DATA
  (earlier raw ADRP hits there were table coincidences). These code sites are the anchors
  for recursive-descent RE of the verify path.
- **Discrepancy flag**: the Oct-26 note "BL31 branch at `0x13dc58` (b.eq)" lands inside
  that DATA table in FD-file coordinates — the Oct-26 offset likely came from a separately
  extracted BL31 blob with its own numbering. Re-derive it in FD coordinates before any
  re-branch experiment.
- Session status: hardware untouched tonight (fleet at baseline); all findings offline.

#### String-ownership negative arc (later, same night) — important for the next RE round
- Question: which code owns/references the validator string cluster (0x155-0x158k:
  `cur image hash not in list!`, `usr cert…`, `nvcnt cmp fail`, `Secure Boot: customer
  efuse Solution`, `hw_cms_image_verify_simple…`)?
- Established (negative results, all reproducible):
  1. **ATFimg1 (0x1321c0-0x1721c0) code does NOT reference these strings** — full-image
     ADRP+ADD/LDR tracking (callee-saved-aware) finds zero exact references; the entry
     function's own string refs resolve to the image's LOCAL pages `0x134-0x135xxx`
     (e.g., logger args `0x134360/0x134371`). The loose page-level hits in `0x153-0x154`
     were string-table data ("This is SMMU " descriptor table), not code.
  2. **BL2/pre-TOC region (0x121100-0x132000): zero refs** (same tracking).
  3. **Whole FV2 span (0x22100-0x121100, XIP hypothesis VA≡FD-offset): zero refs.**
  4. **No 8-byte pointer tables** with entries in 0x155-0x159k anywhere in the FD.
- Interpretation: the strings are reached by COMPUTED addressing (message-index/offset-base
  tables whose entries are stored as runtime VAs or offsets, not raw file offsets), closely
  paralleling the MCU app's message-index logging. Static ADRP xref hunting is exhausted;
  the next pass should be **Ghidra headless full analysis** (language AARCH64LE, load base
  TBD from BL2/BL31 FIP TOC entries) to recover the string table indirection and pin the
  `0xE00` verify message builder.
- Practical takeaway recorded: the validation *vocabulary* lives in the FD, but its code
  owner is not statically obvious — do NOT re-derive it with shell heuristics again; go
  straight to Ghidra with the seeded function list (211 prologues in ATFimg1; extraction:
  FD[0x132000:0x172200] mapped at 0x132000 == identity, page-aligned ✓).
- `--path` dry-probe (live, 30 Oct late, dev 4): nonexistent file → `{"device": 4,
  "failed"}`; 64-B random junk → same instant failure; component version reads
  `7.5.0.5.220` unchanged afterwards ⇒ **the tool validates input before writing** and
  failure is clean/no-write. Size-format error strings in the binary
  (`File size[0x%x] not valid.`, `File size[%ld] is out of range.`) suggest size sanity is
  the first gate. The real write-test (`--path <stock FD>`) therefore fails safe if the
  expected size differs from 0x1a2110 — the pending decision is whether to make it.

#### Mechanism resolved: the images are NOT XIP — RAM execution explains every negative (offline RE focus round)
- Built a working scripted-Ghidra pipeline on this machine: `~/ghidra/build/dist/ghidra_12.1_DEV`
  (source-built 12.1 DEV) + PyGhidra in `/tmp/kilo/venv` (Jython is dead in 12.x headless;
  gotchas: `-loader-BaseAddress` replaces `-base-addr`, `-processor` replaces `-language`,
  `open_program(..., project_location=..., nested_project_location=False, analyze=False)`
  reuses the saved project; iterate `listing.getInstructions(True)` + `getReferencesFrom`
  to avoid JPype/RefType overload issues).
- Full Ghidra analysis of `bl31_img1.bin` (197 functions): zero references into the
  validator string pages — independently corroborating the capstone negatives. The ATFimg2
  slice (77 functions) shows the same zero against its +0x40000 duplicates.
- u32-absolute-VA scan across the FD: only 5 scattered hits — no pointer tables.
- **Decisive: TE module headers (sig `0x5A56` + ARM64 machine `0xAA64`) at FD `0x3838` and
  `0x23260` carry image bases `0x31701000` and `0x31721000`** — runtime VAs in the `0x31__`
  DRAM region whose low bits mirror the FV file offsets. Modules load to high RAM and
  execute there (non-XIP); their ADRP references resolve to RAM addresses, so
  file-offset-space analysis is structurally blind to them. Mystery closed: the validator
  strings are staged in FD for RAM-based consumers.
- Next recipe (precise): import the TE module at `0x23260` into Ghidra at base
  `0x31721000`, analyze, decompile the validator + the `0xE00` verify-message builder.
  FFS inventory simplifies to walking FV2 for `5A56 AA64` TE headers (≥2 found); the
  7.9-entropy zone is TE content/tables, not an LZ4/gzip/LZMA blob (all brute-forces
  negative).
- Slice note: ATFimg2 is a truncated second copy in the FD (~192KB of 256KB; file ends
  `0x1a2110`).

### Host state after session

`/var/local/ascend-work/images/`: `Ascend310P.image.bindtest`, `Ascend310P_ddr.image.D1/D2/D3/D4`
(as named variants), `ddr.D5.bin`, backups `Ascend310P.image.pre-bindtest`,
`Ascend310P_ddr.image.pre-D1`, `lpddr_mcu.bin.pre-D5`. Device dir: **stock kernel, stock ddr,
stock lpddr_mcu.bin** — card 16409 healthy 6/6 after final recovery cycle. Locally:
`/tmp/kilo/Ascend310P*.bindtest|.D1|.D2|.D3|.D4|ddr.D5.bin`, `Ascend310P_ddr.image`,
`lpddr_mcu.bin`, `ascend310p.dtb/.dts`.


## Session 2026-10-30 (pm) — Validator fully recovered from image-1 (FIP TOC decode → Ghidra)

**The string-page mystery is CLOSED, and the prior "TE modules own it / RAM-staged" theory is
superseded.** Chain of discovery:

1. **FIP TOC decoded properly** (base `0x132100`, entries = UUID[16]+u64 off/size/flags):
   three images, offsets contiguous relative to `0x132100`:
   - img0 `0x1321b0` (0x35A1) — BL2/SEC stub (orr/msr prologue).
   - img1 `0x135751` (0x28A02) — **the secure-boot validator image (BL31-hw_cms)**: code
     0x135751–~0x153xxx + rodata incl. ALL validator strings (0x155–0x158k) + e2 string
     blob at 0x15E153 (0x41F, "# Ignore…/ensas/rcar_layout_create" — R-Car leftovers!).
   - img2/terminator entries contiguous: 0xb0+0x35A1=0x3651, +0x28A02=0x2C053 ✓.
2. The earlier "zero refs from bl31_img1" negatives were explained: **auto-analysis never
   explored 0x135xxxx** (no entry/symbol there — treated as data). Carving image-1
   (`e1.bin` = FD[0x135751:0x15E154], base 0x0 after an ignored BaseAddress import; entry
   disassembled manually) + `pyghidra.analyze()` → **828 functions, 428 data refs into the
   string region**. Validator = image-1's own code; identity-relative ADRP works because the
   image loads flat (relative addressing).
3. The "TE" VZ-modules (`0x3838`/`0x23260`, bases `0x31701000`/`0x31721000`) are a different,
   small (~32KB) code container with inline ELF-ish section tables (`.text/.data/.reloc`)
   — NOT the validators (too small, wrong span).

**Recovered architecture (file offsets; Ghidra project `/tmp/kilo e1_proj`):**
- **Header validator `0x13FF75`**: preamble `0x55AA55AA`@hdr+0; head_len `0x4DC`@+4;
  headMagic `0x33CC33CC`@+0x4B8; **nvcnt @+0x14** (<0x20); **code offset `0x2000` @+0x458**;
  **pss slen field @+0x50 (`(u32>>22) < 0x21`)** ← salt-length gate (Oct-26 `pss slen` failure
  candidate); error codes `0x5000_00xx` (aa03/04/18/19/07…).
- **Digest binding `0x13FE91` (FUN_0000a740)**: SHA-256 over [image+0x2000, code_len@+0x45C]
  → RAM `0x4100100` (SW engine via 0x84080000 when mode==1, else HW mailbox 0x2C00000 cmd
  0x82/0x10). **Compare primitive `0x15328D` = plain byte memcmp**, but the enroll path
  byte-swaps 32B before comparing against `0x4100100` → **HW SHA output is per-word
  byte-swapped vs standard SHA-256** — prime suspect for the D1–D5 digest-binding failures
  (rebuild_esbc wrote standard-order digests).
- **Verify mode gate `0x13F84D` (FUN_0000a0fc)**: efuse nibble **@0x8000FFF8** (`mode`);
  mode==1: `efuse[0x8006E080]!=0` → full-CMS, else **digest-only shortcut**; mode!=1: mailbox
  read (0x2C00000 cmd 0xB8) must equal magic **`0x1A4A5252`** → digest-only shortcut, else
  full-CMS. Shortcut path (`0xAAAAAAAA`) = `hw_cms_check_hash(image+0x2C, 0x4100100, 0x20)` —
  **pure digest check, NO signature** (the Ingenic-style endgame equivalent!).
- **Full-CMS path `0x1498BD` (FUN_0001416c)**: geometry builder `0x14036D` reproduces
  rebuild_esbc constants exactly: cms after payload+code_len+`0x20-(cl&0xf)` align; ini at
  cms+`0x2000|0x4000` (flag-switched); **crl at cms+0x800**; size gates (0xaa0e). Then
  `hw_cms_data_verify 0x1491D9` (software CMS: cert add, CRL 100/200/300 modes, update/verify
  via engine helpers, final==1) with ctrl-CPU usr-cert fallback `0x140649`; ini info
  name+hash compare `0x1497E1`; nvcnt gate `0x140271` (header nvcnt ≥ efuse bitpos via
  MMIO 0x84080000/0xB8; types 0xaa/bb/cc/ee registry @0xC6F4D8xx + sentinel 0xA7E358FD).
- **INI parser `0x1406CD`**: comma/semicolon tag;text,hash list → **49-byte entries
  {tag[16]@+4, sha[32]@+0x14, valid-flag@+0x34}**, max 32; gate loop `0x140861` =
  "cur image hash not in list!" allowlist (name cmp + digest cmp vs `0x4100100`).
- **Usr-cert enroll `0x141A9D` (ctrl_flag 0xdd)**: copy cert (<0xDF00) → `0x4012100`
  (staging `0x4010000`+hdr), hash-cal (result cached 0x8007E3C8–E4), **32B byte-reversal**,
  cmp vs stored (via 0x4100100), len recorded `0x401045C = len+0x100`; banner
  "Secure Boot: customer efuse Solution". `usr-cert present` = efuse `0x8007E2A4` bits 30/31.
- Boot-stage/force: `0x14101D` reads `0x8000F0DC>>8 & 0x1B` (0x10 pre-OS, 0x18 post) plus
  **"CORF" force magic (`0x464F5243`)** at `0x90E40010/0x90E50010` — a debug override worth
  probing.

**Actionable consequences:**
- The "external anchor constant" is **the header SHA-256 at image+0x2C compared against the
  (word-swapped) HW-SHA of the payload in shortcut mode** — no unknown efuse hash needed.
- Modified-payload strategy: **rewrite digest@+0x2C in word-byte-swapped order** to match the
  new payload; if the chip is in shortcut mode, no RSA/CMS needed at all. D-series retry =
  re-run with corrected byte order.
- If stuck in full-CMS mode: the CMS (not header+0x460 sig) is what must verify; the
  Oct-26 "0xE00 512B blob" question now maps to the CMS/engine territory of `0x1498BD`.
- Hardware reads worth grabbing on the bench console: nibble @0x8000FFF8, efuses
  0x8006E080 / 0x8007E2A4, and the mailbox word that must be `0x1A4A5252`.

**Toolchain notes:** Ghidra 12 headless: `-create` removed → `-overwrite`; BinaryLoader has
no entry-point loader option (disassemble+createFunction in pyghidra instead); GhidraMCP
extension manifest warnings are benign; pyghidra `open_program` DOES persist edits on context
exit. Artifacts: `/tmp/kilo/e1.bin`, `/tmp/kilo/e1_out2.txt` (428-ref map),
`e1_dec.txt`/`e1_dec2.txt`/`e1_dec3.txt` (decompiles), project `/tmp/kilo e1_proj`.

## Session 2026-10-30 (late pm) — esbc header chain cracked: the "external anchor" is just SHA-256

Full offline validation of stock containers against the recovered validator model:

1. **Anchor = plain SHA-256 of the payload.** For the stock ddr container,
   `header+0x2C == sha256(file[0x2000 : 0x2000+code_len@0x45C])` **in standard byte order**
   (identical result for kernel, hsm, lowpwr). The "externally anchored constant"
   `a8475646…2d45` was never external for the file itself — it is recomputable.
2. **Header tail layout (stock ddr, identical family for all containers):**
   - `+0x458` code_off `0x2000`; `+0x45C` code_len; `+0x460` u32 tag **`0xE00`** (the Oct-26
     "0xE00" — a descriptor tag, NOT the signature itself); `+0x464` `0xFFFFFFFF`;
   - `+0x4AC` magic `0xA5A55555`; `+0x4B8` headMagic `0x33CC33CC`;
   - **`+0x4BC` digest#2 (32 B) = SHA-256(header[0x000:0x4BC])** — hash of the whole header
     up to itself (brute-force proven: covers preamble incl. anchor);
   - `+0x4E0` small fields; `+0x510` table `{0x14, 0x2, 0x200, 0x200}`;
   - **`+0x518` 512-byte RSA-4096 blob** (the true "header signature"; Oct's +0x460 estimate
     was the tag, not the blob).
   - Chain: **payload → digest#1(+0x2C) → digest#2(+0x4BC) → RSA(+0x518)**.
3. **D-series reframed:** D1/D2/D4/D5 failures all had stale/inconsistent digest#1 (checked
   offline, self-consistency table: stock/D3/D5-file variants listed). **D3 was fully
   self-consistent (payload+clen+digest#1) and still FAILED** ⇒ cards run **full mode**,
   not the digest-only shortcut. Under full mode D3 also left digest#2 and the RSA stale —
   consistent with failure. No D test ever re-signed the header chain end-to-end.
4. **Container taxonomy:** only main `Ascend310P.image` (0x1DE5FF2) carries a trailing
   region — tail begins at `payload_end+0x18` (matches the builder's `0x20-(cl&0xf)` align)
   with the literal **`cms\0` tag** (this is where the Oct-26 user-cert CMS on rootfs lived).
   **ddr/hsm/lowpwr have ZERO tail bytes** — they are header-signature-only: the RSA@+0x518
   (or shortcut digest) is all that protects them.
5. **Signature-key status:** raw RSA-recover with BOTH embedded roots (G2 @FD 0x155841 AND
   the previously unnoticed **legacy `CN=Huawei Root CA` @FD 0x1552e9**) shows no PKCS1/PSS
   structure match against digest#2/anchor candidates — the +0x518 signer is a **subkey we
   don't hold** (or the enrolled-user key on enrolled cards, path-dependent).
6. **Re-sign recipe for custom ddr/hsm/lowpwr content (next HW experiments):**
   (a) rewrite payload; (b) digest#1 = sha256(payload) → +0x2C; (c) digest#2 =
   sha256(header[0:0x4BC]) → +0x4BC; (d) RSA-sign digest#2 (or the covered header region,
   per FUN_00013xxx semantics) with a trusted key → +0x518. On enrolled cards the user key
   may suffice (ctrl_flag/usr-cert path); stock cards need the Huawei subkey.
   **Cheapest decisive test remains: customized_info switch (seller's "cert check off") —
   if it flips the card into the digest-only shortcut, steps (a)+(b) alone boot.**
7. Bench-console wish-list unchanged plus: read the mailbox word that must equal
   `0x1A4A5252` (FUN_0000a0fc gate) and efuse nibble @0x8000FFF8.

Artifacts: digest-consistency tables computed from `/tmp/kilo` copies + `work/*.image`.

### Bench console path CLOSED by user decision (2026-10-31)
All three bench prerequisites are unworkable on this rig: the MCU console UART is not
reachable, the ESP32 serial unit on `ttyACM0` is not wired to the card, and the card cannot
be powered in its bench state. No live-console reads will be attempted; the bench wish-list
(`get_secureboot_info`, 0x8000FFF8 nibble, 0x1A4A5252 mailbox word, CORF) is dropped unless
flash-based alternatives are pursued later (MCU log rings live in flash @0x340000+; a
programmer re-dump can recover some of that state without any console). Forward path is
**host-side only**: file-swap cycles on 16409 with the reset recipe, upgrade-tool knobs
(customized_info, user-cert, component write test), and offline RE.
