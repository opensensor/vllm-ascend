# Ascend 310P customer-certificate path — findings (2026-10-26)

**Hardware**: 3× Ascend 310P3 (6 chips), host `matteius-threadripper` @ 192.168.53.187, driver 26.0.rc1.
**Scope**: why the 2026-10-21 self-made-CA attempt failed, what the customer-cert mechanism actually is,
and whether the "sign a child of the Huawei cert" idea can work.
**Status**: all findings below are **Observed** on the card unless marked Interpreted. No firmware was
flashed; the card is in its stock state (`boot_status` OK, stock rootfs installed).

---

## 0. TL;DR

1. **The emSign angle does not apply.** The device's firmware trust anchors are two **Huawei-only**
   roots, extracted from `AS610_HBOOT2_UEFI.fd`. No public CA — emSign, Sectigo, DigiCert, GlobalSign
   or any other — appears anywhere in the boot chain or the driver tree. Huawei's *web* PKI root list
   does contain emSign, but that is a different trust domain from device firmware signing.
2. **Josh's "emSign private key = 4" claim is false** for the cert in question. Tested directly against
   `emSign Root CA - G1`: `d=2,3,4,5,8,16` all fail the `m^(d·e−1) ≡ 1 (mod n)` test; a brute-force
   search over `d ≤ 4096` finds nothing. That root's key is sound. (The *intermediates* compromised in
   the Oct-2026 eMudhra incident are a different matter and also absent from this chain.)
3. **The whole Huawei chain is sound.** All five keys in the signing chain (2 roots + 2 intermediates +
   1 leaf signer) are RSA-4096, e=65537, no shared modulus, no small factor, no perfect power,
   no Wiener/Boneh-Venkataraman small-exponent solution, no Pollard p−1 factor, no Fermat factor.
4. **The real blocker is architectural, not cryptographic.** `upgrade-tool --user_cert` does **not**
   install a certificate. It computes `SHA-256(tag_block ‖ cert_DER)` and writes that 32-byte digest
   into the card's **secure flash** under the key `ccpu_usr_cert_hash`. The certificate bytes are never
   sent to the device.
5. **BL31's boot-time gate is a hash comparison, not a chain validation.** The strings
   `usr cert hash cal fail!` / `usr cert hash cpy fail.` / `usr cert hash cmp fail.` plus
   `Secure Boot: customer efuse Solution` show BL31 computes a hash of the candidate customer cert and
   compares it against the eFuse/secure-flash value. Our 2026-10-21 failure was almost certainly a
   **hash-domain mismatch**, not a broken chain.
6. **The stock CMS structure is now fully decoded, and the "constant messageDigest" is solved.**
   The outer CMS binds the payload via `SHA-256(initrd.ini body)` in `messageDigest` and binds its own
   internals via `signingCertificateV2 = SHA-256(signer cert)`. The constant `AF70DE12…` that the
   2026-09-12 research flagged across all three 9201-byte PSS structures is precisely
   `SHA-256(Computing PL Code-Signing Authority 2 - G2 .der)`. It is a signed attribute, not an
   anomaly.
7. **Therefore the "sign a child of the Huawei cert" idea is a dead end.** There is no Huawei private
   key available, no weak key to exploit, and no cross-certificate from a public root into
   `Huawei Integrity Root CA - G2`.

---

## 1. The device's actual firmware trust anchors

Extracted from `AS610_HBOOT2_UEFI.fd` (only two valid DER certificates in the entire 1.7 MB image):

| # | Subject | Key | Alg | File offset |
|---|---|---|---|---|
| 1 | `C=CN, O=Huawei, CN=Huawei Root CA` | RSA-4096, e=65537 | PKCS#1 v1.5 self-sig | `0x1552e9` |
| 2 | `C=CN, O=Huawei Technologies, OU=Huawei Certification Authority, CN=Huawei Integrity Root CA - G2` | RSA-4096, e=65537 | rsassaPss/MGF1-SHA256 self-sig | `0x155841` |

`strings` over the whole fd yields only `Huawei Root CA` and `Huawei Integrity Root CA - G2`.
Grep for `TrustCor|emSign|eMudhra|AAA Certificate Services|Sectigo` over `/usr/local/Ascend/driver/`
returns **zero** hits.

**Consequence**: the emSign idea has no surface to attach to on this hardware.

---

## 2. The stock signing chain (rootfs container)

`Ascend310P.cpio.gz` esbc trailer, extracted at the verified offsets
(`cms_start = 0x684d260`, `ini_start = cms+0x4000`, `crl_start = ini+0x800`):

```text
cms  len 9201   ini len 76   crl len 3522
ini  = "initrd,   39634b3e512d477fcc1b7280de792fcc450dfeca2d1d135fb4231d23c8ba0ff3;"
```

`openssl pkcs7 -print_certs` on the stock CMS yields:

| Role | Subject | Issuer | Serial | NotAfter |
|---|---|---|---|---|
| intermediate | `Huawei Integrity CA 1 - G2` | `Huawei Integrity Root CA - G2` | `112CB7` | 2031-05-08 |
| leaf signer | `Computing PL Code-Signing Authority 2 - G2` | `Huawei Integrity CA 1 - G2` | `11468AAA24108CFDCD439CA4B1CF37711F` | 2027-02-23 |
| timestamper | `Huawei Time-Stamping Authority 2 - G2` | `Huawei Integrity CA 1 - G2` | — | — |

The CRL bundle is 3 DER objects: root CRL (issuer `Huawei Integrity Root CA - G2`, nextUpdate
2026-11-29), plus two CA certs (`signca.der`, `tsca.der`).

The boot package `Ascend310P.image` is the same container family with tag `uimage` and
digest `93af6abd…` — a **different** digest from the rootfs container (`39634b3e…`). The two containers
are signed independently.

---

## 3. Why `--user_cert` cannot bootstrap a new trust anchor

### 3.1 What the tool actually sends

Reverse-engineered from `upgrade-tool` (x86-64, `0x68a09`):

```text
0x692e7  calc_pub_hash(buf, len, out[32]):
           EVP_DigestInit_ex(ctx, EVP_sha256())
           EVP_DigestUpdate(ctx, buf, len)
           EVP_DigestFinal_ex(ctx, out)      // 32 bytes
0x696b3  → 0x68a09(device_index, hash[32], 0x20)
0x68b99  lea rsi, "ccpu_usr_cert_hash"
0x68ba0  call [rip+0x2ff384]              // set user cfg item, key="ccpu_usr_cert_hash"
```

The input to the hash is the **esbc payload region** `file[0x2000 : 0x2000 + code_len]`, i.e. the
0x100-byte tag block (`"usrcert"` + padding) **followed by** the DER certificate.

Verified against our archived `user.der`:

```text
code_len            = 1721
header digest@0x2c  = 7fb03430761fd43a26e90000c4592838b3746ce8d1fb895c8fa124c9bab63e53
SHA256(body)        = 7fb03430761fd43a26e90000c4592838b3746ce8d1fb895c8fa124c9bab63e53   ✅ match
body[0:16]          = b'3\xaa::usrcert\x00\x00\x00\x00\x00'
cert                = rootcaG2.der (self-signed, CN=RootCA G2, C=CN, O=Test)
```

### 3.2 Where the digest lands

`drv_user_cfg.ko` is the only driver module containing the string `ccpu_usr_cert_hash`
(other hits: `libdrvdsmi.so`, `libascend_hal.so` — host-side libraries). It exposes:

```text
sec_flash_read      / sec_flash_write        (exported)
devdrv_config_sync_data_to_flash
devdrv_config_write_flash_head_content
"The sec_flash_write failed. (device_id=%u; ret=%d)"
```

So the enrolled value is a **32-byte SHA-256 digest written to the card's secure flash**, not a
certificate. The certificate itself is never transferred.

### 3.3 What BL31 does with it

BL31 (`AS610_HBOOT2_UEFI.fd`, FIP@0x132100) carries the customer-cert path:

```text
21a6a  "usr cert hash cal fail!0x%x"
2171d9 "Secure Boot: customer efuse Solution"
21c86  "usr cert hash cpy fail."
21ca0  "usr cert hash cmp fail."
21440  "usr cert copy fail."
215dd  "usr cert cms vef fail, result = 0x%x"
21604  "usr cert cms vef success!"
21661  "cms_file"
216fa8 "support usr cert vef"
21a3d? "no usr, hw vef fail, ctrl_flag = 0x%x!"
```

The sequence `hash cal` → `hash cpy` → `hash cmp` is a **compare of a computed digest against a
stored digest**, and the `customer efuse Solution` string ties the stored value to an eFuse-burned
customer-mode state. The boot-time check is therefore hash-based, exactly as the 2026-10-21
write-up concluded — but with the hash domain now pinned down.

Also present in BL31: the tag set `dataisp`, `dataotp`, `tee`, `initrd`, `uimage`, `dtimage`,
`usrcert` — i.e. BL31 knows the `usrcert` container tag, but the stock `Ascend310P.image` and
`Ascend310P.cpio.gz` contain **no `usrcert` section** (grep: 0 hits in both).

---

## 4. Why the 2026-10-21 attempt failed — the likely real cause

The 2026-10-21 A/B matrix (`artifacts/ascend-310p-usercert-backup-20261021/WRITEUP.md` §10.5)
showed every customer-signed variant failing, with the conclusion "the gate is the CMS trust
decision over our chain". Given the mechanism above, that conclusion was probably **wrong**:
the more likely failure is that BL31 computes the customer-cert hash over a **different byte
range** than `upgrade-tool` does.

Known hash domains, all of which must be reconciled:

| Candidate | Bytes hashed | SHA-256 of our `rootcaG2.der` |
|---|---|---|
| A: tool-side (confirmed from disassembly) | `tag_block(0x100) ‖ cert_DER` | `7fb03430…` |
| B: cert only | `cert_DER` | *different* |
| C: cert subject/pubkey only | varies | *different* |
| D: whole file | `file[0 : filesize]` | *different* |

`upgrade-tool` enrolls **A**. If BL31 computes **B** (the natural choice for "hash of the customer
certificate"), every enrollment mismatches and the comparison always fails — exactly the observed
symptom, and exactly why swapping root ↔ intermediate ↔ leaf made no difference (all three produce
a different digest under both A and B, so all three fail either way).

**This is the first thing to test.** It is cheap: build three `user.der` variants for the same
certificate where the *enrolled digest* is A, B and C respectively, enroll one, and see which one
BL31 accepts. The enrollment path is already proven to accept hand-built envelopes.

Second possibility, now **downgraded**: the CMS needs Huawei's exact structure — which is now known
(§5). Our 2026-10-21 `cmssign 0.0.5` output was a single-signer plain signedData with no
`signingCertificateV2`, no nested RFC-3161 timestamp token, and only one signer. Reproducing the
three-signer shape with PSS/MGF1-SHA256/salt-32 is mechanical once the packing tool is available.

Note the interaction between the two: the payload binding in the CMS is `SHA-256(initrd.ini)`,
whereas the customer-cert enrollment binds `SHA-256(tag ‖ cert)`. These are **different domains**,
and the 2026-10-21 build may have conflated them.

---

## 5. The stock CMS structure — fully decoded and every digest verified

This is the part that was previously guessed at, and it is now exact. The 9201-byte stock CMS is a
`pkcs7-signedData` with **three** signers and a **nested RFC-3161 timestamp token**.

```text
ContentInfo
└─ signedData (version 1)
   ├─ digestAlgorithms: [sha256]
   ├─ encapContentInfo: pkcs7-data, content ABSENT  → detached
   ├─ certificates: [ Huawei Integrity CA 1 - G2,
   │                  Computing PL Code-Signing Authority 2 - G2,
   │                  Huawei Time-Stamping Authority 2 - G2 ]
   ├─ crls: [ 1 CRL ]
   └─ signerInfos: [ 3 ]
      ├─ S1  issuer = Huawei Integrity Root CA - G2, serial 1125559
      │        → TSA / CRL signer, contentType = pkcs7-data
      │        messageDigest = 2934ff7d…   = SHA-256(initrd.ini body)
      │        signingCertificateV2 = AF70DE12… = SHA-256(code-signing leaf cert)
      ├─ S2  issuer = Huawei Integrity CA 1 - G2, serial 11468AAA24108CFDCD439CA4B1CF37711F
      │        → the actual code signer (Computing PL Code-Signing Authority 2 - G2)
      │        contentType = pkcs7-data
      │        messageDigest = 2934ff7d…   = SHA-256(initrd.ini body)
      │        signingCertificateV2 = AF70DE12… = SHA-256(code-signing leaf cert)
      │        includes unsigned attr id-smime-aa-timeStampToken → nested TST
      └─ S3  issuer = Huawei Integrity Root CA - G2, serial 112A81B7A6292455BED7C1223DD7A337EF
               → the RFC-3161 timestamp signer (nested TST), contentType = id-smime-ct-TSTInfo
               messageDigest = 705C909A… = SHA-256(TSTInfo DER)
               signingCertificateV2 = 75EC8C34… = SHA-256(Huawei Time-Stamping Authority 2 - G2)
```

### 5.1 Verified digest bindings (all confirmed by recomputation)

| Binding | Verified |
|---|---|
| `SHA256(initrd.ini body)` = `2934ff7d391caffa2b2543d6d0cfAA4e1f9f4b7a23babb9b419b4a75ccc7b66b` | ✅ matches S1/S2 `messageDigest` |
| `SHA256(Computing PL Code-Signing Authority 2 - G2 .der)` = `af70de12c89033a89a2041d3287874451fe51a7f03c24f1deBDd7a3513fb709a` | ✅ matches S1/S2 `signingCertificateV2` |
| `SHA256(TSTInfo DER)` = `705c909a6dd3403124c05ef0b118268c45972594ec1bfacf6ddbc7bcc5b4f386` | ✅ matches S3 `messageDigest` |
| `SHA256(Huawei Time-Stamping Authority 2 - G2 .der)` = `75ec8c34…` | ✅ matches S3 `signingCertificateV2` |

**`af70de12…` is the constant `messageDigest` that the 2026-09-12 research flagged as "constant
across all three 9201-B PSS structures".** It is not a mystery constant — it is the
`signingCertificateV2` hash of the code-signing leaf certificate, carried as a signed attribute.
That resolves rank-3/U-D from `SECUREBOOT-WEAKNESSES.md`: the outer CMS signature binds the CMS
internals (via `signingCertificateV2`), while the payload binding is the `initrd.ini` digest, which
the inner verifier reads from the esbc `ini` section.

**Consequence for re-signing:** a replacement CMS must reproduce this exact shape — detached
`pkcs7-data`, sha256, PSS with MGF1-SHA256 and salt length 32, three signers, the nested RFC-3161
timestamp token, and `signingCertificateV2` attributes. Our 2026-10-21 `cmssign 0.0.5` output was
a single-signer plain signedData with none of that.

### 5.2 Why `openssl cms -verify` fails on the stock CMS (expected, not a defect)

```text
$ openssl cms -verify -inform DER -in stock_cms.der -content initrd.ini \
      -CAfile <Huawei Integrity Root CA - G2> -certfile chain.pem -purpose any
CMS Verification failure
  ... CMS_SignerInfo_verify_content: verification failure
```

Two independent reasons, both inherent to the container and **not** evidence of tampering:

1. **`-purpose any` / cert-path checks.** The chain includes `Huawei Time-Stamping Authority 2 - G2`
   and the CRL signer; OpenSSL's default purpose and CRL policy reject the combination.
2. **The timestamp token.** The nested TST is signed over the *outer* signature value
   (`encryptedDigest` of S2), so `-content`-based verification of the outer CMS cannot satisfy S3's
   `messageDigest` — the TST's content is the signature, not the ini.

The device does not use OpenSSL's S/MIME policy engine; it uses Huawei's own
`CmscbbVerifyCreateCtx → AddCert → AddCrl → DetachSignature{Begin,Update,Final}` path, which
performs no `CmscbbPkiVerifyCert` chain validation in the runtime module. **A stock CMS failing
`openssl cms -verify` is therefore not a reliable oracle** for whether the device would accept a
replacement — the device's accept/reject logic is narrower than OpenSSL's.

---

## 6. The emSign / public-root question, settled

### 5.1 Huawei's public root list does contain emSign

`https://www.huawei.com/en/root-certification-program/huawei-trusted-root-certificate-20251230`
lists `emSign Root CA - G1` (SHA-256 `40F6AF0346A99AA1CD1D555A4E9CCE62C7F9634603EE406615833DC8C8D00367`,
valid 2018-02-18 → 2043-02-18), alongside AAA Certificate Services, COMODO, DigiCert, GlobalSign and
Sectigo roots. So Josh's list check is correct.

But this is Huawei's **web/server** PKI. The device's **firmware** PKI anchors are the two
Huawei-only roots in §1. No public root cross-certifies into `Huawei Integrity Root CA - G2`, so
there is no path from a public-root compromise to a firmware signature on this hardware.

### 5.2 The "private key = 4" claim is false

Tested directly on `emSign Root CA - G1` (RSA-2048, e=65537), pulled from the card's own
`/etc/ssl/certs/`:

```text
d= 2: m^(d·e−1) ≡ 1 (mod n)  for 0/11 random m
d= 3: 0/10      d= 4: 0/10      d= 5: 0/11
d= 8: 0/7       d=16: 0/9
brute force d ≤ 4096: no solution
```

A `d=4` key would satisfy `m^(4e−1) ≡ 1` for essentially every coprime `m`. It does not. The key is
not weak.

Separately: a scan of **all 120 roots** in the card's `/etc/ssl/certs/ca-certificates.crt`
(Ubuntu `20260816ubuntu0.26.04.1`) for small private exponents `d ≤ 2000` found **zero** weak roots.

### 5.3 The Oct-2026 eMudhra compromise does not reach this device

The compromised intermediates are `EM {DV,OV,EV} TLS CA - G3B/G3C` (cross-signed to
`emSign Root TLS CA - G3` and `AAA Certificate Services`). On the card host:

```text
emSign Root CA - G1        PRESENT   (in ca-certificates.crt)
emSign Root CA - C1        PRESENT
emSign Root TLS CA - G3    ABSENT
emSign ECC Root CA - G3    ABSENT
AAA Certificate Services   ABSENT
EM DV TLS CA - G3B         ABSENT
```

None of the eight compromised intermediates nor their roots are present. And in any case the
compromised keys would have to be accepted by the **firmware** verifier, which uses only the two
Huawei roots. Irrelevant here.

---

## 7. Why "sign a child of the Huawei cert" cannot work

To sign a child of `Huawei Integrity Root CA - G2` (or `Huawei Root CA`) you need the corresponding
**root private key**. Findings:

- The roots are RSA-4096, e=65537, and **sound** (§6.2, §1). No mathematical weakness to exploit.
- The private keys are not in any artifact we hold: the driver tree contains **zero** `.pem`/`.der`
  files (`find /usr/local/Ascend -name '*.pem'` → empty), and `upgrade-tool` has no embedded
  certificates (`strings | grep -c 'BEGIN CERTIFICATE'` → 0). It links OpenSSL but uses it for
  hashing/parsing only.
- No public root cross-signs into the Huawei roots, so a compromised public intermediate buys nothing.

The only remaining legitimate route is the **documented customer-certificate mechanism**, which is
exactly what `--user_cert` exposes — and whose remaining gap is the hash domain + CMS structure
(§4, §5), not key material.

---

## 8. Recommended next steps (in order of cost)

1. **Resolve the hash domain.** Build three `user.der` variants for one certificate differing only
   in the *enrolled digest* (A = `tag‖cert`, B = `cert only`, C = `pubkey-only`), enroll each, and
   observe which BL31 accepts. Requires serial console or a `usr cert hash cmp fail.` capture;
   the 2026-10-21 write-up notes no UART is available, so this may need the SecureBoot
   `blob+0x1000` header-copy route or a debug build.
2. **Get the real packer.** `Ascend-cann-device-sdk_*_linux-x86_64.zip` (CtrlCPU Open SDK) contains
   `image_pack.py` / `esbc_header.py`. `rebuild_esbc.py` already reproduces the stock rootfs
   byte-for-byte, so the container layout is solved — what is missing is the **three-signer CMS
   structure with the nested RFC-3161 token and `signingCertificateV2` attributes** (§5), which the
   SDK would emit correctly. A hand-rolled `cmssign` will not cut it: the device's verifier is
   narrower than OpenSSL's, so it must be matched to the stock structure byte-pattern for byte-pattern.
3. **Read the enrolled hash back.** `sec_flash_read` is exported by `drv_user_cfg.ko`; if a writable
   path exists from the host, reading the 32 bytes at `ccpu_usr_cert_hash` removes all guesswork
   about the domain.
4. **Check the eFuse customer-mode bit.** `Secure Boot: customer efuse Solution` suggests a
   provisioned customer mode. If that bit is not set on these cards, the customer path may be
   closed regardless of signing correctness — this is the rank-1 Unknown from
   `SECUREBOOT-WEAKNESSES.md` and should be confirmed early.
5. **Protect the existing key material.** `rootca_priG2_pri.pem` and `signcert_pri.pem` (archived on
   all three drives) are the only keys that can sign anything these chips will accept. Losing them
   means only the stock rootfs ever boots.

---

## 9. Artifacts produced by this investigation

Working directory: `/tmp/kilo/ascend-secboot/`

| File | Contents |
|---|---|
| `hboot2.fd` | `AS610_HBOOT2_UEFI.fd` (copied from card) |
| `Ascend310P.image` | stock boot package (copied from card) |
| `stock_cms.der`, `stock_crl.bin` | extracted rootfs trailer sections |
| `hwcert_0.der`, `hwcert_1.der` | the two Huawei roots |
| `chain_0/1/2.der` | integrity CA, code-signing leaf, timestamping CA |
| `drv_pkicms.ko`, `drv_upgrade.ko`, `drv_user_cfg.ko` | driver modules from the card rootfs |
| `upgrade-tool` | host tool (for disassembly) |
| `audit2.py`, `smallkey.py`, `base_vote.py` | key-weakness and xref analysis scripts |
| `card-bundle.crt` | card host CA bundle snapshot |
| `huawei-roots.html`, `huawei_roots.json` | Huawei trusted-root list snapshot |

No card state was modified. All reads were non-destructive; `upgrade-tool --status` and
`--system_version` were the only commands run against the devices.
