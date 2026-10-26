# Ascend 310P Secure Boot — Investigation & Breakthrough Write-up

**Date:** 2026-10-21
**Host:** `matteius-threadripper` (192.168.53.187), Ubuntu 26.04.1, kernel 7.0.0-38
**Driver:** 26.0.rc1 · 6× Ascend 310P3 (3 cards × 2 chips)
**Status:** first re-signed-rootfs boot **FAILED** — root causes under investigation

---

## 0. TL;DR

1. **FD patch route is closed.** hboot1 verifies the FD against a **NOR-stored** digest; a
   fully repacked FD (`b.eq→b`) fails even after a hard power-off.
2. **The rootfs is CMS-verified** by BL31: built-in Huawei cert first, then a **customer/user
   cert** fallback.
3. **Breakthrough A:** `upgrade-tool --device_index -1 --user_cert` accepts a **hand-built,
   self-signed** esbc envelope (`user.der`) with **no Huawei `usrcert.ini.cms`**. Accepted on
   all 6 chips (`set user certificate info success`).
4. **Breakthrough B:** the esbc digest format is cracked:
   `digest@0x2c = SHA256(data[off : off+len])`, `off=u32@0x458`, `len=u32@0x45c`.
   Envelopes can be built by hand — the SDK is not required for the envelope.
5. **Open:** the re-signed modified rootfs did not boot. Prime suspects: the retained stock
   Huawei **CRL**, our CMS being a plain PKCS#7 instead of the SPC/`spcIndirectData` form,
   and/or the BL31 **usr-cert hash** comparison.

---

## 1. Trust chain

```
BootROM (on-die)
   └─► hboot1 (NOR)         verifies FD digest vs NOR-stored value      [closed]
          └─► FD / BL31    Huawei Root CA / Huawei Integrity Root CA-G2
                 └─► rootfs (Ascend310P.cpio.gz)  CMS verify
                        built-in Huawei cert  →  else  user cert fallback
```

BL31 strings of interest (carved from `AS610_HBOOT2_UEFI.fd`, FIP@0x132100, BL31@+0x3651,
size 166402):

```
1fbe9  Huawei Root CA
201a6  Huawei Integrity Root CA - G2
215dd  usr cert cms vef fail, result = 0x%x
21604  usr cert cms vef success!
21661  cms_file
2171d9 Secure Boot: customer efuse Solution
21a6a  usr cert hash cal fail!
21c86  usr cert hash cpy fail.
21ca0  usr cert hash cmp fail.
2377a  update ctx add pbroot fail
239d9  add cert fail ret = 0x%x
23a0e  verify fail ret = 0x%x
```

The usr-cert path is **hash-based** (`usr cert hash cmp`, "customer efuse"), not a Huawei
envelope verify. The runtime custom path (`drv_pkicms.ko`) is **chain-less**:
`CmscbbVerifyCreateCtx → AddCert → AddCrl → DetachSignature{Begin,Update,Final}`, with **no**
`CmscbbPkiVerifyCert`.

---

## 2. esbc envelope format (reversed)

For `Ascend310P.image`:

| field | value | meaning |
|---|---|---|
| `u32@0x458` | `0x2000` | signed-range offset |
| `u32@0x45c` | `0x1ddea08` | signed-range length |
| `digest@0x2c` | `93af6abd…` | **SHA256(data[0x2000 : 0x2000+0x1ddea08])** |
| `u32@0x1458/0x145c` | same | second `0x1000` header copy |
| `digest@0x102c` | same | second copy digest |
| `tag@0x2004` | `initrd`/`usrcert`/`uimage`/`hboot2` | tag string |
| payload | `0x2100` | starts here |

Rootfs trailer is a tagged container:

| section | header | length | value |
|---|---|---|---|
| `cms` | 16 B (tag + pad + u32 len@+12) | 9201 (stock) | CKMS/PKCS#7 |
| `ini` | 16 B | 76 | `initrd,   <digest>;\n` |
| `crl` | 16 B | 3522 | `cat rootG2.crl sign.crl ts.crl signca.der tsca.der` |

`initrd.ini` carries the **same** digest as the header (`sha256(range)`).

---

## 3. What was done

### 3.1 Self-made CA (throwaway, now the device trust anchor)

- Root CA: RSA-4096, PSS, 100 years.
- Signing leaf: EKU `codeSigning`, signed directly by the root.
- `openssl verify -CAfile rootcaG2.pem signcert.pem` → **OK**.

### 3.2 `user.der` accepted by the chip

Built with `build_user_der.py` (header template + `rootcaG2.der`, tag `usrcert`, digest fixed).
Enrolled:

```
upgrade-tool --device_index -1 --user_cert --path user.der
Device[0..5] set user certificate info success.
```

No Huawei CMS, no `usrcert.ini` round-trip.

### 3.3 Re-sign of the modified rootfs

- Base: `modified5-filesystem-le.cpio.gz` (110,544,347 B).
    - header digest `38092b04…` **correct**
    - `ini` = `initrd,   38092b04…;` **correct**
    - CMS = **stock Huawei** (signs the *old* digest) → invalid.
- New CMS: `cmssign sign --signer signcert.der --key signcert_pri.pem
  --signerCA rootcaG2.der --in initrd.ini --out initrd.ini.p7s` → 2183 B.
    - Verified: `openssl cms -verify … -purpose any` → **successful**.
    - `cmssign 0.0.5` bug patched: `args.tsca` → `args.tsCA`.
- Spliced into the `cms` slot (0x6967019, len 9201), zero-padded.
- Output `modified6-resigned.cpio.gz`, sha256 `93c587cf…`, installed at
  `/usr/local/Ascend/driver/device/Ascend310P.cpio.gz`.
- **Reboot failed.**

---

## 4. Keys & artefacts

### 4.1 Live copies (threadripper)

```
/home/matteius/usercert-test/
  rootcaG2.der  rootcaG2.pem  rootca_priG2_pri.pem
  signcert.der  signcert.pem  signcert_pri.pem
  user.der     initrd.ini     initrd.ini.p7s
  x509_rootcaG2.cnf  x509_signcert.cnf
```

### 4.2 Backups (3 independent drives, hashes identical)

| drive | path |
|---|---|
| 20 TB | `/run/media/matteius/20TB-drive/vllm-ascend/artifacts/ascend-310p-usercert-backup-20261021/` |
| ai-drive | `/run/media/matteius/ai-drive/usercert-backup-20261021/` |
| 3.6 TB | `/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/usercert-backup-20261021/` |

### 4.3 SHA-256

```
023f6057aad559a70310e700ee2fa277383eb5ac201ea8654bfbfd7d294179fe  rootcaG2.der
39b31e85aa1416b82df6fb4eb137eb860d747e585804e46d3135ebbb71e88aea  rootcaG2.pem
4d5ce7ea215b08f39cdf44cfb59436ae153351ad3a24eab18ae9e62ac9b36548  rootca_priG2_pri.pem
089238e8d1186a6be4b0d77782a8e9fbfaef9436441d8a6fecc54f102b97b135  signcert.der
5aa2119da1df3ce4b6cc05f6cc2a2ac247e47c36d7f3080d03f7bb140966ca8c  signcert_pri.pem
c7806bb6c8c863f668ee414f43111293a0ae61c95a3579f26768ac805b553c48  user.der
93c587cf218d729f5d258db43933b426b1a1a6e6bb9ec3622b7798904db8658f  modified6-resigned.cpio.gz
```

> The two private keys (`rootca_priG2_pri.pem`, `signcert_pri.pem`) are the crown jewels.
> If the chip efuse now pins this CA, losing them means only the stock rootfs will ever boot.

### 4.4 Stock backups

```
/home/matteius/Ascend310P.cpio.gz.stock-backup-20261021   (109,389,874 B)
```

---

## 5. Recovery

```
sudo cp /home/matteius/Ascend310P.cpio.gz.stock-backup-20261021 \
        /usr/local/Ascend/driver/device/Ascend310P.cpio.gz
sudo reboot
```

The user cert (our CA) remains enrolled but is unused once the stock (Huawei-signed) rootfs
passes the built-in cert again.

---

## 6. Why the first attempt likely failed

Ordered by likelihood:

1. **CRL mismatch.** We kept the stock Huawei `initrd.ini.crl`. The chip's `CmscbbVerifyAddCrl`
   may reject a CRL whose issuer is not our root, or require a CRL for `rootcaG2`.
   → Generate `rootcaG2.crl` (empty revocation) and splice it.
2. **CMS content type.** `cmssign` 0.0.5 emits a normal PKCS#7 `signedData`. The stock CMS is
   the Huawei SPC form (contains `spcIndirectData`, the CRL bits, and a timestamp signer).
   → Diff our 2183 B CMS against the 9201 B stock CMS; reproduce the SPC structure, or obtain
   Huawei `image_pack.py`/`cmssign`.
3. **Usr-cert hash target.** BL31 compares a hash of the usr cert against the burned hash.
   If it hashes the **CMS signer (leaf)** while `user.der` enrolled the **root CA**, it
   mismatches. → Try a `user.der` built from `signcert.der` (leaf) instead of `rootcaG2.der`,
   or an intermediate CA that makes the signerCA == the enrolled cert.
4. **Image ID path.** The usr-cert fallback may apply only to specific image IDs (FS/rootfs vs
   initrd). → Confirm which container the customer path covers.
5. **Payload.** The modified5 content itself may be bad, independent of signing.

---

## 7. Next steps

1. **Recover** to the stock rootfs; confirm all 6 chips return (`boot status 16`).
2. Capture the **boot console / firmware log** on the failed boot — the exact error
   (`usr cert cms vef fail` vs `usr cert hash cmp fail` vs CRL) decides the branch.
3. Generate a CRL for `rootcaG2`; splice and retry.
4. If it is the CMS format, obtain the real Huawei `image_pack.py` / `esbc_header.py`
   (CtrlCPU Open SDK, `Ascend-cann-device-sdk_*_linux-x86_64.zip`) and rebuild the trailer
   exactly.
5. Keep every artefact and hash on all three drives; never lose the two private keys.

---

## 8. Known-good static conclusions

- FD route: **dead** (NOR anchor).
- `--user_cert` Huawei-signature gate: **not enforced** (proven on 6 chips).
- Runtime custom-cert chain check: **absent** (`drv_pkicms.ko` 0x943c).
- Boot usr-cert check: **hash-based**, not envelope-based.
- esbc digest: **fully understood**.
- Remaining unknown: the exact **boot-time** acceptance path for a customer-signed rootfs
  (CRL + CMS content type + usr-cert hash target + image ID).

---

## 9. Update — 2026-10-21 07:41 (full chain + stock-payload A/B)

### 9.1 Full chain built (per `instgopen_0051.html`)

Backed up under `full/` on all three drives:

- `signca` (intermediate, codeSigning) ← root
- `signcert` (leaf, codeSigning) ← signca
- `tsca` (timestamp intermediate) ← root
- `ts` (timestamp leaf) ← tsca
- `rootG2.crl`, `sign.crl`, `ts.crl`; `initrd.ini.crl = cat rootG2.crl sign.crl ts.crl signca.der tsca.der`

SHA-256:

```
f31ad1c4e9c9db76df1264243f274e5018b59357fad10a6c7aa3457107d77d8f  signca.der
03016c7268975a78079f44f69f79d229332c0914961b710c73bb490e23bde204  signcert.der
4316631e59b6a1b5d8346676d47cbefbf51d8ff095022dd6e5c9f36f23e70bba  tsca.der
1290eb87c2ecb5606a5833fa769d876ded495e40c11b3e8b112fa1ea27fbc677  ts.der
a58d108d702786d8303c42f2642914886f91685c74c97d4e22e48836ee781dd8  signca_pri.pem
ccf605f7820522bb0ff43cafb4d673f0df161423099b65093ed7035f712f604a  signcert_pri.pem
2ccb02ac6a1e99d038272ff98aa02edd865c03426c4fe4c89b00fe2e65668ef2  tsca_pri.pem
4623bcad3ea3c190b71162d1601282b5425fe8fc0a81e91ad70ede90d973b5b2  ts_pri.pem
```

### 9.2 A/B results

| image | payload | CMS | result |
|---|---|---|---|
| `modified5` | modified | stock Huawei | fail |
| `modified6-resigned` | modified | 2-level, no CRL/ts | fail |
| `modified7-fullchain` | modified | full chain + CRL + ts | **fail** |
| `stock-resigned` | **stock** | full chain + CRL + ts | **fail** |
| `stock-resigned` + `user-signca` | **stock** | full chain + CRL + ts | **fail** |

⇒ The **payload is innocent**. Both the root-CA and the signca `user.der` fail, so the
usr-cert hash target is **not** the cause. Remaining fault domain: the **CMS/trailer bytes**
as constructed by hand (the `image_pack.py` layout), not the payload or the cert choice.

### 9.3 Debug status

- **No UART on the card.** BL31 error not observable; host dmesg only shows the host side.
- `npu-smi` / `upgrade-tool --boot_status` hang whenever the device is down (expected).

### 9.4 Remaining suspects (ranked)

1. **Usr-cert hash target** — BL31 may hash the CMS **signerCA** (`signca`), not the root CA.
   Enroll a `user.der` built from `signca.der` and retry the stock-resigned image.
2. **CMS/trailer byte-exactness** — the hand-rolled trailer may differ from `image_pack.py`
   in padding/section semantics; the only reliable fix is Huawei `image_pack.py` /
   `esbc_header.py` (CtrlCPU Open SDK).
3. **Image-ID path** — customer fallback may apply only to the FS/rootfs container, not initrd.

### 9.5 Next

1. Restore stock, reboot, confirm 6 chips.
2. Blind A/B #1: enroll `user-signca.der`, deploy `stock-resigned`, reboot.
3. If that fails, obtain the CtrlCPU Open SDK and rebuild the trailer with `image_pack.py`.
4. Every artefact/hash is on all three drives; never lose the private keys.

---

## 10. 2026-10-21 late — packing solved, payload fault found, trust path still closed

### 10.1 Real on-card layout (verified byte-exact)

The 310P container is a 4 KiB header (copied at 0x0 and 0x1000) followed by payload and trailer:

| field | offset | value / rule |
|---|---|---|
| preamble | +0x00 | `0x55AA55AA` |
| head_len | +0x04 | `0x4DC` |
| certtype | +0x08 | stock `0xFFFFFFFF` |
| payload digest | +0x2C | SHA-256 of payload (both copies) |
| img_offset | +0x458 | copy0 `0x2000`, copy1 `0x1000` (both stock) |
| code_len | +0x45C | payload length (both copies) |
| sign_offset | +0x460 | `0xE00` |
| head_magic | +0x4B8 | `0x33CC33CC` |
| head_hash | +0x4BC | SHA-256 of header `[base, base+0x4BC)` |
| extern | +0x4E0 | u64 `0xABCD1234AA55AA55`, u32 total size @+0x4E8 |

```
cms_start = 0x2000 + code_len + (32 - code_len % 16)
ini_start = cms_start + 0x4000
crl_start = ini_start + 0x0800
section   = 12-byte tag + u32 LE length at +12
```

`rebuild_esbc.py` reproduces the stock rootfs (`109,389,874 B`) and the stock
`Ascend310P.image` (`31,350,770 B`) **byte-for-byte** (`--in <img>` verify mode).
The header carries **no** trailer offset table; that hypothesis is dead.

### 10.2 Why every earlier splice failed

- `modified2` / `fullchain` / `resigned`: CMS at `0x6967009` instead of `0x6967010` —
  the alignment (`32 - code_len%16`) was computed from the **old** `code_len`
  (23 bytes) instead of the new one (30 bytes), so the whole trailer is 7 bytes early
  and misaligned. The device derives the offset from `code_len` and finds zeros.
- `modified2` additionally: copy-1 `code_len` stale (`0x684b249`), copy-0 `head_hash`
  stale, total-size field stale by 1,154,473 B.
- `stock-resigned`: correct geometry, but the size field was stale by 1,887 B.

### 10.3 Payload defect (independent of signing)

The `modified2` payload is a **broken cpio**: 2,099 entries vs 2,574 in stock, with
476 paths missing — including the usrmerge symlinks `bin`, `lib`, `lib64`, `sbin`
and `etc/pam.d/*`, `etc/crypto-policies/*`. The only addition is `var/tee_probe`.
Without `sbin -> usr/sbin`, `rdinit=/sbin/init` cannot resolve; this payload could
never boot. (Stock has trailing padding after the gzip stream; the modified one does
not — harmless.)

### 10.4 `user.der` envelope defect

All three earlier envelopes had an **invalid head_hash in both copies** and a stale
size field (`31,350,770`, the template size), because the builder only patched
digest/offset/length. `build_user_der.py` now writes certtype, both head hashes,
the extern magic and the real size; all fields validate.

### 10.5 A/B matrix (all with correct packing and size)

| payload | CMS | CRL | certtype | enrolled | result |
|---|---|---|---|---|---|
| stock | stock | stock | HW | – | **boot** |
| stock | ours | 5-object bundle | HW | signca | fail |
| stock | ours | stock-shape 3-object | HW | root | fail |
| stock | ours | stock-shape 3-object | 2 | leaf | fail |
| stock | ours | stock-shape 3-object | 1 | root (valid envelope) | fail |
| stock | ours | stock-shape 3-object | 1 | leaf (valid envelope) | fail |
| stock | stock | ours | HW | – | fail |
| stock | ours | empty (len 0) | 1 | leaf | fail |
| stock + 426 zero bytes, size updated | stock | stock | HW | – | **boot** |

Conclusions:

- Total file size and trailing bytes are **not** checked.
- CRL shape, order, and even presence are not the gate (empty CRL still fails).
- Cert choice (root, intermediate, leaf) and certtype (HW, 0x1, 0x2) are not the gate.
- With the stock CMS the image boots only with the stock CRL, i.e. the CRL is
  validated against the CMS chain as expected.
- The gate is the **CMS trust decision over our chain**; `upgrade-tool --user_cert`
  is accepted but does not enable boot trust for it.

### 10.6 Remaining hypothesis

`support usr cert vef` and `Secure Boot: customer efuse Solution` point at an
eFuse-provisioned customer mode: `--user_cert` stores certificate info, but the boot
path may compare against an **eFuse-burned root/hash** that only a Huawei-signed
provisioning image can set. If so, no self-signed chain can boot this unit.
Next static target: locate the eFuse bit/register read by the `support usr cert vef`
branch in BL31-132100 and determine whether it is writable from the host.

### 10.7 Artifacts

`rebuild_esbc.py` (verified), `build_user_der.py` (fixed), `stock-ourcms*.cpio.gz`,
`modified9-ctff.cpio.gz`, `modified9-ct1.cpio.gz`, `user-*-ct1.der`,
`full/crl-stockshape.bin`. Stock image restored; all six 310P3 chips report OK.
