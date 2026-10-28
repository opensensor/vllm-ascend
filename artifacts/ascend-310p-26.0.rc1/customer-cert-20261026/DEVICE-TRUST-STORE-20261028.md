# The device's trust store: five anchors, and two of them are signing CAs

**Date:** 2026-10-28
**Status:** extracted from the device's own `drv_pkicms.ko` and the vendor's
host-side source. Hardware-validated by the fact that the stock package
verifies against them.

## The question

Is there anything in the CA certificate that exposes other signers or
trusted CAs, or is it just one root?

**Answer: the device's trust store is five certificates, and it is not one
root — it is a set that includes intermediate CAs with the private key
material's public half present and `Certificate Sign` usage.** That is
materially different from "one root that can only sign leaf certs".

## The device's actual trust store

`drv_pkicms.ko` (the device's crypto module, `user_cfg_drv`/`pkicms`) carries
these, verbatim:

| # | Subject | CA? | Key Usage | Role |
|---|---|---|---|---|
| 1 | `CN=Huawei Integrity Root CA - G2` | TRUE | Certificate Sign, CRL Sign | **root** |
| 2 | `CN=Huawei Root CA` (O=Huawei) | TRUE | Certificate Sign, CRL Sign | **root** |
| 3 | `CN=Huawei Integrity CA 1 - G2` | TRUE, **pathlen:0** | Certificate Sign, CRL Sign | **intermediate CA** |
| 4 | `CN=Huawei CSI RSA-PSS Code-Signing Authority 2` | FALSE | Digital Signature | leaf (code signing) |
| 5 | `CN=Huawei Time-Stamping Authority 2 - G2` | FALSE | Digital Signature, Non Repudiation | leaf (timestamps) |

Fingerprints (sha256):

```text
  1  D7:3E:F1:71:39:5D:67:F3:46:C8:99:C0:10:F1:32:03:90:70:85:7A:C9:64:49:EC:B4:45:6F:0D:EE:F6:BB:80
  2  A9:D5:49:E2:14:C1:F8:A2:97:07:90:82:C8:EC:F3:6B:EA:53:9D:AB:A3:1C:7D:56:88:AA:BB:C7:AB:9C:CB:C2
  3  61:B5:7E:BA:58:F8:1C:70:EE:19:2F:23:73:0B:8F:DB:43:66:E4:00:FD:9F:21:7F:8B:47:74:44:E0:03:94:EF
  4  53:42:09:69:86:38:C5:C3:E3:83:50:92:1D:19:8C:FA:B3:7F:00:AA:60:E1:F6:7F:23:56:9B:2F:5B:82:B9:7D
  5  75:EC:8C:34:D3:B3:ED:4C:D1:B9:1B:AD:E3:17:3F:AA:DA:39:9F:86:AE:1F:13:AB:7B:3E:7D:0B:1D:24:34:C9
```

## What the stock package's chain looks like

The `Ascend310P-aicpu_legacy.tar.gz` CMS blob carries two certificates and
chains to the **G2** hierarchy:

```text
  Computing PL Code-Signing Authority 2 - G2      (leaf, Digital Signature)
        issuer: Huawei Integrity CA 1 - G2          (= anchor #3)
  Huawei Integrity CA 1 - G2                     (intermediate, pathlen:0)
        issuer: Huawei Integrity Root CA - G2      (= anchor #1, NOT shipped)
```

So the package's chain terminates at anchor **#1**
(`Huawei Integrity Root CA - G2`), and the intermediate **#3** is what
actually signs the leaf.

Note the *other* package hierarchy that appears in the device's store but
**not** in this package's chain — the legacy one:

```text
  Huawei CSI RSA-PSS Code-Signing Authority 2       (= anchor #4, leaf)
        issuer: Huawei Integrity CA 1 - G2
  Huawei Root CA (O=Huawei)                        (= anchor #2)
```

and `hsm_firmware_update.ko` additionally carries
`+Huawei CSI RSA-PSS Code-Signing Authority 20` and
`+Huawei Certification Authority 1&0$` — i.e. a *second* code-signing
authority with a different OID, which is the one the **kernel** PSS check
(`soc_verify` → `rsa_verify_soc`) uses.

**So there are two independent hierarchies, and they are used by two
different verifiers:**

| hierarchy | root | intermediate | leaf | used by |
|---|---|---|---|---|
| G2 | #1 `Huawei Integrity Root CA - G2` | #3 `Huawei Integrity CA 1 - G2` | `Computing PL Code-Signing Authority 2 - G2` | tsdaemon CMS check (`VerifyAicpuPkgByCms`) |
| legacy | #2 `Huawei Root CA` | #3 `Huawei Integrity CA 1 - G2` (or its own) | #4 `Huawei CSI RSA-PSS Code-Signing Authority 2` | kernel PSS check (`soc_verify`) |

Both share intermediate **#3**, which is a `pathlen:0` CA that can sign.

## The two runtime trust stores (host side)

The vendor's `dms/custom/dms_custom_device.c` writes *two* anchors into the
runtime store at init:

```c
  STATIC unsigned char g_pbRootCert[] = { ... };          /* 3021 bytes */
      subject=CN=RSA-PSS Intergity Root CA 2
      notBefore=Oct 17 2025   notAfter=Oct 17 2075
      Key Usage: critical, Certificate Sign, CRL Sign
      Basic Constraints: critical, CA:TRUE
      sha256: E8:7C:32:6E:D3:97:29:CC:FC:92:39:5C:F4:F6:0F:0D:00:92:AB:23:53:F6:18:11:D2:E3:1A:9F:B0:28:9A:67

  STATIC unsigned char g_community_certificate[] = { ... }; /* 1728 bytes */
      subject=C=CN, ST=Guangdong, L=Shenzhen, O=Hisilicon, OU=Turing,
               CN=Huawei hisilicon ROOT CA
      notBefore=Nov 13 2025   notAfter=Nov 13 2055
      Key Usage: critical, Digital Signature, Certificate Sign, CRL Sign
      Basic Constraints: critical, CA:TRUE
      sha256: 3B:19:CA:67:9E:07:55:6D:58:C1:B1:EC:0C:32:EF:86:5C:D6:1B:9C:9D:64:AF:11:D0:6D:BC:33:69:00:ED:6E
```

These are the `vendor-certificate` and `community-certificate` items, and
the community root is the **`Huawei hisilicon ROOT CA`** — a *community* CA
meant for third-party signers, and it has `Certificate Sign` usage. That is
precisely the root a "customer certificate" would chain to, and it is the
reason the `SIGN_FLAG_*` policy enum has `ONLY_CMTY`, `VENDOR_OR_CMTY`,
`USER_OR_CMTY` values at all.

Neither of these two runtime anchors is in the `drv_pkicms.ko` store — they
are *additional*, installed at runtime, and the 310P build omits the module
that installs them.

## What this means

**Nothing here is a cryptographic weakness.** The PSS parameters are the
strong ones:

```text
  RSA 4096, e = 65537
  hash            : SHA-256
  MGF1            : SHA-256
  salt length     : 32 (== digest length)
  trailer field   : 0xbc (default)
```

and the known PSS pitfalls (salt-length mismatch, PKCS#1 v1.5 confusion,
low-exponent forgery, hash truncation) do not apply.

**What it does mean is that the trust decision is a *policy* decision, and
the policy is partly ours to set.** Specifically:

1. The device holds a **community root** (`Huawei hisilicon ROOT CA`) whose
   job is to trust third-party signers. The `verify_flag` policy file
   selects which of {vendor, user, community} is honoured — and the device's
   default is `1` (`SIGN_FLAG_ONLY_VENDOR`).
2. If the flag were `4` (`ONLY_CMTY`), `5` (`VENDOR_OR_CMTY`) or `6`
   (`USER_OR_CMTY`), a package signed by a certificate chaining to the
   community root would be accepted — and the community root's *private* key
   is not something we have, but a **certificate** chaining to it is
   exactly what the `user-certificate` item is for.
3. The previous session's `install_user_cert.py` already built that
   machinery: it fills `user-certificate` with our own root and sets the
   flag. It targets `/var/asdrv/dev<N>/device-sw-plugin/verify/`, which is
   the **runtime** store on the device.

**And this is the first time the two halves line up.** The `verify_flag`
policy file, the `user-certificate` item, and the community root are three
parts of *one* mechanism — the runtime trust store — and the store's
location (`/var/asdrv/...`) is the same path the device-side
`dms_custom_device.c` reads and writes. The 310P build omits the writer, but
the *reader* is present in `drv_pkicms.ko` and in `libtsdaemon.so`.

So the question that decides everything is unchanged but now much sharper:

> **can we get a file into `/var/asdrv/dev0/device-sw-plugin/verify/` on
> the device?**

If we can write `flag` = `4` (community only) and `user-certificate` = our
own CA, then a package signed by *us* passes the tsdaemon CMS check — and
the whole signature wall comes down without touching the vendor's keys.

The route to that file is what the last few sessions have been circling:
the HDC file channel reaches the device's staging directory but the device
normalises paths, and the runtime store is in `/var/asdrv`, not in the
staging tree.

## Two notes on the wider question

**Does the trust store expose "other signers"?** Yes — deliberately. It
contains a community root and an intermediate CA with signing capability.
That is the vendor's design: an AICPU platform where third parties can
submit signed kernels *if the operator installs a customer CA and sets the
policy flag*. The 310P driver build omits the module that does the
installing, which is why the mechanism exists in the hardware but is
unavailable on this build.

**Is the `pathlen:0` on anchor #3 significant?** It means that CA can sign
leaf certificates but **cannot** create further subordinate CAs. So a
customer certificate would be a leaf under #3, and there is no path to a
"customer CA that can sign customer CAs". That is a correctly scoped
hierarchy, and it means the community-root route is the one with the
right shape for third-party signing.

## Card state

Unchanged by this analysis: `pkcs-enable` Disable on all three cards,
whitelist / package-load ini / host AICPU package at their original md5s,
6/6 chips OK, 0 errors, device log level INFO.
