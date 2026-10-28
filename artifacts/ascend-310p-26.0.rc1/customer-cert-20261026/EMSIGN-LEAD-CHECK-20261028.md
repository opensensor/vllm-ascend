# The emSign lead, and where the device's package trust store really comes from

**Date:** 2026-10-28
**Status:** the published-list lead was checked and **does not apply** to the
package verifier. The decisive mechanism is identified.

## The lead, and why it does not reach the package path

We were pointed at Huawei's published *Trusted Root Certificate* list
(20251230) and the note that an Indian CA used `4` as a private key.

**The list does contain `emSign Root CA - G1`,** with exactly the
fingerprint quoted:

```text
  emSign Root CA - G1   40F6AF0346A99AA1CD1D555A4E9CCE62C7F9634603EE406615833DC8C8D00367
```

and the context is the eMudhra incident reported on 28 Oct 2026: Microsoft
removed six eMudhra "G3" roots plus a Sectigo cross-sign on 26-27 Oct, Google
removed `emSign ECC Root CA - G3` from Chrome, and **Mozilla has not removed
G1** because the weak keys were in the G3 intermediates. G1 is the "other"
eMudhra root still trusted by browsers.

The list has 125 fingerprints. Checks performed against the card:

| set | count | contains emSign G1? |
|---|---|---|
| Huawei's published list (20251230) | 125 | **yes** |
| the **host** system CA bundle (`/etc/ssl/certs/ca-certificates.crt`) | — | **yes** (via `/usr/share/ca-certificates/mozilla/emSign_Root_CA_-_G1.crt`, fingerprint matches exactly) |
| the **device** system CA bundle (`/etc/ssl/certs/ca-certificates.crt`) | 120 | **no** |
| the device's crypto module anchors (`drv_pkicms.ko`) | 5 | **no** |
| the device's runtime anchors (vendor + community, installed by `dms_custom_device.c`) | 2 | **no** |

**None of the device's seven anchors is on Huawei's published list**, and the
published list is not what the device's package verifier consults.

## Where the device's package trust store actually comes from

`libtsdaemon.so` has exactly three cert-file paths and one policy file:

```text
  /var/asdrv/dev<N>/device-sw-plugin/verify/vendor-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/user-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/community-certificate
  /var/asdrv/dev<N>/device-sw-plugin/verify/flag          <- the policy
```

(the prefix literal is `/var/asdrv/dev` and the suffix
`/device-sw-plugin/verify`, in `CmsCbbVerifyPkg::GetVerifyFlagFilePath` and
friends.)

**The system CA bundle is not in this path.** `libtsdaemon.so` contains no
reference to `ca-certificates.crt`, `/etc/ssl/certs`, `ca-bundle`,
`X509_STORE_set_default_paths` or `set_default_paths`. The one
`SSL_CERT_FILE` string is OpenSSL's own compiled-in default
(`.../euleros2.8/aarch64/openssl/ssl/cert.pem`), and `SSL_CERT_DIR` is
beside it — those are the OpenSSL build's defaults, not the package
verifier's store. The package verification path builds its PKI store
explicitly, via `CmscbbPkiStoreAddCert` (`+0x17cf38`), from the three
cert files plus the module's own compiled-in anchors.

**Consequence: adding a root to the device's OS trust store would not make a
package verify.** The store that matters is the runtime one under
`/var/asdrv/...`, and it is exactly the store the 310P build has no writer
for.

## Why that makes the published list interesting after all — but not usable

The list is still worth having, for a different reason than the password
reset. It is the set of roots Huawei has *published* as trusted, and the
device's runtime community root (`CN=Huawei hisilicon ROOT CA`,
O=Hisilicon, OU=Turing) is **not** on it. Neither is the vendor's
`CN=RSA-PSS Intergity Root CA 2`.

So the device's runtime trust store is populated from two roots that are
**not** in Huawei's own published list of trusted roots. That is a fact
about the platform's trust model worth recording, and it is the reason the
`verify_flag` policy and the `user-certificate` item exist as a pair: the
"user" slot is where a customer CA that chains to the community root goes,
and the community root is the one designed for third-party signers.

**It is not a usable forgery route.** Using a publicly-listed root whose
private key is known would require the *device* to trust that root — and it
does not. The device trusts five specific anchors plus the two runtime
ones, none of which is emSign G1 or any other publicly-compromised root.
The published list and the device's trust store are simply different sets.

## The state of the routes

| route | status |
|---|---|
| Forge a package signature with a compromised public root | **closed** — the device does not trust any publicly-compromised root, and does not use the OS CA bundle for package verification |
| Forge a package signature with a vendor key | closed — we do not have it |
| Use the *runtime* trust store: set `flag` to allow community/user certs and install our own CA as `user-certificate` | **still open**, and unchanged — it needs a write to `/var/asdrv/dev<N>/device-sw-plugin/verify/` on the device |
| Kernel PSS check (`soc_verify`) | controlled by `npu-smi set -t pkcs-enable`, which is a **RAM-only** switch (`g_sign_type` at `.bss+0x98`), so it is safe to experiment with |

## What I did not do

- Did not add anything to the device's OS CA store, and did not attempt to
  make the device trust a publicly-listed root. Neither would affect the
  package verifier, and the first would be a gratuitous change to the
  card's configuration.
- Did not install the `emSign` root anywhere.
- Did not modify the runtime trust store.
- `pkcs-enable` remains `Disable` on all three cards, and the whitelist,
  package-load ini and host AICPU package are at their original md5s.

## Card state

- `pkcs-enable`: `Disable` on cards 9, 8217, 16409 (original).
- `ssh-enable`: `True` on 9 and 8217 (the previous session's change, left as
  found), `False` on 16409.
- `/etc/hdcFileTransWhiteList.cfg`: md5 `7baeaca596f22c7ddb711ba91b2bdb2b`.
- `ascend_package_load.ini`: md5 `baca3a99c7034f140c2e7ce89d42b4c0`.
- host AICPU package: md5 `ccf7ed4c6cef44e7e0dac5c2708cf849`.
- 6/6 chips OK, 0 errors, device log level INFO.
