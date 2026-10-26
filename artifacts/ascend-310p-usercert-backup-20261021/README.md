# Ascend 310P user-cert (self-made CA) — 2026-10-21

## What happened

`upgrade-tool --device_index 0 --user_cert --path user.der` accepted a hand-built
esbc envelope containing a self-signed RSA-4096-PSS root CA, with **no Huawei
`usrcert.ini.cms` signature**. Output:

```
Device[0] set user certificate info success.
```

The chip's "customer cert" trust is now bound to this CA. If the hash lands in an
OTP / customer efuse, these private keys are the **only** way to sign a rootfs this
chip will accept. Treat `rootca_priG2_pri.pem` as the crown jewel.

## Files

| File | Purpose |
|------|---------|
| `rootca_priG2_pri.pem` | **root CA PRIVATE KEY — keep secret, mode 600** |
| `rootcaG2.pem`, `rootcaG2.der` | root CA certificate (DER + PEM) |
| `signcert_pri.pem` | **signing PRIVATE KEY — keep secret, mode 600** |
| `signcert.pem`, `signcert.der` | leaf signing cert, chains to `rootcaG2` |
| `user.der` | exact esbc envelope accepted by `--user_cert` |
| `x509_rootcaG2.cnf`, `x509_signcert.cnf` | OpenSSL configs |
| `build_user_der.py` | reproduces `user.der` from a DER cert |

## SHA-256

```
023f6057aad559a70310e700ee2fa277383eb5ac201ea8654bfbfd7d294179fe  rootcaG2.der
39b31e85aa1416b82df6fb4eb137eb860d747e585804e46d3135ebbb71e88aea  rootcaG2.pem
4d5ce7ea215b08f39cdf44cfb59436ae153351ad3a24eab18ae9e62ac9b36548  rootca_priG2_pri.pem
c7806bb6c8c863f668ee414f43111293a0ae61c95a3579f26768ac805b553c48  user.der
3247c98ebd57de31eab4372784c431fba980911932476f74f1fd06e98d3e5f83  x509_rootcaG2.cnf
089238e8d1186a6be4b0d77782a8e9fbfaef9436441d8a6fecc54f102b97b135  signcert.der
5aa2119da1df3ce4b6cc05f6cc2a2ac247e47c36d7f3080d03f7bb140966ca8c  signcert_pri.pem
```

Signing chain: `signcert.pem` → `rootcaG2.pem` (`openssl verify -CAfile rootcaG2.pem signcert.pem`
returns OK). Sign the rootfs CMS (`initrd.ini.p7s`) with `signcert_pri.pem`; the chip trusts
`rootcaG2` as the user cert, so the leaf chains to it.

## esbc envelope format (reversed 2026-10-21)

- `digest @ 0x2c = SHA256(data[off : off+len])`
- `off  = u32 LE @ 0x458` (normally `0x2000`)
- `len  = u32 LE @ 0x45c` (covers tag section + payload)
- header `0x0..0x2000` contains two `0x1000` copies; the second copy's range fields
  (`0x1458`, `0x145c`) and digest (`0x102c`) must be updated too.
- tag section `0x2000..0x2100`; tag string at `0x2004` (`usrcert`).
- payload starts at `0x2100`.

## Device state at backup time

- device 0: user cert = this CA; `boot_status 16` (OS up); `npu-smi` OK.
- stock rootfs still boots: built-in Huawei cert is checked first, user cert is the
  fallback.
- dev guide: `VERSION_ID=26.04` (Ubuntu 26.04.1), driver `26.0.rc1`.

## TODO (to finish the bypass)

Re-sign `Ascend310P.cpio.gz` with this CA (CMS `initrd.ini.p7s` + `initrd.ini` +
CRL trailer), so the built-in cert fails and the user cert is used at boot.
