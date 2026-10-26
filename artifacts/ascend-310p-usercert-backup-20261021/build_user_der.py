#!/usr/bin/env python3
"""Rebuild the esbc 'usrcert' envelope (user.der) with all header fields valid.

Layout (verified against the stock 310P images):
    header  0x0000 / 0x1000   head_len 0x4DC, head_hash @+0x4BC
    body    0x2000            0x100 tag block (tag 'usrcert') + DER cert
    digest  +0x2C             SHA-256(body)
    code_len +0x45C           len(body)
    certtype +0x08            0x1 community, 0x2 client, 0xFFFFFFFF Huawei
    0x4E0                     u64 0xABCD1234AA55AA55, u32 total size @0x4E8
    head_hash +0x4BC          SHA-256(header[base:base+0x4BC])

Usage:
    python3 build_user_der.py [template] [cert.der] [out.der] [certtype]
"""

import hashlib
import struct
import sys

ASSUMED_TEMPLATE = "/usr/local/Ascend/driver/device/Ascend310P.image"
MAGIC_EXTERN = 0xABCD1234AA55AA55

template = sys.argv[1] if len(sys.argv) > 1 else ASSUMED_TEMPLATE
cert_path = sys.argv[2] if len(sys.argv) > 2 else "rootcaG2.der"
out_path = sys.argv[3] if len(sys.argv) > 3 else "user.der"
certtype = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x1

with open(template, "rb") as fh:
    d = fh.read()
with open(cert_path, "rb") as fh:
    cert = fh.read()

hdr = bytearray(d[:0x2000])
tag = bytearray(d[0x2000:0x2100])
tag[0x04:0x14] = b"usrcert" + b"\0" * 9
body = bytes(tag) + cert
digest = hashlib.sha256(body).digest()

for base in (0, 0x1000):
    struct.pack_into("<I", hdr, base + 0x08, certtype)
    struct.pack_into("<I", hdr, base + 0x45C, len(body))
    hdr[base + 0x2C : base + 0x4C] = digest
for base in (0, 0x1000):
    hdr[base + 0x4BC : base + 0x4DC] = hashlib.sha256(bytes(hdr[base : base + 0x4BC])).digest()

out = bytearray(hdr) + body
struct.pack_into("<QI", out, 0x4E0, MAGIC_EXTERN, len(out))
with open(out_path, "wb") as fh:
    fh.write(out)
print(f"wrote {out_path} ({len(out)} bytes, certtype 0x{certtype:x}, digest {digest.hex()}, size {len(out)})")
