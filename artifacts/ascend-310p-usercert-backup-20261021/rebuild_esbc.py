#!/usr/bin/env python3
"""Byte-exact rebuild of the Ascend 310P ESBC image trailer.

Verified against stock Ascend310P.image (31,350,770 B, sha256 of payload
93af6abd...): a --verify run must reproduce the input bit-for-bit.

Layout (offsets are from each 0x1000 header copy):

    file 0x0000        header copy 0
    file 0x1000        header copy 1 (img_offset field is 0x1000)
    0x2000             payload, code_len bytes (field @ copy+0x45C)
    cms_start = 0x2000 + code_len + (32 - code_len % 16)
    ini_start = cms_start + 0x4000
    crl_start = ini_start + 0x0800

Section header: 12-byte tag, NUL padded, u32 LE length at +12.

Integrity fields:
    copy+0x08    u32 certtype (stock 0xFFFFFFFF = Huawei)
    copy+0x2C    32B SHA-256 of payload
    copy+0x45C   u32 code_len
    copy+0x4BC   32B SHA-256 of copy[0:0x4BC]
    0x04E0       u64 0xABCD1234AA55AA55, u32 total file size at 0x04E8
                 (copy 0 only; copy 1 keeps this area zeroed)

certtype values from image_pack.py: 0x1 community, 0x2 client,
0xFFFFFFFF Huawei.
"""

import argparse
import hashlib
import struct
import sys

MAGIC_EXTERN = 0xABCD1234AA55AA55
CMS_SLOT = 0x4000
INI_SLOT = 0x0800
HEAD_HASH_OFF = 0x4BC
HEAD_HASH_END = 0x4DC
CERTTYPE_OFF = 0x08
DIGEST_OFF = 0x2C
CODE_LEN_OFF = 0x45C
EXTERN_OFF = 0x4E0
IMG_OFF = 0x2000


def sha256(data):
    return hashlib.sha256(data).digest()


def section_header(tag, length):
    raw = tag.encode("ascii")
    if len(raw) > 12:
        raise ValueError(f"tag too long: {tag!r}")
    return raw.ljust(12, b"\0") + struct.pack("<I", length)


def align(c):
    return 32 - c % 16


def parse(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < IMG_OFF:
        raise ValueError("file too short")
    code_len = struct.unpack_from("<I", data, CODE_LEN_OFF)[0]
    cms_start = IMG_OFF + code_len + align(code_len)
    ini_start = cms_start + CMS_SLOT
    crl_start = ini_start + INI_SLOT

    def section(start, name):
        tag = data[start : start + 12].rstrip(b"\0")
        if tag != name.encode():
            raise ValueError(f"{name} expected at 0x{start:x}, found {tag!r}")
        length = struct.unpack_from("<I", data, start + 12)[0]
        body = data[start + 16 : start + 16 + length]
        if len(body) != length:
            raise ValueError(f"{name} truncated at 0x{start:x}")
        return body

    return {
        "data": data,
        "header": data[:IMG_OFF],
        "payload": data[IMG_OFF : IMG_OFF + code_len],
        "code_len": code_len,
        "certtype": struct.unpack_from("<I", data, CERTTYPE_OFF)[0],
        "cms": section(cms_start, "cms"),
        "ini": section(ini_start, "ini"),
        "crl": section(crl_start, "crl"),
        "cms_start": cms_start,
        "ini_start": ini_start,
        "crl_start": crl_start,
    }


def parse_salvage(path):
    """Parse an image whose trailer was placed with a stale alignment.

    Locates the cms/ini/crl section headers by tag instead of trusting the
    derived offset, then rebuilds the geometry from the real payload length.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    code_len = struct.unpack_from("<I", data, CODE_LEN_OFF)[0]
    payload = data[IMG_OFF : IMG_OFF + code_len]
    if len(payload) != code_len:
        raise ValueError("payload truncated")

    lo = max(IMG_OFF + code_len, len(data) - 0x10000)
    cms_start = None
    pos = data.find(b"cms\0", lo)
    while pos != -1:
        ini_start = pos + CMS_SLOT
        crl_start = ini_start + INI_SLOT
        if data[ini_start : ini_start + 4] == b"ini\0" and data[crl_start : crl_start + 4] == b"crl\0":
            cms_start = pos
            break
        pos = data.find(b"cms\0", pos + 1)
    if cms_start is None:
        raise ValueError("no consistent cms/ini/crl trailer found")

    def section(start):
        length = struct.unpack_from("<I", data, start + 12)[0]
        body = data[start + 16 : start + 16 + length]
        if len(body) != length:
            raise ValueError(f"section at 0x{start:x} truncated")
        return body

    ini_start = cms_start + CMS_SLOT
    crl_start = ini_start + INI_SLOT
    return {
        "data": data,
        "header": data[:IMG_OFF],
        "payload": payload,
        "code_len": code_len,
        "certtype": struct.unpack_from("<I", data, CERTTYPE_OFF)[0],
        "cms": section(cms_start),
        "ini": section(ini_start),
        "crl": section(crl_start),
        "cms_start": cms_start,
        "ini_start": ini_start,
        "crl_start": crl_start,
    }


def build(image, payload=None, cms=None, ini=None, crl=None, certtype=None):
    payload = image["payload"] if payload is None else payload
    cms = image["cms"] if cms is None else cms
    ini = image["ini"] if ini is None else ini
    crl = image["crl"] if crl is None else crl
    certtype = image["certtype"] if certtype is None else certtype

    if len(cms) + 16 > CMS_SLOT:
        raise ValueError("cms does not fit in the 16 KiB slot")
    if len(ini) + 16 > INI_SLOT:
        raise ValueError("ini does not fit in the 2 KiB slot")

    code_len = len(payload)
    out = bytearray(image["header"])
    out += payload
    out += b"\0" * align(code_len)

    cms_start = len(out)
    out += section_header("cms", len(cms)) + cms
    out += b"\0" * (cms_start + CMS_SLOT - len(out))

    ini_start = len(out)
    assert ini_start == cms_start + CMS_SLOT
    out += section_header("ini", len(ini)) + ini
    out += b"\0" * (ini_start + INI_SLOT - len(out))

    crl_start = len(out)
    assert crl_start == ini_start + INI_SLOT
    out += section_header("crl", len(crl)) + crl

    digest = sha256(payload)
    for base in (0, 0x1000):
        struct.pack_into("<I", out, base + CERTTYPE_OFF, certtype)
        out[base + DIGEST_OFF : base + DIGEST_OFF + 32] = digest
        struct.pack_into("<I", out, base + CODE_LEN_OFF, code_len)
    for base in (0, 0x1000):
        out[base + HEAD_HASH_OFF : base + HEAD_HASH_END] = sha256(bytes(out[base : base + HEAD_HASH_OFF]))
    struct.pack_into("<QI", out, EXTERN_OFF, MAGIC_EXTERN, len(out))
    return bytes(out), digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("-o", "--out")
    ap.add_argument("--payload", help="replacement payload (bytes from 0x2000)")
    ap.add_argument("--cms")
    ap.add_argument("--ini")
    ap.add_argument("--crl")
    ap.add_argument("--certtype", type=lambda s: int(s, 0))
    ap.add_argument("--salvage", action="store_true", help="locate the trailer by tag instead of the derived offset")
    ap.add_argument("--header-from", help="take the 4 KiB header template from another image")
    ap.add_argument("--tag", default="initrd", help="container tag for --regen-ini")
    ap.add_argument("--regen-ini", action="store_true", help="regenerate ini as '<tag>,   <sha256>;\\n'")
    a = ap.parse_args()

    image = parse_salvage(a.src) if a.salvage else parse(a.src)
    if a.header_from:
        with open(a.header_from, "rb") as fh:
            image["header"] = fh.read()[:IMG_OFF]
    if a.payload:
        with open(a.payload, "rb") as fh:
            payload = fh.read()
    else:
        payload = None
    if a.regen_ini:
        base = payload if payload is not None else image["payload"]
        ini = f"{a.tag},   {hashlib.sha256(base).hexdigest()};\n".encode()
    else:
        if a.ini:
            with open(a.ini, "rb") as fh:
                ini = fh.read()
        else:
            ini = None
    if a.cms:
        with open(a.cms, "rb") as fh:
            cms = fh.read()
    else:
        cms = None
    if a.crl:
        with open(a.crl, "rb") as fh:
            crl = fh.read()
    else:
        crl = None

    out, digest = build(image, payload, cms, ini, crl, a.certtype)
    code_len = image["code_len"] if payload is None else len(payload)
    certtype = a.certtype if a.certtype is not None else image["certtype"]
    print(f"code_len {code_len} (0x{code_len:x}), digest {digest.hex()}, certtype 0x{certtype:x}, size {len(out)}")

    if a.out:
        with open(a.out, "wb") as f:
            f.write(out)
        print("wrote", a.out)
    else:
        ok = out == image["data"]
        print("verify:", "byte-identical" if ok else "MISMATCH")
        if not ok:
            diffs = [i for i, (x, y) in enumerate(zip(out, image["data"])) if x != y]
            print(
                "first diffs", [hex(i) for i in diffs[:16]], "count", len(diffs), "size", len(out), len(image["data"])
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
