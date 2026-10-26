#!/usr/bin/env python3
"""Parse and validate an Ascend esbc container.

Format derived from Huawei's own `Ascend_310P_check.bin` (which parses and
verifies these containers) and cross-checked against every image we hold.

OUTER HEADER  (two copies at 0x0000 and 0x1000)

    +0x000  u32   magic          = 0x55aa55aa
    +0x004  u32   head_len       = 0x4dc
    +0x008  u32   cert_type      = 0xffffffff (Huawei) or 0x1 (user)
    +0x02C  u8[32] payload_sha256 = SHA-256(payload)
    +0x04C  u32   sec_blk_size   = 0x500
    +0x050  u32   sec_hdr_off    = 0x108
    +0x054  u32   sec_key_size   = 0x200
    +0x058  u8[sec_blk_size]  security block (zero unless the image is a signed image)
    +0x4BC  u8[32] head_sha256   = SHA-256(header[0x0000:0x04BC])
    +0x458  u32   img_offset     = 0x2000
    +0x45C  u32   code_len       = payload length

PAYLOAD

    +0x000  u32   magic  = 0x33aa3a3a
    +0x004  char[8] tag   ("initrd", "uimage", "dtimage", "hboot2", "usrcert", ...)
    +0x008  u32   tag tail (the last 4 bytes of the 12-byte tag, e.g. "dr" of "initrd")
    +0x100  body  (gzip for initrd, LZMA for uimage, DER for usrcert, FV for hboot2)

SECTION TABLE  (immediately after the payload)

    The payload is followed by a small gap and then a chain of
        tag[12] len u32 body
    entries: cms, ini, crl.

    The gap size is  slot = 0x20 - (code_len & 0xf)
    and the section table proper starts at
        cms_start = img_offset + code_len + slot
    Each entry is 16 bytes of header (tag[12] + len u32) plus its body,
    and the entries are chained.

    ini body = "<tag>,   <sha256(payload)>;"

Usage:
  esbc_info.py <image>            # dump the structure
"""

import hashlib
import re
import struct
import sys

OUTER_MAGIC = 0x55AA55AA
INNER_MAGIC = 0x33AA3A3A


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def parse(path):
    with open(path, "rb") as fh:
        d = fh.read()
    info = {"file": path, "size": len(d)}
    if len(d) < 0x2000 or u32(d, 0) != OUTER_MAGIC:
        info["error"] = "not an esbc container"
        return info, None
    info["head_len"] = u32(d, 4)
    info["cert_type"] = u32(d, 8)
    info["img_offset"] = u32(d, 0x458)
    info["code_len"] = u32(d, 0x45C)
    info["sec_blk"] = (u32(d, 0x4C), u32(d, 0x50), u32(d, 0x54))

    off, ln = info["img_offset"], info["code_len"]
    payload = d[off : off + ln]
    info["payload_sha_stored"] = d[0x2C:0x4C].hex()
    info["payload_sha_actual"] = hashlib.sha256(payload).hexdigest()
    info["head_sha0_stored"] = d[0x4BC:0x4DC].hex()
    info["head_sha0_actual"] = hashlib.sha256(d[:0x4BC]).hexdigest()
    info["head_sha1_stored"] = d[0x14BC:0x14DC].hex()
    info["head_sha1_actual"] = hashlib.sha256(d[0x1000:0x14BC]).hexdigest()

    info["inner_magic_ok"] = u32(payload, 0) == INNER_MAGIC
    info["inner_tag"] = payload[4:12].rstrip(b"\x00").decode("ascii", "replace")
    info["inner_08"] = u32(payload, 8)
    info["inner_tag_tail"] = payload[8:12].rstrip(b"\x00").decode("ascii", "replace")

    slot = 0x20 - (ln & 0xF)
    info["slot"] = slot
    cms_start = off + ln + slot
    info["cms_start"] = cms_start

    secs = {}
    p = cms_start
    for _ in range(4):
        if p + 16 > len(d):
            break
        tag = d[p : p + 12].rstrip(b"\x00").decode("ascii", "replace")
        length = u32(d, p + 12)
        if not tag:
            break
        secs[tag] = {"hdr": p, "off": p + 16, "len": length}
        p += 16 + length
    info["sections"] = secs
    if "ini" in secs:
        info["ini_body"] = d[secs["ini"]["off"] : secs["ini"]["off"] + secs["ini"]["len"]]
    return info, d


def main():
    path = sys.argv[1]
    info, d = parse(path)
    print(f"=== {path}  ({info['size']} bytes) ===")
    if "error" in info:
        print("  ", info["error"])
        return 1
    ok = lambda a, b: "OK" if a == b else "MISMATCH"
    print(f"  head_len     : 0x{info['head_len']:x}")
    print(f"  cert_type    : 0x{info['cert_type']:08x}")
    print(f"  img_offset   : 0x{info['img_offset']:x}")
    print(f"  code_len     : 0x{info['code_len']:x} ({info['code_len']})")
    blk = info["sec_blk"]
    print(f"  sec_blk      : size=0x{blk[0]:x} hdr_off=0x{blk[1]:x} key_size=0x{blk[2]:x}")
    print(f"  payload sha  : {ok(info['payload_sha_stored'], info['payload_sha_actual'])}")
    print(f"  head sha c0  : {ok(info['head_sha0_stored'], info['head_sha0_actual'])}")
    print(f"  head sha c1  : {ok(info['head_sha1_stored'], info['head_sha1_actual'])}")
    magic_ok = "OK" if info["inner_magic_ok"] else "BAD"
    print(f"  inner magic  : {magic_ok}  tag={info['inner_tag']!r}  tail={info['inner_tag_tail']!r}")
    print(f"  slot         : 0x{info['slot']:x}   cms_start = 0x{info['cms_start']:x}")
    for n, s in info["sections"].items():
        print(f"    {n:4s} len={s['len']:8d} @0x{s['off']:x}")
    if "ini_body" in info:
        ib = info["ini_body"].decode("ascii", "replace").strip()
        print(f"  ini body     : {ib!r}")
        m = re.search(r"([0-9a-f]{64})", ib)
        if m:
            print(f"  ini digest   : {ok(m.group(1), info['payload_sha_actual'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
