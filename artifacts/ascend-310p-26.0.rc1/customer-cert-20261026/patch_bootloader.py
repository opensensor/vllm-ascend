#!/usr/bin/env python3
"""Patch an Ascend 310P bootloader image (AS610_HBOOT2_UEFI.fd / Ascend310P.fd).

The bootloader image is an esbc container like the other images, but with a
crucial difference: **it has no `cms` / `ini` / `crl` sections**, so nothing
external signs it.  It is protected only by its own self-consistency:

    0x0000  header copy 0
    0x1000  header copy 1
    0x2000  payload (code_len bytes)
            payload = inner esbc container: 0x100-byte header + body

    @0x02C  SHA-256(payload)                       (both header copies)
    @0x04BC  SHA-256(header[0x0000:0x04BC])        (copy 0)
    @0x14BC  SHA-256(header[0x1000:0x14BC])       (copy 1)

So the image can be modified and re-checksummed without any key.  This is the
mechanism the 2026-10-21 session used to produce
`AS610_HBOOT2_UEFI.fd.patched` (which was never installed).

Usage:
    patch_bootloader.py <in.fd> <out.fd> [--set-byte OFF:HEX]... [--dump]

`--set-byte OFF:HEX` takes a file offset and a byte value; it may be repeated.
Offsets are absolute file offsets.  Dumps the digest self-check before and
after.
"""

import argparse
import hashlib
import struct
import sys

IMG_OFF_FIELD = 0x458
CODE_LEN_FIELD = 0x45C
DIGEST_FIELD = 0x2C
HDR_CHECK_FIELD = 0x4BC
COPY_SIZE = 0x1000


def u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def checks(data):
    off = u32(data, IMG_OFF_FIELD)
    ln = u32(data, CODE_LEN_FIELD)
    return {
        "img_off": off,
        "code_len": ln,
        "payload_digest_ok": data[DIGEST_FIELD : DIGEST_FIELD + 32] == hashlib.sha256(data[off : off + ln]).digest(),
        "hdr0_check_ok": data[HDR_CHECK_FIELD : HDR_CHECK_FIELD + 32]
        == hashlib.sha256(data[0:HDR_CHECK_FIELD]).digest(),
        "hdr1_check_ok": data[COPY_SIZE + HDR_CHECK_FIELD : COPY_SIZE + HDR_CHECK_FIELD + 32]
        == hashlib.sha256(data[COPY_SIZE : COPY_SIZE + HDR_CHECK_FIELD]).digest(),
    }


def reseal(data):
    """Recompute the payload digest and both header self-checksums, in order."""
    off = u32(data, IMG_OFF_FIELD)
    ln = u32(data, CODE_LEN_FIELD)
    payload_digest = hashlib.sha256(data[off : off + ln]).digest()
    for copy in (0, COPY_SIZE):
        data[copy + DIGEST_FIELD : copy + DIGEST_FIELD + 32] = payload_digest
    # the header checksum covers the header up to (but not including) its own
    # field, and therefore includes the payload digest we just wrote
    for copy in (0, COPY_SIZE):
        base = copy + HDR_CHECK_FIELD
        data[base : base + 32] = hashlib.sha256(data[copy : copy + HDR_CHECK_FIELD]).digest()
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument(
        "--set-byte",
        action="append",
        default=[],
        metavar="OFF:HEX",
        help="set a byte: OFF (decimal or 0xHEX) : HEX byte value",
    )
    ap.add_argument("--dump", action="store_true", help="print the structure and exit")
    args = ap.parse_args()

    with open(args.infile, "rb") as fh:
        data = bytearray(fh.read())

    if args.dump:
        c = checks(data)
        off = u32(data, IMG_OFF_FIELD)
        ln = u32(data, CODE_LEN_FIELD)
        print(f"file        : {args.infile} ({len(data)} bytes)")
        print(f"preamble    : 0x{u32(data, 0):08x}")
        print(f"head_len    : 0x{u32(data, 4):x}")
        print(f"certtype    : 0x{u32(data, 8):08x}")
        print(f"img_off     : 0x{off:x}")
        print(f"code_len    : 0x{ln:x} ({ln})")
        print(f"payload[0:16]: {bytes(data[off : off + 16]).hex()}")
        print("checks      :")
        for k, v in c.items():
            print(f"   {k:20s} {v}")
        return 0

    before = checks(data)
    print("before:")
    for k, v in before.items():
        print(f"   {k:20s} {v}")

    for spec in args.set_byte:
        off, _, hexval = spec.partition(":")
        off = int(off, 0)
        val = int(hexval, 16)
        if off >= len(data):
            print(f"offset 0x{off:x} beyond file ({len(data)})", file=sys.stderr)
            return 1
        print(f"   set byte 0x{off:x}: {data[off]:02x} -> {val:02x}")
        data[off] = val

    reseal(data)
    after = checks(data)
    print("after:")
    for k, v in after.items():
        print(f"   {k:20s} {v}")

    if not all(after[k] for k in after if k.endswith("_ok")):
        print("SELF-CHECK FAILED", file=sys.stderr)
        return 1

    with open(args.outfile, "wb") as fh:
        fh.write(bytes(data))
    print(f"wrote {args.outfile} ({len(data)} bytes)")
    print(f"sha256: {hashlib.sha256(bytes(data)).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
