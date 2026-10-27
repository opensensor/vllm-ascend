#!/usr/bin/env python3
"""Assemble an esbc container from a gzip member, padded to an exact length.

The AICPU package install is gated on the package size: the device keeps the
installed package's size saved and compares the size of the file it receives
against it, so the container must come out at exactly that many bytes.

A gzip member tolerates trailing NUL bytes (inflate ignores them), so a member
that is shorter than the target can be padded to length.

Usage: assemble_pkg.py <stock-container> <member.bin> <target-size> <out>
"""
import hashlib
import struct
import sys


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    stock, member_path, target, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

    with open(stock, "rb") as fh:
        container = bytearray(fh.read())
    with open(member_path, "rb") as fh:
        member = fh.read()

    if len(member) > target:
        print(f"member is {len(member)} bytes, larger than the target {target}")
        return 1

    # the member is the gzip stream; pad it to the target with NULs
    padded = member + b"\x00" * (target - len(member))

    inner = bytearray(b"3\xaa::aicpu_kernels")
    inner += b"\x00" * (0x100 - len(inner))
    payload = bytes(inner) + padded

    out_buf = bytearray(container)
    out_buf[0x2000:0x2000 + len(payload)] = payload
    del out_buf[0x2000 + len(payload):]

    struct.pack_into("<I", out_buf, 0x45C, len(payload))
    out_buf[0x2C:0x2C + 32] = hashlib.sha256(bytes(out_buf[0x2000:0x2000 + len(payload)])).digest()
    out_buf[0x4BC:0x4BC + 32] = hashlib.sha256(bytes(out_buf[:0x4BC])).digest()
    out_buf[0x14BC:0x14BC + 32] = hashlib.sha256(bytes(out_buf[0x1000:0x14BC])).digest()

    with open(out, "wb") as fh:
        fh.write(bytes(out_buf))
    print(f"wrote {out}: {len(out_buf)} bytes (target {target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
