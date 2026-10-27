#!/usr/bin/env python3
"""Verify an AICPU kernels package: structure, digests, and the patched library.

The container's payload is the inner header plus a gzip member plus padding, so
Python's gzip.decompress() cannot be used directly -- it tries to parse a second
member out of the trailing NULs.  The member is decoded by trimming the payload
until the first member inflates cleanly, which is exactly what the device's
`tar --no-same-owner -ozxf` sees.

Usage: verify_pkg.py <package> [expect_lib_sha256] [expect_size]
"""
import hashlib
import io
import struct
import sys
import tarfile
import zlib

RUN_CPU_KERNEL_OFF = 0x13E00F0


def main():
    path = sys.argv[1]
    expect_lib = sys.argv[2] if len(sys.argv) > 2 else None
    expect_size = int(sys.argv[3]) if len(sys.argv) > 3 else None

    with open(path, "rb") as fh:
        data = fh.read()

    print(f"file            : {path}")
    print(f"size            : {len(data)}"
          + (f" (expect {expect_size})" if expect_size else ""))
    if expect_size and len(data) != expect_size:
        print("  *** SIZE MISMATCH ***")
    else:
        print("  size OK")

    payload = data[0x2000:]
    gz = payload.find(b"\x1f\x8b\x08")
    if gz < 0:
        print("  *** no gzip stream ***")
        return 1

    # find where the first member ends
    raw = None
    for end in range(len(payload), gz, -1):
        try:
            raw = zlib.decompressobj(31).decompress(payload[gz:end])
            break
        except zlib.error:
            continue
    if raw is None:
        print("  *** member does not decode ***")
        return 1
    print(f"member decodes  : {len(raw)} bytes of tar")

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        members = tf.getmembers()
        print(f"members         : {len(members)}")
        for member in members:
            print(f"  {member.size:>12d}  {member.name}")
        lib = tf.extractfile("aicpu_kernels_device/libcpu_kernels.so").read()
        cfg = tf.extractfile("aicpu_kernels_device/bin_hash.cfg").read().decode()

    lib_hash = hashlib.sha256(lib).hexdigest()
    print(f"lib sha256      : {lib_hash}")
    if expect_lib and lib_hash != expect_lib:
        print("  *** LIB HASH MISMATCH ***")
    elif expect_lib:
        print("  lib hash OK")
    print(f"RunCpuKernel    : {lib[RUN_CPU_KERNEL_OFF:RUN_CPU_KERNEL_OFF + 8].hex()}")

    recorded = [line for line in cfg.splitlines() if line.startswith("libcpu_kernels.so")]
    print(f"bin_hash.cfg    : {recorded}")
    if recorded:
        want = recorded[0].split("=", 1)[1]
        print("  matches library" if want == lib_hash else "  *** DOES NOT MATCH LIBRARY ***")

    code_len = struct.unpack_from("<I", data, 0x45C)[0]
    print(f"code_len        : {code_len} (payload {len(data) - 0x2000})")
    print(f"payload sha256  : "
          f"{'OK' if data[0x2C:0x2C + 32] == hashlib.sha256(data[0x2000:0x2000 + code_len]).digest() else 'BAD'}")
    print(f"head sha0       : "
          f"{'OK' if data[0x4BC:0x4BC + 32] == hashlib.sha256(bytes(data[:0x4BC])).digest() else 'BAD'}")
    print(f"head sha1       : "
          f"{'OK' if data[0x14BC:0x14BC + 32] == hashlib.sha256(bytes(data[0x1000:0x14BC])).digest() else 'BAD'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
