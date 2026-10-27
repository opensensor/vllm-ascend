#!/usr/bin/env python3
"""Compress a tar with a chosen zlib and report the gzip member length.

The AICPU package container's payload is a gzip stream of a newc tar, and the
device gates the package install on the package *size*: it keeps the stock
package's size saved and compares the size of the file it receives against it.
Every package that loaded had pkgSize equal to the saved size (53559460); every
one that stalled had a different size.  So the container has to come out at
exactly that length.

Re-compressing with the host's zlib (1.3.1) does not reproduce the vendor's
member length -- it comes out about 13 KiB longer -- so this loads a specific
zlib with ctypes and compresses through it.  The device ships zlib 1.2.13 and
the host has 1.2.11 (from snapd), both of which are worth trying.

Usage: zcompress.py <zlib.so> <in.tar> [level] [memLevel]
Prints: <member_length>
"""
import ctypes
import ctypes.util
import struct
import sys
import zlib


def load_zlib(path):
    lib = ctypes.CDLL(path)
    lib.deflateInit2_ = lib.deflateInit2_
    # int deflateInit2_(z_streamp, level, method, windowBits, memLevel,
    #                    strategy, const char *version, int stream_size)
    lib.deflateInit2_.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.deflateInit2_.restype = ctypes.c_int
    lib.deflate = lib.deflate
    lib.deflate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.deflate.restype = ctypes.c_int
    lib.deflateEnd = lib.deflateEnd
    lib.deflateEnd.argtypes = [ctypes.c_void_p]
    lib.deflateEnd.restype = ctypes.c_int
    return lib


class ZStream(ctypes.Structure):
    _fields_ = [
        ("next_in", ctypes.c_void_p),
        ("avail_in", ctypes.c_uint),
        ("total_in", ctypes.c_ulong),
        ("next_out", ctypes.c_void_p),
        ("avail_out", ctypes.c_uint),
        ("total_out", ctypes.c_ulong),
        ("msg", ctypes.c_char_p),
        ("state", ctypes.c_void_p),
        ("zalloc", ctypes.c_void_p),
        ("zfree", ctypes.c_void_p),
        ("opaque", ctypes.c_void_p),
        ("data_type", ctypes.c_int),
        ("adler", ctypes.c_ulong),
        ("reserved", ctypes.c_ulong),
    ]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    zlib_path, tar_path = sys.argv[1], sys.argv[2]
    level = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    memlevel = int(sys.argv[4]) if len(sys.argv) > 4 else 9

    with open(tar_path, "rb") as fh:
        data = fh.read()

    lib = load_zlib(zlib_path)
    st = ZStream()
    ret = lib.deflateInit2_(ctypes.byref(st), level, 8, 15, memlevel, 0,
                            zlib.ZLIB_VERSION.encode(), ctypes.sizeof(ZStream))
    if ret != 0:
        print(f"deflateInit2 failed: {ret}")
        return 1

    out = bytearray()
    chunk = 1 << 20
    src = ctypes.create_string_buffer(data, len(data))
    dst = ctypes.create_string_buffer(chunk)
    st.next_in = ctypes.cast(src, ctypes.c_void_p)
    st.avail_in = len(data)
    # Z_NO_FLUSH while there is input left, Z_FINISH once it is exhausted.
    # Passing Z_FINISH with input still pending makes zlib error out.
    while True:
        st.next_out = ctypes.cast(dst, ctypes.c_void_p)
        st.avail_out = chunk
        if st.avail_in == 0:
            flush = 4      # Z_FINISH
        else:
            flush = 0      # Z_NO_FLUSH
        r = lib.deflate(ctypes.byref(st), flush)
        out += dst.raw[:chunk - st.avail_out]
        if r == 1:  # Z_STREAM_END
            break
        if r != 0:
            print(f"deflate failed: {r}")
            return 1
    lib.deflateEnd(ctypes.byref(st))

    member_len = 10 + len(out) + 8
    print(member_len)

    header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03"
    trailer = struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF)
    with open("/tmp/member.bin", "wb") as fh:
        fh.write(header)
        fh.write(bytes(out))
        fh.write(trailer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
