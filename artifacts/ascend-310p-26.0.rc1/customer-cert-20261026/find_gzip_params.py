#!/usr/bin/env python3
"""Find a deflate encoding whose length matches the original container exactly.

The AICPU package install is gated on the package *size*: the device keeps the
stock package's size saved and compares the size of the file it receives
against it.  Every package that loaded had pkgSize equal to the saved size
(53559460); every one that stalled had a different size.  So a rebuilt package
has to be exactly that many bytes.

Re-compressing the unmodified tar with gzip.compress(level=9) comes out about
13 KiB larger than the original member, which suggests the vendor's deflate
stream was produced slightly differently (a different zlib version, or a
different flush behaviour).  This searches for an encoding that reproduces the
original length exactly, so the same encoding can then be applied to the
modified tar.
"""
import gzip
import sys
import zlib

SRC = sys.argv[1] if len(sys.argv) > 1 else "stock_sysk.tar.gz"


def member_len_of(container):
    payload = bytes(container[0x2000:])
    gz = payload.find(b"\x1f\x8b\x08")
    return len(container) - (0x2000 + gz)


def try_encodings(raw, target):
    """Yield (name, bytes) for a range of deflate encodings."""
    # gzip via zlib with an explicit header/trailer
    for level in range(1, 10):
        for memlevel in range(1, 10):
            for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED,
                             zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED):
                for wbits in (15, 14, 13, 12, 11, 10, 9):
                    co = zlib.compressobj(level, zlib.DEFLATED, wbits,
                                          memlevel, strategy)
                    body = co.compress(raw) + co.flush()
                    size = 10 + len(body) + 8
                    yield (f"lvl{level} mem{memlevel} strat{strategy} wb{wbits}",
                           size, body)

    # gzip.compress variants
    for level in range(1, 10):
        blob = gzip.compress(raw, level, mtime=0)
        yield (f"gzip.compress lvl{level}", len(blob), blob)

    # zlib.compress (zlib wrapper, not gzip)
    for level in range(1, 10):
        blob = zlib.compress(raw, level)
        yield (f"zlib.compress lvl{level}", len(blob), blob)


def main():
    with open(SRC, "rb") as fh:
        container = bytearray(fh.read())
    payload = bytes(container[0x2000:])
    gz = payload.find(b"\x1f\x8b\x08")
    raw = gzip.decompress(payload[gz:])
    target = member_len_of(container)
    print(f"original member len : {target}", flush=True)

    best = None
    for name, size, _blob in try_encodings(raw, target):
        delta = size - target
        if delta == 0:
            print(f"EXACT MATCH: {name}", flush=True)
            return 0
        if best is None or abs(delta) < abs(best[0]):
            best = (delta, name, size)
        print(f"  {name:34s} size {size} delta {delta}", flush=True)
    print(f"best: {best[1]} delta {best[0]}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
