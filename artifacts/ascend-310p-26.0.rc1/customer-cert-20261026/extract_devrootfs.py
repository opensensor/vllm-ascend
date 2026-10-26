#!/usr/bin/env python3
"""Extract the Ascend device rootfs from its esbc container.

(renamed from cpio3aa.py once the real payload format was established.)

Ascend310P.cpio.gz is an esbc container (magic 0x55aa55aa, head_len 0x4dc)
whose payload is NOT a CPIO archive directly.  The layout is:

    0x0000  esbc header
    0x1000  the payload begins here, as a gzip stream
            the decompressed data is a standard SVR4 "newc" CPIO archive
            (magic 070701)

(The bytes just after the gzip header spell "::initrd", which is why naive
scanning of the container for a CPIO magic fails.)

This tool extracts the payload and unpacks it with the system cpio.

A 3aa archive is a flat sequence of entries, each laid out as:

    offset  size  field
    0       6     magic "3\xaa::"
    6       6     file name length (hex, big endian)
    12      6     file mode    (hex, big endian)
    18      6     file size   (hex, big endian)
    24      6     uid         (hex, big endian)
    30      6     gid         (hex, big endian)
    36      6     nlink       (hex, big endian)
    42      6     mtime       (hex, big endian)
    48      6     devmajor    (hex, big endian)
    54      6     devminor    (hex, big endian)
    60      6     rdevmajor   (hex, big endian)
    66      6     rdevminor   (hex, big endian)
    72      6     namesize    (hex, big endian)
    78      6     check       (hex, big endian)
    84      N     file name (namesize bytes, NUL padded to a 4-byte boundary)
    ...     S     file data (size bytes, padded to a 4-byte boundary)

The archive ends with an entry whose name is "TRAILER!!!".

Usage: cpio3aa.py <archive> [outdir]
"""
import contextlib
import os
import struct
import sys
import zlib

MAGIC = b"070701"
HDR = 110


def _hex(text):
    return int(text, 16)


def extract(archive, outdir):
    """Unpack a SVR4 "newc" CPIO archive into outdir."""
    data = archive
    pos = 0
    count = 0

    while pos + HDR <= len(data):
        if data[pos:pos + 6] != MAGIC:
            pos += 1
            continue
        f = [_hex(data[pos + 6 + i * 8:pos + 6 + (i + 1) * 8].decode())
             for i in range(13)]
        mode, filesize, namesize = f[1], f[6], f[11]
        name_off = pos + HDR
        name = data[name_off:name_off + namesize - 1].decode(errors="replace")
        data_off = (name_off + namesize + 3) & ~3
        content = data[data_off:data_off + filesize]
        next_off = (data_off + filesize + 3) & ~3

        if name == "TRAILER!!!":
            break

        if (mode & 0o170000) == 0o040000:            # directory
            os.makedirs(os.path.join(outdir, name.lstrip("/")), exist_ok=True)
        elif (mode & 0o170000) == 0o120000:          # symlink
            target = os.path.join(outdir, name.lstrip("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            link = content.decode(errors="replace")
            with contextlib.suppress(OSError):
                if os.path.islink(target) or os.path.exists(target):
                    os.remove(target)
                os.symlink(link, target)
        else:                                        # regular file
            target = os.path.join(outdir, name.lstrip("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(content)
            with contextlib.suppress(OSError):
                os.chmod(target, mode & 0o7777)

        count += 1
        pos = next_off

    return count, None


def find_payload(blob):
    """Return (offset, size) of the gzip payload inside the esbc container.

    Layout measured on Ascend310P.cpio.gz:
        0x0000  esbc header (magic 0x55aa55aa, head_len 0x4dc)
        0x2000  a short "3\xaa::initrd" sub-header
        0x2100  the gzip stream holding the newc CPIO archive
    """
    if blob[:3] == b"\x1f\x8b\x08":          # already a bare gzip stream
        return 0, len(blob)
    idx = blob.find(b"\x1f\x8b\x08")
    if idx < 0:
        raise ValueError("no gzip payload found in the container")
    return idx, len(blob) - idx


def inflate_single_member(data):
    """Inflate the first gzip member in `data`.

    gzip.decompress() walks *all* members and then tries to parse the
    trailing bytes as another member; in an esbc container those trailing
    bytes are the next container header (magic 0x55aa55aa), so it raises
    "Not a gzipped file".  Only the first member is wanted here, so the
    header is parsed directly and the raw deflate stream is inflated.
    """
    if data[:3] != b"\x1f\x8b\x08":
        raise ValueError("not a gzip stream")
    flg = data[3]
    pos = 10
    if flg & 4:                                    # FEXTRA
        xlen = struct.unpack("<H", data[pos:pos + 2])[0]
        pos += 2 + xlen
    if flg & 8:                                    # FNAME
        pos = data.index(b"\x00", pos) + 1
    if flg & 16:                                   # FCOMMENT
        pos = data.index(b"\x00", pos) + 1
    if flg & 2:                                    # FHCRC
        pos += 2
    obj = zlib.decompressobj(-zlib.MAX_WBITS)
    return obj.decompress(data[pos:])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "devfs"

    with open(path, "rb") as fh:
        blob = fh.read()

    off, size = find_payload(blob)
    payload = inflate_single_member(blob[off:off + size])
    inner = payload.find(MAGIC)
    if inner < 0:
        raise ValueError("no newc CPIO magic in the decompressed payload")
    payload = payload[inner:]

    os.makedirs(outdir, exist_ok=True)
    count, written = extract(payload, outdir)
    print(f"extracted {count} entries into {outdir} "
          f"(payload {size} bytes gzip -> {len(payload)} bytes cpio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
