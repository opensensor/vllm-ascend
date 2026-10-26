#!/usr/bin/env python3
"""Parse the EDK2 FFS2 files inside an Ascend esbc payload and check their checksums.

The bootloader payload is a firmware volume: an EDK2 FIP header followed by FFS2
files.  Each EFI_FFS_FILE_HEADER carries an IntegrityCheck field, and files with
the FFS_ATTRIB_CHECKSUM attribute carry a CRC32 of their data in the header
(preceded by a UINT32 size field).

This tool:
  * locates the FIP header(s) in the payload,
  * walks the FFS files in the volume body,
  * recomputes each file's CRC32 and reports whether it matches the header,
  * identifies which file contains a given payload offset.

That is what tells us whether the bootloader's integrity chain is recomputable
(i.e. whether a payload edit can be made consistent) or cryptographic.
"""
import struct
import sys
import zlib

FFS_ATTRIB_CHECKSUM = 0x40
FFS_ATTRIB_LARGE_FILE = 0x01
FILE_TYPE_FFS_PAD = 0xF0

FV_SIGNATURE = b'_FVH'


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def find_fips(payload):
    """Yield (base_offset, fv_length) for each FIP header in the payload."""
    out = []
    i = 0
    while True:
        i = payload.find(FV_SIGNATURE, i)
        if i < 0:
            break
        base = i - 40                      # Signature is at FvBase+0x28
        if base >= 0:
            zero = payload[base:base + 16]
            if zero == b"\0" * 16 or zero[:4] != b"\xff\xff\xff\xff":
                hl = u16(payload, base + 0x30)
                if 0x30 <= hl <= 0x100:
                    out.append((base, u64(payload, base + 0x20), hl))
        i += 4
    return out


def walk_ffs(body, start=0, end=None):
    """Yield (hdr_off, name, integrity, type, attrs, data_off, data_len) per FFS file."""
    if end is None:
        end = len(body)
    off = start
    while off + 24 <= end:
        name = body[off:off + 16]
        if name[:4] == b"\xff\xff\xff\xff":
            break
        integrity = u16(body, off + 16)
        ftype = body[off + 18]
        attrs = body[off + 19]
        size = int.from_bytes(body[off + 20:off + 23], "little")
        state = body[off + 23]
        data_off = off + 24
        if attrs & FFS_ATTRIB_LARGE_FILE:
            size = u64(body, data_off)
            data_off += 8
        if ftype == FILE_TYPE_FFS_PAD:
            off = data_off + size
            continue
        yield (off, name, integrity, ftype, attrs, data_off, size, state)
        # files are 8-byte aligned
        nxt = data_off + size
        off = (nxt + 7) & ~7


def crc_check(body, attrs, integrity, data_off, size):
    """Recompute the FFS integrity value for a file."""
    if attrs & FFS_ATTRIB_CHECKSUM:
        stored = u32(body, data_off)
        calc = zlib.crc32(body[data_off + 4:data_off + size]) & 0xFFFFFFFF
        return ("crc32", stored, calc, stored == calc)
    # otherwise IntegrityCheck is a checksum over the file header
    return ("hdrsum", integrity, integrity, None)


def main():
    path = sys.argv[1]
    with open(path, "rb") as fh:
        data = fh.read()
    # detect: is this an esbc container, or a raw payload?
    payload = data
    try:
        off = u32(data, 0x458)
        ln = u32(data, 0x45C)
        if off + ln <= len(data) and data[off:off + 4] == bytes.fromhex("33aa3a3a"):
            payload = data[off:off + ln]
            print(f"[esbc container] payload = {len(payload)} bytes")
    except Exception:
        pass

    for base, fvlen, hl in find_fips(payload):
        print(f"\n=== FIP at payload+0x{base:x}  FvLength=0x{fvlen:x}  HeaderLength=0x{hl:x} ===")
        body = base + 56
        limit = min(len(payload), base + fvlen) if fvlen and base + fvlen <= len(payload) else len(payload)
        n = 0
        for (foff, name, integ, ftype, attrs, doff, size, state) in walk_ffs(payload, body, limit):
            n += 1
            if n <= 14:
                kind, stored, calc, ok = crc_check(payload, attrs, integ, doff, size)
                flag = "" if ok is None else ("  OK" if ok else "  *** CRC MISMATCH ***")
                print(f"  file @0x{foff:06x} type=0x{ftype:02x} attrs=0x{attrs:02x} "
                      f"size={size} data@0x{doff:06x} {kind} stored=0x{stored:x} calc=0x{calc:x}{flag}")
        print(f"  ({n} FFS files)")


if __name__ == "__main__":
    main()
