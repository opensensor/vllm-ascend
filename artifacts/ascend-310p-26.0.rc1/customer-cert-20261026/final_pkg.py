#!/usr/bin/env python3
"""Build the final AICPU kernels package: patched library, exact target size.

The device gates the AICPU package install on the package size: it keeps the
installed package's size saved and compares the size of the file it receives
against it, so the container has to come out at exactly that many bytes.

Steps:
  1. rebuild the tar with the patched libcpu_kernels.so and the matching
     bin_hash.cfg, in GNU tar format (which reproduces the original tar byte for
     byte);
  2. compress it at a memLevel whose output is *smaller* than the target --
     the vendor's own compressor produced a stream ~13 KiB shorter than any
     zlib here does, so the member is padded back up to length afterwards;
  3. assemble the container at exactly the target length and fix up its three
     digests (payload sha256, head sha0, head sha1) and code_len.

Container layout:

    0x0000  esbc header
    0x2000  inner header (0x100 bytes) + gzip member + padding
    0x02C   u8[32]  sha256 of the payload
    0x45C   u32      payload length   (0x2000 + code_len == file length)
    0x4BC   u8[32]  sha256 of container[0:0x4BC]
    0x14BC  u8[32]  sha256 of container[0x1000:0x14BC]

Usage: final_pkg.py <stock-container> <target-size> <memLevel> <out>
"""
import gzip
import hashlib
import io
import struct
import sys
import tarfile
import zlib

RUN_CPU_KERNEL_OFF = 0x13E00F0
PATCH = bytes.fromhex("000080d2c0035fd6")   # mov x0,#0 ; ret


def main():
    stock, target, memlevel, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

    with open(stock, "rb") as fh:
        container = bytearray(fh.read())
    payload = bytes(container[0x2000:])
    gz = payload.find(b"\x1f\x8b\x08")
    with gzip.GzipFile(fileobj=io.BytesIO(payload[gz:])) as gz_fh:
        raw = gz_fh.read()

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src:
        lib_member = [m for m in src.getmembers() if m.name.endswith("libcpu_kernels.so")][0]
        lib = bytearray(src.extractfile(lib_member).read())
    lib[RUN_CPU_KERNEL_OFF:RUN_CPU_KERNEL_OFF + 8] = PATCH
    lib = bytes(lib)
    new_hash = hashlib.sha256(lib).hexdigest()
    print(f"patched libcpu_kernels.so sha256 {new_hash}", flush=True)

    buf = io.BytesIO()
    dst = tarfile.open(fileobj=buf, mode="w:", format=tarfile.GNU_FORMAT)  # noqa: SIM115
    for member in src.getmembers():
        if member.name.endswith("libcpu_kernels.so"):
            info = tarfile.TarInfo(member.name)
            info.size = len(lib)
            info.mode = member.mode
            info.mtime = member.mtime
            info.uid = member.uid
            info.gid = member.gid
            info.uname = member.uname
            info.gname = member.gname
            dst.addfile(info, io.BytesIO(lib))
            continue
        if member.name.endswith("bin_hash.cfg"):
            cfg = src.extractfile(member).read().decode()
            new_cfg = "".join(
                f"libcpu_kernels.so={new_hash}\n"
                if line.startswith("libcpu_kernels.so") else line
                for line in cfg.splitlines(keepends=True)
            )
            blob = new_cfg.encode()
            info = tarfile.TarInfo(member.name)
            info.size = len(blob)
            info.mode = member.mode
            info.mtime = member.mtime
            info.uid = member.uid
            info.gid = member.gid
            info.uname = member.uname
            info.gname = member.gname
            dst.addfile(info, io.BytesIO(blob))
            continue
        if member.isfile():
            dst.addfile(member, io.BytesIO(src.extractfile(member).read()))
        else:
            dst.addfile(member)
    dst.close()
    tar2 = buf.getvalue()
    print(f"tar {len(tar2)} bytes", flush=True)

    compressor = zlib.compressobj(9, zlib.DEFLATED, 15, memlevel, zlib.Z_DEFAULT_STRATEGY)
    body = compressor.compress(tar2) + compressor.flush()
    header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03"
    trailer = struct.pack("<II", zlib.crc32(tar2) & 0xFFFFFFFF, len(tar2) & 0xFFFFFFFF)
    member_bytes = header + body + trailer
    print(f"gzip member {len(member_bytes)} bytes "
          f"(target {target}, {'under' if len(member_bytes) <= target else 'OVER'})", flush=True)
    if len(member_bytes) > target:
        print("member too long -- try a lower memLevel")
        return 1

    inner = bytearray(b"3\xaa::aicpu_kernels")
    inner += b"\x00" * (0x100 - len(inner))
    avail = (target - 0x2000) - len(inner)
    new_payload = bytes(inner) + member_bytes + b"\x00" * (avail - len(member_bytes))

    result = bytearray(container)
    result[0x2000:0x2000 + len(new_payload)] = new_payload
    del result[0x2000 + len(new_payload):]

    struct.pack_into("<I", result, 0x45C, len(new_payload))
    result[0x2C:0x2C + 32] = hashlib.sha256(bytes(result[0x2000:0x2000 + len(new_payload)])).digest()
    result[0x4BC:0x4BC + 32] = hashlib.sha256(bytes(result[:0x4BC])).digest()
    result[0x14BC:0x14BC + 32] = hashlib.sha256(bytes(result[0x1000:0x14BC])).digest()

    with open(out, "wb") as fh:
        fh.write(bytes(result))
    print(f"wrote {out}: {len(result)} bytes (target {target}, "
          f"{'EXACT MATCH' if len(result) == target else 'MISMATCH'})", flush=True)

    # self-check: the first member must inflate and carry our patched library
    with open(out, "rb") as fh:
        check = fh.read()
    payload2 = bytes(check[0x2000:])
    gz2 = payload2.find(b"\x1f\x8b\x08")
    do = zlib.decompressobj(31)
    tar3 = do.decompress(payload2[gz2:])
    print(f"  first member inflates to {len(tar3)} bytes", flush=True)
    with tarfile.open(fileobj=io.BytesIO(tar3), mode="r:") as tf:
        lib2 = tf.extractfile("aicpu_kernels_device/libcpu_kernels.so").read()
    print(f"  lib sha256   {hashlib.sha256(lib2).hexdigest()}", flush=True)
    print(f"  RunCpuKernel {lib2[RUN_CPU_KERNEL_OFF:RUN_CPU_KERNEL_OFF + 8].hex()}", flush=True)
    return 0 if len(result) == target else 1


if __name__ == "__main__":
    sys.exit(main())
