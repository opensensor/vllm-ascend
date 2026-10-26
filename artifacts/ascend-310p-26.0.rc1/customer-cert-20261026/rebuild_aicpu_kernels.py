#!/usr/bin/env python3
"""Build an AICPU kernels package with a substituted library.

Takes the stock container, replaces one library with a modified one, and
rebuilds the container so every vendor-defined integrity check still passes:

  * bin_hash.cfg         -- one SHA-256 per library, recomputed
  * payload_sha256       -- SHA-256 of the 3aa::aicpu_kernels payload
  * head_sha0 / head_sha1 -- SHA-256 over the two halves of the header

The container carries no certificate (cert_type 0xffffffff, CMS section past
EOF), so this is a complete rebuild, not a re-signing.

Usage:
    rebuild_aicpu_kernels.py <container> <lib.so> <libname.so> <out> [--payload-dir DIR]

--payload-dir replaces the whole aicpu_kernels_device/ directory (for when the
substitution is more than one file).
"""
import gzip
import hashlib
import io
import os
import struct
import sys
import tarfile

PAYLOAD_OFF = 0x2000
PAYLOAD_SHA_OFF = 0x2C
CODE_LEN_OFF = 0x45C
HEAD_SHA0_OFF = 0x4BC
HEAD_SHA1_OFF = 0x14BC
HEAD_SHA1_FROM = 0x1000
HEAD_SHA1_TO = 0x14BC
PREFIX = "aicpu_kernels_device"


def sha256(data):
    return hashlib.sha256(data).digest()


def parse_payload(container):
    payload = bytes(container[PAYLOAD_OFF:])
    gz = payload.find(b"\x1f\x8b\x08")
    if gz < 0:
        raise ValueError("no gzip stream in the payload")
    return gzip.decompress(payload[gz:]), gz


def build_payload(tar_bytes):
    inner = bytearray(b"3\xaa::aicpu_kernels")
    inner += b"\x00" * (0x100 - len(inner))
    return bytes(inner) + gzip.compress(tar_bytes, 9)


def read_tar(tar_bytes):
    return tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")


def repack(tar_bytes, replace_name, new_bytes, extra_dir=None):
    """Return a new tar with one member replaced (and optionally a whole
    directory substituted)."""
    src = read_tar(tar_bytes)
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:", format=src.format) as dst:
        for member in src.getmembers():
            if extra_dir and member.name.startswith(PREFIX + "/"):
                rel = member.name[len(PREFIX) + 1:]
                src_file = os.path.join(extra_dir, rel)
                if not os.path.isfile(src_file):
                    continue
                with open(src_file, "rb") as fh:
                    blob = fh.read()
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
            if member.name == replace_name:
                info = tarfile.TarInfo(member.name)
                info.size = len(new_bytes)
                info.mode = member.mode
                info.mtime = member.mtime
                info.uid = member.uid
                info.gid = member.gid
                info.uname = member.uname
                info.gname = member.gname
                dst.addfile(info, io.BytesIO(new_bytes))
                continue
            if member.isfile():
                dst.addfile(member, io.BytesIO(src.extractfile(member).read()))
            else:
                dst.addfile(member)
    return out.getvalue()


def rewrite_bin_hash(tar_bytes):
    """Recompute bin_hash.cfg from the libraries present in the tar."""
    src = read_tar(tar_bytes)
    entries = []
    for member in src.getmembers():
        if member.isfile() and member.name.endswith(".so"):
            blob = src.extractfile(member).read()
            entries.append((member.name[len(PREFIX) + 1:], hashlib.sha256(blob).hexdigest()))
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:", format=src.format) as dst:
        for member in src.getmembers():
            if member.name == PREFIX + "/bin_hash.cfg":
                body = "".join(f"{n}={h}\n" for n, h in entries).encode()
                info = tarfile.TarInfo(member.name)
                info.size = len(body)
                info.mode = member.mode
                info.mtime = member.mtime
                info.uid = member.uid
                info.gid = member.gid
                info.uname = member.uname
                info.gname = member.gname
                dst.addfile(info, io.BytesIO(body))
                continue
            if member.isfile():
                dst.addfile(member, io.BytesIO(src.extractfile(member).read()))
            else:
                dst.addfile(member)
    return out.getvalue()


def rebuild(container, replace_name, new_bytes, extra_dir=None):
    tar_bytes, _gz = parse_payload(container)
    tar2 = repack(tar_bytes, replace_name, new_bytes, extra_dir)
    tar3 = rewrite_bin_hash(tar2)
    payload = build_payload(tar3)

    out = bytearray(container)
    out[PAYLOAD_OFF:PAYLOAD_OFF + len(payload)] = payload
    del out[PAYLOAD_OFF + len(payload):]

    out[CODE_LEN_OFF:CODE_LEN_OFF + 4] = struct.pack("<I", len(payload))
    out[PAYLOAD_SHA_OFF:PAYLOAD_SHA_OFF + 32] = sha256(payload)
    out[HEAD_SHA0_OFF:HEAD_SHA0_OFF + 32] = sha256(bytes(out[:HEAD_SHA0_OFF]))
    out[HEAD_SHA1_OFF:HEAD_SHA1_OFF + 32] = sha256(bytes(out[HEAD_SHA1_FROM:HEAD_SHA1_TO]))
    return bytes(out)


def verify(container):
    """Re-verify every digest in a container; return (ok, report)."""
    report = []
    payload = bytes(container[PAYLOAD_OFF:])

    report.append(("payload sha256",
                    container[PAYLOAD_SHA_OFF:PAYLOAD_SHA_OFF + 32] == sha256(payload)))
    report.append(("head sha c0",
                    container[HEAD_SHA0_OFF:HEAD_SHA0_OFF + 32] == sha256(bytes(container[:HEAD_SHA0_OFF]))))
    stored1 = container[HEAD_SHA1_OFF:HEAD_SHA1_OFF + 32]
    actual1 = sha256(bytes(container[HEAD_SHA1_FROM:HEAD_SHA1_TO]))
    report.append(("head sha c1", stored1 == actual1))

    gz = payload.find(b"\x1f\x8b\x08")
    tar_bytes = gzip.decompress(payload[gz:])
    lib_ok = True
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as src:
        cfg = src.extractfile(PREFIX + "/bin_hash.cfg").read().decode()
        for line in cfg.splitlines():
            if "=" not in line:
                continue
            name, expect = line.split("=", 1)
            try:
                blob = src.extractfile(PREFIX + "/" + name.strip()).read()
            except Exception:
                lib_ok = False
                continue
            if hashlib.sha256(blob).hexdigest() != expect.strip():
                lib_ok = False
    report.append(("bin_hash.cfg (per-library sha256)", lib_ok))
    unsigned = struct.unpack_from("<I", container, 8)[0] == 0xFFFFFFFF
    report.append(("cert_type == 0xffffffff (unsigned)", unsigned))
    return all(ok for _n, ok in report), report


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    extra_dir = None
    for a in sys.argv[1:]:
        if a.startswith("--payload-dir"):
            extra_dir = a.split("=", 1)[1] if "=" in a else args[-1]
    if len(args) < 4:
        print(__doc__)
        return 2
    src, lib_path, lib_name, out_path = args[:4]

    with open(src, "rb") as fh:
        container = bytearray(fh.read())
    ok, report = verify(container)
    print("=== original container ===")
    for name, good in report:
        print(f"  {name:42s} {'OK' if good else 'BAD'}")
    if not ok:
        print("original container does not verify -- aborting")
        return 1

    with open(lib_path, "rb") as fh:
        new_bytes = fh.read()
    print(f"\nsubstituting {lib_name} ({len(new_bytes)} bytes)"
          + (f"  [whole dir {extra_dir}]" if extra_dir else ""))

    rebuilt = rebuild(container, PREFIX + "/" + lib_name, new_bytes, extra_dir)
    ok2, report2 = verify(rebuilt)
    print("\n=== rebuilt container ===")
    for name, good in report2:
        print(f"  {name:42s} {'OK' if good else 'BAD'}")

    with open(out_path, "wb") as fh:
        fh.write(rebuilt)
    print(f"\nwrote {out_path} ({len(rebuilt)} bytes)")
    return 0 if ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
