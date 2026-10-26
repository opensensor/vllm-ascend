#!/usr/bin/env python3
"""Rebuild an Ascend AICPU-kernels container with a substituted library.

The AICPU kernels the card's AI CPU executes are shipped as

    Ascend310P-aicpu_syskernels.tar.gz
      = esbc container (magic 0x55aa55aa, head_len 0x4dc)
        payload at 0x2000: "3\\xaa::aicpu_kernels" + gzip(tar)
          aicpu_kernels_device/
            libaicpu_kernels.so  libcpu_kernels.so  libtensorflow.so
            libpt_kernels.so     libtf_kernels.so
            bin_hash.cfg         version.info

and the container's *entire* integrity mechanism is a set of plain SHA-256
digests -- no certificate, no key, no signature:

    +0x02C  u8[32]  payload_sha256 = SHA-256(payload)
    +0x4BC  u8[32]  head_sha0     = SHA-256(container[0 : 0x4BC])
    +0x14BC u8[32]  head_sha1     = SHA-256(container[0x1000 : 0x14BC])

`cert_type` is 0xffffffff and the CMS section start lies past EOF, so there is
nothing to forge.  Inside the payload, `bin_hash.cfg` holds one SHA-256 per
library:

    libaicpu_kernels.so=74f888c0cb887efc84543be7556d0670efcac445bcc66d445b475b01586fdd5f

which is verified to be exactly sha256 of the file.

So the whole container can be rebuilt after substituting a library: repack the
tar, recompute bin_hash.cfg, recompute the payload digest, recompute the two
head digests.

This script performs that rebuild and re-verifies every digest.  It does NOT
deploy anything -- deployment is a separate, explicit step.

Usage:
    python3 rebuild_aicpu_kernels.py <container.tar.gz> <lib.so> <libname.so> <out.tar.gz>
"""
import gzip
import hashlib
import io
import struct
import sys
import tarfile

ESBC_MAGIC = 0x55AA55AA
PAYLOAD_OFF = 0x2000
HEAD_LEN = 0x4DC
PAYLOAD_SHA_OFF = 0x2C
CODE_LEN_OFF = 0x45C
HEAD_SHA0_OFF = 0x4BC
HEAD_SHA1_OFF = 0x14BC
HEAD_SHA1_FROM = 0x1000
HEAD_SHA1_TO = 0x14BC


def sha256(data):
    return hashlib.sha256(data).digest()


def read_container(path):
    with open(path, "rb") as fh:
        return bytearray(fh.read())


def parse_payload(container):
    """Return (tar_bytes, offset_of_gzip) for the 3aa payload."""
    payload = bytes(container[PAYLOAD_OFF:])
    gz = payload.find(b"\x1f\x8b\x08")
    if gz < 0:
        raise ValueError("no gzip stream in the payload")
    return gzip.decompress(payload[gz:]), gz


def build_payload(tar_bytes):
    """Rebuild the 3aa::aicpu_kernels + gzip wrapper."""
    inner = bytearray(b"3\xaa::aicpu_kernels")
    inner += b"\x00" * (0x100 - len(inner))
    return bytes(inner) + gzip.compress(tar_bytes, 9)


def repack_tar(original_tar, replace_name, new_bytes):
    """Return a new tar with one member replaced, preserving the others."""
    src = tarfile.open(fileobj=io.BytesIO(original_tar), mode="r:")
    out = io.BytesIO()
    dst = tarfile.open(fileobj=out, mode="w:", format=src.format)
    for member in src.getmembers():
        data = src.extractfile(member).read() if member.isfile() else b""
        if member.name == replace_name:
            member.size = len(new_bytes)
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
            dst.addfile(member, io.BytesIO(data))
        else:
            dst.addfile(member)
    dst.close()
    return out.getvalue()


def rewrite_bin_hash(tar_bytes, prefix):
    """Recompute bin_hash.cfg from the libraries present in the tar."""
    src = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")
    cfg_name = prefix + "/bin_hash.cfg"
    entries = []
    for member in src.getmembers():
        if not member.isfile() or not member.name.endswith(".so"):
            continue
        blob = src.extractfile(member).read()
        entries.append((member.name[len(prefix) + 1:], hashlib.sha256(blob).hexdigest()))
    out = io.BytesIO()
    dst = tarfile.open(fileobj=out, mode="w:", format=src.format)
    for member in src.getmembers():
        if member.name == cfg_name:
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
        data = src.extractfile(member).read() if member.isfile() else b""
        if member.isfile():
            dst.addfile(member, io.BytesIO(data))
        else:
            dst.addfile(member)
    dst.close()
    return out.getvalue()


def rebuild(container, replace_name, new_bytes):
    tar_bytes, _gz = parse_payload(container)
    prefix = "aicpu_kernels_device"
    tar2 = repack_tar(tar_bytes, replace_name, new_bytes)
    tar3 = rewrite_bin_hash(tar2, prefix)
    payload = build_payload(tar3)

    out = bytearray(container)
    out[PAYLOAD_OFF:PAYLOAD_OFF + len(payload)] = payload
    del out[PAYLOAD_OFF + len(payload):]

    # the header records the payload length; keep it consistent
    out[CODE_LEN_OFF:CODE_LEN_OFF + 4] = struct.pack("<I", len(payload))

    # recompute the three digests
    out[PAYLOAD_SHA_OFF:PAYLOAD_SHA_OFF + 32] = sha256(payload)
    out[HEAD_SHA0_OFF:HEAD_SHA0_OFF + 32] = sha256(bytes(out[:HEAD_SHA0_OFF]))
    out[HEAD_SHA1_OFF:HEAD_SHA1_OFF + 32] = sha256(bytes(out[HEAD_SHA1_FROM:HEAD_SHA1_TO]))
    return bytes(out)


def verify(container):
    """Re-verify every digest in a container; return (ok, report)."""
    report = []
    payload = bytes(container[PAYLOAD_OFF:])
    ln = struct.unpack_from("<I", container, 4)[0]
    off = PAYLOAD_OFF
    slot = 0x20 - (ln & 0xF)

    stored = container[PAYLOAD_SHA_OFF:PAYLOAD_SHA_OFF + 32]
    actual = sha256(payload)
    report.append(("payload sha256", stored == actual))

    stored0 = container[HEAD_SHA0_OFF:HEAD_SHA0_OFF + 32]
    actual0 = sha256(bytes(container[:HEAD_SHA0_OFF]))
    report.append(("head sha c0", stored0 == actual0))

    stored1 = container[HEAD_SHA1_OFF:HEAD_SHA1_OFF + 32]
    actual1 = sha256(bytes(container[HEAD_SHA1_FROM:HEAD_SHA1_TO]))
    report.append(("head sha c1", stored1 == actual1))

    gz = payload.find(b"\x1f\x8b\x08")
    tar_bytes = gzip.decompress(payload[gz:])
    src = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")
    cfg = src.extractfile("aicpu_kernels_device/bin_hash.cfg").read().decode()
    lib_ok = True
    for line in cfg.splitlines():
        if "=" not in line:
            continue
        name, expect = line.split("=", 1)
        try:
            blob = src.extractfile("aicpu_kernels_device/" + name.strip()).read()
        except Exception:
            lib_ok = False
            continue
        if hashlib.sha256(blob).hexdigest() != expect.strip():
            lib_ok = False
    report.append(("bin_hash.cfg (per-library sha256)", lib_ok))
    report.append(("cert_type == 0xffffffff (unsigned)",
                   struct.unpack_from("<I", container, 8)[0] == 0xFFFFFFFF))
    return all(ok for _n, ok in report), report


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    src, lib_path, lib_name, out_path = sys.argv[1:5]

    container = read_container(src)
    ok, report = verify(container)
    print("=== original container ===")
    for name, good in report:
        print(f"  {name:42s} {'OK' if good else 'BAD'}")
    if not ok:
        print("original container does not verify -- aborting")
        return 1

    with open(lib_path, "rb") as fh:
        new_bytes = fh.read()
    print(f"\nsubstituting {lib_name} ({len(new_bytes)} bytes)")

    rebuilt = rebuild(container, "aicpu_kernels_device/" + lib_name, new_bytes)
    ok2, report2 = verify(rebuilt)
    print("\n=== rebuilt container ===")
    for name, good in report2:
        print(f"  {name:42s} {'OK' if good else 'BAD'}")

    with open(out_path, "wb") as fh:
        fh.write(rebuilt)
    print(f"\nwrote {out_path} ({len(rebuilt)} bytes)")

    # show the substituted library is present with the expected hash
    payload = rebuilt[PAYLOAD_OFF:]
    gz = payload.find(b"\x1f\x8b\x08")
    tar_bytes = gzip.decompress(payload[gz:])
    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")
    blob = tf.extractfile("aicpu_kernels_device/" + lib_name).read()
    print(f"  {lib_name} in the rebuilt tar: {len(blob)} bytes, "
          f"sha256={hashlib.sha256(blob).hexdigest()}")
    return 0 if ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
