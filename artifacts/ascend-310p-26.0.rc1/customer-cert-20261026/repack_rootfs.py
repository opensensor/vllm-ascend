#!/usr/bin/env python3
"""Repack the Ascend 310P card rootfs (esbc container) with modified files.

The card rootfs `/usr/local/Ascend/driver/device/Ascend310P.cpio.gz` is an esbc
container whose payload is a gzip-compressed SVR4 cpio archive. This tool:

  1. extracts the container and the cpio payload
  2. applies file overrides (replace / add / delete inside the archive)
  3. repacks the cpio and gzip
  4. rebuilds the esbc container with correct geometry and digests
     (header copies, payload digest, cms/ini/crl slots)

The `cms`, `ini` and `crl` sections are carried over **unchanged** from the input
container unless explicitly overridden, so the image keeps its existing
signature. That is the point: the rootfs is fully rewritable, but re-signing it
requires a CA the bootloader trusts (see CUSTOM-FIRMWARE-PROGRESS.md).

Usage:
  repack_rootfs.py --container <Ascend310P.cpio.gz> --out <new.cpio.gz>
                    [--set path=localfile]... [--add path=localfile]...
                    [--del path]...
"""
import argparse
import gzip
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

IMG_OFF = 0x2000
CODE_LEN_OFF = 0x45C
IMG_OFF_FIELD = 0x458
CERT_TYPE_OFF = 0x08
DIGEST_OFF = 0x2C
CMS_SLOT = 0x4000
INI_SLOT = 0x800
COPY_SIZE = 0x1000


def u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def align(n, a):
    return (n + a - 1) // a * a


def read_container(path):
    with open(path, "rb") as fh:
        data = fh.read()
    code_len = u32(data, CODE_LEN_OFF)
    if len(data) < IMG_OFF + code_len:
        raise ValueError("container truncated")
    payload = data[IMG_OFF:IMG_OFF + code_len]
    # The three tagged slots sit at fixed offsets after the payload:
    #   cms @ cms_start, ini @ cms_start+0x4000, crl @ cms_start+0x4800
    # Each slot is: 12-byte tag, 4-byte length, then the body.
    cms_start = IMG_OFF + code_len + ((32 - code_len % 16) % 16)

    def section(start, name):
        if start + 16 > len(data):
            return b""
        tag = data[start:start + 12].rstrip(b"\0")
        if tag != name:
            return b""
        length = u32(data, start + 12)
        return data[start + 16:start + 16 + length]

    return {
        "data": data,
        "header": data[:IMG_OFF],
        "payload": payload,
        "code_len": code_len,
        "certtype": u32(data, CERT_TYPE_OFF),
        "cms": section(cms_start + 0x10, b"cms"),
        "ini": section(cms_start + CMS_SLOT + 0x10, b"ini"),
        "crl": section(cms_start + CMS_SLOT + INI_SLOT + 0x10, b"crl"),
    }


def build_container(header, payload, cms, ini, crl, certtype):
    out = bytearray(header)
    out += payload
    out += b"\0" * ((32 - len(payload) % 16) % 16)

    # reproduce the original slot layout: a 16-byte gap, then the slot header,
    # then the body, then zero padding to the next slot.
    # Slot geometry, matching the stock container exactly.  cms_start is the
    # first byte after the (16-aligned) payload.  Each slot is:
    #     16-byte zero gap | 12-byte tag | 4-byte length | body
    # and the slots are pinned at absolute offsets:
    #     cms @ cms_start+0x0010   (header at +0x10, body at +0x20)
    #     ini @ cms_start+0x4010   (body at +0x4020)
    #     crl @ cms_start+0x4810   (body at +0x4820)
    cms_start = len(out)

    def put_slot(at, tag, body, slot_end):
        nonlocal out
        out += b"\0" * (at - len(out))
        out += tag.ljust(12, b"\0") + struct.pack("<I", len(body)) + body
        out += b"\0" * (slot_end - len(out))

    put_slot(cms_start + 0x10, b"cms", cms, cms_start + CMS_SLOT)
    put_slot(cms_start + CMS_SLOT + 0x10, b"ini", ini,
             cms_start + CMS_SLOT + INI_SLOT)
    put_slot(cms_start + CMS_SLOT + INI_SLOT + 0x10, b"crl", crl,
             cms_start + CMS_SLOT + INI_SLOT + 0x20 + len(crl))

    # header copies: both must carry the new payload digest
    digest = hashlib.sha256(payload).digest()
    for copy in (0, COPY_SIZE):
        out[copy + DIGEST_OFF:copy + DIGEST_OFF + 32] = digest
        struct.pack_into("<I", out, copy + CODE_LEN_OFF, len(payload))
        struct.pack_into("<I", out, copy + IMG_OFF_FIELD, IMG_OFF)
        struct.pack_into("<I", out, copy + CERT_TYPE_OFF, certtype)
    return bytes(out)


def cpio_extract(blob, dest):
    """Extract a newc cpio, preserving the archived mode bits.

    `cpio -idm` cannot restore modes when running unprivileged, so the modes are
    parsed out of the archive and re-applied with chmod afterwards.
    """
    os.makedirs(dest, exist_ok=True)
    # NOTE: -idm (preserve mode) is required, but it must NOT be combined with
    # --no-preserve-owner, which makes cpio skip entries it cannot chown.
    p = subprocess.run(["cpio", "-idm", "--quiet", "--no-absolute-filenames"],
                       cwd=dest, input=blob, capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode()[:400] + "\n")
        return False
    # walk the archive and restore modes
    pos = 0
    while pos < len(blob):
        if blob[pos:pos + 6] != b"070701":
            break
        f = [int(blob[pos + 6 + i * 8:pos + 14 + i * 8], 16) for i in range(13)]
        mode, namesize, filesize = f[1], f[11], f[6]
        name = blob[pos + 110:pos + 110 + namesize - 1].decode("utf-8", "replace")
        pos += 110 + namesize
        pos += (-pos) % 4
        data = blob[pos:pos + filesize]
        pos += filesize
        pos += (-pos) % 4
        if name == "TRAILER!!!":
            break
        target = os.path.join(dest, name.lstrip("/"))
        try:
            if (mode & 0o170000) == 0o120000:      # symlink
                if os.path.lexists(target):
                    os.remove(target)
                os.symlink(data.decode("utf-8", "replace"), target)
            elif (mode & 0o170000) == 0o040000:    # dir
                os.makedirs(target, exist_ok=True)
                os.chmod(target, mode & 0o7777)
            elif (mode & 0o170000) == 0o100000:    # regular
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(data)
                os.chmod(target, mode & 0o7777)
        except OSError:
            pass
    return True


def cpio_pack(root):
    """Pack a directory tree into an SVR4 cpio archive using the real cpio tool.

    Hand-rolling the newc header is error-prone (13 hex fields in a fixed
    order), and getting it wrong produces an archive that the device cannot
    parse.  Using cpio(1) also gets uid/gid/mode/nlink handling right.
    """
    listing = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for n in sorted(dirnames) + sorted(filenames):
            rel = os.path.join(dirpath, n)[len(root):]
            if os.path.lexists(os.path.join(root, rel)):
                listing.append(rel)
    listing.sort()
    stdin = ("\n".join(listing) + "\n").encode()
    p = subprocess.run(["cpio", "-o", "-H", "newc", "--quiet", "--owner=0:0"],
                       cwd=root, input=stdin, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("cpio pack failed: {}".format(p.stderr.decode()[:400]))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[],
                    help="path=localfile (replace an existing file in the archive)")
    ap.add_argument("--add", action="append", default=[],
                    help="path=localfile (add a new file)")
    ap.add_argument("--delete", action="append", default=[], help="delete a path")
    ap.add_argument("--certtype", type=lambda s: int(s, 0))
    args = ap.parse_args()

    img = read_container(args.container)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        # The payload is itself an esbc container: [0x100 header][gzip(cpio)].
        # The inner header starts with magic 33aa3a3a and the tag at +0x04.
        inner = img["payload"]
        if inner[:4] != b"\x33\xaa\x3a\x3a":
            print("unexpected inner magic {!r} - not an esbc-wrapped rootfs".format(inner[:4]),
                  file=sys.stderr)
            return 1
        gz = inner[0x100:]
        dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        blob = dobj.decompress(gz) + dobj.flush()
        if not cpio_extract(blob, root):
            print("cpio extract failed", file=sys.stderr)
            return 1

        for spec in args.set + args.add:
            path, _, local = spec.partition("=")
            dst = os.path.join(root, path.lstrip("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(local, dst)
            print(f"  {'set' if spec in args.set else 'add'} {path} <- {local}")
        for path in args.delete:
            dst = os.path.join(root, path.lstrip("/"))
            if os.path.exists(dst):
                os.remove(dst)
                print(f"  del {path}")

        newblob = cpio_pack(root)
        newgz = gzip.compress(newblob, 9)
        # inner esbc: 0x100-byte header (copy the original, update the size fields)
        inner_hdr = bytearray(img["payload"][:0x100])
        struct.pack_into("<I", inner_hdr, 0x30, len(newgz))
        newpayload = bytes(inner_hdr) + newgz

    certtype = args.certtype if args.certtype is not None else img["certtype"]
    out = build_container(img["header"], newpayload, img["cms"], img["ini"],
                          img["crl"], certtype)
    with open(args.out, "wb") as fh:
        fh.write(out)
    print(f"wrote {args.out} ({len(out)} bytes)")
    print(f"  payload: {len(img['payload'])} -> {len(newpayload)} bytes")
    print(f"  certtype: 0x{certtype:08x}")
    print(f"  cms/ini/crl carried over unchanged "
          f"({len(img['cms'])}/{len(img['ini'])}/{len(img['crl'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
