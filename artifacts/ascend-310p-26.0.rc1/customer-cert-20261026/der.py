#!/usr/bin/env python3
"""Minimal DER reader for CMS SignedData, sufficient to verify SignerInfos."""


class TLV:
    __slots__ = ("tag", "cls", "constructed", "num", "hdr", "content", "children", "raw")

    def __init__(self, tag, cls, constructed, num, hdr, content, raw):
        self.tag = tag
        self.cls = cls
        self.constructed = constructed
        self.num = num
        self.hdr = hdr
        self.content = content
        self.raw = raw
        self.children = []

    def __repr__(self):
        return f"<TLV tag=0x{self.tag:02x} len={len(self.content)}>"

    def find(self, num, cls=0, constructed=None):
        for c in self.children:
            if c.num == num and c.cls == cls and (constructed is None or c.constructed == constructed):
                return c
        return None

    def findall(self, num, cls=0):
        return [c for c in self.children if c.num == num and c.cls == cls]


def parse(buf, off=0, end=None):
    """Parse one TLV at buf[off]; returns (TLV, next_off)."""
    if end is None:
        end = len(buf)
    p = off
    b0 = buf[p]
    cls = b0 >> 6
    constructed = bool(b0 & 0x20)
    num = b0 & 0x1F
    p += 1
    if num == 0x1F:
        num = 0
        while True:
            bb = buf[p]
            p += 1
            num = (num << 7) | (bb & 0x7F)
            if not (bb & 0x80):
                break
    ln = buf[p]
    p += 1
    if ln & 0x80:
        nb = ln & 0x7F
        ln = int.from_bytes(buf[p : p + nb], "big")
        p += nb
    hdr = p - off
    content = buf[p : p + ln]
    t = TLV(b0, cls, constructed, num, hdr, content, buf[off : p + ln])
    if constructed:
        q = 0
        while q < len(content):
            child, q = parse(content, q)
            t.children.append(child)
    return t, p + ln


def walk(node, depth=0, path=""):
    for i, c in enumerate(node.children):
        yield path + f"/{i}", c
        yield from walk(c, depth + 1, path + f"/{i}")


def oid_str(content):
    if not content:
        return ""
    first = content[0]
    parts = [str(first // 40), str(first % 40)]
    v = 0
    for b in content[1:]:
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            parts.append(str(v))
            v = 0
    return ".".join(parts)
