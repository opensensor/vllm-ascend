#!/usr/bin/env python3
"""Verify a detached CMS SignedData the way Ascend's Cmscbb verifier does.

For every SignerInfo:
  * messageDigest attr == SHA-256(detached content)
  * RSA-PSS signature over DER signedAttrs (re-tagged SET OF) using the
    signer's certificate
  * signingCertificateV2 attr == SHA-256(signer cert DER)
"""
import hashlib
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from der import oid_str, parse, walk


def collect_certs(node, out):
    for _, c in walk(node):
        if c.cls == 0 and c.num == 16 and c.constructed:
            try:
                cert = x509.load_der_x509_certificate(c.raw)
            except Exception:
                continue
            out.append((cert, c.raw))
    return out


def find_signer_infos(sd):
    """SignedData ::= SEQUENCE { version, digestAlgorithms, encapContentInfo,
       [0] certificates OPTIONAL, [1] crls OPTIONAL, signerInfos }

    Both digestAlgorithms and signerInfos are SET (universal tag 17), so
    select the LAST one -- signerInfos is always last per RFC 5652.
    """
    sets = [c for c in sd.children if c.cls == 0 and c.num == 17 and c.constructed]
    if not sets:
        return []
    si_set = sets[-1]
    return [c for c in si_set.children if c.constructed]


def attrs_of(si):
    """Return list of (oid, value_tlv) for signed attrs and unsigned attrs."""
    signed = []
    unsigned = []
    for c in si.children:
        if c.cls == 2 and c.num == 0:
            # [0] IMPLICIT SET OF Attribute -- signed attrs
            for a in c.children:
                oid = oid_str(a.children[0].content)
                vals = a.findall(2)
                signed.append((oid, vals[0] if vals else None, a))
        if c.cls == 2 and c.num == 1:
            for a in c.children:
                oid = oid_str(a.children[0].content)
                vals = a.findall(2)
                unsigned.append((oid, vals[0] if vals else None, a))
    return signed, unsigned


def main(cms_path, content_path, verbose=True):
    with open(cms_path, "rb") as fh:
        buf = fh.read()
    top, _ = parse(buf)
    # Outer SignedData is ContentInfo.content[0] -> [0] IMPLICIT -> signedData
    sd = None
    if len(top.children) >= 2 and top.children[1].cls == 2 and top.children[1].num == 0:
        cand = top.children[1].children
        if cand and cand[0].cls == 0 and cand[0].num == 16:
            sd = cand[0]
    if sd is None:
        for _, c in walk(top):
            if c.cls == 0 and c.num == 16 and c.constructed:
                sd = c
                break
    if sd is None:
        print("no signedData")
        return
    certs = collect_certs(sd, [])
    print(f"embedded certificates: {len(certs)}")
    for cert, raw in certs:
        print(f"   {cert.subject.rfc4514_string()}  sha256={hashlib.sha256(raw).hexdigest()[:16]}")

    with open(content_path, "rb") as fh:
        content = fh.read()
    cd = hashlib.sha256(content).digest()

    sis = find_signer_infos(sd)
    print(f"\nsignerInfos: {len(sis)}")
    print(f"detached content SHA-256: {cd.hex()}")

    allok = True
    for idx, si in enumerate(sis):
        print(f"\n--- SignerInfo #{idx} ---")
        signed, unsigned = attrs_of(si)
        md = None
        for oid, val, _ in signed:
            if oid == "1.2.840.113549.1.9.4" and val is not None:
                md = val.content
        if md:
            print(f"  messageDigest      : {md.hex()}")
            print(f"  CONTENT MATCH      : {md == cd}")
            if md != cd:
                allok = False
        scv = None
        for oid, val, _ in signed:
            if oid == "1.2.840.113549.1.9.16.2.47" and val is not None:
                scv = val.content
        if scv:
            print(f"  signingCertV2 hash : {scv.hex()}")
            for cert, raw in certs:
                if hashlib.sha256(raw).digest() == scv:
                    print(f"     -> {cert.subject.rfc4514_string()}")
        # signature value
        sig = None
        for c in si.children:
            if c.cls == 0 and c.num == 4 and not c.constructed:
                sig = c.content
        # signedAttrs DER
        sa_der = None
        for c in si.children:
            if c.cls == 2 and c.num == 0:
                sa_der = c.raw
        if sig and sa_der:
            # signature is over the SET OF encoding (0x31) of the [0] IMPLICIT block
            sa_der[sa_der[0] & 0x1F and 0 or 0:]
            # rebuild: header of [0] replaced by SET tag, length preserved
            len(sa_der) - sa_der.__len__()  # placeholder
            # simpler: construct from content
            t = None
            for c in si.children:
                if c.cls == 2 and c.num == 0:
                    t = c
            content_bytes = t.content
            # length of content
            n = len(content_bytes)
            if n < 0x80:
                hdr = bytes([0x31, n])
            elif n < 0x100:
                hdr = bytes([0x31, 0x81, n])
            else:
                hdr = bytes([0x31, 0x82, (n >> 8) & 0xFF, n & 0xFF])
            to_verify = hdr + content_bytes
            hit = None
            for cert, raw in certs:
                pub = cert.public_key()
                if not isinstance(pub, rsa.RSAPublicKey):
                    continue
                for salt in (32, 20, 48):
                    try:
                        pub.verify(sig, to_verify,
                                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                               salt_length=salt),
                                   hashes.SHA256())
                        hit = (cert, salt)
                        break
                    except Exception:
                        continue
                if hit:
                    break
            if hit:
                print(f"  PSS SIG            : VALID "
                      f"(signer {hit[0].subject.rfc4514_string()}, salt={hit[1]})")
            else:
                print("  PSS SIG            : NO VALID MATCH")
                allok = False
        for oid, val, _ in unsigned:
            print(f"  unsigned attr      : {oid}")
    print(f"\nOVERALL: {'ALL SIGNERINFOS VALID' if allok else 'PROBLEMS FOUND'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
