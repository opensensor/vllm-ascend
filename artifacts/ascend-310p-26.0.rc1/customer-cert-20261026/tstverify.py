#!/usr/bin/env python3
"""Extract and verify the nested RFC-3161 timestamp token in a CMS.

The token lives in the SignerInfo's unsignedAttrs under
id-smime-aa-timeStampToken (1.2.840.113549.1.9.16.2.14).  It is itself a
ContentInfo/SignedData, so verifying it means repeating the SignerInfo
verification on the nested structure: RSA-PSS over the re-tagged SET OF
signedAttrs, with the messageDigest attribute checked against the TSTInfo.

Structure of the token, as walked by `der.walk` (paths are child indices):

    TST/1              SEQUENCE   -- the [0] EXPLICIT ContentInfo
    TST/1/0            SEQUENCE   -- the inner ContentInfo
    TST/1/0/1          [0]        -- EXPLICIT wrapper
    TST/1/0/1/0        SEQUENCE   -- SignedData
    TST/1/0/1/0/1      [0]        -- encapContentInfo (carries the certs)
    TST/1/0/1/0/3      [0]        -- unsignedAttrs of the nested SignerInfo
    TST/1/0/1/0/3/0/0/3/1  SET    -- the nested signerInfos
"""

import contextlib
import hashlib
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from der import oid_str, parse, walk

OID_TIME_STAMP_TOKEN = "1.2.840.113549.1.9.16.2.14"
OID_TST_INFO = "1.2.840.113549.1.9.16.1.4"
OID_MESSAGE_DIGEST = "1.2.840.113549.1.9.4"

TAG_SEQUENCE = 16
TAG_SET = 17
CLASS_CONTEXT = 2


def find_time_stamp_token(top):
    """Return the unsignedAttrs element holding the timestamp token."""
    for _path, node in walk(top):
        if node.cls == 0 and node.num == TAG_SEQUENCE and node.constructed and node.children:
            if oid_str(node.children[0].content) == OID_TIME_STAMP_TOKEN:
                return node
    return None


def find_nested_signed_data(tst):
    """Return the nested SignedData SEQUENCE inside the timestamp token.

    The timestamp token is an AttributeValue holding a ContentInfo:

        TST            SEQUENCE { OID, SET { SEQUENCE { OID, [0] { SignedData } } } }

    so the SignedData is reached by descending: TST -> SET -> inner
    ContentInfo -> its [0] EXPLICIT -> the SignedData SEQUENCE.
    """
    # TST/1 is the SET holding the ContentInfo
    wrapper = tst.find(TAG_SET)
    if wrapper is None or not wrapper.children:
        return None
    content_info = wrapper.children[0]
    if content_info is None:
        return None
    # ContentInfo = SEQUENCE { OID, [0] EXPLICIT SignedData }
    explicit = content_info.find(0, cls=CLASS_CONTEXT)
    if explicit is None or not explicit.children:
        return None
    signed_data = explicit.children[0]
    if signed_data is None or signed_data.num != TAG_SEQUENCE:
        return None
    return signed_data


def collect_certs(sd):
    """Collect every X.509 certificate carried by the nested SignedData.

    A CertificateChoices is SEQUENCE { OID, [0] EXPLICIT Certificate }, so the
    walk looks for any SEQUENCE whose first child is a SEQUENCE and then tries
    to parse it as an X.509 certificate.
    """
    certs = []
    for _path, node in walk(sd):
        if (
            node.cls == 0
            and node.num == TAG_SEQUENCE
            and node.constructed
            and node.children
            and node.children[0].num == TAG_SEQUENCE
        ):
            with contextlib.suppress(Exception):
                certs.append((x509.load_der_x509_certificate(node.raw), node.raw))
    return certs


def find_tst_info(sd):
    """Return the encapsulated TSTInfo element, if present.

    The TSTInfo is carried inside the encapContentInfo as
    SEQUENCE { OID id-ct-TSTInfo, [0] EXPLICIT OCTET STRING }, so the OID is
    a direct child of the encapContentInfo SEQUENCE rather than the first
    child of a nested SEQUENCE.
    """
    for _path, node in walk(sd):
        if node.cls == 0 and node.num == 6 and not node.constructed:
            if oid_str(node.content) == OID_TST_INFO:
                parent = node
                return parent
    return None


def tst_info_bytes(sd):
    """Return the raw TSTInfo content (the OCTET STRING payload)."""
    for _path, node in walk(sd):
        if node.cls == 0 and node.num == 6 and not node.constructed:
            if oid_str(node.content) != OID_TST_INFO:
                continue
            # the OID's parent SEQUENCE holds [0] EXPLICIT { OCTET STRING }
            for _p2, parent in walk(sd):
                if parent.cls != 0 or parent.num != TAG_SEQUENCE or not parent.children:
                    continue
                if parent.children[0] is not node:
                    continue
                for _p3, wrapper in walk(parent):
                    if wrapper.cls == CLASS_CONTEXT and wrapper.num == 0 and wrapper.children:
                        inner = wrapper.children[0]
                        if inner.cls == 0 and inner.num == 4:
                            return inner.content
    return None


def find_signer_infos(sd):
    """Return the SignerInfo SEQUENCEs of the nested SignedData.

    SignedData is SEQUENCE { version, digestAlgorithms, encapContentInfo,
    [0] certificates, [1] crls, signerInfos }.  There are two SET children --
    digestAlgorithms and signerInfos -- and only the latter holds SignerInfos,
    which are recognised by carrying version + digestAlgorithm + the rest.
    """
    sets = [node for node in sd.children if node.cls == 0 and node.num == TAG_SET and node.constructed]
    for node in sets:
        candidates = [
            child for child in node.children if child.cls == 0 and child.num == TAG_SEQUENCE and child.constructed
        ]
        # a SignerInfo starts with an INTEGER version and an AlgorithmIdentifier
        if candidates and candidates[0].children:
            first = candidates[0].children[0]
            if first.cls == 0 and first.num == 2 and not first.constructed:
                return candidates
    return []


def extract_signer_fields(si):
    """Pull messageDigest, signedAttrs and the signature out of one SignerInfo.

    A SignerInfo is SEQUENCE { version, sid, digestAlgorithm, signedAttrs [0],
    signatureAlgorithm, signature, unsignedAttrs [1] }.  signedAttrs is
    [0] IMPLICIT, and the signature is the OCTET STRING.
    """
    message_digest = None
    signed_attrs = None
    signature = None
    for child in si.children:
        if child.cls == CLASS_CONTEXT and child.num == 0 and child.constructed:
            signed_attrs = child
            for attr in child.children:
                if not attr.children:
                    continue
                if oid_str(attr.children[0].content) == OID_MESSAGE_DIGEST:
                    # Attribute = SEQUENCE { OID, SET OF OCTET STRING }
                    for values in attr.findall(17):
                        for value in values.children:
                            if value.cls == 0 and value.num == 4:
                                message_digest = value.content
        elif child.cls == 0 and child.num == 4 and not child.constructed:
            signature = child.content
    return message_digest, signed_attrs, signature


def retag_as_set(signed_attrs):
    """Re-tag the [0] IMPLICIT signedAttrs as a SET OF, as CMS requires.

    The signature is computed over the *DER encoding* of the signedAttrs with
    the context tag replaced by the universal SET tag -- the original length
    bytes must be kept verbatim, so the tag byte is patched in place rather
    than the length being recomputed.
    """
    raw = bytearray(signed_attrs.raw)
    raw[0] = 0x31
    return bytes(raw)


def main():
    with open(sys.argv[1], "rb") as handle:
        buf = handle.read()
    top, _ = parse(buf)

    tst = find_time_stamp_token(top)
    if tst is None:
        print("no timestamp token")
        return 1

    sd = find_nested_signed_data(tst)
    if sd is None:
        print("timestamp token found, but no nested SignedData")
        return 1
    print("timestamp token found")

    certs = collect_certs(sd)
    for cert, raw in certs:
        print("  cert:", cert.subject.rfc4514_string(), hashlib.sha256(raw).hexdigest()[:16])

    tstinfo = tst_info_bytes(sd)
    if tstinfo is not None:
        print("  TSTInfo sha256:", hashlib.sha256(tstinfo).hexdigest())

    signer_infos = find_signer_infos(sd)
    print("  TST signerInfos:", len(signer_infos))

    for si in signer_infos:
        message_digest, signed_attrs, signature = extract_signer_fields(si)
        if signed_attrs is None or signature is None:
            print("    incomplete SignerInfo")
            continue
        to_verify = retag_as_set(signed_attrs)

        matched = None
        for cert, _raw in certs:
            pub = cert.public_key()
            if not isinstance(pub, rsa.RSAPublicKey):
                continue
            try:
                pub.verify(
                    signature,
                    to_verify,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                    hashes.SHA256(),
                )
            except Exception:
                continue
            matched = cert
            break

        print(f"    msgDigest={message_digest.hex() if message_digest else None}")
        if tstinfo is not None and message_digest:
            actual = hashlib.sha256(tstinfo).hexdigest()
            print(f"    TSTInfo digest match: {actual == message_digest.hex()}")
        if matched is not None:
            print(f"    PSS SIG: VALID {matched.subject.rfc4514_string()}")
        else:
            print("    PSS SIG: NO MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
