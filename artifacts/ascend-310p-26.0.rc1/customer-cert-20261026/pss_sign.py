#!/usr/bin/env python3
"""Produce a CMS SignedData with rsassaPss, matching Ascend's stock structure.

Stock structure (verified from stock_cms.der):
  signedData {
    version 1
    digestAlgorithms { sha256 }                       (with NULL params)
    encapContentInfo { pkcs7-data }                   (detached)
    [0] certificates { signca, signcert }
    [1] crls { root.crl, signca.crl }
    signerInfos { SignerInfo {
      version 1
      issuer, serial
      digestAlgorithm sha256
      [0] signedAttrs {
          contentType    = pkcs7-data
          signingTime
          messageDigest  = SHA-256(content)
          signingCertificateV2 = SHA-256(signer cert DER)
      }
      signatureAlgorithm rsassaPss { sha256, mgf1-sha256, salt 32 }
      signature  (RSA-PSS over DER SET OF signedAttrs)
      [1] unsignedAttrs { id-smime-aa-timeStampToken }
    } }
  }

The timestamp token is optional here (the verifier does not require it); it is
included only if a pre-built TST is supplied.
"""

import hashlib
import sys
from datetime import datetime, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ---- minimal DER writer ----


def tlv(tag, content):
    if len(content) < 0x80:
        return bytes([tag, len(content)]) + content
    if len(content) < 0x100:
        return bytes([tag, 0x81, len(content)]) + content
    return bytes([tag, 0x82, len(content) >> 8, len(content) & 0xFF]) + content


def seq(*parts):
    return tlv(0x30, b"".join(parts))


def setof(*parts):
    return tlv(0x31, b"".join(parts))


def oid(dotted):
    a = [int(x) for x in dotted.split(".")]
    out = bytes([a[0] * 40 + a[1]])
    for v in a[2:]:
        chunk = []
        chunk.append(v & 0x7F)
        v >>= 7
        while v:
            chunk.append((v & 0x7F) | 0x80)
            v >>= 7
        out += bytes(reversed(chunk))
    return tlv(0x06, out)


def integer(n):
    if n == 0:
        return tlv(0x02, b"\x00")
    b = n.to_bytes((n.bit_length() + 8) // 8, "big")
    return tlv(0x02, b)


def octetstring(b):
    return tlv(0x04, b)


def null():
    return b"\x05\x00"


def ctx(n, content, constructed=True):
    tag = 0xA0 | n if constructed else 0x80 | n
    return tlv(tag, content)


def utctime(dt):
    s = dt.strftime("%y%m%d%H%M%SZ").encode()
    return tlv(0x17, s)


def explicit(n, content):
    return tlv(0xA0 | n, content)


def implicit_set(n, content):
    return tlv(0xA0 | n, content)


def algid_sha256():
    # AlgorithmIdentifier with NULL params (as Huawei emits)
    return seq(oid("2.16.840.1.101.3.4.2.1"), null())


def algid_pss():
    # rsassaPss: hashAlgorithm sha256 (NULL params), maskGenAlgorithm mgf1 sha256,
    # saltLength 32, trailerField 1
    mgf1 = seq(oid("1.2.840.113549.1.1.8"), algid_sha256())
    # The stock Huawei container OMITS trailerField (it ends at saltLength),
    # so match that byte-for-byte.
    pss = seq(explicit(0, algid_sha256()), explicit(1, mgf1), explicit(2, integer(32)))
    return seq(oid("1.2.840.113549.1.1.10"), pss)


def strip_tlv_header(t):
    """Return the CONTENT of a TLV (without its tag/length header)."""
    b = t
    i = 1
    ln = b[i]
    i += 1
    if ln & 0x80:
        nb = ln & 0x7F
        ln = int.from_bytes(b[i : i + nb], "big")
        i += nb
    return b[i : i + ln]


def gentime(dt):
    """DER GeneralizedTime (tag 0x18), UTC, with Z."""
    return tlv(0x18, dt.strftime("%Y%m%d%H%M%SZ").encode())


def attr(oid_str, values):
    # `values` is a list of already-encoded DER elements; wrap them in ONE SET OF.
    return seq(oid(oid_str), setof(*values))


def attr_raw(oid_str, value_der):
    # value_der is a single already-encoded DER element (e.g. a ContentInfo).
    return seq(oid(oid_str), setof(value_der))


# ---- main ----


def main(
    ini_path, out_path, signer_pem, key_pem, cacert_pem, crl_files=(), tst_der=None, ts_cert_pem=None, ts_key_pem=None
):
    with open(ini_path, "rb") as fh:
        content = fh.read()
    digest = hashlib.sha256(content).digest()

    with open(signer_pem, "rb") as fh:
        signer = x509.load_pem_x509_certificate(fh.read())
    with open(cacert_pem, "rb") as fh:
        ca = x509.load_pem_x509_certificate(fh.read())
    with open(key_pem, "rb") as fh:
        key = serialization.load_pem_private_key(fh.read(), None)

    now = datetime.now(timezone.utc).replace(microsecond=0)

    signed_attrs = setof(
        attr("1.2.840.113549.1.9.3", [oid("1.2.840.113549.1.7.1")]),
        attr("1.2.840.113549.1.9.5", [utctime(now)]),
        attr("1.2.840.113549.1.9.4", [octetstring(digest)]),
        # signingCertificateV2: SEQUENCE { SEQUENCE { SEQUENCE {
        #   digestAlgorithm AlgorithmIdentifier, digest OCTET STRING } } }
        # (three levels, matching the stock container exactly)
        attr(
            "1.2.840.113549.1.9.16.2.47",
            [
                seq(
                    seq(
                        seq(
                            algid_sha256(),
                            octetstring(hashlib.sha256(signer.public_bytes(serialization.Encoding.DER)).digest()),
                        )
                    )
                )
            ],
        ),
    )

    # The signature covers the DER encoding of signedAttrs with the
    # context-specific [0] tag replaced by the universal SET OF tag (0x31),
    # per RFC 5652 section 5.4.
    attrs_content = strip_tlv_header(signed_attrs)
    n = len(attrs_content)
    if n < 0x80:
        set_hdr = bytes([0x31, n])
    elif n < 0x100:
        set_hdr = bytes([0x31, 0x81, n])
    else:
        set_hdr = bytes([0x31, 0x82, (n >> 8) & 0xFF, n & 0xFF])
    sig = key.sign(
        set_hdr + attrs_content,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )

    # SignerInfo.version, sid (issuerAndSerialNumber = SEQUENCE{Issuer, INTEGER}),
    # digestAlgorithm, [0] signedAttrs, signatureAlgorithm, signature
    issuer_and_serial = seq(signer.issuer.public_bytes(), integer(signer.serial_number))
    signer_info = seq(
        integer(1),
        issuer_and_serial,
        algid_sha256(),
        implicit_set(0, strip_tlv_header(signed_attrs)),
        algid_pss(),
        octetstring(sig),
    )

    certs = implicit_set(
        0, ca.public_bytes(serialization.Encoding.DER) + signer.public_bytes(serialization.Encoding.DER)
    )
    crls = b""
    if crl_files:
        blobs = []
        for f in crl_files:
            with open(f, "rb") as fh:
                blobs.append(fh.read())
        blob = b"".join(blobs)
        crls = implicit_set(1, blob)

    unsigned = b""
    if tst_der:
        unsigned = implicit_set(1, attr_raw("1.2.840.113549.1.9.16.2.14", tst_der))
    elif ts_cert_pem and ts_key_pem:
        # Build an RFC-3161 timestamp token over the signature value.
        with open(ts_cert_pem, "rb") as fh:
            ts_cert = x509.load_pem_x509_certificate(fh.read())
        with open(ts_key_pem, "rb") as fh:
            ts_key = serialization.load_pem_private_key(fh.read(), None)
        # RFC3161 TSTInfo, exactly the stock shape:
        #   version INTEGER(1)
        #   policy  OID
        #   messageImprint SEQUENCE { AlgorithmIdentifier, OCTET STRING }
        #   serialNumber INTEGER
        #   genTime GeneralizedTime
        tst_info = seq(
            integer(1),
            oid("1.3.6.1.4.1.2011.201.1.1.1"),
            seq(algid_sha256(), octetstring(hashlib.sha256(sig).digest())),
            integer(ts_cert.serial_number),
            gentime(now),
        )
        tst_attrs = setof(
            attr("1.2.840.113549.1.9.5", [utctime(now)]),
            attr("1.2.840.113549.1.9.4", [octetstring(hashlib.sha256(tst_info).digest())]),
        )
        tst_sig = ts_key.sign(
            (bytes([0x31, len(strip_tlv_header(tst_attrs))]) + strip_tlv_header(tst_attrs)),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )
        tst_si = seq(
            integer(1),
            seq(ts_cert.issuer.public_bytes(), integer(ts_cert.serial_number)),
            algid_sha256(),
            implicit_set(0, strip_tlv_header(tst_attrs)),
            algid_pss(),
            octetstring(tst_sig),
        )
        tst_sd = seq(
            integer(3),
            setof(algid_sha256()),
            seq(oid("1.2.840.113549.1.9.16.1.4"), explicit(0, tst_info)),
            implicit_set(0, ts_cert.public_bytes(serialization.Encoding.DER)),
            setof(tst_si),
        )
        tst_der = seq(oid("1.2.840.113549.1.7.2"), ctx(0, tst_sd))
        unsigned = implicit_set(1, attr_raw("1.2.840.113549.1.9.16.2.14", tst_der))

    signed_data = seq(
        integer(1),
        setof(algid_sha256()),
        seq(oid("1.2.840.113549.1.7.1")),
        certs,
        crls,
        setof(signer_info),
    )
    if unsigned:
        # rebuild signerInfo with unsigned attrs
        si = seq(
            integer(1),
            issuer_and_serial,
            algid_sha256(),
            implicit_set(0, strip_tlv_header(signed_attrs)),
            algid_pss(),
            octetstring(sig),
            unsigned,
        )
        signed_data = seq(integer(1), setof(algid_sha256()), seq(oid("1.2.840.113549.1.7.1")), certs, crls, setof(si))

    content_info = seq(oid("1.2.840.113549.1.7.2"), ctx(0, signed_data))
    with open(out_path, "wb") as fh:
        fh.write(content_info)
    print(f"wrote {out_path} ({len(content_info)} bytes)")
    print(f"  messageDigest = {digest.hex()}")


if __name__ == "__main__":
    FULL = "/run/media/matteius/20TB-drive/vllm-ascend/artifacts/ascend-310p-usercert-backup-20261021/full"
    ini, out = sys.argv[1], sys.argv[2]
    use_tst = "--tst" in sys.argv
    main(
        ini,
        out,
        f"{FULL}/signcert.pem",
        f"{FULL}/signcert_pri.pem",
        f"{FULL}/signca.pem",
        crl_files=[f"{FULL}/rootG2.crl", f"{FULL}/sign.crl"],
        ts_cert_pem=f"{FULL}/ts.pem" if use_tst else None,
        ts_key_pem=f"{FULL}/ts_pri.pem" if use_tst else None,
    )
