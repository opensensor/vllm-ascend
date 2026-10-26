#!/usr/bin/env python3
"""Extract and verify the nested RFC-3161 timestamp token in a CMS."""
import hashlib, sys
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from der import parse, walk, oid_str

buf=open(sys.argv[1],'rb').read()
top,_=parse(buf)
# find id-smime-aa-timeStampToken 1.2.840.113549.1.9.16.2.14
tst=None
for path,c in walk(top):
    if c.cls==0 and c.num==16 and c.constructed and c.children:
        oid=oid_str(c.children[0].content)
        if oid=="1.2.840.113549.1.9.16.2.14":
            tst=c; break
if tst is None:
    print("no timestamp token"); sys.exit(1)
# the token is a ContentInfo -> SignedData
sd=None
for path,c in walk(tst):
    if c.cls==0 and c.num==16 and c.constructed:
        sd=c; break
print("timestamp token found")
certs=[]
for path,c in walk(sd):
    if c.cls==0 and c.num==16 and c.constructed and c.children and c.children[0].num==16:
        try: certs.append((x509.load_der_x509_certificate(c.raw), c.raw))
        except Exception: pass
for cert,raw in certs:
    print("  cert:",cert.subject.rfc4514_string(), hashlib.sha256(raw).hexdigest()[:16])
# TSTInfo content
tstinfo=None
for path,c in walk(sd):
    if c.cls==0 and c.num==16 and c.constructed and c.children:
        oid=oid_str(c.children[0].content)
        if oid=="1.2.840.113549.1.9.16.1.4":
            tstinfo=c; break
if tstinfo:
    val=tstinfo.findall(2)
    if val:
        info=val[0].content
        print("  TSTInfo sha256:",hashlib.sha256(info).hexdigest())
# signerinfo of the TST
si_set=None
for c in sd.children:
    if c.cls==0 and c.num==17 and c.constructed: si_set=c
sis=[c for c in si_set.children if c.constructed]
print("  TST signerInfos:",len(sis))
for si in sis:
    md=None; sig=None; sa=None
    for c in si.children:
        if c.cls==2 and c.num==0:
            sa=c
            for a in c.children:
                o=oid_str(a.children[0].content)
                if o=="1.2.840.113549.1.9.4":
                    v=a.findall(2)
                    if v: md=v[0].content
        if c.cls==0 and c.num==4 and not c.constructed: sig=c.content
    n=len(sa.content)
    hdr=bytes([0x31,n]) if n<0x80 else bytes([0x31,0x82,(n>>8)&0xFF,n&0xFF])
    tv=hdr+sa.content
    hit=None
    for cert,raw in certs:
        pub=cert.public_key()
        if not isinstance(pub,rsa.RSAPublicKey): continue
        try:
            pub.verify(sig,tv,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=32),hashes.SHA256())
            hit=cert; break
        except Exception: pass
    print(f"    msgDigest={md.hex() if md else None}")
    if tstinfo and md:
        val=tstinfo.findall(2)
        if val:
            print(f"    TSTInfo digest match: {hashlib.sha256(val[0].content).hexdigest()==md.hex()}")
    print(f"    PSS SIG: {'VALID '+hit.subject.rfc4514_string() if hit else 'NO MATCH'}")
