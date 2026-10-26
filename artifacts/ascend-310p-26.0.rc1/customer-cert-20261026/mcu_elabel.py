#!/usr/bin/env python3
"""Analyse the Ascend 310P MCU electronic label (elabel).

The card's MCU firmware keeps a structured "electronic label" record in the
`BALE` region of its SPI flash.  Its layout, recovered from the MCU firmware
image (ascend_220000_verified.bin, ARM Cortex-M / Thumb-2, base 0x08028000):

    elabel record base r8        -> the 'BALE' region
    item table at 0x0802aed8     -> 21 entries of 0x10 bytes:
                                       u32  flags | item id
                                       u32  max length
                                       u32  offset of the item within the record
                                       u32  string pointer (label text)
    per item at r8 + offset:
        u16 crc16                 -> CRC-16/XMODEM of the item data
        u16 length
        u8  data[length]

`elabel_init` (0x08049aa4) verifies each item with:

    0x08049b8a  ldrh sl, [r8, r3]        ; expected CRC
    0x08049b8e  bl   0x805b2c0           ; crc16 over (r8+r3+4), length r1
    0x08049b92  cmp  sl, r0
    0x08049b94  bne -> "fail to read and verify elabel."

The CRC table lives at 0x08032338 and is the standard CRC-16/CCITT-FALSE table
(poly 0x1021, init 0, no reflection) -- i.e. CRC-16/XMODEM.

**There is no signature anywhere in this path.** The electronic label -- which
carries the board model and serial number -- is protected only by a 16-bit
checksum, and the write path (SPI_FLASH_WritElectronicLabel, 0x0805d088)
explicitly disables the sector write-protect before writing.

Usage:
    python3 mcu_elabel.py <16MB-flash-dump>
    python3 mcu_elabel.py <16MB-flash-dump> --bale 0x2c0000
"""
import struct
import sys

MCU_IMAGE_BASE = 0x08028000
ELABEL_ITEM_TABLE = 0x0802AED8
CRC_TABLE_ADDR = 0x08032338  # CRC-16/CCITT-FALSE table in the MCU image
DEFAULT_BALE = 0x2C0000


def crc16_xmodem(data):
    """CRC-16/XMODEM: poly 0x1021, init 0x0000, no reflection, no final xor."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def load_item_table(mcu_image):
    offset = ELABEL_ITEM_TABLE - MCU_IMAGE_BASE
    items = []
    for index in range(64):
        base = offset + index * 0x10
        if base + 16 > len(mcu_image):
            break
        w0, w1, w2, w3 = struct.unpack_from("<4I", mcu_image, base)
        if w1 == 0 and w2 == 0:
            break
        label = ""
        if MCU_IMAGE_BASE <= w3 < MCU_IMAGE_BASE + len(mcu_image):
            start = w3 - MCU_IMAGE_BASE
            end = mcu_image.find(b"\x00", start)
            label = mcu_image[start:end].decode(errors="replace")
        items.append({"index": index, "id": w0 & 0xFFFF, "flags": w0 >> 16,
                      "maxlen": w1 & 0xFFFF, "offset": w2, "label": label})
    return items


def analyse(flash, mcu_image, bale=DEFAULT_BALE):
    items = load_item_table(mcu_image)

    # The MCU's table-driven CRC is the reflected form; XMODEM is the
    # non-reflected equivalent, so use the bitwise implementation.
    magic = flash[bale:bale + 4]
    print(f"elabel record base : 0x{bale:08x}")
    print(f"region magic       : {magic!r}")
    print()

    results = []
    for item in items:
        pos = bale + item["offset"]
        if pos + 4 > len(flash):
            continue
        crc = struct.unpack_from("<H", flash, pos)[0]
        length = struct.unpack_from("<H", flash, pos + 2)[0]
        data = flash[pos + 4:pos + 4 + length]
        calc = crc16_xmodem(data) if length else 0
        ok = crc == calc and length <= item["maxlen"]
        results.append((item, crc, length, calc, ok, data))

    print(f"{'idx':>3} {'id':>6} {'max':>4} {'off':>7} {'crc':>6} {'len':>5} "
          f"{'calc':>6}  {'status':<6} value")
    print("-" * 100)
    for item, crc, length, calc, ok, data in results:
        shown = data[:34].decode(errors="replace") if data else ""
        print(f"{item['index']:>3} 0x{item['id']:04x} {item['maxlen']:>4} "
              f"0x{item['offset']:06x} 0x{crc:>04x} {length:>5} 0x{calc:>04x}  "
              f"{'OK' if ok else 'BAD':<6} {shown!r}")

    good = sum(1 for r in results if r[4])
    print(f"\n{good}/{len(results)} items validate with CRC-16/XMODEM only.")
    print("The electronic label carries NO signature: a 16-bit checksum is the")
    print("entire integrity mechanism on this path.")
    return 0 if good == len(results) else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    bale = DEFAULT_BALE
    for arg in sys.argv[1:]:
        if arg.startswith("--bale"):
            bale = int(arg.split("=", 1)[1], 0) if "=" in arg else DEFAULT_BALE

    with open(args[0], "rb") as fh:
        flash = fh.read()

    # The MCU image is the 256 KB window at flash offset 0x220000.
    mcu_image = flash[0x220000:0x260000]
    if len(mcu_image) < 0x10000:
        print("MCU image window not found in the dump", file=sys.stderr)
        return 2

    return analyse(flash, mcu_image, bale)


if __name__ == "__main__":
    sys.exit(main())
