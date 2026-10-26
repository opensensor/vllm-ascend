#!/usr/bin/env python3
"""Resolve PC-relative literal loads in the Ascend 310P MCU firmware.

ascend_220000_verified.bin is the card's MCU flash (Cortex-M, Thumb-2,
LiteOS, mbedTLS).  Its code region is 0x00000..0x02800 and its string table
starts around 0x02800.

The firmware addresses strings through PC-relative literal pools:
    ldr  rN, [pc, #imm]      ; load the address of a string
with the Thumb rule  base = (pc_of_ldr + 4) & ~3,  target = base + imm.

This script walks the code, resolves every such literal, and reports which
function-scope code references each string -- so the trust-relevant functions
(customized_verify, WriteCertToFlash, CheckPubKeyHash, trustlist, ...) can be
located and disassembled.
"""
import re
import struct
import sys

try:
    from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
except ImportError:
    print("capstone is required", file=sys.stderr)
    sys.exit(1)

FLASH = "ascend_220000_verified.bin"
CODE_END = 0x2800


def load(path):
    with open(path, "rb") as fh:
        return fh.read()


def build_strings(blob):
    return {m.start(): m.group().decode(errors="replace")
            for m in re.finditer(rb"[\x20-\x7e]{4,}", blob[:0x10000])}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else FLASH
    blob = load(path)
    strings = build_strings(blob)

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True

    refs = {}
    for insn in md.disasm(blob[:CODE_END], 0):
        if not insn.mnemonic.startswith("ldr") or len(insn.operands) != 2:
            continue
        mem = insn.operands[1]
        if mem.type != 3:            # ARM_OP_MEM
            continue
        # PC-relative: base register is PC (capstone id 11)
        if mem.mem.base != 11:
            continue
        base = (insn.address + 4) & ~3
        target = base + mem.mem.disp
        if not (0 <= target < len(blob) - 3):
            continue
        value = struct.unpack_from("<I", blob, target)[0]
        if value in strings:
            refs.setdefault(strings[value], []).append(insn.address)

    print(f"=== {len(refs)} strings referenced from code ===\n")
    keywords = ("cert", "trust", "block", "verify", "sign", "key", "hash",
                "custom", "random", "switch", "region", "label", "rsa", "pub",
                "boot", "update", "info", "type", "switch", "mps", "vrd")
    for name, addrs in sorted(refs.items(), key=lambda kv: min(kv[1])):
        if any(k in name.lower() for k in keywords):
            print(f"  0x{min(addrs):04x}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
