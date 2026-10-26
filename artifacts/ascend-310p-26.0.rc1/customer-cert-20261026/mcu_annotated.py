#!/usr/bin/env python3
"""Resolve string literals referenced by instructions, so the MCU trust
logic can be read directly from the disassembly.

For a PC-relative `ldr rN, [pc, #imm]` at address A:
    slot   = ((A + 4) & ~3) + imm          (address of the literal slot)
    value  = u32 at slot                    (the absolute address it loads)

and `value` is then usually the address of a NUL-terminated string in the
image.  This prints, for a given address range, each instruction together with
the string it loads -- which is what makes the MCU's log messages (and thus
the control flow) legible.
"""
import re
import struct
import sys

try:
    from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
except ImportError:
    print("capstone is required", file=sys.stderr)
    sys.exit(1)

BASE = 0x08028000


def main():
    path = sys.argv[1]
    start = int(sys.argv[2], 0)
    end = int(sys.argv[3], 0)

    with open(path, "rb") as fh:
        blob = fh.read()

    strings = {}
    for m in re.finditer(rb"[\x20-\x7e]{3,}", blob[:0x10000]):
        strings[BASE + m.start()] = m.group().decode(errors="replace")

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True
    md.skipdata = True

    for insn in md.disasm(blob[start - BASE:end - BASE], start):
        if insn.id == 0:
            continue
        extra = ""
        if insn.mnemonic.startswith("ldr") and len(insn.operands) == 2:
            mem = insn.operands[1]
            if mem.type == 3 and mem.mem.base == 11:
                slot = ((insn.address + 4) & ~3) + mem.mem.disp
                foff = slot - BASE
                if 0 <= foff < len(blob) - 3:
                    val = struct.unpack_from("<I", blob, foff)[0]
                    s = strings.get(val)
                    if s:
                        extra = f'   ; "{s[:60]}"'
                    else:
                        extra = f"   ; -> 0x{val:08x}"
        print(f"  {insn.address:08x}: {insn.bytes.hex():10s} {insn.mnemonic:8s} {insn.op_str}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
