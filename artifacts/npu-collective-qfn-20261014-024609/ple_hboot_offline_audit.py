"""Reproduce the bounded HBOOT analysis; never accesses hardware.

Usage: python3 ple_hboot_offline_audit.py /tmp/kilo/hdk2611 OUTPUT_DIR
Requires capstone. Offsets in the text output are file offsets, not VMAs.
This audit does not validate RSA signatures or determine a live fuse state.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from capstone import CS_ARCH_ARM, CS_MODE_ARM, CS_OP_MEM, Cs


def header_audit(path):
    data = path.read_bytes()
    raw_offset, raw_size, signature_offset = struct.unpack_from("<III", data, 0x458)
    assert struct.unpack_from("<I", data)[0] == 0x55AA55AA
    assert struct.unpack_from("<I", data, 4)[0] == 0x4DC
    assert struct.unpack_from("<I", data, 0x4B8)[0] == 0x33CC33CC
    assert raw_offset + raw_size <= len(data)
    assert hashlib.sha256(data[:0x4BC]).digest() == data[0x4BC:0x4DC]
    assert hashlib.sha256(data[raw_offset:raw_offset + raw_size]).digest() == data[0x2C:0x4C]
    return data, {
        "file": str(path), "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "raw_offset": hex(raw_offset),
        "raw_size": hex(raw_size), "signature_offset_field": hex(signature_offset),
        "header_sha256_matches": True, "payload_sha256_matches": True,
        "rsa_signature_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data, first = header_audit(args.input_dir / "HBOOT1_b.bin")
    _, second = header_audit(args.input_dir / "AS610_HBOOT2_UEFI.fd")
    # Pin the disassembly claims to this exact firmware, rather than applying
    # hard-coded offsets to another release with superficially similar headers.
    assert first["sha256"] == "d1c93aba287adbd9b6a3137fa7c8272fc731f9e5a2410c250fea8dc15c71564d"
    decoder = Cs(CS_ARCH_ARM, CS_MODE_ARM)
    decoder.detail = True
    entry = next(decoder.disasm(data[0x2100:0x2104], 0x2100))
    assert entry.mnemonic == "b" and entry.operands[0].imm == 0x2110
    assert data[0x2108:0x2110] == b"XLOADER!"
    string_offset = data.index(b"compatibility sign verify ret=0x%x\n\0")
    assert string_offset == 0x198D2
    instruction = next(decoder.disasm(data[0x3C68:0x3C6C], 0x3C68))
    assert instruction.mnemonic == "ldrne"
    literal_offset = instruction.address + 8 + instruction.operands[1].mem.disp
    assert literal_offset == 0x4BB4
    assert struct.unpack_from("<I", data, literal_offset)[0] == 0x1300000 + string_offset
    lines = ["HBOOT1_b.bin; ARM little endian; FILE OFFSETS (not runtime addresses).",
             "Selected reachable blocks; inline literal pools may still decode as instructions."]
    for start, end in [(0x2100, 0x2120), (0x2124, 0x2188), (0x3724, 0x37B4),
                       (0x3C28, 0x3C80), (0x10B98, 0x10C08), (0x10C14, 0x11164),
                       (0x16610, 0x16630)]:
        lines.append(f"\nRange {start:#x}..{end:#x}")
        for insn in decoder.disasm(data[start:end], start):
            annotation = ""
            if insn.mnemonic.startswith("ldr") and len(insn.operands) > 1:
                operand = insn.operands[1]
                if operand.type == CS_OP_MEM and insn.reg_name(operand.mem.base) == "pc":
                    slot = insn.address + 8 + operand.mem.disp
                    if 0 <= slot <= len(data) - 4:
                        value = struct.unpack_from("<I", data, slot)[0]
                        annotation = f" ; [{slot:#x}] = {value:#x}"
                        offset = value - 0x1300000
                        if 0x17400 <= offset < 0x19C80:
                            stop = data.find(b"\0", offset)
                            if stop >= offset:
                                annotation += " " + repr(data[offset:stop])
            lines.append(f"{insn.address:08x}: {insn.mnemonic:8} {insn.op_str}{annotation}")
    report = {
        "scope": "Offline analysis of downloaded firmware; not confirmed running firmware",
        "images": [first, second],
        "hboot1_entry_isa": "AArch32 ARM little endian",
        "entry_file_offset": "0x2100",
        "string_pointer_mapping_observed": "runtime address = file offset + 0x1300000",
        "compatibility_message_reference": {"instruction": "0x3c68", "literal": "0x4bb4", "string": "0x198d2"},
        "verification_routine_file_offset": "0x10c14",
        "branch_observed": {
            "comparison_file_offset": "0x10d20",
            "reads_address": "0x10106040", "compares_value": "0x1a4a5252",
            "equal": "payload digest comparison, then logs non-sec boot! on success",
            "unequal": "root public key hash comparison and signature verification path",
            "source_of_state": "Unknown; not established as fuse, register, or writable setting",
        },
        "live_secure_boot_enforcement": "Unknown",
        "runtime_cms_policy_bypassed": False,
        "hardware_access_or_modification": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ple-hboot-offline-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "ple-hboot1-arm-disassembly.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
