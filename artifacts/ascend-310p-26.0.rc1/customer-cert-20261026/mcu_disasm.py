#!/usr/bin/env python3
"""Disassemble named functions in the Ascend 310P MCU firmware.

ascend_220000_verified.bin is the card's MCU flash (Cortex-M, Thumb-2,
LiteOS, mbedTLS), mapped at base 0x08028000.

The firmware's log macro passes the function name as a string, so every named
function has at least one code site that loads its name.  Those sites identify
the *body* of the function; the entry point is found by scanning backwards to
the nearest Thumb prologue (push {..,lr}).

Usage:
    python3 mcu_disasm.py <dump> [function ...]
    python3 mcu_disasm.py <dump> --list
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

WANTED = [
    "customized_verify", "customized_msg_check", "customized_info_read",
    "customized_info_write", "customized_info_clear", "customized_info_update",
    "customized_last_frame_handle", "customized_info_init",
    "WriteCertToFlash", "ClearCertToFlash", "cert_update_result",
    "get_cert_region_id", "write_cert_info", "renew_cert_info",
    "write_cert_frame_data", "get_certificate_info", "cert_data_init",
    "get_random_info", "write_boot_random", "set_switch_state",
    "UpdateCertSwitchState", "SaveCertRandom", "UpdateCertRandom",
    "ReadCertRandomData", "get_boot_param", "set_boot_param_to_flash",
    "McuSignatureVerify", "CheckPubKeyHash", "RsaSignatureVerify",
    "RsaVerify", "VerifySig", "CalHash", "CheckFileHead",
    "read_rsa_sign_info", "read_rsa_head_item",
    "get_secureboot_data", "set_secureboot_info_callback",
    "get_secureboot_info_callback", "set_secureboot_param_check",
    "check_input_secureboot_param",
    "elabel_init", "elabel_read", "elabel_update", "elabel_write",
    "elabel_exit", "elabel_set_default", "elabel_init_default",
    "board_read_elable_response_callback",
    "read_customized_info_callback", "write_customized_info_callback",
    "clear_customized_info_callback",
    "comm_distribute_handle", "MCU_COMM_Init", "comm_create_message",
    "comm_async_remote_message_handle",
    "mps_ver_check", "get_mps_ver",
    "verify_firmware_file", "check_firmware_info", "read_cms_file_info",
    "read_cms_file_info_item", "verify_flash_info",
    "flash_sha256_check", "sha256_check", "sha256_get",
    "SPI_FLASH_ReadRegion", "SPI_FLASH_WriteRegion", "get_flash_region",
    "RsaSignatureVerify",
]


def build_index(blob):
    """Return {name: [code addresses where the name string is loaded]}."""
    strings = {}
    for m in re.finditer(rb"[\x20-\x7e]{3,}", blob[:0x10000]):
        strings[BASE + m.start()] = m.group().decode(errors="replace")

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True
    md.skipdata = True

    refs = {}
    for insn in md.disasm(blob[0x20000:0x40000], BASE + 0x20000):
        if insn.id == 0:
            continue
        if not insn.mnemonic.startswith("ldr") or len(insn.operands) != 2:
            continue
        mem = insn.operands[1]
        if mem.type != 3 or mem.mem.base != 11:
            continue
        pc_base = (insn.address + 4) & ~3
        foff = (pc_base + mem.mem.disp) - BASE
        if not (0 <= foff < len(blob) - 3):
            continue
        val = struct.unpack_from("<I", blob, foff)[0]
        if val in strings:
            refs.setdefault(strings[val], []).append(insn.address)
    return refs


def find_prologue(blob, addr):
    """Scan backwards from addr for a Thumb function prologue (push {..,lr})."""
    off = addr - BASE
    for back in range(0, 0x2000, 2):
        p = off - back
        if p < 0:
            break
        b = blob[p]
        # PUSH {reglist, lr}  -> 0xB5xx ; or STMDB sp!, {...} 0xE92D xxxx
        if b == 0xB5 and (blob[p + 1] & 0x80):
            return BASE + p
        if b == 0xE9 and (blob[p + 1] & 0xF0) == 0x20:
            return BASE + p
    return None


def main():
    path = sys.argv[1]
    with open(path, "rb") as fh:
        blob = fh.read()

    refs = build_index(blob)

    if "--list" in sys.argv:
        for name in WANTED:
            if name in refs:
                print(f"{name:<40} {', '.join(hex(a) for a in refs[name][:4])}")
        return 0

    wanted = [a for a in sys.argv[2:] if not a.startswith("--")]
    if not wanted:
        wanted = WANTED

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = False

    for name in wanted:
        if name not in refs:
            print(f"### {name}: not found")
            continue
        print(f"\n{'=' * 78}\n### {name}   string loaded at {', '.join(hex(a) for a in refs[name][:4])}\n{'=' * 78}")
        for site in refs[name]:
            start = find_prologue(blob, site)
            if start is None:
                print(f"  (no prologue found for site 0x{site:08x})")
                continue
            print(f"  -- function body from 0x{start:08x} (file 0x{start - BASE:05x})")
            foff = start - BASE
            for count, insn in enumerate(md.disasm(blob[foff:foff + 0x600], start)):
                mark = "  <<<" if insn.address == site else ""
                print(f"    {insn.address:08x}: {insn.bytes.hex():10s} {insn.mnemonic:8s} {insn.op_str}{mark}")
                if count > 120:
                    print("    ...")
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
