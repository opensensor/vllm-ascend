#!/usr/bin/env python3
"""Locate the MCU trust-verification functions in ascend_220000_verified.bin.

The dump is the card's MCU flash (Cortex-M, Thumb-2, LiteOS, mbedTLS) and is
mapped at base address 0x08028000.

The firmware reaches its strings through a literal pool of absolute 32-bit
addresses.  To go from a string to the function that uses it:

  1. find the literal-pool slots whose value equals the string's address;
  2. find the `ldr rN, [pc, #imm]` instructions that load from those slots
     (Thumb rule: base = (pc + 4) & ~3, target = base + imm);
  3. report the containing function.

This gives the code address of the trust machinery:
    customized_verify / customized_msg_check
    WriteCertToFlash / ClearCertToFlash
    McuSignatureVerify / CheckPubKeyHash / RsaVerify / VerifySig
    get_cert_region_id / write_cert_info / renew_cert_info
    board_read_elable_response_callback
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
CODE_END = 0x2800

TARGETS = [
    "customized_verify",
    "customized_msg_check",
    "customized_info_read",
    "customized_info_write",
    "customized_info_clear",
    "customized_info_update",
    "customized_last_frame_handle",
    "WriteCertToFlash",
    "ClearCertToFlash",
    "cert_update_result",
    "get_cert_region_id",
    "write_cert_info",
    "renew_cert_info",
    "write_cert_frame_data",
    "get_certificate_info",
    "McuSignatureVerify",
    "CheckPubKeyHash",
    "RsaSignatureVerify",
    "RsaVerify",
    "VerifySig",
    "CalHash",
    "CheckFileHead",
    "read_rsa_sign_info",
    "read_rsa_head_item",
    "trustlist",
    "blocklist",
    "https",
    "https crl",
    "set_switch_state",
    "set_secureboot_info_callback",
    "get_secureboot_data",
    "board_read_elable_response_callback",
    "comm_distribute_handle",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ascend_220000_verified.bin"
    with open(path, "rb") as fh:
        blob = fh.read()

    strings = {m.start(): m.group().decode(errors="replace")
               for m in re.finditer(rb"[\x20-\x7e]{4,}", blob[:0x10000])}
    addr_of = {BASE + off: name for off, name in strings.items()}

    # 1. literal-pool slots holding a string address
    pool = {}
    for off in range(0, len(blob) - 3, 4):
        val = struct.unpack_from("<I", blob, off)[0]
        if val in addr_of:
            pool.setdefault(addr_of[val], []).append(off)

    # 2. LDR-literal instructions that read from those slots
    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True
    slot_users = {}
    for insn in md.disasm(blob[:CODE_END], 0):
        if not insn.mnemonic.startswith("ldr") or len(insn.operands) != 2:
            continue
        mem = insn.operands[1]
        if mem.type != 3 or mem.mem.base != 11:
            continue
        base = (insn.address + 4) & ~3
        target = base + mem.mem.disp
        if 0 <= target < len(blob) - 3:
            slot_users.setdefault(target, []).append(insn.address)

    print(f"{'function/const':<40} {'string addr':<12} {'pool slot':<12} 'ldr sites'")
    print("-" * 100)
    found = {}
    for name in TARGETS:
        if name not in pool:
            continue
        for slot in pool[name]:
            users = slot_users.get(slot, [])
            sites = ", ".join(f"0x{u:04x}" for u in users[:6]) or "(none)"
            print(f"{name:<40} 0x{BASE+slot+0:08x}  0x{slot:04x}      {sites}")
            for u in users:
                found.setdefault(name, []).append(u)

    print("\n=== code addresses of the trust machinery ===")
    for name in TARGETS:
        if name in found:
            for u in found[name]:
                print(f"  0x{u:04x}  (addr 0x{BASE+u:08x})  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
