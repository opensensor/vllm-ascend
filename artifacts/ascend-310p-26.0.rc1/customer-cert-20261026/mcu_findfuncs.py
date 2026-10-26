#!/usr/bin/env python3
"""Locate the MCU trust-verification code in ascend_220000_verified.bin.

The MCU firmware (Cortex-M, Thumb-2, LiteOS, mbedTLS) is mapped at base
0x08028000.  Its code region is 0x0000..0x2800 and it references strings
through literal pools of absolute 32-bit addresses.

Strategy:
  1. Build the set of string addresses.
  2. Find the literal pools (runs of string-address words) and note which
     string each pool slot holds.
  3. Walk the code, and for every PC-relative `ldr rN, [pc, #imm]` resolve the
     pool slot it reads.  If the slot holds a string address, we have found
     the code that uses that string.
  4. Report, for each interesting name, the code addresses that reference it.

The trust machinery we want is exactly the set of names the previous session
listed from strings alone; this turns them into code locations.
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
    # trust store
    "trustlist", "blocklist", "https", "https crl", "switch", "null",
    # customised-certificate path
    "customized_verify", "customized_msg_check", "customized_info_read",
    "customized_info_write", "customized_info_clear", "customized_info_update",
    "customized_last_frame_handle", "customized_info_init",
    "read_customized_info_callback", "write_customized_info_callback",
    "clear_customized_info_callback",
    # certificate write path
    "WriteCertToFlash", "ClearCertToFlash", "cert_update_result",
    "get_cert_region_id", "write_cert_info", "renew_cert_info",
    "write_cert_frame_data", "get_certificate_info", "cert_data_init",
    "get_random_info", "write_boot_random", "set_switch_state",
    "UpdateCertSwitchState", "SaveCertRandom", "UpdateCertRandom",
    "ReadCertRandomData",
    # signature verification
    "McuSignatureVerify", "CheckPubKeyHash", "RsaSignatureVerify",
    "RsaVerify", "VerifySig", "CalHash", "CheckFileHead",
    "read_rsa_sign_info", "read_rsa_head_item",
    # secureboot info
    "get_secureboot_data", "set_secureboot_info_callback",
    "get_secureboot_info_callback", "set_secureboot_param_check",
    "check_input_secureboot_param",
    # elabel
    "elabel_init", "elabel_read", "elabel_update", "elabel_write",
    "elabel_exit", "elabel_set_default", "elabel_init_default",
    "board_read_elable_response_callback",
    # comm
    "comm_distribute_handle", "MCU_COMM_Init", "comm_create_message",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ascend_220000_verified.bin"
    with open(path, "rb") as fh:
        blob = fh.read()

    strings = {}
    for m in re.finditer(rb"[\x20-\x7e]{3,}", blob[:0x10000]):
        strings[BASE + m.start()] = m.group().decode(errors="replace")

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True

    refs = {}
    for insn in md.disasm(blob[:CODE_END], BASE):
        if not insn.mnemonic.startswith("ldr") or len(insn.operands) != 2:
            continue
        mem = insn.operands[1]
        if mem.type != 3 or mem.mem.base != 11:
            continue
        pc_base = (insn.address + 4) & ~3
        slot = pc_base + mem.mem.disp
        if not (0 <= slot < len(blob) - 3):
            continue
        val = struct.unpack_from("<I", blob, slot)[0]
        if val in strings:
            refs.setdefault(strings[val], []).append(insn.address)

    print(f"=== {len(refs)} strings referenced from code ===")
    hit = 0
    for name in TARGETS:
        if name not in refs:
            continue
        hit += 1
        addrs = refs[name]
        shown = ", ".join(f"0x{a:08x}" for a in addrs[:8])
        print(f"  {name:<40} {shown}")
    print(f"\n{hit}/{len(TARGETS)} targets located in code")
    return 0


if __name__ == "__main__":
    sys.exit(main())
