#!/usr/bin/env python3
"""Build a symbol map for the Ascend 310P MCU firmware from ascend_220000_verified.bin.

The dump is the card's MCU flash, mapped at base address 0x08028000.  The
firmware keeps a table of {name-string-address, function-pointer} pairs; the
name strings are the ones the firmware also uses in its log messages
("customized_verify", "WriteCertToFlash", ...).

Walking that table recovers the real code address of every named function,
which is what makes the trust machinery disassemblable.

Layout observed in the dump:

    file 0x21490: 0x0802a88e   <- name string address
    file 0x21494: 0x0802ab6a   <- function pointer
    file 0x21498: 0x0802a944
    file 0x2149c: 0x0802a8d3
    file 0x214a0: 0x0802ab58   <- "customized_verify"

i.e. pairs of (u32 name-addr, u32 func-addr), terminated by a non-image word.
"""
import re
import struct
import sys

BASE = 0x08028000


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ascend_220000_verified.bin"
    with open(path, "rb") as fh:
        blob = fh.read()

    strings = {}
    for m in re.finditer(rb"[\x20-\x7e]{3,}", blob[:0x10000]):
        strings[m.start()] = m.group().decode(errors="replace")

    addr_to_name = {BASE + off: name for off, name in strings.items()}

    def in_image(v):
        return BASE <= v < BASE + len(blob)

    # Collect every (name_addr, func_addr) pair where both are in-image.
    pairs = []
    for off in range(0, len(blob) - 7, 4):
        a = struct.unpack_from("<I", blob, off)[0]
        b = struct.unpack_from("<I", blob, off + 4)[0]
        if a in addr_to_name and in_image(b):
            name = addr_to_name[a]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                pairs.append((name, b, off))

    # The name table is a dense run; keep the first occurrence of each name
    # but prefer the one inside the longest contiguous table.
    by_name = {}
    for name, func, off in pairs:
        by_name.setdefault(name, []).append((func, off))

    print(f"=== {len(by_name)} named functions recovered ===\n")
    wanted = [
        "customized_verify", "customized_msg_check", "customized_info_read",
        "customized_info_write", "customized_info_clear", "customized_info_update",
        "customized_last_frame_handle", "customized_info_init",
        "WriteCertToFlash", "ClearCertToFlash", "cert_update_result",
        "get_cert_region_id", "write_cert_info", "renew_cert_info",
        "write_cert_frame_data", "get_certificate_info", "cert_data_init",
        "get_random_info", "write_boot_random", "set_switch_state",
        "UpdateCertSwitchState", "SaveCertRandom", "UpdateCertRandom",
        "ReadCertRandomData", "get_boot_param", "set_boot_param",
        "WriteBackBootParam", "set_boot_param_to_flash",
        "McuSignatureVerify", "CheckPubKeyHash", "RsaSignatureVerify",
        "RsaVerify", "VerifySig", "CalHash", "CheckFileHead",
        "read_rsa_sign_info", "read_rsa_head_item",
        "get_secureboot_data", "set_secureboot_info_callback",
        "get_secureboot_info_callback", "set_secureboot_param_check",
        "check_input_secureboot_param",
        "board_read_elable_response_callback",
        "read_customized_info_callback", "write_customized_info_callback",
        "clear_customized_info_callback",
        "comm_distribute_handle", "comm_create_message",
        "comm_async_remote_message_handle", "MCU_COMM_Init",
        "mps_ver_check", "get_mps_ver", "check_vrd_chip_cfg",
        "switch_vrd_page", "upgrade_mps_vrd", "validate_vrd_process",
        "verify_firmware_file", "check_firmware_info",
        "read_cms_file_info", "read_cms_file_info_item",
        "get_upgrade_file_len", "verify_flash_info",
        "flash_sha256_check", "sha256_check", "sha256_get",
        "elabel_init", "elabel_read", "elabel_update", "elabel_write",
        "elabel_exit", "elabel_set_default", "elabel_init_default",
        "SPI_FLASH_ReadRegion", "SPI_FLASH_WriteRegion",
        "SPI_FLASH_EraseRegion", "get_flash_region",
    ]

    resolved = {}
    for name in wanted:
        cands = by_name.get(name)
        if not cands:
            continue
        # pick the candidate whose file offset sits in the dense name table
        func, off = min(cands, key=lambda c: c[1])
        resolved[name] = (func, off)

    print(f"{'function':<40} {'code addr':<12} {'file off':<10}")
    print("-" * 64)
    for name, (func, off) in resolved.items():
        print(f"{name:<40} 0x{func:08x}   0x{func - BASE:05x}")

    missing = [n for n in wanted if n not in resolved]
    if missing:
        print("\nnot located:", ", ".join(missing))

    print("\n=== all recovered names (count) ===")
    print(", ".join(sorted(by_name)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
