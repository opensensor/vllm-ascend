"""Offline HSM boot-state query audit. Requires capstone and pyelftools.

Usage: python3 ple_hsm_boot_query_audit.py RECOVERED_DRIVER_DIR OUTPUT_DIR
RECOVERED_DRIVER_DIR contains extracted/driver/device and device-rootfs/var.
No register reads, driver calls, or firmware modifications are performed.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from capstone import CS_ARCH_ARM, CS_ARCH_ARM64, CS_MODE_ARM, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile


def module_disassembly(path, ranges):
    lines = [str(path), "Section-relative .text offsets; relocation targets annotated."]
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        symbols = elf.get_section_by_name(".symtab")
        code = elf.get_section_by_name(".text").data()
        notes = {}
        for relocation in elf.get_section_by_name(".rela.text").iter_relocations():
            symbol = symbols.get_symbol(relocation["r_info_sym"])
            note = symbol.name
            section_index = symbol["st_shndx"]
            if isinstance(section_index, int):
                section = elf.get_section(section_index)
                offset = symbol["st_value"] + relocation["r_addend"]
                note += f" {section.name}+{offset:#x}"
                if "rodata" in section.name:
                    content = section.data()
                    if 0 <= offset < len(content):
                        stop = content.find(b"\0", offset)
                        if stop >= offset:
                            note += " " + repr(content[offset:stop][:200])
            notes[relocation["r_offset"]] = note
        decoder = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        for start, end in ranges:
            lines.append(f"Range {start:#x}..{end:#x}")
            for instruction in decoder.disasm(code[start:end], start):
                lines.append(f"{instruction.address:08x}: {instruction.mnemonic:8} "
                             f"{instruction.op_str} ; {notes.get(instruction.address, '')}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    firmware = args.driver_dir / "extracted/driver/device/hisserika.bin"
    data = firmware.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == "a7ca05f24a718db449a2db0da58f412b21cc77c31f6e84d6b3790fa6c2ce0a2b"
    mapping_bias = 0x10A0BF00
    request_id, handler_pointer = struct.unpack_from("<II", data, 0x16B60)
    assert request_id == 0x99CC99CC
    assert handler_pointer == mapping_bias + 0x11B09  # Thumb entry bit.
    assert struct.unpack_from("<II", data, 0x11B24) == (0x4F4FF4F4, 0x10106040)
    assert data[0x1B44D:].startswith(b"service_uefi_func\0")
    assert struct.unpack_from("<I", data, 0x79D0)[0] == 0x10160200
    assert struct.unpack_from("<I", data, 0x7A10)[0] == 0x10160028
    assert struct.unpack_from("<I", data, 0x12068)[0] == 0x10A2E0C0
    assert struct.unpack_from("<I", data, 0x104E4)[0] == 0x10A2E0C0
    assert struct.unpack_from("<I", data, 0x16A4C)[0] == 0xAA000000
    assert struct.unpack_from("<I", data, 0x16A5C)[0] == mapping_bias + 0x103F5
    decoder = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    decoded = list(decoder.disasm(data[0x11B08:0x11B24], 0x11B08))
    assert any(i.address == 0x11B1C and i.mnemonic == "ldr" and i.op_str == "r2, [r2]" for i in decoded)
    lines = [str(firmware), f"sha256={digest}", "Thumb code; FILE OFFSETS, not runtime addresses."]
    for start, end in [(0x11B08, 0x11B24), (0x11F98, 0x12028), (0x120CC, 0x1211A),
                       (0x1221C, 0x122A0), (0x11E3C, 0x11F58),
                       (0xC260, 0xC2B4), (0xC30C, 0xC38A), (0xC424, 0xC49C),
                       (0x79B8, 0x79D0), (0x79D4, 0x79E6), (0x79EC, 0x79FE),
                       (0x7A04, 0x7A10), (0x103F4, 0x104E2),
                       (0x15E38, 0x15E46), (0x15D30, 0x15D50)]:
        lines.append(f"Range {start:#x}..{end:#x}")
        lines.extend(f"{i.address:08x}: {i.mnemonic:8} {i.op_str}"
                     for i in decoder.disasm(data[start:end], start))
    modules = args.driver_dir / "device-rootfs/var"
    for filename, ranges in [
        ("drv_smbus_intf.ko", [(0x1C48, 0x22D8)]),
        ("drv_ascend_devmng.ko", [(0x4448, 0x44E0), (0x4634, 0x4854)]),
        ("hsm_ras.ko", [(0x278, 0x2D0), (0x484, 0x804)]),
    ]:
        lines.extend(module_disassembly(modules / filename, ranges))
    report = {
        "scope": "Offline firmware and recovered device kernel-module analysis",
        "firmware": str(firmware), "sha256": digest,
        "hsm_file_to_runtime_bias": hex(mapping_bias),
        "query": {
            "dispatcher_name_from_log_string": "service_uefi_func",
            "dispatch_table_file_offset": "0x16b50",
            "entry_file_offset": "0x16b60", "request_id": hex(request_id),
            "handler_pointer": hex(handler_pointer), "handler_file_offset": "0x11b08",
            "isa": "Thumb", "read_address": "0x10106040",
            "response_words": ["0x4f4ff4f4", "value read from 0x10106040"],
            "response_length": 8, "handler_success_return": "0x01000000",
            "writes_boot_state_word": False,
            "host_runtime_transport_validated": False,
        },
        "transport_trace": {
            "firmware_names": ["scmi_task_register", "scmi_recv_data", "scmi_send_data"],
            "protocol_standard_compliance": "Not established from these names",
            "uefi_channel": 1,
            "mailbox_address_as_seen_by_hsm": "0x10160200 + channel * 0x80",
            "hsm_notification_write_address": "0x10160028",
            "uefi_receive_bytes": 64,
            "send_wrapper_validates": "non-null buffer, at most 128 bytes, length multiple of 4",
            "lifetime_guard_address": "0x10a2e0c0",
            "lifetime_guard_reader": "0x11e9a..0x11ea2: when word equals 1, calls task-control routine 0x15e38 before request handling",
            "lifetime_guard_writer": "set_dc_time_stamp at 0x103f4; writes 1 at 0x10408 when previously zero",
            "timestamp_dispatch_entry": "0x16a4c: command 0xaa000000, Thumb pointer at 0x16a5c targets 0x103f4",
            "task_control_semantics": "Routine 0x15e38 calls 0x15d30; task scheduling/control observed, exact API name and resume behavior unconfirmed",
            "timestamp_invocation_on_live_device": "Not observed",
            "linux_scmi_candidate": "hsm_ras.ko maps device-tree compatible int_scmi_hiss and registers a fault-reporting IRQ; not proven to expose UEFI requests",
        },
        "separate_smbus_path": {
            "module": "drv_smbus_intf.ko", "query": "smbus_secure_boot_query",
            "initialization": "smbus_secure_boot_init starts certificate-info worker",
            "flag_source": "devdrv_driver_get_cert_info, selector 4, first returned byte",
            "implementation": "drv_ascend_devmng.ko reads a 64 KiB flash block via dev_flash_read_block",
            "equivalence_to_hboot_state": "Not established",
        },
        "host_candidates": {
            "Ascend_310P_check.bin": "Inspected strings describe image/CRL verification, not an established arbitrary register-read interface",
            "DEBUG_READ_REG": "Host debug command dispatched to TS; not established as CtrlCPU physical-memory access",
            "dsmi_get_last_bootstate": "Header documents normal/abnormal result, not enforcement-enabled state; support list excludes 310P",
            "dsmi_get_hiss_status": "Header reports HISS_NOT_INITIALIZED/HISS_ERROR/HISS_OK; not the raw boot-policy word",
        },
        "limitations": [
            "No known writer for the boot-state word",
            "Literal-search absence does not exclude synthesized addresses or another address space",
            "Current runtime state need not equal boot-time state",
            "HSM UEFI query availability after OS startup is unproven",
            "Boot-ROM policy and runtime CMS policy remain distinct",
        ],
        "live_boot_state_read": False, "hardware_modified": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ple-hsm-boot-query-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "ple-hsm-boot-query-disassembly.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
