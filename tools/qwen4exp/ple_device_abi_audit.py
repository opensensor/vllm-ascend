# SPDX-License-Identifier: Apache-2.0
"""Audit extracted device-kernel ABI evidence offline; never load a module.

Requires pyelftools. Matching symbol versions establish consistency of supplied
artifacts, not compatibility of a newly compiled module or device queue access.
"""

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

from elftools.elf.elffile import ELFFile

VERSION_RECORD = struct.Struct("<Q56s")
CRC_RECORD = struct.Struct("<I")


def parse_symvers(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("malformed symbol-version row")
        crc, name = int(fields[0], 16), fields[1]
        if not 0 <= crc <= 0xFFFFFFFF:
            raise ValueError("symbol CRC exceeds 32 bits")
        if name in result and result[name] != crc:
            raise ValueError(f"conflicting CRC for {name}")
        result[name] = crc
    if not result:
        raise ValueError("empty symbol-version table")
    return result


def parse_versions(data):
    if not data or len(data) % VERSION_RECORD.size:
        raise ValueError("invalid ARM64 __versions length")
    result = {}
    for crc, raw_name in VERSION_RECORD.iter_unpack(data):
        if b"\0" not in raw_name or crc > 0xFFFFFFFF:
            raise ValueError("invalid ARM64 symbol-version record")
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        if not name or name in result:
            raise ValueError("empty or duplicate imported symbol")
        result[name] = crc
    return result


def compare_versions(imports, providers):
    matched, missing, mismatched = [], [], []
    for name, crc in sorted(imports.items()):
        if name not in providers:
            missing.append(name)
        elif providers[name] != crc:
            mismatched.append({"symbol": name, "import": hex(crc), "provider": hex(providers[name])})
        else:
            matched.append(name)
    return {"matched": matched, "missing": missing, "mismatched": mismatched}


def module_evidence(path):
    with path.open("rb") as stream:
        elf = ELFFile(stream)
        if elf.elfclass != 64 or not elf.little_endian or elf["e_machine"] != "EM_AARCH64":
            raise ValueError(f"not a little-endian ARM64 ELF: {path}")
        metadata = elf.get_section_by_name(".modinfo")
        versions = elf.get_section_by_name("__versions")
        symbols = elf.get_section_by_name(".symtab")
        if metadata is None or versions is None or symbols is None:
            raise ValueError(f"missing module ABI sections: {path}")
        vermagic = next(
            entry.removeprefix(b"vermagic=").decode()
            for entry in metadata.data().split(b"\0")
            if entry.startswith(b"vermagic=")
        )
        exports = {}
        for symbol in symbols.iter_symbols():
            if not symbol.name.startswith("__crc_"):
                continue
            index = symbol["st_shndx"]
            if not isinstance(index, int):
                raise ValueError(f"unsupported CRC encoding: {symbol.name}")
            section = elf.get_section(index)
            if section.name not in ("__kcrctab", "__kcrctab_gpl"):
                raise ValueError(f"unexpected CRC section: {section.name}")
            offset = symbol["st_value"]
            crc = CRC_RECORD.unpack_from(section.data(), offset)[0]
            exports[symbol.name.removeprefix("__crc_")] = crc
        return {"vermagic": vermagic, "imports": parse_versions(versions.data()), "exports": exports}


def file_evidence(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def audit(rootfs, provider_modules=()):
    tables = sorted((rootfs / "boot").glob("symvers-*.gz"))
    if len(tables) != 1:
        raise ValueError("expected exactly one boot symbol-version archive")
    table = tables[0]
    release = table.name.removeprefix("symvers-").removesuffix(".gz")
    kernel = parse_symvers(gzip.decompress(table.read_bytes()).decode())
    module = rootfs / "var/drv_pcie.ko"
    evidence = module_evidence(module)
    comparison = compare_versions(evidence["imports"], kernel)
    providers = dict(kernel)
    provider_records = []
    for path in provider_modules:
        record = module_evidence(path)
        if record["vermagic"].split()[0] != release:
            raise ValueError(f"provider kernel release mismatch: {path}")
        for name, crc in record["exports"].items():
            if name in providers and providers[name] != crc:
                raise ValueError(f"conflicting provider CRC for {name}")
            providers[name] = crc
        provider_records.append({**file_evidence(path), "vermagic": record["vermagic"], "exports": record["exports"]})
    all_imports = compare_versions(evidence["imports"], providers)
    build = rootfs / "usr/src/kernels" / release
    required = ("Makefile", "include/generated/autoconf.h", "include/generated/utsrelease.h", "scripts/mod/modpost")
    missing = [name for name in required if not (build / name).is_file()]
    consistent = evidence["vermagic"].split()[0] == release and not all_imports["mismatched"]
    return {
        "kernel_release": release,
        "kernel_symbol_count": len(kernel),
        "artifact_versions_consistent": consistent,
        "build_compatibility_proven": False,
        "device_queue_access_proven": False,
        "kernel_table": file_evidence(table),
        "kernel_config": file_evidence(rootfs / "boot" / f"config-{release}"),
        "pcie_module": {**file_evidence(module), **evidence},
        "kernel_import_comparison": comparison,
        "additional_providers": provider_records,
        "all_import_comparison": all_imports,
        "all_import_crcs_resolved": not all_imports["missing"] and not all_imports["mismatched"],
        "missing_build_files": missing,
        "interpretation": "Missing kernel-table symbols may be supplied by other modules; verify separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-module", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = audit(args.rootfs, args.provider_module)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "kernel_release",
                    "kernel_symbol_count",
                    "artifact_versions_consistent",
                    "missing_build_files",
                )
            }
        )
    )
    return 0 if result["artifact_versions_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
