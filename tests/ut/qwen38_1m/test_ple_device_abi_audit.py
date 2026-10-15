# SPDX-License-Identifier: Apache-2.0
"""Offline ABI evidence must distinguish unknown providers from wrong CRCs."""

import gzip

import pytest

from tools.qwen4exp import ple_device_abi_audit as audit


def test_symbol_versions_and_provider_comparison():
    providers = audit.parse_symvers("0x1234 first vmlinux EXPORT_SYMBOL\n0x5678 second vmlinux EXPORT_SYMBOL_GPL\n")
    imports = audit.parse_versions(
        b"".join(
            audit.VERSION_RECORD.pack(crc, name)
            for crc, name in [(0x1234, b"first"), (0x5679, b"second"), (42, b"external")]
        )
    )
    assert audit.compare_versions(imports, providers) == {
        "matched": ["first"],
        "missing": ["external"],
        "mismatched": [{"symbol": "second", "import": "0x5679", "provider": "0x5678"}],
    }


@pytest.mark.parametrize(
    "text",
    ["", "garbage", "0x100000000 a kernel EXPORT_SYMBOL", "0x1 a kernel EXPORT_SYMBOL\n0x2 a kernel EXPORT_SYMBOL"],
)
def test_malformed_provider_tables(text):
    with pytest.raises(ValueError):
        audit.parse_symvers(text)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"x",
        audit.VERSION_RECORD.pack(1, b"x" * 56),
        audit.VERSION_RECORD.pack(1, b""),
        audit.VERSION_RECORD.pack(1 << 32, b"x"),
        audit.VERSION_RECORD.pack(1, b"x") * 2,
    ],
)
def test_malformed_import_tables(data):
    with pytest.raises(ValueError):
        audit.parse_versions(data)


def test_missing_boot_archive(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        audit.audit(tmp_path)


def test_artifact_consistency_is_not_build_compatibility(tmp_path, monkeypatch):
    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "symvers-test.gz").write_bytes(gzip.compress(b"0x1 first vmlinux EXPORT_SYMBOL\n"))
    (boot / "config-test").write_text("CONFIG_MODVERSIONS=y\n")
    (tmp_path / "var").mkdir()
    module = tmp_path / "var/drv_pcie.ko"
    module.write_bytes(b"test module placeholder")
    monkeypatch.setattr(
        audit,
        "module_evidence",
        lambda _path: {"vermagic": "test SMP", "imports": {"first": 1, "other": 2}, "exports": {}},
    )
    result = audit.audit(tmp_path)
    assert result["artifact_versions_consistent"]
    assert not result["all_import_crcs_resolved"]
    assert not result["build_compatibility_proven"]
    assert not result["device_queue_access_proven"]
    assert len(result["missing_build_files"]) == 4
    assert result["all_import_comparison"]["missing"] == ["other"]

    provider = tmp_path / "provider.ko"
    provider.write_bytes(b"provider placeholder")
    monkeypatch.setattr(
        audit,
        "module_evidence",
        lambda path: {
            "vermagic": "test SMP",
            "imports": {"first": 1, "other": 2},
            "exports": {"other": 2} if path == provider else {},
        },
    )
    result = audit.audit(tmp_path, [provider])
    assert result["all_import_crcs_resolved"]
    assert not result["build_compatibility_proven"]

    monkeypatch.setattr(
        audit,
        "module_evidence",
        lambda path: {
            "vermagic": "wrong SMP" if path == provider else "test SMP",
            "imports": {"first": 1},
            "exports": {},
        },
    )
    with pytest.raises(ValueError, match="release mismatch"):
        audit.audit(tmp_path, [provider])
