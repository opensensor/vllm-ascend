# SPDX-License-Identifier: Apache-2.0
"""Manifest creation must reject altered binaries and mismatched bindings."""

import json
from pathlib import Path

import pytest

from tools.qwen4exp.build_native_prefill import digest
from tools.qwen4exp.prepare_native_prefill import make_manifest

ROOT = Path(__file__).resolve().parents[3]


def fake_build(directory, version=2):
    entries = {}
    for name in ("native_wy", "native_route_gather", "native_local_swiglu"):
        path = directory / f"{name}.bin"
        path.write_bytes(name.encode())
        entries[name] = {"path": str(path), "sha256": digest(path)}
    for name in (
        "native_wy.cpp",
        "native_route_gather.cpp",
        "native_local_swiglu.cpp",
        "reconstruction_bridge.cpp",
        "compile_reconstruction.cpp",
    ):
        (directory / name).write_text("// fixture source\n")
    library = directory / "library.so"
    library.write_bytes(b"fixture library")
    value = {
        "namespace": f"qwen_prefill_v{version}",
        "binaries": entries,
        "bridge": {"path": str(library), "sha256": digest(library)},
        "sources": {p.name: digest(p) for p in directory.glob("*.cpp")},
    }
    (directory / "provenance.json").write_text(json.dumps(value))
    return value


def test_manifest_is_data_only_and_retains_frozen_sources(tmp_path):
    fake_build(tmp_path)
    value = make_manifest(tmp_path, ROOT)
    assert value["name"] == "qwen_prefill_v2"
    assert value["operators"] == ["qwen_prefill_v2::launch"]
    assert any(entry["path"].endswith("native_wy.cpp") for entry in value["assets"])
    assert "load_library=False" in value["validation_source"]


def test_changed_binary_and_resource_version_rejected_before_loading(tmp_path):
    fake_build(tmp_path)
    (tmp_path / "native_wy.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        make_manifest(tmp_path, ROOT)
    fake_build(tmp_path, version=1)
    with pytest.raises(ValueError, match="version differs"):
        make_manifest(tmp_path, ROOT)
