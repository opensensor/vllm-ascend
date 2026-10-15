# SPDX-License-Identifier: Apache-2.0
"""Host-only build isolation and native ABI guards; not numerical NPU gates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp import build_mamba_commit as builder


def test_requires_backend_autoload_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    with pytest.raises(RuntimeError, match="AUTOLOAD=0"):
        builder.build(tmp_path / "out", tmp_path, tmp_path)
    assert not (tmp_path / "out").exists()


def test_build_freezes_explicit_sources_and_no_drain_contract(monkeypatch, tmp_path):
    import torch.utils.cpp_extension

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    root = Path(__file__).resolve().parents[3]
    destination = tmp_path / "fresh"
    calls = []
    actual_find = builder.importlib.util.find_spec
    monkeypatch.setattr(
        builder.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(tmp_path / "torch_npu/__init__.py"))
            if name == "torch_npu"
            else actual_find(name)
        ),
    )
    monkeypatch.setattr(torch.utils.cpp_extension, "include_paths", lambda: [])
    monkeypatch.setattr(torch.utils.cpp_extension, "library_paths", lambda: [])

    def compile_mock(command, **kwargs):
        calls.append(command)
        path = Path(command[command.index("-o") + 1]) if command[0] == "c++" else Path(command[2])
        path.write_bytes(b"compiled mock")

    monkeypatch.setattr(builder.subprocess, "run", compile_mock)
    report = builder.build(destination, tmp_path, root)
    assert report["namespace"] == report["entry"] == "qwen_mamba_commit_v1"
    assert report["eager_only"] is report["flush_host_queue"] is False
    assert report["hardware_validated"] is report["server_modified"] is False
    assert report["bridge_source"] == "native_owned_bridge.cpp"
    assert set(report["sources"]) == {Path(path).name for path in builder.SOURCES}
    for path in builder.SOURCES:
        assert (destination / Path(path).name).read_bytes() == (root / path).read_bytes()
        assert report["sources"][Path(path).name] == builder.digest(destination / Path(path).name)
    assert "-DQWEN_NATIVE_FLUSH_HOST_QUEUE=0" in calls[-1]
    assert "-DGLM_RECONSTRUCTION_NAMESPACE=qwen_mamba_commit_v1" in calls[-1]
    assert str(destination / "native_owned_bridge.cpp") in calls[-1]
    assert calls[1][1] == str(destination / "native_mamba_commit.cpp")
    assert json.loads((destination / "provenance.json").read_text()) == report
    previous = (destination / "provenance.json").read_bytes()
    with pytest.raises(FileExistsError):
        builder.build(destination, tmp_path, root)
    assert len(calls) == 3 and (destination / "provenance.json").read_bytes() == previous
