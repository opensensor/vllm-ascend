# SPDX-License-Identifier: Apache-2.0
"""Host-only build isolation and native ABI guards; not numerical NPU gates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp import build_hc_combine as builder


def test_requires_backend_autoload_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    with pytest.raises(RuntimeError, match="AUTOLOAD=0"):
        builder.build(tmp_path / "out", tmp_path, tmp_path)
    assert not (tmp_path / "out").exists()


def test_build_freezes_explicit_sources_and_queue_drain_contract(monkeypatch, tmp_path):
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
    assert report["namespace"] == report["entry"] == "qwen_hc_combine_v1"
    assert report["eager_only"] is report["flush_host_queue"] is True
    assert report["hardware_validated"] is report["server_modified"] is False
    assert report["bridge_source"] == "native_owned_bridge.cpp"
    assert set(report["sources"]) == {Path(path).name for path in builder.SOURCES}
    for path in builder.SOURCES:
        assert (destination / Path(path).name).read_bytes() == (root / path).read_bytes()
        assert report["sources"][Path(path).name] == builder.digest(destination / Path(path).name)
    assert "-DQWEN_NATIVE_FLUSH_HOST_QUEUE=1" in calls[-1]
    assert "-DGLM_RECONSTRUCTION_NAMESPACE=qwen_hc_combine_v1" in calls[-1]
    assert str(destination / "native_owned_bridge.cpp") in calls[-1]
    assert calls[1][1] == str(destination / "native_hc_combine.cpp")
    assert json.loads((destination / "provenance.json").read_text()) == report
    previous = (destination / "provenance.json").read_bytes()
    with pytest.raises(FileExistsError):
        builder.build(destination, tmp_path, root)
    assert len(calls) == 3 and (destination / "provenance.json").read_bytes() == previous


def test_native_contract_keeps_fp32_product_then_add_and_bounded_descriptor():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/qwen4exp/native_hc_combine.cpp").read_text()
    assert "const Tiling td{raw->rows};" in source
    assert "STREAMS = 4, WIDTH = 2560, MAX_ROWS = 2560" in source
    assert "if (td.rows <= 0 || td.rows > MAX_ROWS) return;" in source
    assert "GlobalTensor<float> block_, injection_;" in source
    assert "GlobalTensor<half> hyper_, output_;" in source
    assert "Muls(product, input, coefficient, count);\n        PipeBarrier<PIPE_V>();\n        Add(" in source
    assert "Mins(residual, residual, FP16_MAX" in source
    assert "Maxs(residual, residual, -FP16_MAX" in source
    assert "Cast(outputHalf, residual, RoundMode::CAST_NONE" in source
    assert "inputHalf" not in source


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 68, 640, 2048, 2560])
def test_eight_block_row_partition_has_unique_complete_output_ownership(rows):
    # Each native task owns one token and its four whole 2560-element rows.
    ownership = [row * 4 + stream for core in range(8) for row in range(core, rows, 8) for stream in range(4)]
    assert sorted(ownership) == list(range(rows * 4))
