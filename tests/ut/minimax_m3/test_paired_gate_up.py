# SPDX-License-Identifier: Apache-2.0
"""Offline launch ABI, opt-in packaging, and compact gate/up admission tests."""

import json
import shutil
import subprocess
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3 import build_resident_native
from tools.minimax_m3.int8_native import GroupedW4Kernel, paired_geometry


def inputs():
    return [
        torch.zeros(1, 1024).half(),
        torch.zeros(2, 128, 512, dtype=torch.int8),
        torch.ones(2, 4, 32).half(),
        torch.ones(2, 128, 512, dtype=torch.int8),
        torch.full((2, 4, 32), 2.0).half(),
        torch.tensor([2, 1, 0, 2], dtype=torch.int32),
    ]


@pytest.mark.parametrize(
    "index,value",
    [
        (0, torch.zeros(2, 1024).half()),
        (0, torch.zeros(1, 1024)),
        (1, torch.zeros(2, 128, 512, dtype=torch.uint8)),
        (3, torch.zeros(2, 256, 512, dtype=torch.int8)),
        (4, torch.zeros(2, 4, 16).half()),
        (5, torch.zeros(4, dtype=torch.int64)),
        (5, torch.zeros(5, dtype=torch.int32)),
        (5, torch.zeros(0, dtype=torch.int32)),
        (5, torch.zeros(8, dtype=torch.int32)[::2]),
    ],
)
def test_reject_bad_pair_before_preparing_metadata_or_launching(index, value):
    args = inputs()
    args[index] = value
    kernel = GroupedW4Kernel.__new__(GroupedW4Kernel)
    kernel.bank_scales = Mock(side_effect=AssertionError("invalid input prepared metadata"))
    kernel.launch = Mock(side_effect=AssertionError("invalid input reached NPU"))
    with pytest.raises(ValueError, match="paired gate/up"):
        kernel._gate_up(*args)
    kernel.bank_scales.assert_not_called()
    kernel.launch.assert_not_called()


def test_pair_abi_keeps_banks_separate_and_reuses_cached_metadata():
    kernel = GroupedW4Kernel.__new__(GroupedW4Kernel)
    kernel.paired_kernel = object()
    kernel.paired_tilings, kernel.banks = {}, {}
    kernel.offsets = torch.empty(4096, dtype=torch.uint32)
    kernel.launch = Mock()
    args = inputs()
    assert paired_geometry(*args) == (4, 2, 128, 1024)
    gate, up = kernel._gate_up(*args)
    first = kernel.launch.call_args
    launch_args = first.args[1]
    assert first.args[0] is kernel.paired_kernel and first.args[2] == 1
    assert len(launch_args) == 14
    for index, tensor in (
        (0, args[0]),
        (1, args[1]),
        (3, args[5]),
        (4, gate),
        (5, gate),
        (9, kernel.offsets),
        (10, args[3]),
        (13, up),
    ):
        assert launch_args[index] is tensor
    assert launch_args[6].tolist() == [4, 2, 128, 1024, 2, 3]
    assert gate.shape == up.shape == (4, 128)
    assert gate.data_ptr() != up.data_ptr()
    assert gate.dtype == up.dtype == torch.float16
    kernel._gate_up(*args)
    second = kernel.launch.call_args.args[1]
    assert all(second[i] is launch_args[i] for i in (2, 6, 8, 11, 12))
    assert len(kernel.banks) == 2 and len(kernel.paired_tilings) == 1


@pytest.mark.parametrize("paired", [False, True])
def test_constructor_exposes_pair_only_with_explicit_binary(monkeypatch, paired):
    constructor = Mock(side_effect=lambda *args: object())
    monkeypatch.setattr(torch, "classes", types.SimpleNamespace(test=types.SimpleNamespace(Kernel=constructor)))
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(test=types.SimpleNamespace(launch=Mock())))
    monkeypatch.setattr(torch.Tensor, "npu", lambda value: value, raising=False)
    monkeypatch.setattr("tools.minimax_m3.int8_native.importlib.import_module", lambda name: object())
    kwargs = {"paired_binary": "pair.bin", "paired_entry": "pair"} if paired else {}
    kernel = GroupedW4Kernel("test", "normal.bin", "normal", **kwargs)
    assert callable(getattr(kernel, "gate_up", None)) is paired
    assert constructor.call_count == (2 if paired else 1)
    with pytest.raises(ValueError, match="supplied together"):
        GroupedW4Kernel("test", "normal.bin", "normal", paired_binary="pair.bin")


@pytest.mark.parametrize("options", [{}, {"int8_compute": True, "int8_original_layout": True}])
def test_pair_build_rejects_incompatible_storage(tmp_path, options):
    with pytest.raises(ValueError, match="paired gate/up"):
        build_resident_native.build(
            tmp_path / "out", tmp_path, "minimax_w4_pair", tmp_path, paired_gate_up=True, **options
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("paired", [False, True])
def test_offline_build_manifest_seals_exactly_the_selected_kernels(tmp_path, monkeypatch, paired):
    # Stub only the unavailable compiler/bridge. Exercise real source selection,
    # entry renaming, tar packaging, helper snapshot and manifest hash checks.
    source = tmp_path / "csrc"
    repo = Path(__file__).resolve().parents[3]
    for directory in (
        "gmm/w2_blocked_dequant_matmul_v310/op_kernel",
        "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel",
        "third_party/catlass/include",
        "moe/common/kernel_utils",
        "common/include",
    ):
        (source / directory).mkdir(parents=True)
    grouped = "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel"
    for name in ("w4a8_grouped.cpp", "w4a8_gate_up.cpp"):
        (source / grouped / name).write_text((repo / "csrc" / grouped / name).read_text())
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "c++":
            destination = command[command.index("-o") + 1]
        else:
            destination = command[2]
            assert "GetUserWorkspace" not in Path(command[1]).read_text()
            assert "KERNEL_TYPE_MIX_AIC" not in Path(command[1]).read_text()
        Path(destination).write_bytes(b"offline compiler stub")

    monkeypatch.setattr(build_resident_native.subprocess, "run", run)
    monkeypatch.setattr(build_resident_native, "compile_bridge", lambda path, *args: path.write_bytes(b"stub"))
    output = tmp_path / "out"
    path = build_resident_native.build(
        output,
        source,
        "minimax_w4_pair",
        tmp_path,
        int8_compute=True,
        wide_l1=True,
        rint_unpack=True,
        scale_cache=True,
        paired_gate_up=paired,
    )
    manifest = NativeManifest(json.loads(path.read_text()))
    manifest.verify_files()
    assert len(commands) == (3 if paired else 2)
    assert (output / "gate_up_w4.bin").exists() is paired
    assert ("paired_binary=" in manifest.value["validation_source"]) is paired
    assert json.loads((output / "build-options.json").read_text())["paired_gate_up"] is paired
    if paired:
        (output / "gate_up_w4.bin").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="digest|hash|SHA"):
            manifest.verify_files()


def test_native_entry_pointer_selection_and_activation_lifetime_on_host(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[3]
    entry = root / "csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w4a8_gate_up.cpp"
    (tmp_path / "entry.cpp").write_text(entry.read_text())
    for header in ("compat_310p.h", "w4a8_tile.h", "w2_grouped_blocked_dequant_matmul_v310_tiling_data.h"):
        (tmp_path / header).write_text("// Host stubs are declared by the harness.\n")
    harness = Path(__file__).with_name("paired_gate_up_host.cpp")
    executable = tmp_path / "entry-test"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", f"-I{tmp_path}", str(harness), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    cases = result.stdout.split("BASE ")[1:]
    for case, ids in zip(cases, ([2, 1, 0, 1], [0], [2, 2, -1, 2], [1, 1, 0, 0]), strict=True):
        lines = case.splitlines()
        x, gate, up, gate_ratios, up_ratios, gate_y, up_y, gate_scales, up_scales = map(int, lines[0].split())
        expected = []
        for row, expert in enumerate(ids):
            if not 0 <= expert < 2:
                continue
            for bank, ratios, output, scales in (
                (gate, gate_ratios, gate_y, gate_scales),
                (up, up_ratios, up_y, up_scales),
            ):
                expected.append(
                    [
                        x,
                        bank + expert * 128 * 1024 // 2,
                        ratios + expert * 4 * 32 * 4,
                        output + row * 128 * 2,
                        scales + expert * 4 * 4,
                        int(bool(expected)),
                    ]
                )
        actual = [list(map(int, line.removeprefix("CALL ").split())) for line in lines[1:]]
        assert actual == expected
