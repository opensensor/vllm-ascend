# SPDX-License-Identifier: Apache-2.0
"""Compile actual streaming body against an independent synchronous CPU fixture.

The fixture checks integer layout and FP32 correction, not NPU task scheduling,
DMA timing, native CAST_NONE rounding, hardware parity or performance.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp import native_streaming

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "streaming_cpu_stubs"


@pytest.fixture(scope="module")
def native_cpu(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host g++ unavailable")
    path = tmp_path_factory.mktemp("streaming-projection") / "projection.so"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-I",
            str(STUBS),
            "-I",
            str(ROOT / "tools/qwen4exp"),
            str(STUBS / "harness.cpp"),
            "-o",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    library = ctypes.CDLL(str(path))
    library.run_projection.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64)]
    library.run_projection.restype = ctypes.c_int
    library.run_columns.argtypes = library.run_projection.argtypes
    library.run_columns.restype = ctypes.c_int
    library.projection_error.restype = ctypes.c_char_p
    library.projection_peak.argtypes = [ctypes.c_uint32]
    library.projection_peak.restype = ctypes.c_uint64
    return library


def pack_nibbles(values):
    values = np.asarray(values, dtype=np.int8)
    return np.ascontiguousarray(
        ((values[..., ::2] & 15) | ((values[..., 1::2] & 15) << 4)).astype(np.uint8).view(np.int8)
    )


def fixture(rows, outputs, k, ends, seed):
    rng = np.random.default_rng(seed)
    experts = len(ends)
    groups = k // 128
    low_values = rng.integers(-8, 8, (rows, k), dtype=np.int8)
    high_values = rng.integers(-8, 8, (rows, k), dtype=np.int8)
    weights = rng.integers(-8, 8, (experts, outputs, k), dtype=np.int8)
    # Scale/sum lanes are replicated in the production quantizer.
    xs = rng.uniform(0.001, 0.025, (rows, groups)).astype(np.float32)
    sums = rng.uniform(-80, 80, (rows, groups)).astype(np.float32)
    sw = rng.uniform(0.002, 0.06, (experts, outputs, groups)).astype(np.float16)
    zw = rng.uniform(-1, 1, (experts, outputs, groups)).astype(np.float16)
    ws = rng.uniform(-100, 100, (experts, outputs, groups)).astype(np.float16)
    # Checkpoint N16 strips are [expert,N16,groups,K64,16,32] packed bytes.
    codes = pack_nibbles(weights).reshape(experts, outputs // 16, 16, groups, 2, 32)
    codes = np.ascontiguousarray(codes.transpose(0, 1, 3, 4, 2, 5))

    def metadata(value):
        return np.ascontiguousarray(value.reshape(experts, outputs // 16, 16, groups).transpose(0, 1, 3, 2))

    low, high = pack_nibbles(low_values), pack_nibbles(high_values)
    scale_lanes = np.ascontiguousarray(np.repeat(xs[..., None], 8, axis=-1))
    sum_lanes = np.ascontiguousarray(np.repeat(sums[..., None], 8, axis=-1))
    end_array = np.asarray(ends, dtype=np.int64)
    storage = np.full(rows * outputs + 32, np.float16(123.5))
    output = storage[16:-16].reshape(rows, outputs)
    config = np.asarray((rows, experts, outputs, k, 8, 0, 1), dtype=np.int64)
    arguments = [
        low,
        high,
        scale_lanes,
        sum_lanes,
        codes,
        metadata(sw),
        metadata(zw),
        metadata(ws),
        end_array,
        output,
        config,
    ]
    reference = np.zeros((rows, outputs), dtype=np.float32)
    start = 0
    for expert, end in enumerate(ends):
        if end > start:
            for group in range(groups):
                section = slice(group * 128, (group + 1) * 128)
                dot_low = (
                    low_values[start:end, section].astype(np.int32) @ weights[expert, :, section].astype(np.int32).T
                )
                dot_high = (
                    high_values[start:end, section].astype(np.int32) @ weights[expert, :, section].astype(np.int32).T
                )
                corrected = dot_low.astype(np.float32) + dot_high.astype(np.float32) * np.float32(16)
                corrected = corrected - zw[expert, :, group].astype(np.float32) * sums[start:end, group, None]
                corrected = corrected + ws[expert, :, group].astype(np.float32) * np.float32(8)
                corrected = corrected * sw[expert, :, group].astype(np.float32)
                corrected = corrected * xs[start:end, group, None]
                reference[start:end] = reference[start:end] + corrected
        start = end
    return arguments, reference.astype(np.float16), storage


def run(native_cpu, arguments, columns=False):
    pointers = (ctypes.c_void_p * 11)(*[a.ctypes.data for a in arguments])
    lengths = (ctypes.c_int64 * 11)(*[a.nbytes for a in arguments])
    entry = native_cpu.run_columns if columns else native_cpu.run_projection
    status = entry(pointers, lengths)
    return status, native_cpu.projection_error().decode()


@pytest.mark.parametrize(
    "rows,outputs,k,active", [(17, 2560, 640, 17), (17, 2560, 640, 1), (17, 2560, 640, 0), (1, 1280, 2560, 1)]
)
def test_actual_columns_concatenate_to_full_reference(native_cpu, rows, outputs, k, active):
    arguments, expected, _ = fixture(rows, outputs, k, [0, active // 2, active // 2, active], 89)
    originals = [a.copy() for a in arguments[:9]]
    pieces = []
    for first in range(0, outputs // 128, 8):
        count = min(8, outputs // 128 - first)
        storage = np.full(rows * count * 128 + 32, np.float16(123.5))
        output = storage[16:-16].reshape(rows, count * 128)
        config = np.concatenate((arguments[10], np.asarray([first, count], dtype=np.int64)))
        window = [*arguments[:9], output, config]
        status, error = run(native_cpu, window, columns=True)
        assert status == 0, error
        np.testing.assert_array_equal(output, expected[:, first * 128 : (first + count) * 128])
        np.testing.assert_array_equal(storage[:16], np.float16(123.5))
        np.testing.assert_array_equal(storage[-16:], np.float16(123.5))
        pieces.append(output)
    np.testing.assert_array_equal(np.concatenate(pieces, axis=-1), expected)
    for argument, original in zip(arguments[:9], originals):
        np.testing.assert_array_equal(argument, original)


@pytest.mark.parametrize("first,count", [(2, 3), (8, 1), (19, 1)])
def test_interior_and_single_tile_windows_use_full_bank_stride(native_cpu, first, count):
    arguments, expected, _ = fixture(15, 2560, 640, [0, 15], 29)
    output = np.full((15, count * 128), np.float16(19.5))
    config = np.concatenate((arguments[10], np.asarray([first, count], dtype=np.int64)))
    status, error = run(native_cpu, [*arguments[:9], output, config], columns=True)
    assert status == 0, error
    np.testing.assert_array_equal(output, expected[:, first * 128 : (first + count) * 128])


@pytest.mark.parametrize("first,count", [(-1, 1), (20, 1), (19, 2), (0, 0), (0, 9)])
def test_native_column_bounds_rejected(native_cpu, first, count):
    arguments, _, _ = fixture(1, 2560, 640, [1], 7)
    arguments[10] = np.concatenate((arguments[10], np.asarray([first, count], dtype=np.int64)))
    status, error = run(native_cpu, arguments, columns=True)
    assert status != 0 and "assertion" in error


@pytest.mark.parametrize(
    "rows,k,outputs",
    [(1, 128, 128), (15, 640, 256), (16, 2560, 128), (17, 640, 128), (33, 128, 256), (1, 2560, 1280), (1, 640, 2560)],
)
@pytest.mark.parametrize("seed", [3, 41])
def test_actual_entry_projection_matches_independent_fp32_reference(native_cpu, rows, k, outputs, seed):
    ends = [0, rows // 2, rows // 2, rows]
    arguments, expected, storage = fixture(rows, outputs, k, ends, seed)
    originals = [a.copy() for a in arguments[:9]]
    status, error = run(native_cpu, arguments)
    assert status == 0, error
    np.testing.assert_array_equal(arguments[9], expected)
    np.testing.assert_array_equal(storage[:16], np.float16(123.5))
    np.testing.assert_array_equal(storage[-16:], np.float16(123.5))
    for argument, original in zip(arguments[:9], originals):
        np.testing.assert_array_equal(argument, original)
    assert native_cpu.projection_peak(0) == 140032
    assert native_cpu.projection_peak(1) + native_cpu.projection_peak(2) == 167936
    assert native_cpu.projection_peak(3) == 4096
    assert native_cpu.projection_peak(4) == 16384
    assert native_cpu.projection_peak(5) == 16384


@pytest.mark.parametrize("ends", [[0, 0, 0, 0], [1, 1, 1, 1], [15, 15, 15, 15], [17, 17, 17, 17]])
def test_all_peer_and_partial_local_prefix_zero_without_stale_output(native_cpu, ends):
    arguments, expected, _ = fixture(17, 128, 640, ends, 17)
    status, error = run(native_cpu, arguments)
    assert status == 0, error
    np.testing.assert_array_equal(arguments[9], expected)
    assert np.count_nonzero(arguments[9][ends[-1] :]) == 0


def test_many_empty_experts_production_widths(native_cpu):
    ends = [0] * 128
    ends[17:] = [1] * (128 - 17)
    arguments, expected, _ = fixture(1, 128, 2560, ends, 83)
    status, error = run(native_cpu, arguments)
    assert status == 0, error
    np.testing.assert_array_equal(arguments[9], expected)


def test_shrink_grow_reuses_output_without_stale_peer_rows(native_cpu):
    arguments, _, storage = fixture(17, 256, 640, [0, 0, 0, 17], 83)
    for active in (17, 1, 0, 16, 17):
        ends = [0, 0, 0, active]
        arguments[8][:] = ends
        _, expected, _ = fixture(17, 256, 640, ends, 83)
        status, error = run(native_cpu, arguments)
        assert status == 0, error
        np.testing.assert_array_equal(arguments[9], expected)
        np.testing.assert_array_equal(storage[:16], np.float16(123.5))
        np.testing.assert_array_equal(storage[-16:], np.float16(123.5))


@pytest.mark.parametrize("ends", [[1, 0], [0, 18], [-1, 1]])
def test_native_bad_boundaries_fail_offline(native_cpu, ends):
    arguments, _, _ = fixture(17, 128, 128, [0, 17], 2)
    arguments[8][:] = ends
    status, error = run(native_cpu, arguments)
    assert status != 0 and "assertion" in error


@pytest.mark.parametrize(
    "index,value", [(0, 0), (0, 25601), (1, 129), (2, 129), (2, 2688), (3, 64), (3, 2688), (4, 4), (5, 1), (6, 2)]
)
def test_actual_entry_geometry_assertions(native_cpu, index, value):
    arguments, _, _ = fixture(1, 128, 128, [1], 2)
    arguments[10][index] = value
    status, error = run(native_cpu, arguments)
    assert status != 0 and "assertion" in error


def python_fixture():
    rows, experts, outputs, groups = 17, 4, 128, 5
    bank = SimpleNamespace(weight=torch.zeros(experts, outputs, 320, dtype=torch.int8))
    for name in ("weight_scale", "weight_offset", "weight_sum"):
        setattr(bank, name, torch.zeros(experts, outputs, groups, dtype=torch.float16))
    prepared = [torch.zeros(rows, 320, dtype=torch.int8), torch.zeros(rows, 320, dtype=torch.int8)]
    prepared += [torch.zeros(rows, groups, 8, dtype=torch.float32), torch.zeros(rows, groups, 8, dtype=torch.float32)]
    ends = torch.full((experts,), rows, dtype=torch.int64)
    return bank, prepared, ends


@pytest.mark.parametrize(
    "broken",
    [
        "rows",
        "experts",
        "outputs",
        "outputs_large",
        "k",
        "weight_k",
        "high_shape",
        "low_dtype",
        "weight_dtype",
        "scale_dtype",
        "sum_shape",
        "metadata_dtype",
        "metadata_shape",
        "ends_dtype",
        "ends_shape",
    ],
)
def test_python_resource_metadata_guards(broken):
    bank, prepared, ends = python_fixture()
    if broken == "rows":
        prepared[0] = torch.empty(0, 320, dtype=torch.int8)
    elif broken == "experts":
        bank.weight = torch.zeros(129, 128, 320, dtype=torch.int8)
    elif broken == "outputs":
        bank.weight = torch.zeros(4, 129, 320, dtype=torch.int8)
    elif broken == "outputs_large":
        bank.weight = torch.zeros(4, 2688, 320, dtype=torch.int8)
        for name in ("weight_scale", "weight_offset", "weight_sum"):
            setattr(bank, name, torch.zeros(4, 2688, 5, dtype=torch.float16))
    elif broken == "k":
        prepared[0] = torch.empty(17, 321, dtype=torch.int8)
    elif broken == "weight_k":
        bank.weight = bank.weight[..., :-1]
    elif broken == "high_shape":
        prepared[1] = prepared[1][:-1]
    elif broken == "low_dtype":
        prepared[0] = prepared[0].float()
    elif broken == "weight_dtype":
        bank.weight = bank.weight.float()
    elif broken == "scale_dtype":
        prepared[2] = prepared[2].half()
    elif broken == "sum_shape":
        prepared[3] = prepared[3][..., :-1]
    elif broken == "metadata_dtype":
        bank.weight_scale = bank.weight_scale.float()
    elif broken == "metadata_shape":
        bank.weight_offset = bank.weight_offset[..., :-1]
    elif broken == "ends_dtype":
        ends = ends.int()
    else:
        ends = ends[:-1]
    resource = native_streaming.NativeStreamingProjection(None, lambda *args: pytest.fail("invalid shape launched"))
    with pytest.raises(ValueError):
        resource(bank, prepared, ends)


def test_python_resource_rejects_cpu_and_noncontiguous_metadata():
    bank, prepared, ends = python_fixture()
    resource = native_streaming.NativeStreamingProjection(None, lambda *args: pytest.fail("CPU metadata launched"))
    with pytest.raises(ValueError, match="one NPU"):
        resource(bank, prepared, ends)
    prepared[0] = torch.zeros(17, 640, dtype=torch.int8)[:, ::2]
    with pytest.raises(ValueError, match="one NPU"):
        resource(bank, prepared, ends)


def test_python_launch_abi_metadata_only(monkeypatch):
    bank, prepared, ends = python_fixture()
    calls = []
    monkeypatch.setattr(native_streaming, "_tensors_on_one_device", lambda *args: None)
    resource = native_streaming.NativeStreamingProjection("CPU launch spy", lambda *args: calls.append(args))
    output = resource(bank, prepared, ends)
    assert output.shape == (17, 128) and output.dtype == torch.float16
    assert len(calls) == 1 and calls[0][2] == 8
    args = calls[0][1]
    assert len(args) == 11 and args[-1].tolist() == [17, 4, 128, 640, 8, 0, 1]
    assert all(original is actual for original, actual in zip(prepared, args[:4]))


@pytest.mark.parametrize("first,count", [(-1, 1), (1, 1), (0, 0), (0, 9), (True, 1), (0, 1.0)])
def test_python_column_bounds_metadata_only(monkeypatch, first, count):
    bank, prepared, ends = python_fixture()
    monkeypatch.setattr(native_streaming, "_tensors_on_one_device", lambda *args: None)
    resource = native_streaming.NativeStreamingProjection(
        None, lambda *args: pytest.fail("invalid column window launched"), "column spy"
    )
    with pytest.raises(ValueError, match="column window"):
        resource.columns(bank, prepared, ends, first, count)


def test_missing_column_resource_does_not_launch(monkeypatch):
    bank, prepared, ends = python_fixture()
    monkeypatch.setattr(native_streaming, "_tensors_on_one_device", lambda *args: None)
    resource = native_streaming.NativeStreamingProjection(None, lambda *args: pytest.fail("missing resource launched"))
    with pytest.raises(RuntimeError, match="missing"):
        resource.columns(bank, prepared, ends, 0, 1)


def test_python_columns_launch_abi_metadata_only(monkeypatch):
    bank, prepared, ends = python_fixture()
    monkeypatch.setattr(native_streaming, "_tensors_on_one_device", lambda *args: None)
    calls = []
    resource = native_streaming.NativeStreamingProjection("full spy", lambda *args: calls.append(args), "columns spy")
    output = resource.columns(bank, prepared, ends, 0, 1)
    assert output.shape == (17, 128) and output.dtype == torch.float16
    assert len(calls) == 1 and calls[0][0] == "columns spy" and calls[0][2] == 8
    args = calls[0][1]
    assert len(args) == 11 and args[-1].tolist() == [17, 4, 128, 640, 8, 0, 1, 0, 1]
