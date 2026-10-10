# SPDX-License-Identifier: Apache-2.0
"""Actual v2 native body under synchronous CPU bounds stubs; no NPU claims."""

import builtins
import ctypes
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.qwen38_1m.test_streaming_projection import fixture, run
from tools.qwen4exp import benchmark_streaming_next_310 as benchmark
from tools.qwen4exp import native_streaming_next
from tools.qwen4exp.streaming_next_memory import contract, regions, render_header

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "streaming_cpu_stubs"


@pytest.fixture(scope="module")
def native_next(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host g++ unavailable")
    path = tmp_path_factory.mktemp("streaming-next") / "projection.so"
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
            str(STUBS / "harness_next.cpp"),
            "-o",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    library = ctypes.CDLL(str(path))
    for name in ("run_projection", "run_columns"):
        getattr(library, name).argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64)]
        getattr(library, name).restype = ctypes.c_int
    library.projection_error.restype = ctypes.c_char_p
    library.projection_peak.argtypes = [ctypes.c_uint32]
    library.projection_peak.restype = ctypes.c_uint64
    return library


def test_projection_only_contract():
    assert (ROOT / "tools/qwen4exp/qwen_streaming_next_contract.h").read_text() == render_header()
    assert contract()["abi_version"] == 2
    assert max(r.end for r in regions() if r.space == "UB") == 248064
    assert not {"quantizer_scratch", "persistent_gate", "metadata_stage"} & {r.name for r in regions()}
    for space in {r.space for r in regions()}:
        ordered = sorted((r for r in regions() if r.space == space), key=lambda r: r.offset)
        assert all(a.end <= b.offset for a, b in zip(ordered, ordered[1:]))


@pytest.mark.parametrize(
    "rows,k,outputs",
    [
        (1, 128, 160),
        (9, 640, 320),
        (10, 2560, 160),
        (11, 640, 160),
        (31, 128, 320),
        (32, 640, 160),
        (33, 128, 160),
        (65, 640, 160),
        (33, 2560, 1280),
        (33, 640, 2560),
    ],
)
@pytest.mark.parametrize("seed", [3, 41])
def test_exact_projection_with_mblocks_metadata_slots_and_tails(native_next, rows, k, outputs, seed):
    args, expected, storage = fixture(rows, outputs, k, [0, rows], seed)
    originals = [a.copy() for a in args[:9]]
    status, error = run(native_next, args)
    assert status == 0, error
    np.testing.assert_array_equal(args[9], expected)
    np.testing.assert_array_equal(storage[:16], np.float16(123.5))
    np.testing.assert_array_equal(storage[-16:], np.float16(123.5))
    for a, b in zip(args[:9], originals):
        np.testing.assert_array_equal(a, b)
    assert native_next.projection_peak(0) == 248064


@pytest.mark.parametrize("experts", [1, 3, 4, 31, 32, 33, 85, 86, 128])
def test_end_cache_aligned_dma_and_scalar_tail_never_overread(native_next, experts):
    args, expected, _ = fixture(33, 160, 640, [0] * (experts - 1) + [17], 17)
    status, error = run(native_next, args)
    assert status == 0, error
    np.testing.assert_array_equal(args[9], expected)
    assert np.count_nonzero(args[9][17:]) == 0


@pytest.mark.parametrize("first,count", [(0, 8), (8, 8)])
def test_two_down_windows_match_complete_reference(native_next, first, count):
    args, expected, _ = fixture(33, 2560, 640, [0, 33], 29)
    output = np.full((33, count * 160), np.float16(19.5))
    config = np.concatenate((args[10], np.asarray([first, count], dtype=np.int64)))
    status, error = run(native_next, [*args[:9], output, config], columns=True)
    assert status == 0, error
    np.testing.assert_array_equal(output, expected[:, first * 160 : (first + count) * 160])


@pytest.mark.parametrize("first,count", [(-1, 1), (16, 1), (15, 2), (0, 0), (0, 9)])
def test_next_column_bounds(native_next, first, count):
    args, _, _ = fixture(1, 2560, 640, [1], 7)
    args[10] = np.concatenate((args[10], np.asarray([first, count], dtype=np.int64)))
    status, error = run(native_next, args, columns=True)
    assert status != 0 and "assertion" in error


@pytest.mark.parametrize("experts", [85, 86])
def test_native_resource_accepts_tp6_bank_sizes_without_device_reads(monkeypatch, experts):
    monkeypatch.setattr(native_streaming_next, "_tensors_on_one_device", lambda *_: None)
    bank = SimpleNamespace(weight=torch.zeros(experts, 160, 64, dtype=torch.int8))
    for name in ("weight_scale", "weight_offset", "weight_sum"):
        setattr(bank, name, torch.zeros(experts, 160, 1, dtype=torch.float16))
    prepared = (
        torch.zeros(2, 64, dtype=torch.int8),
        torch.zeros(2, 64, dtype=torch.int8),
        torch.zeros(2, 1, 8),
        torch.zeros(2, 1, 8),
    )
    calls = []
    resource = native_streaming_next.NativeStreamingProjectionNext(
        "kernel", lambda *args: calls.append(args), "columns"
    )
    output = resource(bank, prepared, torch.tensor([2] * experts))
    assert output.shape == (2, 160) and resource.tile_columns == 160
    assert len(calls) == 1 and calls[0][0] == "kernel" and calls[0][-1] == 8


def test_benchmark_requires_execution_before_any_device_library_or_sensor_access(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name != "torch_npu"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *_args, **_kw: pytest.fail("unexpected sensor call")
    )
    monkeypatch.setattr(sys, "argv", ["benchmark", "--bundle", "missing", "--model", "missing", "--output", "missing"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("ends", [[32, 33, 65, 66, 98], [1, 33, 34, 66, 98], [32, 64, 65, 97, 98]])
def test_full_tiles_overwrite_reused_operands_and_tails_clear_padding(native_next, ends):
    # Alternate full and partial tiles across experts and metadata-slot wraps.
    args, expected, storage = fixture(98, 160, 640, ends, 811)
    status, error = run(native_next, args)
    assert status == 0, error
    np.testing.assert_array_equal(args[9], expected)
    np.testing.assert_array_equal(storage[:16], np.float16(123.5))
    np.testing.assert_array_equal(storage[-16:], np.float16(123.5))
