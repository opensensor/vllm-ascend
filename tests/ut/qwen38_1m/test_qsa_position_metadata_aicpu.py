"""Host parity gate for the isolated QSA AI CPU metadata candidate."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/qwen4exp/qsa_position_metadata_host.cpp"


@pytest.fixture(scope="module")
def compute(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is required for the host C++ parity gate")
    library = tmp_path_factory.mktemp("qsa_position_metadata") / "metadata.so"
    subprocess.run(["g++", "-O2", "-std=c++17", "-fPIC", "-shared", str(SOURCE), "-o", str(library)], check=True)
    function = ctypes.CDLL(str(library)).qsa_position_metadata_host
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
    ]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("ratio,capacity,width", [(4, 10000, 512), (3, 5, 3), (1, 0, 0)])
def test_metadata_matches_qsa_reference(compute, ratio, capacity, width):
    positions = torch.tensor([-5, -2, -1, 0, 1, 2, 3, 4, 5, 2047, 39999, 262143], dtype=torch.int32)
    input_positions = (ctypes.c_int32 * positions.numel())(*positions.tolist())
    output = (ctypes.c_int32 * (4 * positions.numel()))()
    assert compute(input_positions, positions.numel(), ratio, capacity, width, output) == 0
    next_positions = positions + 1
    complete = torch.div(next_positions, ratio, rounding_mode="floor")
    expected = torch.stack(
        (
            complete.clamp_max(capacity),
            complete.clamp_max(capacity).clamp_max(width),
            complete * ratio,
            next_positions - complete * ratio,
        )
    )
    actual = torch.tensor(list(output), dtype=torch.int32).view(4, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratio,capacity,width", [(0, 10, 2), (4, -1, 0), (4, 10, 11)])
def test_rejects_bad_geometry(compute, ratio, capacity, width):
    positions = (ctypes.c_int32 * 1)(0)
    output = (ctypes.c_int32 * 4)()
    assert compute(positions, 1, ratio, capacity, width, output) != 0


def test_rejects_output_overflow(compute):
    positions = (ctypes.c_int32 * 1)(2**31 - 1)
    output = (ctypes.c_int32 * 4)()
    assert compute(positions, 1, 4, 10000, 512, output) != 0


def test_prefill_chunk_matches_reference(compute):
    positions = torch.arange(37952, 40000, dtype=torch.int32)
    input_positions = (ctypes.c_int32 * positions.numel())(*positions.tolist())
    output = (ctypes.c_int32 * (4 * positions.numel()))()
    assert compute(input_positions, positions.numel(), 4, 10000, 512, output) == 0
    complete = torch.div(positions + 1, 4, rounding_mode="floor")
    expected = torch.stack((complete, complete.clamp_max(512), complete * 4, positions + 1 - complete * 4))
    torch.testing.assert_close(torch.tensor(list(output), dtype=torch.int32).view(4, -1), expected, rtol=0, atol=0)
