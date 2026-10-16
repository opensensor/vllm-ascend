# SPDX-License-Identifier: Apache-2.0
"""Execute the actual operand helper with a bounded CPU copy/event model."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "streaming_operands_cpu_stubs"


@pytest.fixture(scope="module")
def operands_program(tmp_path_factory):
    executable = tmp_path_factory.mktemp("streaming-operands") / "check-streaming-operands"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{STUBS}",
            f"-I{ROOT / 'tools/glm_perf'}",
            str(Path(__file__).parent / "reduce_kernel_cpu_stubs/check_streaming_operands.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("rows", [1, 4, 5, 15, 16, 17, 30, 31])
@pytest.mark.parametrize("pairs", [0, 1, 3, 4, 32, 64])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_operand_bytes_order_reuse_and_drain(operands_program, rows, pairs, reuse, bits):
    result = subprocess.run(
        [str(operands_program), str(rows), str(pairs), str(int(reuse)), str(bits)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "operand bytes, event ownership, cache reuse and drain passed" in result.stdout


@pytest.mark.parametrize("rows", [1, 4, 5, 15, 16, 17, 30, 31])
@pytest.mark.parametrize("pairs", [0, 1, 3, 4, 32, 64])
def test_down_reuses_bounded_l1_staging_slots(operands_program, rows, pairs):
    result = subprocess.run(
        [str(operands_program), str(rows), str(pairs), "0", "4", "bounded"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "operand bytes, event ownership, cache reuse and drain passed" in result.stdout


@pytest.mark.parametrize(
    "mode",
    [
        "unmatched",
        "unready",
        "duplicate",
        "release",
        "slot",
        "overwrite",
        "l1-reader",
        "l1-backward",
        "source-bounds",
        "destination-bounds",
    ],
)
def test_invalid_instruction_order_or_bounds_is_detected(operands_program, mode):
    result = subprocess.run([str(operands_program), mode], capture_output=True, text=True, check=True)
    assert "rejected:" in result.stdout
