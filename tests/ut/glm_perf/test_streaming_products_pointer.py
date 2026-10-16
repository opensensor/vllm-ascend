# SPDX-License-Identifier: Apache-2.0
"""Pinned compiler-pointer helper with independent CPU instruction models."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pointer_checker(tmp_path_factory):
    tests = Path(__file__).parent
    root = tests.parents[2]
    stubs = tests / "streaming_products_pointer_cpu_stubs"
    executable = tmp_path_factory.mktemp("streaming-pointer-products") / "check-pointer"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-fno-strict-aliasing",
            "-DGLM_STREAMING_POINTER_CPU_MODEL",
            f"-I{stubs}",
            f"-I{root / 'tools/glm_perf'}",
            str(stubs / "check_streaming_products_pointer.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("stride", [1, 4, 128])
@pytest.mark.parametrize("rows", range(1, 32))
def test_pointer_consumer_all_rows_layouts_slots_masks_and_changed_batches(pointer_checker, rows, stride, k):
    result = subprocess.run(
        [str(pointer_checker), str(rows), str(stride), str(k)], capture_output=True, text=True, check=True
    )
    assert f"retained guards passed: {7 * k // 64} pairs" in result.stdout
    assert "FMA and signed-zero witnesses passed" in result.stdout


def test_pointer_faults_require_masks_bounds_raw_readiness_and_vector_fences(pointer_checker):
    result = subprocess.run([str(pointer_checker), "faults"], capture_output=True, text=True, check=True)
    assert "7 mask, pointer-bound, raw-lifetime and dependency faults rejected" in result.stdout
