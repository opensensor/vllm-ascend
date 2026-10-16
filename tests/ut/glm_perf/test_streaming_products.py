# SPDX-License-Identifier: Apache-2.0
"""Execute the actual streamed consumer with independent bounded CPU semantics."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def products_checker(tmp_path_factory):
    tests = Path(__file__).parent
    root = tests.parents[2]
    executable = tmp_path_factory.mktemp("streaming-products") / "check-products"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-strict-aliasing",
            "-DGLM_STREAMING_DEVICE_API",
            f"-I{tests / 'streaming_products_cpu_stubs'}",
            f"-I{root / 'tools/glm_perf'}",
            str(tests / "reduce_kernel_cpu_stubs/check_streaming_products.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("rows", range(1, 32))
@pytest.mark.parametrize("group_major", [False, True])
def test_complete_paired_consumer_all_rows_layouts_and_slot_replays(products_checker, rows, group_major, k):
    result = subprocess.run(
        [str(products_checker), str(rows), str(int(group_major)), str(k)], capture_output=True, text=True, check=True
    )
    assert (
        f"direct strided casts, ordered K32 factors, slot reuse and guards passed: {7 * k // 64} pairs" in result.stdout
    )


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("rows", range(1, 32))
@pytest.mark.parametrize("stride", [1, 4, 128])
def test_gather_scale_consumer_all_rows_strides_raw_slots_and_changed_batches(products_checker, rows, stride, k):
    result = subprocess.run(
        [str(products_checker), str(rows), str(int(stride == 1)), str(k), str(stride)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"slot reuse and guards passed: {7 * k // 64} pairs" in result.stdout
    assert "expanded scale Gather, exact bits, route-cache independence and two cast masks passed" in result.stdout
