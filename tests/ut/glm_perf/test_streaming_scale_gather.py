# SPDX-License-Identifier: Apache-2.0
"""Actual public-operator helper: exact bits, bounded retained layout and replay."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def scale_gather_checker(tmp_path_factory):
    tests = Path(__file__).parent
    root = tests.parents[2]
    executable = tmp_path_factory.mktemp("streaming-scale-gather") / "check-scale-gather"
    stubs = tests / "streaming_scale_gather_cpu_stubs"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{stubs}",
            f"-I{root / 'tools/glm_perf'}",
            str(stubs / "check_streaming_scale_gather.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("rows", range(1, 32))
@pytest.mark.parametrize("stride", [1, 4, 128])
def test_exact_scale_replication_with_retained_indices_and_changed_batches(scale_gather_checker, rows, stride):
    result = subprocess.run(
        [str(scale_gather_checker), str(rows), str(stride)], capture_output=True, text=True, check=True
    )
    assert "exact bits, retained route cache and changed batches passed: 768 broadcasts" in result.stdout


def test_reject_bounds_alias_route_cache_and_missing_dependency_fence(scale_gather_checker):
    result = subprocess.run([str(scale_gather_checker), "faults"], capture_output=True, text=True, check=True)
    assert "10 bounds, alias, retained-cache and dependency faults rejected" in result.stdout
