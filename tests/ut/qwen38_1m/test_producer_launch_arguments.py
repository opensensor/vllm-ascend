# SPDX-License-Identifier: Apache-2.0
"""Host threads exercise callback copies, failure quarantine and destruction."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_only_producer_releases_operands_after_success(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).parent / "cpp/test_producer_launch_arguments.cpp"
    binary = tmp_path / "ownership"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            f"-I{root}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
