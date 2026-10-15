# SPDX-License-Identifier: Apache-2.0
"""Compile and exercise the exact integer helper used by the AICore kernel."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_integer_decisions_across_context(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / "integer-gate"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            f"-I{root / 'tools/qwen4exp'}",
            str(Path(__file__).parent / "cpp/test_mamba_commit_integer.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
