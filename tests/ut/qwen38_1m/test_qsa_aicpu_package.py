# SPDX-License-Identifier: Apache-2.0
"""Regression checks for AI CPU-only custom-op packaging."""

import shutil
import subprocess
from pathlib import Path

import pytest

PACKAGE_CMAKE = Path(__file__).resolve().parents[3] / "csrc/cmake/aicpu_package.cmake"


def detect_package(tmp_path: Path, op_names: list[str], enabled: bool) -> bool:
    script = tmp_path / "check.cmake"
    selected = " ".join(f'"{name}"' for name in op_names)
    script.write_text(
        f'include("{PACKAGE_CMAKE}")\n'
        f'detect_aicpu_only_package(result "{tmp_path}" '
        f"{'ON' if enabled else 'OFF'} {selected})\n"
        'message(STATUS "AICPU_ONLY=${result}")\n'
    )
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True, check=True)
    return "AICPU_ONLY=ON" in result.stdout


@pytest.mark.skipif(shutil.which("cmake") is None, reason="CMake is required")
def test_aicpu_only_package_skips_ai_core_metadata(tmp_path):
    op_dir = tmp_path / "attention/qsa_exact_topk_aicpu_v310"
    (op_dir / "op_kernel_aicpu").mkdir(parents=True)

    assert detect_package(tmp_path, [op_dir.name], enabled=True)
    assert not detect_package(tmp_path, [op_dir.name], enabled=False)


@pytest.mark.skipif(shutil.which("cmake") is None, reason="CMake is required")
def test_mixed_and_unknown_packages_keep_ai_core_metadata(tmp_path):
    cpu_op = tmp_path / "attention/qsa_exact_topk_aicpu_v310"
    (cpu_op / "op_kernel_aicpu").mkdir(parents=True)
    core_op = tmp_path / "attention/qsa_indexer_score_v310"
    (core_op / "op_kernel").mkdir(parents=True)

    assert not detect_package(tmp_path, [cpu_op.name, core_op.name], enabled=True)
    assert not detect_package(tmp_path, [cpu_op.name, "missing_op"], enabled=True)
    assert not detect_package(tmp_path, ["ALL"], enabled=True)
