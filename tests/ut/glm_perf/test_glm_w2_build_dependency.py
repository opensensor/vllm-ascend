"""Keep grouped W2/W4 builds sensitive to their shared kernel header."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_grouped_kernel_registers_standalone_source_dependency(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")

    cmake_lists = Path(__file__).parents[3] / "csrc/gmm/w2_grouped_blocked_dequant_matmul_v310/op_host/CMakeLists.txt"
    script = tmp_path / "check-dependency.cmake"
    script.write_text(
        "macro(add_op_to_compiled_list)\nendmacro()\n"
        "function(add_ops_compile_options)\nendfunction()\n"
        "set(BUILD_OPS_RTY_KERNEL ON)\n"
        f'include("{cmake_lists.as_posix()}")\n'
        'if(NOT "${w2_grouped_blocked_dequant_matmul_v310_depends}" '
        'STREQUAL "gmm/w2_blocked_dequant_matmul_v310")\n'
        '  message(FATAL_ERROR "Grouped W2/W4 shared header is not a build-cache dependency")\n'
        "endif()\n",
        encoding="utf-8",
    )
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
