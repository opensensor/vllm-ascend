# SPDX-License-Identifier: Apache-2.0
"""Production core-slot ownership, tile reuse and original-order combination."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("cached", [False, True])
def test_production_prefill_tile_protocol_and_guard_bytes(tmp_path, cached):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    stubs = Path(__file__).parent / "channel_scale_cpu_stubs"
    source = (root / "tools/glm_perf/glm_channel_down_prefill.cpp").read_text()
    source = source.replace('#include "glm_channel_projection.cpp"', '#include "projection_stub.h"')
    (tmp_path / "production_down_prefill.cpp").write_text(source)
    binary = tmp_path / "check-prefill-protocol"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            *(["-DGLM_CHANNEL_CACHE_PREFILL_ROUTES"] if cached else []),
            f"-I{tmp_path}",
            f"-I{stubs}",
            f"-I{root / 'tools/glm_perf'}",
            str(stubs / "check_prefill.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
