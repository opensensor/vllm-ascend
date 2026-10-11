# SPDX-License-Identifier: Apache-2.0
"""Execute production token-window loops; replace only Cube math on CPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_token_owned_prefill_route_order_peer_nan_tails_and_no_gm_route_writes(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    stubs = Path(__file__).parent / "channel_scale_cpu_stubs"
    source = (root / "tools/glm_perf/glm_channel_down_token.cpp").read_text()
    source = source.replace('#include "glm_channel_projection.cpp"', '#include "projection_stub.h"')
    (tmp_path / "production_down_token.cpp").write_text(source)
    binary = tmp_path / "check-token-prefill"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{stubs}",
            str(stubs / "check_token_prefill.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
