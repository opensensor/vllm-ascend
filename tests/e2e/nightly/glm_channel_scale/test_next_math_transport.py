# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU process integration; never opens an NPU or claims model parity."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("tp,world", [(2, 6), (4, 4)])
def test_router_packets_and_global_head_score_reduction(request, tmp_path, tp, world):
    if not request.config.getoption("--glm-next-math-cpu-transport"):
        pytest.skip("requires explicit CPU process gate; no NPU validation")
    root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "-m",
        "tools.glm_perf.probe_rank_math",
        "--tp-size",
        str(tp),
        "--output",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "torchrun.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, str(tmp_path / "torchrun.log")
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(world)]
    assert all(report["owner_rank"] == tp - 1 and report["backend"] == "gloo" for report in reports)
    assert all(case["argmax_agreement"] == 1 for report in reports for case in report["cases"])
