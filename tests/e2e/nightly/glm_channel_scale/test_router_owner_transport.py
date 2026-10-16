# SPDX-License-Identifier: Apache-2.0
"""Opt-in device packet and graph gate; synthetic routes do not qualify GLM."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_router_hccl_preserves_changed_int64_ids_and_float32_weights(request, tmp_path):
    if not request.config.getoption("--glm-router-owner-hccl"):
        pytest.skip("requires explicit reservation; ordinary collection stays offline")
    tp = request.config.getoption("--glm-router-owner-tp-size")
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={tp}",
            "-m",
            "tools.glm_perf.probe_router_owner",
            "--backend",
            "hccl",
            "--tp-size",
            str(tp),
            "--graphs",
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / "torchrun.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, str(tmp_path / "torchrun.log")
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(tp)]
    assert all(report["owner_rank"] == tp - 1 for report in reports)
    assert all(case["byte_parity"] for report in reports for case in report["cases"])
    assert all(case["graph_parity"] for report in reports for case in report["cases"] if case["tokens"] <= 4)
