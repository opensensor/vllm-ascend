# SPDX-License-Identifier: Apache-2.0
"""Explicit six/four-chip transport and asymmetric capture gate; no real model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_indexer_owner_hccl_and_changed_input_graphs(request, tmp_path):
    if not request.config.getoption("--glm-indexer-owner-hccl"):
        pytest.skip("requires explicit exclusive chip reservation; ordinary collection stays offline")
    tp = request.config.getoption("--glm-indexer-owner-tp-size")
    owner = request.config.getoption("--glm-indexer-owner-rank")
    world = 6 if tp == 2 else 4
    root = Path(__file__).resolve().parents[4]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "-m",
        "tools.glm_perf.probe_indexer_owner",
        "--backend",
        "hccl",
        "--tp-size",
        str(tp),
        "--owner-rank",
        str(owner),
        "--tokens",
        "1",
        "4",
        "37",
        "640",
        "--graphs",
        "--output",
        str(tmp_path),
    ]
    if request.config.getoption("--glm-indexer-overlap-prefill"):
        command.append("--overlap-prefill")
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / "torchrun.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, str(tmp_path / "torchrun.log")
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(world)]
    assert len(reports) == world
    assert all(case["changed_input_byte_parity"] for report in reports for case in report["cases"])
    assert all(case["graph_parity"] is True for report in reports for case in report["cases"] if case["tokens"] <= 4)
