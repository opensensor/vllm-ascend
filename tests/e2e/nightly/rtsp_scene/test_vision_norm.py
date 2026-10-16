"""Require checkpoint parity and faster paired vision-norm graph execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_checkpoint_vision_normalization_parity_and_performance(request, tmp_path):
    model = request.config.getoption("--scene-dense-model")
    device = request.config.getoption("--scene-dense-rank")
    if not model:
        pytest.skip("Provide the real checkpoint; requires an isolated 310P")
    output = tmp_path / "vision-norm.json"
    root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rtsp_scene.vision_norm_bench",
            "--model",
            str(Path(model).resolve()),
            "--output",
            str(output),
            "--device",
            str(device),
        ],
        cwd=root,
        check=True,
        timeout=240,
    )
    result = json.loads(output.read_text())
    assert len(result["records"]) == 4
    for record in result["records"]:
        assert record["passed"] and record["graph_passed"]
        assert all(all(case["bitwise_equal"]) for case in record["checks"])
        assert record["graph_candidate"]["median_us"] < record["graph_control"]["median_us"]
