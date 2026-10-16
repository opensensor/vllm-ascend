"""Actual checkpoint and changed-input graph gate for the active-row epilogue."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_active_epilogue_preserves_real_projection_and_ffn_outputs(request, tmp_path):
    model = request.config.getoption("--scene-dense-model")
    control = request.config.getoption("--scene-dense-build")
    candidate = request.config.getoption("--scene-active-epilogue-build")
    rank = request.config.getoption("--scene-dense-rank")
    if not model or not control or not candidate:
        pytest.skip("Requires an isolated scene NPU, real checkpoint and both frozen builds")
    manifest = json.loads((Path(candidate) / "build.json").read_text())
    assert manifest["active_epilogue"] is True
    output = tmp_path / "active-epilogue.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rtsp_scene.kernel_compare",
            "--model",
            model,
            "--control",
            control,
            "--candidate",
            candidate,
            "--output",
            str(output),
            "--rank",
            str(rank),
            "--device",
            str(rank),
            "--tokens",
            "1",
            "2",
            "4",
            "8",
            "--iterations",
            "100",
            "--repeats",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        timeout=300,
    )
    report = json.loads(output.read_text())
    assert report["device"] == rank and report["rank"] == rank
    records = report["records"]
    assert len(records) == 12
    for row in records:
        assert row["passed"] and row["graph_passed"]
        assert len(row["checks"]) == 5 and len(row["graph_checks"]) == 3
        assert all(all(check["bitwise_equal"]) for check in row["checks"] + row["graph_checks"])
        assert row["graph_candidate"]["median_us"] > 0
        assert row["graph_control"]["median_us"] > 0
