"""Hardware gate for worker transport, mixed image ordering and chunk limits."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_worker_bytes_match_real_model_inputs(request, tmp_path):
    model = request.config.getoption("--scene-dense-model")
    matrix = request.config.getoption("--scene-dense-build")
    build = request.config.getoption("--scene-image-prepare-build")
    device = request.config.getoption("--scene-dense-rank")
    if not model or not matrix or not build:
        pytest.skip("Provide the real checkpoint and both builds; requires an isolated 310P")
    output = tmp_path / "worker-image.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rtsp_scene.worker_image_bench",
            "--model",
            model,
            "--matrix-build",
            matrix,
            "--build",
            build,
            "--device",
            str(device),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        timeout=240,
    )
    rows = json.loads(output.read_text())
    assert {r["cohort"] for r in rows} == {"mixed_order", "chunk_boundary", "live_4_3", "live_16_9"}
    assert all(r["bitwise_equal"] and r["max_abs_error"] == 0 for r in rows)
    for row in rows:
        assert row["transport_bytes"] < row["baseline_float_bytes"] * 0.20
        assert row["worker"]["median_us"] < row["cpu"]["median_us"] * 0.90
