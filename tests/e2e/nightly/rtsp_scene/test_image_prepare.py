"""Gate exact device image preparation and its isolated performance."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

MAX_TIME_RATIO = 0.90


@pytest.mark.parametrize("input_width", [320, 640])
def test_device_image_preparation_matches_model_inputs(request, tmp_path, input_width):
    model = request.config.getoption("--scene-dense-model")
    matrix = request.config.getoption("--scene-dense-build")
    build = request.config.getoption("--scene-image-prepare-build")
    device = request.config.getoption("--scene-dense-rank")
    if not model or not matrix or not build:
        pytest.skip("Provide the local checkpoint and both builds; requires an isolated 310P")
    output = tmp_path / "image-prepare.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rtsp_scene.image_prepare_bench",
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
            "--input-width",
            str(input_width),
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        timeout=240,
    )
    results = json.loads(output.read_text())
    assert len(results) == 4
    heights = (180, 240) if input_width == 320 else (360, 480)
    assert {(row["height"], row["frames"]) for row in results} == {(height, n) for height in heights for n in (1, 8)}
    for row in results:
        assert row["passed"] and len(row["checks"]) == 4
        assert all(check["bitwise_equal"] and check["max_abs_error"] == 0 for check in row["checks"])
        assert row["npu"]["median_us"] < row["cpu"]["median_us"] * MAX_TIME_RATIO
