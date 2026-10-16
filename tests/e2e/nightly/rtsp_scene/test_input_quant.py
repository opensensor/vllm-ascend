"""Gate exact NZ input quantization and complete checkpoint projections."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

MAX_REGRESSION_RATIO = 1.03
MAX_AGGREGATE_RATIO = 0.99


def test_checkpoint_input_quantization_parity_and_performance(request, tmp_path):
    model = request.config.getoption("--scene-dense-model")
    build = request.config.getoption("--scene-dense-build")
    quant = request.config.getoption("--scene-input-quant-build")
    device = request.config.getoption("--scene-dense-rank")
    if not model or not build or not quant:
        pytest.skip("Provide checkpoint and both frozen builds; requires an isolated 310P")
    output = tmp_path / "input-quant.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rtsp_scene.input_quant_bench",
            "--model",
            str(Path(model).resolve()),
            "--build",
            str(Path(build).resolve()),
            "--input-build",
            str(Path(quant).resolve()),
            "--device",
            str(device),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
        timeout=240,
    )
    result = json.loads(output.read_text())
    assert len(result["codes"]) == 75 and len(result["projections"]) == 20
    assert all(row["codes_equal"] and row["scales_equal"] for row in result["codes"])
    assert len(result["layouts"]) == 15
    assert all(row["codes_equal"] and row["scales_equal"] for row in result["layouts"])
    original, candidate = 0, 0
    for row in result["projections"]:
        assert row["passed"] and row["graph_passed"]
        assert all(all(check["bitwise_equal"]) for check in row["checks"] + row["graph_checks"])
        before, after = row["graph_control"]["median_us"], row["graph_candidate"]["median_us"]
        assert after <= before * MAX_REGRESSION_RATIO
        # The deployed FFN down input was already NZ-quantized. Count only
        # projections where this new fusion replaces the live input path.
        if row["projection"] != "down":
            original += before
            candidate += after
    assert candidate < original * MAX_AGGREGATE_RATIO
