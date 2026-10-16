"""Hardware parity and performance gates for the isolated scene kernel stack."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_tp_shard_dense_and_nz_ffn(request, tmp_path):
    model = request.config.getoption("--scene-dense-model")
    build = request.config.getoption("--scene-dense-build")
    rank = request.config.getoption("--scene-dense-rank")
    if not model or not build:
        pytest.skip("Provide the real checkpoint and frozen RTC build; requires an isolated 310P")
    manifest = json.loads((Path(build) / "build.json").read_text())
    tokens = [1, 4, 8]
    if manifest.get("max_decode_rows", 8) == 16:
        tokens.extend((9, 16))
    root = Path(__file__).resolve().parents[4]
    for module, filename in (("dense_bench", "dense.json"), ("nz_bench", "nz.json")):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.rtsp_scene." + module,
                "--model",
                str(Path(model).resolve()),
                "--build",
                str(Path(build).resolve()),
                "--rank",
                str(rank),
                "--output",
                str(tmp_path / filename),
                "--iterations",
                "100",
                "--repeats",
                "8",
                "--tokens",
                *map(str, tokens),
            ],
            cwd=root,
            check=True,
            timeout=180,
        )
    dense = json.loads((tmp_path / "dense.json").read_text())
    nz = json.loads((tmp_path / "nz.json").read_text())
    assert len(dense["records"]) == len(tokens) * 4 * 2
    assert all(item["passed"] and item["graph_changed_input_passed"] for item in dense["records"])
    assert all(check["max_absolute_error"] == 0 for item in dense["records"] for check in item["checks"])
    assert len(nz["activation"]) == len(tokens) * 6 and len(nz["ffn"]) == len(tokens)
    assert all(item["passed"] and item["max_code_error"] == 0 for item in nz["activation"])
    assert all(item["passed"] and item["graph_changed_input_passed"] for item in nz["ffn"])
    # Include conversion removal AND both projections, not an activation-only gain.
    for item in nz["ffn"]:
        assert item["graph_custom"]["median_us"] <= item["graph_reference"]["median_us"]
