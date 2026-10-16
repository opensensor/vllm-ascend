import json

import pytest

from tools.rtsp_scene.serving_compare import compare


def make_run(directory, decode, fingerprint="same"):
    directory.mkdir()
    group = {
        "concurrency": 1,
        "requests": 1,
        "batch_seconds": [decode + 1],
        "fixed_completion_tokens": 64,
        "diagnostic_only": True,
        "server_mean_seconds": {"request_prefill_time_seconds": 1, "request_decode_time_seconds": decode},
    }
    (directory / "summary.json").write_text(json.dumps([group]))
    record = {"body_sha256": fingerprint, "usage": {"prompt_tokens": 123, "completion_tokens": 64}}
    (directory / "c1-r0-0.json").write_text(json.dumps(record))


def test_matched_workload_comparison(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control, 2)
    make_run(candidate, 1)
    result = compare(control, candidate)["results"][0]
    assert result["matched_requests"] == 1
    assert result["server_mean_seconds"]["request_decode_time_seconds"]["time_reduction_percent"] == 50
    assert result["batch_time_reduction_percent"] == pytest.approx(100 / 3)


def test_rejects_different_request_bodies(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control, 2)
    make_run(candidate, 1, "changed")
    with pytest.raises(ValueError, match="workload differs"):
        compare(control, candidate)


def test_rejects_different_token_budget(tmp_path):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    make_run(control, 2)
    make_run(candidate, 1)
    record = candidate / "c1-r0-0.json"
    payload = json.loads(record.read_text())
    payload["usage"]["completion_tokens"] = 63
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="workload differs"):
        compare(control, candidate)
