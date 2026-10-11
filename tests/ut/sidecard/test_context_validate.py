# SPDX-License-Identifier: Apache-2.0
"""Client budgets and boundary admission are distinct from completed prefill."""

import io
import json
import urllib.error
from unittest.mock import Mock

import pytest

from tools.sidecard.context_validate import model_limits, run


def test_model_limits_reserve_output_inside_the_context():
    assert model_limits({"data": [{"id": "glm-sidecard", "max_model_len": 360000}]}) == {
        "context": 360000,
        "input": 328000,
        "output": 32000,
    }


@pytest.mark.parametrize(
    "models",
    [
        {"data": []},
        {"data": [{"id": "glm-sidecard", "max_model_len": 8192}]},
        {"data": [{"id": "glm-sidecard", "max_model_len": 32000}]},
        {"data": [{"id": "glm-sidecard", "max_model_len": True}]},
    ],
)
def test_incompatible_service_limits_are_rejected(models):
    with pytest.raises(ValueError):
        model_limits(models)


def test_boundary_admission_never_claims_long_prefill_or_quality(tmp_path, monkeypatch):
    responses = [
        {"response": {"data": [{"id": "glm-sidecard", "max_model_len": 360000}]}},
        {"response": {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}},
        {"response": {"tokens": [12]}},
    ]
    api = Mock(side_effect=responses)
    monkeypatch.setattr("tools.sidecard.context_validate.request", api)
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    rejected = urllib.error.HTTPError("url", 400, "too long", {}, io.BytesIO(b'{"error":"too long"}'))
    urlopen = Mock(side_effect=[response, rejected])
    monkeypatch.setattr("tools.sidecard.context_validate.urllib.request.urlopen", urlopen)
    output = tmp_path / "report.json"
    report = run("http://localhost:8001", output, expected_context=360000)
    assert report["passed"]
    assert not report["full_context_prefill_completed"]
    assert not report["long_context_quality_validated"]
    assert not report["near_capacity_admission"]["generation_completed"]
    assert json.loads(output.read_text()) == report
    body = json.loads(urlopen.call_args_list[0].args[0].data)
    assert len(body["prompt"]) + body["max_tokens"] == 360000
    assert api.call_args_list[1].args[2]["max_tokens"] == 32000


def test_context_mismatch_preserves_failed_report_before_generation(tmp_path, monkeypatch):
    api = Mock(return_value={"response": {"data": [{"id": "glm-sidecard", "max_model_len": 360000}]}})
    monkeypatch.setattr("tools.sidecard.context_validate.request", api)
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="expected admission"):
        run("http://localhost:8001", output, expected_context=360001)
    report = json.loads(output.read_text())
    assert not report["passed"]
    assert report["error"]["type"] == "ValueError"
    assert api.call_count == 1
