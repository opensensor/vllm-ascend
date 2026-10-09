# SPDX-License-Identifier: Apache-2.0
"""Reject missing sensor coverage, premature resumes and unsafe MTP receipts."""

import json

import pytest

from tools.qwen4exp.benchmark_speculation_sweep import thermal_evidence


def sample(time, temperature, action="none", cooling=False):
    return {
        "time": time,
        "temperatures_c": [temperature] * 4,
        "sensor_valid": True,
        "action": action,
        "cooling": cooling,
    }


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_hold_hysteresis_and_resume_with_complete_coverage(tmp_path):
    path = tmp_path / "thermal.jsonl"
    rows = [
        sample(100, 90),
        sample(102, 94, "pause", True),
        sample(104, 92, "holding", True),
        sample(106, 85, "resume"),
        sample(108, 86),
    ]
    write(path, rows)
    assert thermal_evidence(path, 101, 109) == {
        "max_core_c": 94,
        "thermal_policy_pass": True,
        "thermal_shutdown": None,
        "hard_thermal_limit_exceeded": False,
    }


@pytest.mark.parametrize("fault", ["no_hold", "resume_hot", "gap", "missing_sensor", "control_error", "stale"])
def test_faults_cannot_qualify_sustained_arm(tmp_path, fault):
    path = tmp_path / "thermal.jsonl"
    rows = [sample(100, 90), sample(102, 94, "pause", True), sample(104, 85, "resume")]
    if fault == "no_hold":
        rows[1] = sample(102, 94)
    elif fault == "resume_hot":
        rows[2] = sample(104, 86, "resume")
    elif fault == "gap":
        rows[1]["time"] = 120
        rows[2]["time"] = 122
    elif fault == "missing_sensor":
        rows[1]["temperatures_c"].pop()
    elif fault == "control_error":
        rows[1]["action"] = "control_error"
    end = 125 if fault in ("gap", "stale") else 105
    write(path, rows)
    assert thermal_evidence(path, 101, end)["thermal_policy_pass"] is False


def test_empty_and_partial_log_cannot_supply_temperatures(tmp_path):
    path = tmp_path / "thermal.jsonl"
    path.write_text('{"time":')
    assert thermal_evidence(path, 100, 101)["max_core_c"] is None
