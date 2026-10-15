# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.qwen4exp.mixed_prefill_profile import Recorder


def test_instance_hook_restores_inherited_method_and_bounds_records():
    class Target:
        def calculate(self, value):
            return value + 1

    target = Target()
    recorder = Recorder(limit=2)
    recorder.wrap(target, "calculate", "call", lambda value: dict(value=value))
    assert [target.calculate(value) for value in range(3)] == [1, 2, 3]
    result = recorder.snapshot()
    assert result["total"] == 3 and result["dropped"] == 1
    assert [row["value"] for row in result["records"]] == [1, 2]
    assert all(not row["failed"] and row["duration_ns"] >= 0 for row in result["records"])
    recorder.restore()
    assert "calculate" not in vars(target)
    assert target.calculate(4) == 5


def test_class_hook_restores_exact_descriptor_and_records_failure():
    class Target:
        def calculate(self):
            raise ValueError("original failure")

    original = Target.calculate
    recorder = Recorder()
    recorder.wrap(Target, "calculate", "call")
    with pytest.raises(ValueError, match="original failure"):
        Target().calculate()
    assert recorder.snapshot()["records"][0]["failed"]
    recorder.restore()
    assert Target.calculate is original


def test_changed_hook_prevents_partial_restoration():
    target = SimpleNamespace(first=lambda: 1, second=lambda: 2)
    recorder = Recorder()
    recorder.wrap(target, "first", "first")
    recorder.wrap(target, "second", "second")
    first_wrapper = target.first
    target.second = lambda: 3
    with pytest.raises(RuntimeError, match="ownership changed"):
        recorder.restore()
    assert target.first is first_wrapper
    assert len(recorder.hooks) == 2
