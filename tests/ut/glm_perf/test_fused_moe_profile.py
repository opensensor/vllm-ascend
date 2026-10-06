# SPDX-License-Identifier: Apache-2.0
"""Profiler failure cleanup and frozen-source validation."""

import json
from types import SimpleNamespace

import pytest

from tools.glm_perf import fused_moe_profile


def test_event_profiling_restores_launcher_after_pipeline_failure(monkeypatch):
    class Event:
        def record(self):
            pass

    monkeypatch.setattr(
        fused_moe_profile.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None, Event=lambda **kwargs: Event()),
        raising=False,
    )
    launch = lambda *args: None
    native = SimpleNamespace(launch=launch, pack_kernel=object(), gate_kernel=object(), down_kernel=object())
    calls = []

    def pipeline():
        calls.append(native.launch)
        native.launch(native.gate_kernel, [], 8)
        if len(calls) == 2:
            raise RuntimeError("device submission failed")

    with pytest.raises(RuntimeError, match="device submission"):
        fused_moe_profile.measure(native, pipeline, warmups=1, samples=3)
    assert calls[0] is launch and calls[1] is not launch
    assert native.launch is launch


def test_frozen_profiler_rejects_changed_helper_before_loading_device_code(tmp_path):
    root = tmp_path / "test_frozen_helpers"
    root.mkdir()
    (root / "glm_fused_moe.py").write_text("changed helper")
    (tmp_path / "provenance.json").write_text(
        json.dumps({"_build": {"helper_package": root.name}, "_helpers": {"glm_fused_moe.py": "0" * 64}})
    )
    with pytest.raises(ValueError, match="frozen helper changed"):
        fused_moe_profile.frozen_helper(tmp_path)
