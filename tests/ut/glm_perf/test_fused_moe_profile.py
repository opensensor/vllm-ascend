# SPDX-License-Identifier: Apache-2.0
"""Profiler failure cleanup and frozen-source validation."""

import json
from types import SimpleNamespace

import pytest

from tools.glm_perf import fused_moe_profile


@pytest.mark.parametrize("specialization", [None, 3, 4])
def test_event_profile_uses_bounded_pairs_and_resolves_every_sample(monkeypatch, specialization):
    clock = [0]
    created = []
    syncs = []

    class Event:
        def __init__(self, **kwargs):
            created.append(self)

        def record(self):
            self.timestamp = clock[0]

        def elapsed_time(self, other):
            return other.timestamp - self.timestamp

    monkeypatch.setattr(
        fused_moe_profile.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: syncs.append(clock[0]), Event=Event),
        raising=False,
    )
    kernels = [object() for _ in range(4)]
    durations = {id(kernel): index + 1 for index, kernel in enumerate(kernels)}

    def launch(kernel, args, blocks):
        clock[0] += durations[id(kernel)]

    native = SimpleNamespace(
        launch=launch,
        pack_kernel=kernels[0],
        gate_kernel=kernels[1],
        down_kernel=kernels[2],
        reduce_kernel=kernels[3],
    )

    selected = kernels
    if specialization:
        gate, down = object(), object()
        setattr(native, f"gate_w{specialization}_kernel", gate)
        setattr(native, f"down_w{specialization}_kernel", down)
        durations[id(gate)] = 2
        durations[id(down)] = 3
        selected = (native.pack_kernel, gate, down, native.reduce_kernel)

    def pipeline():
        for kernel in selected:
            native.launch(kernel, [], 8)

    report = fused_moe_profile.measure(native, pipeline, warmups=2, samples=9)
    assert len(created) == (12 if specialization else 8)
    assert len(syncs) == 10
    assert sorted(value["median_ms"] for value in report.values()) == [1, 2, 3, 4]
    assert all(len(value["samples_ms"]) == 9 for value in report.values())
    assert native.launch is launch


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
