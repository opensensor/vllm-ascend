# SPDX-License-Identifier: Apache-2.0
"""Drain/graph ordering and fail-closed operator-mode transitions."""

import dataclasses
import types

import pytest
import torch

from tools.glm_perf.resident_control import Control
from tools.minimax_m3.resident_determinism import flag_replacements, transition


class Client:
    def __init__(self, fail=False, baseline=False):
        self.source = "" if baseline else "def replacements(native_resources): return {}\n"
        candidate = "baseline" if baseline else "m3_candidate"
        control = Control("old", "direct-target", candidate, self.source)
        self.status = [
            dict(
                rank=0,
                pid=12,
                weight_storage_digest="weights",
                generation="old",
                candidate=candidate,
                digest=control.digest,
                native_failed=False,
            )
        ]
        self.paused, self.fail, self.events = False, fail, []

    def request(self, path, **kwargs):
        if path == "/is_paused":
            return {"is_paused": self.paused}
        assert path == "/pause?mode=wait&clear_cache=true"
        self.paused = True
        self.events.append("drain")

    def rpc(self, method, *args):
        if method == "resident_status":
            return self.status
        assert self.paused
        self.events.append(method)
        if method == "resident_capture":
            assert self.events[-2] == "resident_apply"
            return [{"determinism": {"passed": not self.fail}}]
        return self.status

    def switch(self, control):
        Control.from_dict(dataclasses.asdict(control))
        assert self.paused and control.source == self.source
        self.events.append("recapture")
        return self.status

    def resume(self):
        self.events.append("resume")
        self.paused = False


@pytest.mark.parametrize("baseline", [False, True])
def test_flags_change_after_graph_release_and_before_recapture(baseline):
    client = Client(baseline=baseline)
    assert transition(client, client.source, True)["enabled"]
    assert client.events == ["drain", "resident_prepare", "resident_apply", "resident_capture", "recapture", "resume"]


def test_failed_flag_change_leaves_server_paused():
    client = Client(fail=True)
    with pytest.raises(RuntimeError, match="remains paused"):
        transition(client, client.source, True)
    assert client.paused
    assert "recapture" not in client.events and "resume" not in client.events


def test_wrong_source_rejected_before_pause():
    client = Client()
    with pytest.raises(ValueError, match="source must match"):
        transition(client, client.source + "\n", True)
    assert not client.events


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_reports_actual_flags_after_synchronizing(monkeypatch, enabled):
    state, events = {"enabled": not enabled, "warn_only": False}, []
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: state["enabled"])
    monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: state["warn_only"])
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=lambda: events.append("sync")), raising=False)

    def set_flags(value, warn_only):
        assert events == ["idle", "sync"]
        state.update(enabled=value, warn_only=warn_only)

    monkeypatch.setattr(torch, "use_deterministic_algorithms", set_flags)
    worker = types.SimpleNamespace(_m3_idle=lambda: events.append("idle"), resident_status=lambda: {"rank": 0})
    result = next(iter(flag_replacements(enabled).values()))(worker)["determinism"]
    assert result["passed"] and result["previous"]["enabled"] is not enabled
    assert result["current"] == {"enabled": enabled, "warn_only": enabled}


def test_worker_collects_flag_failure_without_escaping_rpc(monkeypatch):
    def fail():
        raise RuntimeError("injected sync failure")

    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=fail), raising=False)
    worker = types.SimpleNamespace(_m3_idle=lambda: None, resident_status=lambda: {"rank": 0})
    result = next(iter(flag_replacements(True).values()))(worker)
    assert not result["determinism"]["passed"]
    assert "injected sync failure" in result["determinism"]["error"]
