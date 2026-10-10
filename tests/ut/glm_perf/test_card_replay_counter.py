# SPDX-License-Identifier: Apache-2.0
"""Execute actual diagnostic replay hooks without NPU imports."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def worker(model):
    path = (
        Path(__file__).resolve().parents[3]
        / "artifacts/glm-perf-310p/hardware-card-profiles-20261010/card_worker_extension.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"os": os}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    result = namespace["CardWorkerExtension"]()
    result.rank = 0
    result.model_runner = SimpleNamespace(model=model)
    return result


def test_replay_counter_is_idempotent_and_restores_original():
    calls = []
    original = lambda *args, **kwargs: calls.append((args, kwargs)) or 42
    model = SimpleNamespace(_replay=original)
    probe = worker(model)
    assert probe.card_start_replay_count()["replays"] == 0
    assert model._replay(1, flag=True) == 42
    assert probe.card_start_replay_count()["replays"] == 1
    assert calls == [((1,), {"flag": True})]
    assert probe.card_stop_replay_count()["replays"] == 1
    assert model._replay is original
    assert probe.card_replay_status()["replays"] is None
    probe.card_stop_replay_count()


def test_failed_replay_is_not_counted():
    def fail(*args):
        raise RuntimeError("replay failed")

    probe = worker(SimpleNamespace(_replay=fail))
    probe.card_start_replay_count()
    with pytest.raises(RuntimeError, match="replay failed"):
        probe.model_runner.model._replay(1)
    assert probe.card_replay_status()["replays"] == 0
    probe.card_stop_replay_count()


def test_eager_wrapper_is_reported_without_mutation():
    model = SimpleNamespace()
    probe = worker(model)
    assert "error" in probe.card_start_replay_count()
    assert not hasattr(model, "_replay")
