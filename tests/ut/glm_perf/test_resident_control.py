"""Regression tests for staged Python changes and resident mode selection."""

import dataclasses
import sys
import types
import uuid

import pytest

from tools.glm_perf.resident_control import Control, PatchSession


@pytest.fixture
def target(monkeypatch):
    module = types.ModuleType("vllm_ascend._resident_fixture")
    module.projection = lambda value: value + 1
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def control(mode="graph", source="", **kwargs):
    return Control.from_dict(
        {
            "generation": uuid.uuid4().hex,
            "mode": mode,
            "candidate": "candidate" if source else "baseline",
            "source": source,
            **kwargs,
        }
    )


def source(offset):
    return f"""def projection(value):
    return value + {offset}
def replacements():
    return {{"vllm_ascend._resident_fixture:projection": projection}}
"""


def test_prepare_does_not_patch_and_baseline_restores_original(target):
    original = target.projection
    session = PatchSession()
    candidate = control(source=source(2))
    receipt = session.prepare(dataclasses.asdict(candidate))
    assert target.projection is original
    assert receipt["digest"] == candidate.digest
    assert session.apply(candidate.generation)
    assert target.projection(5) == 7
    baseline = control()
    session.prepare(dataclasses.asdict(baseline))
    session.apply(baseline.generation)
    assert target.projection is original


def test_rapid_source_edits_are_applied_and_modes_reuse_graphs(target):
    session = PatchSession()
    for offset in (2, 3):
        candidate = control(source=source(offset))
        session.prepare(dataclasses.asdict(candidate))
        assert session.apply(candidate.generation)
        assert target.projection(0) == offset
        session.graphs_dirty = False
    selected = target.projection
    mode = control("direct-draft", source=source(3))
    session.prepare(dataclasses.asdict(mode))
    assert not session.apply(mode.generation)
    assert target.projection is selected


def test_invalid_candidate_leaves_active_patch_and_graphs_untouched(target):
    session = PatchSession()
    candidate = control(source=source(2))
    session.prepare(dataclasses.asdict(candidate))
    session.apply(candidate.generation)
    session.graphs_dirty = False
    selected = target.projection
    with pytest.raises(AttributeError):
        session.prepare(dataclasses.asdict(control(source=source(3).replace(":projection", ":missing"))))
    assert session.current == candidate
    assert target.projection is selected
    assert not session.graphs_dirty


def test_generation_must_be_prepared_and_recapture_can_be_requested():
    session = PatchSession()
    with pytest.raises(ValueError, match="not been prepared"):
        session.apply(uuid.uuid4().hex)
    mode = control(recapture=True)
    session.prepare(dataclasses.asdict(mode))
    assert session.apply(mode.generation)


@pytest.mark.parametrize(
    "fields",
    [
        {"mode": []},
        {"mode": "typo"},
        {"generation": ""},
        {"candidate": "../candidate"},
        {"source": False},
        {"recapture": "yes"},
        {"unknown": True},
        {"candidate": "baseline", "source": "print('wrong')"},
        {"candidate": "candidate", "source": ""},
    ],
)
def test_malformed_controls_are_rejected(fields):
    value = dataclasses.asdict(control())
    value.update(fields)
    with pytest.raises(ValueError):
        Control.from_dict(value)
