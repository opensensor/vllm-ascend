# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One selective-prediction module (task T3.1).

The module (``vllm_ascend/system_one/select.py``) decides when a System-One
answer is trustworthy enough to keep vs. **abstain** (later escalated to the W2
MoE by T3.2). It provides:

* :class:`AbstainRule` — a confidence-threshold accept/abstain decision over a
  prediction's per-field calibrated confidences, aggregated across fields
  (``min`` = the weakest field gates, or ``mean``). ``decide`` returns an
  :class:`AbstainDecision` carrying ``accept``, a ``reason``, and the reported
  ``min_confidence``.
* :class:`ConformalAbstainRule` — the conformal variant: abstain when any field's
  conformal prediction set (from T2.4's ``ConformalCalibrator.predict_set``) has
  size > 1 (or is empty). Duck-typed on ``predict_set`` so this module never
  imports ``calibrate`` (keeps the import path clean).
* :func:`risk_coverage_curve` — the risk–coverage curve: for each threshold, the
  coverage (fraction accepted) and the selective risk (error rate among the
  accepted). Monotone: higher threshold → lower coverage.
* :func:`tune_threshold` — pick the operating-point threshold hitting a target
  **risk** (max coverage with risk ≤ target) or a target **coverage**.

RED-first contract. The module is loaded **by file path** (not
``import vllm_ascend...``) so the test stays pure-Python, host-side: no torch, no
torch_npu, no triton. The conformal test loads T2.4's ``calibrate`` by path too.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_select.py``
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SELECT_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "select.py"
_CAL_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "calibrate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sel = _load_module("system_one_select", _SELECT_PATH)

AbstainRule = sel.AbstainRule
ConformalAbstainRule = sel.ConformalAbstainRule
AbstainDecision = sel.AbstainDecision
risk_coverage_curve = sel.risk_coverage_curve
tune_threshold = sel.tune_threshold


# --------------------------------------------------------------------------- #
# Synthetic, perfectly separated validation set: correctness is a step at 0.5.
# Confidence i/100 for i in 0..99; the record is correct iff confidence >= 0.5.
# So dropping the lowest-confidence records (raising the threshold) strictly
# removes errors first — risk is non-increasing, coverage is decreasing, and the
# risk-0 operating point sits exactly at threshold 0.5 / coverage 0.5.
# --------------------------------------------------------------------------- #
def _separated_set(n: int = 100):
    scores = [i / n for i in range(n)]
    correct = [s >= 0.5 for s in scores]
    return scores, correct


def _apply(scores, correct, threshold):
    """Coverage + selective risk when accepting confidence >= threshold."""
    accepted = [ok for s, ok in zip(scores, correct) if s >= threshold]
    coverage = len(accepted) / len(scores)
    risk = (sum(1 for ok in accepted if not ok) / len(accepted)) if accepted else 0.0
    return coverage, risk


# --------------------------------------------------------------------------- #
# AbstainRule — confidence threshold
# --------------------------------------------------------------------------- #
def test_below_threshold_abstains():
    rule = AbstainRule(threshold=0.5)
    d = rule.decide({"$.tool": 0.4})
    assert d.accept is False
    assert d.reason == "below_threshold"
    assert d.min_confidence == pytest.approx(0.4)


def test_at_threshold_accepts():
    rule = AbstainRule(threshold=0.5)
    d = rule.decide({"$.tool": 0.5})
    assert d.accept is True
    assert d.reason == "accept"
    assert d.min_confidence == pytest.approx(0.5)


def test_above_threshold_accepts_and_reports_min():
    rule = AbstainRule(threshold=0.5)
    d = rule.decide({"$.tool": 0.95, "$.arg": 0.8})
    assert d.accept is True
    assert d.reason == "accept"
    assert d.min_confidence == pytest.approx(0.8)  # weakest field reported


def test_min_aggregation_weakest_field_gates():
    # One weak field forces abstain even though the mean is high.
    rule = AbstainRule(threshold=0.5, aggregation="min")
    d = rule.decide({"$.tool": 0.99, "$.arg": 0.2})
    assert d.accept is False
    assert d.reason == "below_threshold"
    assert d.min_confidence == pytest.approx(0.2)
    assert d.field == "$.arg"


def test_mean_aggregation_accepts_when_average_clears():
    rule = AbstainRule(threshold=0.5, aggregation="mean")
    d = rule.decide({"$.tool": 0.99, "$.arg": 0.2})  # mean 0.595 >= 0.5
    assert d.accept is True
    assert d.aggregate_confidence == pytest.approx(0.595)
    # min_confidence is still reported for transparency
    assert d.min_confidence == pytest.approx(0.2)


def test_accepts_prediction_object_with_confidences():
    # Duck-typed on a T0.3-style Prediction: object exposing `.confidences`.
    pred = SimpleNamespace(value={"tool": "search"}, confidences={"$.tool": 0.9})
    d = AbstainRule(threshold=0.5).decide(pred)
    assert d.accept is True


def test_scalar_and_sequence_confidence_inputs():
    assert AbstainRule(threshold=0.5).decide(0.9).accept is True
    assert AbstainRule(threshold=0.5).decide([0.9, 0.2]).accept is False


def test_empty_confidences_abstains():
    d = AbstainRule(threshold=0.5).decide({})
    assert d.accept is False
    assert d.reason == "no_confidence"


def test_invalid_threshold_and_aggregation_rejected():
    with pytest.raises(ValueError):
        AbstainRule(threshold=1.5)
    with pytest.raises(ValueError):
        AbstainRule(threshold=0.5, aggregation="max")


# --------------------------------------------------------------------------- #
# ConformalAbstainRule — set-size > 1 abstains (uses T2.4 ConformalCalibrator)
# --------------------------------------------------------------------------- #
def _fitted_conformal():
    cal = _load_module("system_one_calibrate_for_select", _CAL_PATH)
    # All calibration nonconformity scores == 0.1. At alpha=0.1, qhat == 0.1, so
    # predict_set keeps every label with (1 - p) <= 0.1, i.e. p >= 0.9.
    conformal = cal.ConformalCalibrator().fit([0.1] * 100)
    return conformal


def test_conformal_singleton_accepts():
    rule = ConformalAbstainRule(_fitted_conformal(), alpha=0.1)
    d = rule.decide({"$.tool": {"search": 0.95, "noop": 0.05}})
    assert d.accept is True
    assert d.reason == "accept"
    assert d.max_set_size == 1


def test_conformal_set_gt_1_abstains():
    rule = ConformalAbstainRule(_fitted_conformal(), alpha=0.1)
    # Two labels both clear p >= 0.9 -> set size 2 -> abstain.
    d = rule.decide({"$.tool": {"search": 0.95, "calculate": 0.92}})
    assert d.accept is False
    assert d.reason == "conformal_set_gt_1"
    assert d.max_set_size == 2
    assert d.field == "$.tool"


def test_conformal_any_field_set_gt_1_abstains():
    rule = ConformalAbstainRule(_fitted_conformal(), alpha=0.1)
    d = rule.decide(
        {
            "$.tool": {"search": 0.95, "noop": 0.05},   # singleton
            "$.arg": {"a": 0.95, "b": 0.93},            # size 2
        }
    )
    assert d.accept is False
    assert d.max_set_size == 2


# --------------------------------------------------------------------------- #
# risk_coverage_curve — endpoints + monotonicity
# --------------------------------------------------------------------------- #
def test_curve_empty_input():
    assert risk_coverage_curve([], []) == []


def test_curve_endpoints():
    scores, correct = _separated_set()
    curve = risk_coverage_curve(scores, correct)
    # Full-coverage endpoint at threshold 0.0: everything accepted, full risk.
    full = curve[0]
    assert full.threshold == pytest.approx(0.0)
    assert full.coverage == pytest.approx(1.0)
    full_risk = sum(1 for ok in correct if not ok) / len(correct)
    assert full.risk == pytest.approx(full_risk)
    # Zero-coverage endpoint: very high threshold accepts nothing.
    zero = curve[-1]
    assert zero.coverage == pytest.approx(0.0)
    assert math.isinf(zero.threshold)


def test_curve_is_monotone():
    scores, correct = _separated_set()
    curve = risk_coverage_curve(scores, correct)
    coverages = [p.coverage for p in curve]
    risks = [p.risk for p in curve]
    # Coverage decreases (non-increasing) as threshold rises.
    assert all(b <= a + 1e-12 for a, b in zip(coverages, coverages[1:]))
    # Selective risk is non-increasing on well-separated data.
    assert all(b <= a + 1e-12 for a, b in zip(risks, risks[1:]))
    # Thresholds are ascending.
    assert all(b >= a for a, b in zip([p.threshold for p in curve], [p.threshold for p in curve][1:]))


# --------------------------------------------------------------------------- #
# tune_threshold — hit a target risk / a target coverage
# --------------------------------------------------------------------------- #
def test_tune_hits_target_risk():
    scores, correct = _separated_set()
    thr = tune_threshold(scores, correct, target_risk=0.0)
    coverage, risk = _apply(scores, correct, thr)
    assert risk <= 0.0 + 1e-9              # accepted-set error within tolerance
    assert coverage == pytest.approx(0.5)  # and it keeps as much as it can
    assert thr == pytest.approx(0.5)


def test_tune_hits_target_risk_nonzero():
    scores, correct = _separated_set()
    target = 0.1
    thr = tune_threshold(scores, correct, target_risk=target)
    coverage, risk = _apply(scores, correct, thr)
    assert risk <= target + 1e-9
    # A non-trivial amount is still accepted.
    assert coverage > 0.4


def test_tune_hits_target_coverage():
    scores, correct = _separated_set()
    thr = tune_threshold(scores, correct, target_coverage=0.7)
    coverage, _ = _apply(scores, correct, thr)
    assert coverage == pytest.approx(0.7, abs=0.02)


def test_tune_requires_exactly_one_target():
    scores, correct = _separated_set()
    with pytest.raises(ValueError):
        tune_threshold(scores, correct)
    with pytest.raises(ValueError):
        tune_threshold(scores, correct, target_risk=0.1, target_coverage=0.5)


# --------------------------------------------------------------------------- #
# Edge cases: all-correct, all-wrong, single record — no crash.
# --------------------------------------------------------------------------- #
def test_all_correct_any_coverage_zero_risk():
    scores = [i / 10 for i in range(10)]
    correct = [True] * 10
    curve = risk_coverage_curve(scores, correct)
    assert all(p.risk == pytest.approx(0.0) for p in curve)
    # Target risk 0 keeps everything.
    thr = tune_threshold(scores, correct, target_risk=0.0)
    coverage, risk = _apply(scores, correct, thr)
    assert coverage == pytest.approx(1.0)
    assert risk == pytest.approx(0.0)


def test_all_wrong_forces_full_abstain_for_zero_risk():
    scores = [i / 10 for i in range(10)]
    correct = [False] * 10
    thr = tune_threshold(scores, correct, target_risk=0.0)
    coverage, _ = _apply(scores, correct, thr)
    assert coverage == pytest.approx(0.0)   # only abstaining hits risk 0
    assert math.isinf(thr)


def test_single_record_no_crash():
    curve = risk_coverage_curve([0.7], [True])
    assert curve[0].coverage == pytest.approx(1.0)
    assert tune_threshold([0.7], [True], target_coverage=1.0) == pytest.approx(0.0)
    # AbstainRule on a single-field prediction.
    assert AbstainRule(threshold=0.5).decide({"$.tool": 0.7}).accept is True


# --------------------------------------------------------------------------- #
# Import-hygiene ast-gate: no torch / torch_npu / triton / vllm on import path.
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    source = _SELECT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"select.py must not import {sorted(offenders)}"
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, f"forbidden import: {stripped!r}"
