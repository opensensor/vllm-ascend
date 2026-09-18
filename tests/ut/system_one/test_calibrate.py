# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One post-hoc calibration module (task T2.4).

The module (``vllm_ascend/system_one/calibrate.py``) maps raw model confidences
to calibrated probabilities, fit on a *held-out* split, and provides:

* :class:`TemperatureScaler` / :class:`PlattScaler` — scalar temperature (or
  Platt ``a, b``) fit by minimizing NLL on ``(confidence, correct)`` pairs;
  ``transform`` divides logits by ``T`` (or applies the Platt sigmoid).
* :class:`ConformalCalibrator` — split-conformal prediction that, at target
  coverage ``1 - alpha``, returns a **prediction set** per field using held-out
  nonconformity scores (``1 - p_true``), with an empirical-coverage check.
* :func:`calibration_report` — before/after ECE using the **same** binned-ECE
  math as the T0.3 harness (``tools/system_one/bench.py``), so an improvement is
  *quantified* rather than asserted.

These are the RED-first contract. Miscalibration is injected with a *known*
temperature distortion so the recovered ``T`` and the ECE reduction have known
signs; conformal coverage is checked against its finite-sample guarantee.

The module is loaded **by file path** (not ``import vllm_ascend...``) so the test
stays pure-Python and host-side: no torch, no torch_npu, no triton. It also loads
the T0.3 harness by path to cross-check that the report's ECE is byte-identical to
the harness ECE.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_calibrate.py``
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CAL_PATH = _REPO_ROOT / "vllm_ascend" / "system_one" / "calibrate.py"
_BENCH_PATH = _REPO_ROOT / "tools" / "system_one" / "bench.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-ref resolution works on 3.12+.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cal = _load_module("system_one_calibrate", _CAL_PATH)
bench = _load_module("system_one_bench_for_calibrate", _BENCH_PATH)

TemperatureScaler = cal.TemperatureScaler
PlattScaler = cal.PlattScaler
ConformalCalibrator = cal.ConformalCalibrator
calibration_report = cal.calibration_report
sigmoid = cal.sigmoid
logit = cal.logit
harness_ece = bench.expected_calibration_error


# --------------------------------------------------------------------------- #
# Synthetic fixtures with a KNOWN miscalibration.
#
# Draw a true logit ``z``; correctness ~ Bernoulli(sigmoid(z)); the *reported*
# confidence is ``sigmoid(k * z)``. ``k > 1`` is over-confident (probs pushed to
# the extremes vs the true rate), ``k < 1`` under-confident, ``k == 1`` already
# calibrated. Temperature scaling with ``T == k`` recovers perfect calibration,
# so the sign of the fitted ``T`` and the ECE drop are both known in advance.
# --------------------------------------------------------------------------- #
def _make_confidence_set(distortion_k: float, n: int = 6000, seed: int = 1):
    rng = random.Random(seed)
    conf: list[float] = []
    correct: list[bool] = []
    for _ in range(n):
        z = rng.uniform(-4.0, 4.0)
        correct.append(rng.random() < sigmoid(z))
        conf.append(sigmoid(distortion_k * z))
    return conf, correct


def _make_field_probs_stream(n: int, seed: int, n_classes: int = 5):
    """A stream of ``(class_probs, true_label)`` from a noisy softmax model."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        true = rng.randrange(n_classes)
        logits = [rng.gauss(0.0, 1.5) for _ in range(n_classes)]
        logits[true] += rng.gauss(1.5, 1.0)  # true label is higher, but noisy
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        probs = {i: exps[i] / s for i in range(n_classes)}
        out.append((probs, true))
    return out


# --------------------------------------------------------------------------- #
# sigmoid / logit round-trip
# --------------------------------------------------------------------------- #
def test_sigmoid_logit_roundtrip():
    for p in (0.01, 0.2, 0.5, 0.73, 0.99):
        assert math.isclose(sigmoid(logit(p)), p, rel_tol=1e-9, abs_tol=1e-9)
    # sigmoid is numerically stable at extremes (no overflow; underflows to 0).
    assert 0.0 <= sigmoid(-1000.0) < 1e-6
    assert 1.0 - 1e-6 < sigmoid(1000.0) <= 1.0


# --------------------------------------------------------------------------- #
# Temperature scaling: reduces ECE on over-confident data; T has the right sign.
# --------------------------------------------------------------------------- #
def test_temperature_scaling_reduces_ece_overconfident():
    conf, correct = _make_confidence_set(distortion_k=2.5, seed=1)
    scaler = TemperatureScaler().fit(conf, correct)

    before = harness_ece(conf, correct)
    after = harness_ece(scaler.transform(conf), correct)

    # Over-confidence -> temperature must be > 1 (softens the logits).
    assert scaler.T > 1.0
    assert scaler.T > 1.5
    # ECE must drop materially (measured, not asserted): here ~0.10 -> ~0.015.
    assert after < before
    assert after < 0.5 * before
    assert after < 0.04


def test_temperature_scaling_underconfident_gives_T_below_one():
    conf, correct = _make_confidence_set(distortion_k=0.4, seed=2)
    scaler = TemperatureScaler().fit(conf, correct)
    after = harness_ece(scaler.transform(conf), correct)
    before = harness_ece(conf, correct)

    assert scaler.T < 1.0
    assert scaler.T < 0.8
    assert after < before


def test_temperature_is_noop_on_calibrated_data():
    conf, correct = _make_confidence_set(distortion_k=1.0, seed=3)
    scaler = TemperatureScaler().fit(conf, correct)

    # Already calibrated -> T ~ 1 within tolerance, ECE essentially unchanged.
    assert 0.8 < scaler.T < 1.25
    before = harness_ece(conf, correct)
    after = harness_ece(scaler.transform(conf), correct)
    assert after < 0.03
    assert after <= before + 0.01


def test_temperature_transform_preserves_mapping_shape():
    conf, correct = _make_confidence_set(distortion_k=2.0, seed=4)
    scaler = TemperatureScaler().fit(conf, correct)

    # Sequence in -> list out (same length).
    out_seq = scaler.transform(conf[:10])
    assert isinstance(out_seq, list) and len(out_seq) == 10
    assert all(0.0 <= p <= 1.0 for p in out_seq)

    # Mapping in (IR leaf paths) -> mapping out (same keys), the T2.5/T3.1 surface.
    field_conf = {"$.tool": 0.95, "$.args.n": 0.6}
    out_map = scaler.transform(field_conf)
    assert isinstance(out_map, dict)
    assert set(out_map) == set(field_conf)
    # Over-confident T>1 pulls high confidence toward 0.5.
    assert out_map["$.tool"] < field_conf["$.tool"]


def test_temperature_fit_is_deterministic():
    conf, correct = _make_confidence_set(distortion_k=2.5, seed=5)
    t1 = TemperatureScaler().fit(conf, correct).T
    t2 = TemperatureScaler().fit(conf, correct).T
    assert t1 == t2


def test_temperature_accepts_logits_directly():
    conf, correct = _make_confidence_set(distortion_k=2.5, seed=6)
    logits = [logit(c) for c in conf]
    from_prob = TemperatureScaler().fit(conf, correct).T
    from_logit = TemperatureScaler().fit(logits, correct, is_logit=True).T
    assert math.isclose(from_prob, from_logit, rel_tol=1e-6, abs_tol=1e-6)


def test_temperature_edge_cases_do_not_crash():
    # All correct, all wrong, single sample -> fit returns a finite positive T.
    for conf, correct in (
        ([0.9, 0.8, 0.95], [True, True, True]),
        ([0.9, 0.8, 0.95], [False, False, False]),
        ([0.7], [True]),
    ):
        scaler = TemperatureScaler().fit(conf, correct)
        assert scaler.T > 0.0 and math.isfinite(scaler.T)
        out = scaler.transform(conf)
        assert all(0.0 <= p <= 1.0 for p in out)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform([0.5])


# --------------------------------------------------------------------------- #
# Platt scaling (2-parameter logistic) reduces ECE too.
# --------------------------------------------------------------------------- #
def test_platt_scaling_reduces_ece():
    conf, correct = _make_confidence_set(distortion_k=2.5, seed=7)
    scaler = PlattScaler().fit(conf, correct)
    before = harness_ece(conf, correct)
    after = harness_ece(scaler.transform(conf), correct)
    assert after < before
    assert after < 0.05
    # Temperature is the Platt special case b == 0; over-confidence -> a < 1.
    assert scaler.a < 1.0


# --------------------------------------------------------------------------- #
# Conformal prediction: coverage guarantee + monotone set size.
# --------------------------------------------------------------------------- #
def test_conformal_coverage_meets_target():
    calib = _make_field_probs_stream(2000, seed=10)
    test = _make_field_probs_stream(4000, seed=11)

    scores = [ConformalCalibrator.nonconformity(p, y) for p, y in calib]
    conf = ConformalCalibrator().fit(scores)

    alpha = 0.1
    covered = sum(1 for p, y in test if y in conf.predict_set(p, alpha))
    coverage = covered / len(test)

    # Split-conformal guarantees E[coverage] >= 1 - alpha under exchangeability;
    # a single split may dip slightly below, so allow a stated finite-sample tol.
    assert coverage >= (1 - alpha) - 0.04

    # The module's own empirical-coverage helper agrees.
    helper = conf.empirical_coverage([p for p, _ in test], [y for _, y in test], alpha)
    assert math.isclose(helper, coverage, rel_tol=0, abs_tol=1e-12)


def test_conformal_sets_grow_as_alpha_shrinks():
    calib = _make_field_probs_stream(2000, seed=12)
    test = _make_field_probs_stream(3000, seed=13)
    conf = ConformalCalibrator().fit([ConformalCalibrator.nonconformity(p, y) for p, y in calib])

    def avg_set(alpha):
        return sum(len(conf.predict_set(p, alpha)) for p, _ in test) / len(test)

    # Smaller alpha (higher target coverage) -> larger prediction sets (monotone).
    s_loose = avg_set(0.2)
    s_mid = avg_set(0.1)
    s_tight = avg_set(0.05)
    assert s_loose <= s_mid <= s_tight
    assert s_tight > s_loose  # strictly grows across this range


def test_conformal_quantile_monotone_in_alpha():
    conf = ConformalCalibrator().fit([i / 100.0 for i in range(100)])
    # qhat is non-increasing in alpha.
    assert conf.quantile(0.05) >= conf.quantile(0.1) >= conf.quantile(0.2)


def test_conformal_edge_cases_do_not_crash():
    # Single class: predict_set always returns that class (full coverage).
    conf = ConformalCalibrator().fit([0.0, 0.0, 0.0])
    s = conf.predict_set({0: 1.0}, alpha=0.1)
    assert s == {0}

    # All-wrong calibration (scores near 1) -> large qhat -> sets include all.
    conf_wrong = ConformalCalibrator().fit([1.0] * 50)
    full = conf_wrong.predict_set({0: 0.1, 1: 0.2, 2: 0.7}, alpha=0.1)
    assert full == {0, 1, 2}

    # Tiny calibration set with tight alpha -> qhat is +inf -> trivial full set.
    conf_tiny = ConformalCalibrator().fit([0.3])
    assert conf_tiny.quantile(0.01) == float("inf")
    assert conf_tiny.predict_set({0: 0.0, 1: 0.0}, alpha=0.01) == {0, 1}


def test_conformal_fit_before_predict_raises():
    with pytest.raises(RuntimeError):
        ConformalCalibrator().predict_set({0: 1.0}, alpha=0.1)


# --------------------------------------------------------------------------- #
# calibration_report: before/after ECE using the T0.3 harness math.
# --------------------------------------------------------------------------- #
def test_calibration_report_matches_harness_math():
    conf, correct = _make_confidence_set(distortion_k=2.5, seed=20)
    scaler = TemperatureScaler().fit(conf, correct)
    after_conf = scaler.transform(conf)

    report = calibration_report(conf, after_conf, correct, n_bins=10)

    # Byte-identical to the T0.3 harness ECE on the same inputs.
    assert report["before"]["ece"] == harness_ece(conf, correct, 10)
    assert report["after"]["ece"] == harness_ece(after_conf, correct, 10)
    assert report["delta_ece"] == report["before"]["ece"] - report["after"]["ece"]
    assert report["delta_ece"] > 0.0  # calibration improved
    assert report["n"] == len(conf)
    assert report["n_bins"] == 10


def test_module_ece_matches_harness():
    conf, correct = _make_confidence_set(distortion_k=1.7, seed=21)
    assert cal.expected_calibration_error(conf, correct, 10) == harness_ece(conf, correct, 10)


# --------------------------------------------------------------------------- #
# Import-hygiene ast-gate: no torch_npu / triton on the import path.
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    source = _CAL_PATH.read_text(encoding="utf-8")
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
    assert not offenders, f"calibrate.py must not import {sorted(offenders)}"
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, f"forbidden import: {stripped!r}"
