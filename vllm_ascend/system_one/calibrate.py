# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-hoc confidence calibration for the System-One runtime (task T2.4).

Maps a runtime's **raw** per-field confidences to **calibrated** probabilities,
fit on a *held-out* split, so the PRD's calibration requirement (§6.2/§7.2) is a
*measured* Expected Calibration Error (ECE) rather than an assertion. Two methods,
in the PRD's order:

1. :class:`TemperatureScaler` / :class:`PlattScaler` — scalar **temperature**
   ``T`` (or Platt ``a, b``) fit by minimizing negative log-likelihood on held-out
   ``(confidence, correct)`` pairs. ``transform`` divides the logit by ``T``
   (Platt applies ``sigmoid(a * z + b)``). Over-confidence yields ``T > 1``
   (softens the logits toward 0.5); under-confidence yields ``T < 1``.

2. :class:`ConformalCalibrator` — **split-conformal** prediction. At a target
   coverage ``1 - alpha`` it returns a *prediction set* per field (for the
   classification / enum heads) built from held-out nonconformity scores
   ``s = 1 - p_true``. This gives a *distribution-free* marginal-coverage
   guarantee ``P(y_true in C) >= 1 - alpha`` **under exchangeability** of the
   calibration and test points (see :class:`ConformalCalibrator` for the exact
   assumption and its finite-sample form).

A common :class:`Calibrator` protocol (``fit`` + ``transform``) covers the
scalers; conformal exposes ``fit`` + ``predict_set``. :func:`calibration_report`
quantifies before/after ECE using the **same** binned-ECE math as the T0.3
benchmark harness (``tools/system_one/bench.py``) so improvements are comparable
across the two — the implementation here is kept byte-identical and the unit
tests cross-check it against the harness.

Design constraints (mirroring the rest of ``system_one``): **pure Python, stdlib
only**. No ``torch``, no ``torch_npu``, no ``triton`` — the module imports and
unit-tests host-side, CPU-only. Downstream, T3.1 (abstain rule) and T2.5
(single-forward assembly) consume :class:`TemperatureScaler` /
:class:`ConformalCalibrator` and this module's calibrated outputs.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "sigmoid",
    "logit",
    "expected_calibration_error",
    "Calibrator",
    "TemperatureScaler",
    "PlattScaler",
    "ConformalCalibrator",
    "calibration_report",
]

# Clip probabilities to this open interval before taking a logit, so 0/1 inputs
# do not blow up to +-inf. 1e-6 matches the harness's numeric hygiene.
_PROB_EPS = 1e-6
# Default number of equal-width calibration bins over [0, 1] (matches T0.3).
DEFAULT_N_BINS = 10
# Bounds for the 1/T search in temperature fitting. The NLL is convex in
# ``w = 1 / T``; these bracket every realistic temperature (T in [1e-3, 1e3]).
_INV_T_LO = 1e-3
_INV_T_HI = 1e3
# Golden-section iteration budget / convergence width for the 1-D searches.
_GOLDEN_ITERS = 200
_GOLDEN_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Numerically stable sigmoid / logit
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-x))``."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    """Inverse sigmoid ``log(p / (1 - p))``; clips ``p`` to avoid infinities."""
    p = min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)
    return math.log(p / (1.0 - p))


def _clip01(p: float) -> float:
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


# --------------------------------------------------------------------------- #
# ECE (kept byte-identical to the T0.3 harness so reports are comparable)
# --------------------------------------------------------------------------- #
def _bin_index(confidence: float, n_bins: int) -> int:
    idx = int(confidence * n_bins)
    if idx < 0:
        return 0
    if idx >= n_bins:
        return n_bins - 1
    return idx


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_N_BINS
) -> float:
    """Binned Expected Calibration Error over equal-width bins on ``[0, 1]``.

    ``ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|``. Identical math to
    ``tools/system_one/bench.py`` (task T0.3); reproduced here only so this module
    stays self-contained. Empty input returns ``0.0``.
    """
    n = len(confidences)
    if n == 0:
        return 0.0
    counts = [0] * n_bins
    sum_conf = [0.0] * n_bins
    sum_correct = [0] * n_bins
    for conf, ok in zip(confidences, correct):
        b = _bin_index(conf, n_bins)
        counts[b] += 1
        sum_conf[b] += conf
        sum_correct[b] += 1 if ok else 0
    ece = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        acc = sum_correct[b] / counts[b]
        conf = sum_conf[b] / counts[b]
        ece += (counts[b] / n) * abs(acc - conf)
    return ece


# --------------------------------------------------------------------------- #
# Common surface
# --------------------------------------------------------------------------- #
@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc calibrator: ``fit`` on held-out data, then ``transform``.

    The scalar scalers (:class:`TemperatureScaler`, :class:`PlattScaler`) satisfy
    this. :class:`ConformalCalibrator` is set-valued and exposes ``predict_set``
    instead of ``transform``.
    """

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> Calibrator: ...

    def transform(self, confidences: Any) -> Any: ...


def _to_logits(values: Sequence[float], is_logit: bool) -> list[float]:
    return [float(v) for v in values] if is_logit else [logit(float(v)) for v in values]


def _binary_nll(logits: Sequence[float], labels: Sequence[float], scale: float, bias: float) -> float:
    """NLL of ``sigmoid(scale * z + bias)`` vs binary ``labels``.

    Uses the stable form ``log(1 + exp(-|a|)) + max(a, 0) - y * a`` with
    ``a = scale * z + bias`` so large logits never overflow.
    """
    total = 0.0
    for z, y in zip(logits, labels):
        a = scale * z + bias
        total += math.log1p(math.exp(-abs(a))) + max(a, 0.0) - y * a
    return total


def _golden_min(func, lo: float, hi: float) -> float:
    """Minimize a unimodal (here convex) 1-D ``func`` on ``[lo, hi]``.

    Deterministic golden-section search — no dependencies, no randomness — so the
    fitted parameter is reproducible across runs.
    """
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = func(c)
    fd = func(d)
    for _ in range(_GOLDEN_ITERS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = func(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = func(d)
        if hi - lo < _GOLDEN_TOL:
            break
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Temperature scaling (1 parameter: T)
# --------------------------------------------------------------------------- #
class TemperatureScaler:
    """Scalar **temperature scaling**: ``p_cal = sigmoid(logit(p) / T)``.

    ``T`` is fit on held-out ``(confidence, correct)`` pairs by minimizing the
    binary NLL of ``sigmoid(z / T)`` against the correctness labels, where
    ``z = logit(p)`` (or ``z = p`` when ``is_logit=True``). The NLL is convex in
    ``w = 1 / T``, so a deterministic golden-section search finds the global
    optimum. ``T > 1`` softens over-confident probabilities toward ``0.5``;
    ``T < 1`` sharpens under-confident ones. ``T == 1`` is a no-op.
    """

    def __init__(self) -> None:
        self.T: float | None = None

    def fit(
        self,
        confidences: Sequence[float],
        correct: Sequence[bool],
        *,
        is_logit: bool = False,
    ) -> TemperatureScaler:
        """Fit ``T`` on held-out data. Returns ``self`` for chaining."""
        logits = _to_logits(confidences, is_logit)
        labels = [1.0 if c else 0.0 for c in correct]
        if not logits:
            self.T = 1.0
            return self

        def nll(w: float) -> float:
            return _binary_nll(logits, labels, w, 0.0)

        w_star = _golden_min(nll, _INV_T_LO, _INV_T_HI)
        # Guard against a degenerate ~0 optimum (e.g. all logits ~0).
        self.T = 1.0 / w_star if w_star > 0.0 else 1.0
        return self

    def transform(self, confidences: Any, *, is_logit: bool = False) -> Any:
        """Apply the fitted temperature.

        Accepts a ``Sequence[float]`` (returns a ``list``) or a ``Mapping`` of IR
        leaf path -> confidence (returns a ``dict`` with the same keys, the
        T2.5/T3.1 surface). With ``is_logit`` the inputs are treated as logits.
        """
        if self.T is None:
            raise RuntimeError("TemperatureScaler.transform called before fit")
        t = self.T

        def _one(v: float) -> float:
            z = float(v) if is_logit else logit(float(v))
            return _clip01(sigmoid(z / t))

        if isinstance(confidences, Mapping):
            return {k: _one(v) for k, v in confidences.items()}
        return [_one(v) for v in confidences]


# --------------------------------------------------------------------------- #
# Platt scaling (2 parameters: a, b)
# --------------------------------------------------------------------------- #
class PlattScaler:
    """**Platt scaling**: ``p_cal = sigmoid(a * logit(p) + b)``.

    A 2-parameter logistic fit (slope ``a``, bias ``b``) on held-out
    ``(confidence, correct)`` pairs — the strict generalization of temperature
    scaling, which is the special case ``b == 0`` with ``a == 1 / T``. Fit by
    convex coordinate descent (alternating golden-section searches on ``a`` and
    ``b``), so it is deterministic and dependency-free.
    """

    _A_LO, _A_HI = 1e-3, 1e3
    _B_LO, _B_HI = -20.0, 20.0
    _COORD_ROUNDS = 40

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(
        self,
        confidences: Sequence[float],
        correct: Sequence[bool],
        *,
        is_logit: bool = False,
    ) -> PlattScaler:
        logits = _to_logits(confidences, is_logit)
        labels = [1.0 if c else 0.0 for c in correct]
        if not logits:
            self.a, self.b = 1.0, 0.0
            return self

        a, b = 1.0, 0.0
        for _ in range(self._COORD_ROUNDS):
            a = _golden_min(lambda av, b=b: _binary_nll(logits, labels, av, b), self._A_LO, self._A_HI)
            b = _golden_min(lambda bv, a=a: _binary_nll(logits, labels, a, bv), self._B_LO, self._B_HI)
        self.a, self.b = a, b
        return self

    def transform(self, confidences: Any, *, is_logit: bool = False) -> Any:
        if self.a is None or self.b is None:
            raise RuntimeError("PlattScaler.transform called before fit")
        a, b = self.a, self.b

        def _one(v: float) -> float:
            z = float(v) if is_logit else logit(float(v))
            return _clip01(sigmoid(a * z + b))

        if isinstance(confidences, Mapping):
            return {k: _one(v) for k, v in confidences.items()}
        return [_one(v) for v in confidences]


# --------------------------------------------------------------------------- #
# Split-conformal prediction (coverage guarantee, set-valued outputs)
# --------------------------------------------------------------------------- #
class ConformalCalibrator:
    """Split-conformal prediction for the classification / enum heads.

    Given held-out **nonconformity scores** ``s = 1 - p_true`` (higher = worse),
    ``fit`` stores their empirical distribution. At test time, for a target
    coverage ``1 - alpha``, :meth:`predict_set` returns every label whose score is
    at most the conformal quantile ``qhat``::

        C(x) = { y : 1 - p(y) <= qhat }

    where ``qhat`` is the ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile
    of the calibration scores (the ``ceil((n+1)(1-alpha))``-th smallest, 1-indexed;
    ``+inf`` — the trivial full set — when that rank exceeds ``n``).

    **Guarantee & assumption.** If the calibration scores and a test score are
    *exchangeable* (in particular, i.i.d. from the same distribution — no
    distribution shift between the held-out split and serving), then the
    prediction set has marginal coverage

        ``1 - alpha <= P(y_true in C(X)) <= 1 - alpha + 1 / (n + 1)``.

    The guarantee is *marginal* (averaged over test points), *distribution-free*
    (no assumption on the model's quality), and holds for any nonconformity score.
    A single calibration draw may realize coverage slightly below ``1 - alpha``;
    coverage concentrates around the target as ``n`` grows. Smaller ``alpha`` ->
    larger ``qhat`` -> larger sets (monotone).
    """

    def __init__(self) -> None:
        self._scores: list[float] | None = None
        self.n: int = 0

    @staticmethod
    def nonconformity(field_probs: Mapping[Hashable, float], true_label: Hashable) -> float:
        """Standard nonconformity score ``1 - p(true_label)`` for one datum.

        A label absent from ``field_probs`` is treated as probability ``0`` (score
        ``1``), the maximally nonconforming case.
        """
        return 1.0 - float(field_probs.get(true_label, 0.0))

    def fit(self, scores: Sequence[float]) -> ConformalCalibrator:
        """Store held-out nonconformity scores. Returns ``self`` for chaining."""
        self._scores = sorted(float(s) for s in scores)
        self.n = len(self._scores)
        return self

    def quantile(self, alpha: float) -> float:
        """The conformal quantile ``qhat`` at miscoverage ``alpha``.

        Returns ``+inf`` when the finite-sample rank exceeds ``n`` (i.e. the
        calibration set is too small for the requested coverage), which makes
        :meth:`predict_set` return the trivial full set.
        """
        if self._scores is None:
            raise RuntimeError("ConformalCalibrator.quantile called before fit")
        n = self.n
        if n == 0:
            return float("inf")
        rank = math.ceil((n + 1) * (1.0 - alpha))
        if rank > n:
            return float("inf")
        if rank < 1:
            rank = 1
        return self._scores[rank - 1]

    def predict_set(self, field_probs: Mapping[Hashable, float], alpha: float) -> set:
        """Prediction set at target coverage ``1 - alpha`` for one field.

        Returns ``{ y : 1 - p(y) <= qhat }``. With ``qhat == +inf`` every offered
        label is included (trivial full-coverage set).
        """
        if self._scores is None:
            raise RuntimeError("ConformalCalibrator.predict_set called before fit")
        qhat = self.quantile(alpha)
        return {label for label, prob in field_probs.items() if (1.0 - float(prob)) <= qhat}

    def empirical_coverage(
        self,
        field_probs_list: Sequence[Mapping[Hashable, float]],
        true_labels: Sequence[Hashable],
        alpha: float,
    ) -> float:
        """Fraction of ``(probs, true_label)`` whose true label lands in its set.

        The empirical check of the coverage guarantee on a held-out / test stream.
        Returns ``0.0`` on empty input.
        """
        n = len(field_probs_list)
        if n == 0:
            return 0.0
        covered = sum(
            1 for probs, y in zip(field_probs_list, true_labels) if y in self.predict_set(probs, alpha)
        )
        return covered / n


# --------------------------------------------------------------------------- #
# Before / after report (quantifies the improvement via the T0.3 ECE)
# --------------------------------------------------------------------------- #
def calibration_report(
    before: Sequence[float],
    after: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Before/after ECE + reliability bins using the T0.3 harness math.

    ``before`` and ``after`` are raw and calibrated confidences over the same
    held-out points as ``correct``. The returned dict quantifies the improvement::

        {
          "n": int, "n_bins": int,
          "before": {"ece": float},
          "after":  {"ece": float},
          "delta_ece": before_ece - after_ece,   # > 0 means calibration helped
        }
    """
    before_ece = expected_calibration_error(before, correct, n_bins)
    after_ece = expected_calibration_error(after, correct, n_bins)
    return {
        "n": len(before),
        "n_bins": n_bins,
        "before": {"ece": before_ece},
        "after": {"ece": after_ece},
        "delta_ece": before_ece - after_ece,
    }
