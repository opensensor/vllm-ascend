# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective prediction for the System-One runtime (task T3.1).

Decides when a System-One answer is trustworthy enough to **keep** vs. **abstain**
— an abstain is escalated to the W2 MoE ("System-Two") by T3.2. This implements
the PRD's selective-prediction requirement (§6.3 / §7.4): route a request to the
slow tier when the on-card answer is not confident enough, and *tune the operating
point* on a validation risk–coverage curve rather than guessing a threshold.

Two abstain rules and a tuner:

1. :class:`AbstainRule` — a **confidence-threshold** rule over a prediction's
   per-field *calibrated* confidences (the T2.4 output). The per-record decision
   aggregates the fields explicitly: ``"min"`` (default) makes the **weakest**
   field gate the record — a record is only as trustworthy as its least-confident
   leaf — while ``"mean"`` averages. ``decide`` returns an :class:`AbstainDecision`
   with ``accept``, a ``reason``, and the reported ``min_confidence`` (plus the
   aggregate and the gating field).

2. :class:`ConformalAbstainRule` — the **conformal** variant. Using T2.4's
   :class:`ConformalCalibrator.predict_set`, it abstains when **any** field's
   prediction set is larger than one label (ambiguous) — or empty (no conforming
   label). This is the set-valued "unsure" signal with a distribution-free
   coverage guarantee. The calibrator is **duck-typed** on ``predict_set`` so this
   module never imports ``calibrate`` and stays off the heavy import path.

3. :func:`risk_coverage_curve` + :func:`tune_threshold` — the **risk–coverage
   tuner**. The curve gives, for each candidate threshold, the *coverage*
   (fraction accepted) and the *selective risk* (error rate among the accepted).
   It is monotone by construction: a higher threshold accepts fewer records
   (coverage decreases), and on well-separated confidences the errors are dropped
   first (selective risk is non-increasing). :func:`tune_threshold` picks the
   operating point that hits a **target risk** (max coverage with risk ≤ target)
   or a **target coverage**.

Design constraints (mirroring the rest of ``system_one``): **pure Python, stdlib
only**. No ``torch``, no ``torch_npu``, no ``triton`` — the module imports and
unit-tests host-side, CPU-only, and it does not import the sibling ``calibrate`` /
``bench`` modules (it operates on plain confidence sequences and per-field
probability maps), so it stays decoupled and import-clean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = [
    "AbstainDecision",
    "AbstainRule",
    "ConformalAbstainRule",
    "RiskCoveragePoint",
    "risk_coverage_curve",
    "tune_threshold",
]

# Aggregation strategies for collapsing per-field confidences into a per-record
# score. "min" is the conservative default: the weakest leaf gates the record.
_VALID_AGGREGATIONS = ("min", "mean")

# The conformal rule accepts only when every field's prediction set is a singleton
# (exactly one conforming label). A larger set is ambiguous; an empty set means no
# label conforms — both abstain.
_ACCEPT_SET_SIZE = 1

# Numeric slack when comparing a measured selective risk against a target.
_RISK_TOL = 1e-9

# Reason codes carried on an AbstainDecision (stable strings the T3.2 router logs).
_REASON_ACCEPT = "accept"
_REASON_BELOW_THRESHOLD = "below_threshold"
_REASON_NO_CONFIDENCE = "no_confidence"
_REASON_SET_GT_1 = "conformal_set_gt_1"
_REASON_EMPTY_SET = "conformal_empty_set"


# --------------------------------------------------------------------------- #
# Decision record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AbstainDecision:
    """The accept/abstain outcome for one record.

    ``accept`` is the keep-vs-abstain verdict (``False`` -> escalate to W2).
    ``reason`` is a stable code (``"accept"``, ``"below_threshold"``,
    ``"no_confidence"``, ``"conformal_set_gt_1"``, ``"conformal_empty_set"``).
    ``min_confidence`` is the weakest field confidence for the threshold rule
    (``None`` when unavailable / for the conformal rule); ``aggregate_confidence``
    is the value actually compared against the threshold; ``max_set_size`` is the
    largest per-field conformal set (conformal rule); ``field`` names the field
    that drove the decision (weakest confidence / largest set).
    """

    accept: bool
    reason: str
    min_confidence: float | None = None
    aggregate_confidence: float | None = None
    max_set_size: int | None = None
    field: Any = None


# --------------------------------------------------------------------------- #
# Confidence extraction (accept a Mapping, a Sequence, a scalar, or a Prediction)
# --------------------------------------------------------------------------- #
def _confidence_map(prediction: Any) -> dict[Any, float]:
    """Normalize a prediction's confidences into ``{field_key: confidence}``.

    Accepts, in order: an object exposing ``.confidences`` (a T0.3-style
    ``Prediction``); a ``Mapping`` of IR leaf path -> confidence (the T2.4 /
    T2.5 per-field surface); a bare scalar confidence (mapped under key ``"$"``);
    or any other sequence of confidences (keyed by position).
    """
    confidences = getattr(prediction, "confidences", prediction)
    if isinstance(confidences, Mapping):
        return {k: float(v) for k, v in confidences.items()}
    if isinstance(confidences, bool):  # bool is an int subclass; treat as scalar
        return {"$": float(confidences)}
    if isinstance(confidences, (int, float)):
        return {"$": float(confidences)}
    try:
        return {i: float(v) for i, v in enumerate(confidences)}
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"cannot extract confidences from {prediction!r}") from exc


# --------------------------------------------------------------------------- #
# Confidence-threshold abstain rule
# --------------------------------------------------------------------------- #
class AbstainRule:
    """Accept/abstain by a **confidence threshold** on aggregated field confidences.

    ``threshold`` is the minimum aggregate calibrated confidence in ``[0, 1]`` to
    **accept**; below it the record abstains (escalate to W2). ``aggregation``
    collapses the per-field confidences into the scalar compared to the threshold:

    * ``"min"`` (default) — the **weakest** field gates the record. Conservative:
      a single low-confidence leaf abstains the whole record.
    * ``"mean"`` — the average field confidence.

    ``min_confidence`` (the weakest field) is always reported for transparency,
    independent of the aggregation used.
    """

    def __init__(self, threshold: float, *, aggregation: str = "min") -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        self.threshold = float(threshold)
        self.aggregation = aggregation

    def decide(self, prediction: Any) -> AbstainDecision:
        """Decide accept/abstain for one prediction."""
        confidences = _confidence_map(prediction)
        if not confidences:
            return AbstainDecision(
                accept=False,
                reason=_REASON_NO_CONFIDENCE,
                min_confidence=None,
                aggregate_confidence=None,
            )

        weakest_field = min(confidences, key=lambda k: confidences[k])
        min_confidence = confidences[weakest_field]
        if self.aggregation == "min":
            aggregate = min_confidence
        else:  # "mean"
            aggregate = sum(confidences.values()) / len(confidences)

        accept = aggregate >= self.threshold
        return AbstainDecision(
            accept=accept,
            reason=_REASON_ACCEPT if accept else _REASON_BELOW_THRESHOLD,
            min_confidence=min_confidence,
            aggregate_confidence=aggregate,
            field=weakest_field,
        )


# --------------------------------------------------------------------------- #
# Conformal (set-size) abstain rule
# --------------------------------------------------------------------------- #
class ConformalAbstainRule:
    """Accept/abstain by **conformal set size** > 1 (any ambiguous field abstains).

    Wraps a fitted split-conformal calibrator (T2.4's
    :class:`ConformalCalibrator`, duck-typed on ``predict_set(field_probs, alpha)
    -> set``). For each field's offered label->probability map, it builds the
    prediction set at coverage ``1 - alpha`` and abstains when **any** field's set
    is not a singleton — a set larger than one label is ambiguous, an empty set
    means no label conforms. The reported ``max_set_size`` / ``field`` identify the
    most ambiguous field.
    """

    def __init__(self, calibrator: Any, alpha: float) -> None:
        if not hasattr(calibrator, "predict_set"):
            raise TypeError("calibrator must expose predict_set(field_probs, alpha)")
        self.calibrator = calibrator
        self.alpha = float(alpha)

    def decide(self, field_probs_by_path: Mapping[Any, Mapping[Any, float]]) -> AbstainDecision:
        """Decide accept/abstain given per-field label->probability maps."""
        sizes = {
            path: len(self.calibrator.predict_set(probs, self.alpha))
            for path, probs in field_probs_by_path.items()
        }
        if not sizes:
            return AbstainDecision(
                accept=False, reason=_REASON_NO_CONFIDENCE, max_set_size=None
            )

        worst_field = max(sizes, key=lambda k: sizes[k])
        max_set_size = sizes[worst_field]
        accept = max_set_size == _ACCEPT_SET_SIZE
        if accept:
            reason = _REASON_ACCEPT
        elif max_set_size == 0:
            reason = _REASON_EMPTY_SET
        else:
            reason = _REASON_SET_GT_1
        return AbstainDecision(
            accept=accept,
            reason=reason,
            max_set_size=max_set_size,
            field=worst_field,
        )


# --------------------------------------------------------------------------- #
# Risk–coverage curve + threshold tuner
# --------------------------------------------------------------------------- #
class RiskCoveragePoint(NamedTuple):
    """One operating point: ``(coverage, risk, threshold)``.

    ``coverage`` is the fraction of records accepted (confidence >= ``threshold``);
    ``risk`` is the selective risk — the error rate among the accepted records
    (``0.0`` by convention when nothing is accepted).
    """

    coverage: float
    risk: float
    threshold: float


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[RiskCoveragePoint]:
    """Risk–coverage curve over accept-if-``score >= threshold`` operating points.

    ``scores`` are per-record confidences (e.g. the aggregate the abstain rule
    compares) and ``correct`` the matching correctness labels. Returns points in
    **ascending threshold** order, so coverage is non-increasing and — on
    well-separated data — selective risk is non-increasing too.

    Endpoints are always present: threshold ``0.0`` accepts everything (coverage
    ``1.0``, full risk), and a final ``+inf`` threshold accepts nothing (coverage
    ``0.0``). Empty input returns ``[]``.
    """
    scores = [float(s) for s in scores]
    correct = [bool(c) for c in correct]
    n = len(scores)
    if n == 0:
        return []

    # Candidate thresholds: 0.0 (guaranteed full-coverage endpoint) + every
    # distinct score. Accepting score >= threshold makes coverage step down as the
    # threshold rises past each score value; ties are handled by the >= test.
    thresholds = sorted({0.0, *scores})
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        accepted = [ok for s, ok in zip(scores, correct) if s >= threshold]
        k = len(accepted)
        coverage = k / n
        risk = (sum(1 for ok in accepted if not ok) / k) if k else 0.0
        points.append(RiskCoveragePoint(coverage, risk, threshold))
    # Zero-coverage endpoint: a threshold above every score accepts nothing.
    points.append(RiskCoveragePoint(0.0, 0.0, float("inf")))
    return points


def tune_threshold(
    scores: Sequence[float],
    correct: Sequence[bool],
    *,
    target_risk: float | None = None,
    target_coverage: float | None = None,
) -> float:
    """Pick the operating-point threshold on the validation risk–coverage curve.

    Exactly one target must be given:

    * ``target_risk`` — return the threshold with the **maximum coverage** whose
      selective risk is ``<= target_risk`` (accept as much as possible while
      keeping the accepted-set error under the target). The trivial abstain-all
      point (risk ``0``) always qualifies, so a threshold is always returned —
      ``+inf`` when even that is required (e.g. all-wrong data, target risk ``0``).
    * ``target_coverage`` — return the threshold whose coverage is **closest** to
      ``target_coverage`` (ties, i.e. a coverage plateau, broken toward the lower
      threshold — the inclusive boundary of that plateau).

    Raises ``ValueError`` if not exactly one target is given, or on empty input.
    """
    if (target_risk is None) == (target_coverage is None):
        raise ValueError("pass exactly one of target_risk / target_coverage")
    curve = risk_coverage_curve(scores, correct)
    if not curve:
        raise ValueError("cannot tune a threshold on an empty validation set")

    if target_risk is not None:
        eligible = [p for p in curve if p.risk <= target_risk + _RISK_TOL]
        # The +inf abstain-all point (risk 0.0) is always eligible for any
        # target_risk >= 0, so `eligible` is never empty in practice.
        best = max(eligible, key=lambda p: (p.coverage, -p.threshold))
        return best.threshold

    # target_coverage: nearest coverage; a coverage plateau breaks to the lower
    # (inclusive) threshold.
    best = min(curve, key=lambda p: (abs(p.coverage - target_coverage), p.threshold))
    return best.threshold
