# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""System-One benchmark harness + task-family loader (task T0.3).

A reproducible, host-side harness that scores a **pluggable** runtime on
``(context, schema, gold_value)`` records across the four metric families the
PRD (§7/§8) requires and the downstream device gates (T0.4/T1.3/T2.6/T3.3)
consume:

1. **Latency** — median + p95 (+ mean) per-request wall time; the harness times
   each ``runtime.predict(context, schema)`` call.
2. **Task accuracy** — exact-match per record (predicted typed value == gold)
   plus per-field accuracy.
3. **Schema-validity %** — fraction of predictions satisfying the schema. If the
   T0.2 checker (``vllm_ascend/system_one/validate.py``) is importable at run
   time it is preferred; otherwise a minimal IR-driven fallback runs. The
   dependency is intentionally **soft** so T0.3 does not hard-block on T0.2.
4. **Calibration** — Expected Calibration Error (binned) + reliability-diagram
   bins, computed from per-prediction confidence vs correctness. Pure math,
   implemented from scratch and unit-tested against hand-computed cases.

Design surface (the importable API is primary; ``main`` is a thin CLI):

* :class:`Record` — one ``(context, schema, gold_value)`` datum.
* :class:`Prediction` — ``(value, confidences)``; ``confidences`` maps a leaf
  field path (``"$.tool"``) to a probability in ``[0, 1]``.
* :class:`Runtime` — protocol: ``predict(context, schema) -> Prediction``.
* :class:`StubRuntime` — deterministic reference runtime (gold-with-noise) that
  exercises all four metrics.
* :class:`TaskFamily` / :func:`load_task_family` — a pluggable loader yielding
  ``Record``s. An in-repo tiny synthetic family (typed function-call / intent
  routing) ships as a fixture so the harness runs with **no network**.
* :func:`run_benchmark` — returns a JSON-serializable report ``dict``.

**Dataset pluggability.** A real dataset (e.g. a BFCL-style function-calling set
or a JSON-Schema extraction corpus) plugs in by registering a loader with
:func:`register_task_family` that yields :class:`Record`s ``(context, schema,
gold)``. The harness never fetches anything; a downloader/adapter would live in
the loader and cache to disk, keeping this module import-clean and offline.

Design constraints: **pure Python, stdlib only.** No ``torch``, no ``torch_npu``,
no ``triton`` — the module imports and unit-tests host-side, CPU-only. The T0.1
IR compiler is loaded **by file path** (not ``import vllm_ascend...``) so pulling
it in never triggers the heavy package ``__init__``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Record",
    "Prediction",
    "Runtime",
    "StubRuntime",
    "TaskFamily",
    "register_task_family",
    "load_task_family",
    "expected_calibration_error",
    "reliability_bins",
    "latency_stats",
    "exact_match",
    "per_field_accuracy",
    "validate_value",
    "run_benchmark",
    "main",
]

# Default number of calibration bins over [0, 1].
DEFAULT_N_BINS = 10
# Percentile the latency summary reports alongside the median.
LATENCY_PERCENTILE = 95.0
# Confidence a StubRuntime attaches to a field it emits correctly / incorrectly.
_STUB_CORRECT_CONF = 0.9
_STUB_WRONG_CONF = 0.3


# --------------------------------------------------------------------------- #
# T0.1 IR compiler — loaded by file path to stay host-side / import-clean.
# --------------------------------------------------------------------------- #
def _load_schema_ir():
    """Load the T0.1 ``schema_ir`` module by file path.

    Loading by path (rather than ``import vllm_ascend.system_one.schema_ir``)
    keeps this harness free of the shippable package ``__init__`` side effects,
    so it stays pure-Python and CPU-only.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm_ascend" / "system_one" / "schema_ir.py"
    spec = importlib.util.spec_from_file_location("system_one_schema_ir_t03", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load schema_ir from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so nested-dataclass forward refs resolve on 3.12+.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sir = _load_schema_ir()
compile_schema = _sir.compile_schema
FieldKind = _sir.FieldKind


# --------------------------------------------------------------------------- #
# Core data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Record:
    """One benchmark datum: a context, an output schema, and the gold value.

    ``schema`` is a JSON-Schema-subset ``dict`` (T0.1 compilable); ``gold`` is a
    nested ``dict`` of the expected typed value keyed by field name.
    """

    context: str
    schema: dict[str, Any]
    gold: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    """A runtime's output: the typed ``value`` and per-field ``confidences``.

    ``confidences`` maps a leaf field **path** (``"$.tool"``, ``"$.args.n"``) to
    a probability in ``[0, 1]``. A runtime that only produces a scalar confidence
    may map every leaf path to the same value.
    """

    value: dict[str, Any]
    confidences: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Runtime(Protocol):
    """The pluggable runtime interface the harness scores."""

    def predict(self, context: str, schema: dict[str, Any]) -> Prediction: ...


# --------------------------------------------------------------------------- #
# Path helpers (navigate nested value dicts by IR path, e.g. "$.args.n")
# --------------------------------------------------------------------------- #
def _split_path(path: str) -> list[str]:
    """Turn an IR path (``"$"`` root, ``"$.a.b"``) into field-name segments."""
    return [seg for seg in path.lstrip("$").split(".") if seg]


def _lookup(value: Any, path: str) -> tuple[bool, Any]:
    """Return ``(present, value_at_path)`` for a leaf ``path`` in ``value``."""
    node = value
    for seg in _split_path(path):
        if not isinstance(node, Mapping) or seg not in node:
            return (False, None)
        node = node[seg]
    return (True, node)


def _set_path(value: dict[str, Any], path: str, new: Any) -> None:
    """Set a leaf ``path`` in a (possibly nested) mutable ``value`` dict."""
    segs = _split_path(path)
    node = value
    for seg in segs[:-1]:
        node = node.setdefault(seg, {})
    node[segs[-1]] = new


# --------------------------------------------------------------------------- #
# Validity: prefer T0.2 validate.py, else an IR-driven fallback.
# --------------------------------------------------------------------------- #
def _load_t0_2_validator() -> Callable[[Any, Any], tuple[bool, str | None]] | None:
    """Try to load the T0.2 checker and adapt it to ``(ir, value) -> (bool, path)``.

    The soft dependency: if ``validate.py`` is absent, or exposes no recognizable
    entry point, or raises when called, we return ``None`` and the caller uses
    the fallback. This keeps T0.3 from hard-blocking on T0.2's exact API.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "vllm_ascend" / "system_one" / "validate.py"
    if not module_path.exists():
        return None
    try:  # pragma: no cover - only exercised once T0.2 lands
        spec = importlib.util.spec_from_file_location("system_one_validate_t03", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        return None

    for name in ("validate_value", "validate", "is_valid", "check_value", "check"):
        fn = getattr(module, name, None)
        if callable(fn):
            return _adapt_validator(fn)
    return None


def _adapt_validator(fn: Callable[..., Any]) -> Callable[[Any, Any], tuple[bool, str | None]]:
    """Normalize a T0.2 validator's return into ``(valid, offending_path)``."""

    def _call(ir: Any, value: Any) -> tuple[bool, str | None]:  # pragma: no cover
        result = fn(ir, value)
        if isinstance(result, tuple):
            valid = bool(result[0])
            offending = result[1] if len(result) > 1 else None
            return (valid, offending)
        if isinstance(result, bool):
            return (result, None)
        # Objects exposing ``.valid`` / ``.offending`` (or ``.path``).
        valid = bool(getattr(result, "valid", result))
        offending = getattr(result, "offending", getattr(result, "path", None))
        return (valid, offending)

    return _call


def _fallback_validate(ir: Any, value: Any) -> tuple[bool, str | None]:
    """Minimal IR-driven validity check: ``(valid, offending_path)``.

    Drives off the T0.1 leaf validator descriptors: required-presence, enum
    membership, numeric range/integrality, string type + length, boolean type.
    """
    for desc in ir.validator_descriptors():
        path = desc["path"]
        present, v = _lookup(value, path)
        if not present:
            if desc["required"]:
                return (False, path)
            continue
        kind = desc["kind"]
        if kind == FieldKind.ENUM.value:
            # bool is an int subclass; enum members are str/int, never bool.
            if isinstance(v, bool) or v not in desc["members"]:
                return (False, path)
        elif kind in (FieldKind.INTEGER.value, FieldKind.NUMBER.value):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return (False, path)
            if kind == FieldKind.INTEGER.value and not isinstance(v, int):
                return (False, path)
            if v < desc["minimum"] or v > desc["maximum"]:
                return (False, path)
        elif kind == FieldKind.STRING.value:
            if not isinstance(v, str) or len(v) > desc["max_length"]:
                return (False, path)
        elif kind == FieldKind.BOOLEAN.value:
            if not isinstance(v, bool):
                return (False, path)
    return (True, None)


# Resolve the validator once at import; expose which one is in use.
_T0_2_VALIDATOR = _load_t0_2_validator()
VALIDATOR_NAME = "t0.2" if _T0_2_VALIDATOR is not None else "fallback"


def validate_value(ir: Any, value: Any) -> tuple[bool, str | None]:
    """Validate ``value`` against a compiled ``ir``: ``(valid, offending_path)``.

    Prefers the T0.2 checker when importable; falls back to the IR-driven check.
    A T0.2 call that raises degrades gracefully to the fallback for that record.
    """
    if _T0_2_VALIDATOR is not None:
        try:  # pragma: no cover - only once T0.2 lands
            return _T0_2_VALIDATOR(ir, value)
        except Exception:
            return _fallback_validate(ir, value)
    return _fallback_validate(ir, value)


# --------------------------------------------------------------------------- #
# Calibration math (from scratch): ECE + reliability bins
# --------------------------------------------------------------------------- #
def _bin_index(confidence: float, n_bins: int) -> int:
    """Map a confidence in [0, 1] to an equal-width bin index in [0, n_bins)."""
    idx = int(confidence * n_bins)
    if idx < 0:
        return 0
    if idx >= n_bins:
        return n_bins - 1  # confidence == 1.0 falls in the last bin
    return idx


def _bin_accumulate(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int
) -> tuple[list[int], list[float], list[int]]:
    counts = [0] * n_bins
    sum_conf = [0.0] * n_bins
    sum_correct = [0] * n_bins
    for conf, ok in zip(confidences, correct):
        b = _bin_index(conf, n_bins)
        counts[b] += 1
        sum_conf[b] += conf
        sum_correct[b] += 1 if ok else 0
    return counts, sum_conf, sum_correct


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_N_BINS
) -> float:
    """Binned Expected Calibration Error.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)| over equal-width confidence
    bins. Perfect calibration (per-bin accuracy == mean confidence) gives 0.
    Empty input returns 0.0.
    """
    n = len(confidences)
    if n == 0:
        return 0.0
    counts, sum_conf, sum_correct = _bin_accumulate(confidences, correct, n_bins)
    ece = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        acc = sum_correct[b] / counts[b]
        conf = sum_conf[b] / counts[b]
        ece += (counts[b] / n) * abs(acc - conf)
    return ece


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_N_BINS
) -> list[dict[str, Any]]:
    """Reliability-diagram bins: one dict per equal-width bin over [0, 1].

    Each carries ``bin_index``, ``lower``/``upper`` edges, ``count``, and the
    bin's ``avg_confidence`` and ``accuracy`` (``None`` for empty bins). Counts
    sum to ``len(confidences)``.
    """
    counts, sum_conf, sum_correct = _bin_accumulate(confidences, correct, n_bins)
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        cnt = counts[b]
        bins.append(
            {
                "bin_index": b,
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": cnt,
                "avg_confidence": (sum_conf[b] / cnt) if cnt else None,
                "accuracy": (sum_correct[b] / cnt) if cnt else None,
            }
        )
    return bins


# --------------------------------------------------------------------------- #
# Latency math
# --------------------------------------------------------------------------- #
def _percentile_nearest_rank(sorted_vals: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: rank = ceil(pct/100 * N), 1-indexed."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    rank = math.ceil((pct / 100.0) * n)
    rank = max(1, min(rank, n))
    return sorted_vals[rank - 1]


def latency_stats(durations: Sequence[float]) -> dict[str, Any]:
    """Summarize per-request wall times: median, p95 (nearest-rank), mean, n."""
    n = len(durations)
    if n == 0:
        return {"median_s": 0.0, "p95_s": 0.0, "mean_s": 0.0, "n": 0}
    ordered = sorted(durations)
    return {
        "median_s": statistics.median(ordered),
        "p95_s": _percentile_nearest_rank(ordered, LATENCY_PERCENTILE),
        "mean_s": statistics.fmean(ordered),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Accuracy math
# --------------------------------------------------------------------------- #
def exact_match(prediction_value: Any, gold: Any) -> bool:
    """Whole-record exact match of the typed value against the gold value."""
    return prediction_value == gold


def per_field_accuracy(predictions: Sequence[Prediction], records: Sequence[Record]) -> dict[str, float]:
    """Per-leaf-field accuracy across all records that declare that field."""
    hits: dict[str, int] = {}
    totals: dict[str, int] = {}
    for pred, rec in zip(predictions, records):
        ir = compile_schema(rec.schema)
        for descriptor in ir.validator_descriptors():
            path = descriptor["path"]
            _, gold_v = _lookup(rec.gold, path)
            present, pred_v = _lookup(pred.value, path)
            totals[path] = totals.get(path, 0) + 1
            if present and pred_v == gold_v:
                hits[path] = hits.get(path, 0) + 1
    return {path: hits.get(path, 0) / totals[path] for path in totals}


# --------------------------------------------------------------------------- #
# StubRuntime — deterministic reference runtime (gold-with-noise)
# --------------------------------------------------------------------------- #
class StubRuntime:
    """A deterministic runtime that emits gold, optionally corrupted per record.

    ``overrides`` maps a record index to ``{leaf_path: replacement_value}``; a
    replacement may be a different valid value (accuracy error) or an
    out-of-domain value (validity error). ``confidences`` optionally maps
    ``(record_index, leaf_path)`` to an explicit probability; otherwise a
    correctly-emitted leaf gets ``correct_conf`` and an overridden one gets
    ``wrong_conf`` — so the stub exercises accuracy, validity and calibration at
    once.

    Records are matched by ``context`` (unique per record), so the stub is robust
    to iteration order.
    """

    def __init__(
        self,
        records: Sequence[Record],
        *,
        overrides: Mapping[int, Mapping[str, Any]] | None = None,
        confidences: Mapping[tuple[int, str], float] | None = None,
        correct_conf: float = _STUB_CORRECT_CONF,
        wrong_conf: float = _STUB_WRONG_CONF,
    ) -> None:
        self._by_context = {rec.context: (i, rec) for i, rec in enumerate(records)}
        self._overrides = {int(k): dict(v) for k, v in (overrides or {}).items()}
        self._confidences = dict(confidences or {})
        self._correct_conf = correct_conf
        self._wrong_conf = wrong_conf

    def predict(self, context: str, schema: dict[str, Any]) -> Prediction:
        idx, rec = self._by_context[context]
        value = copy.deepcopy(rec.gold)
        overridden = self._overrides.get(idx, {})
        for path, replacement in overridden.items():
            _set_path(value, path, replacement)

        ir = compile_schema(schema)
        confidences: dict[str, float] = {}
        for descriptor in ir.validator_descriptors():
            path = descriptor["path"]
            if (idx, path) in self._confidences:
                confidences[path] = self._confidences[(idx, path)]
            elif path in overridden:
                confidences[path] = self._wrong_conf
            else:
                confidences[path] = self._correct_conf
        return Prediction(value=value, confidences=confidences)


# --------------------------------------------------------------------------- #
# Task-family loader (pluggable) + in-repo synthetic fixture
# --------------------------------------------------------------------------- #
@runtime_checkable
class TaskFamily(Protocol):
    """A task family loader: ``load(name) -> Iterable[Record]``."""

    def load(self, name: str) -> Iterable[Record]: ...


_TASK_FAMILIES: dict[str, Callable[[], Iterable[Record]]] = {}


def register_task_family(name: str, loader: Callable[[], Iterable[Record]]) -> None:
    """Register a task-family loader under ``name`` (idempotent overwrite).

    A real dataset (BFCL-style function-calling, JSON-Schema extraction, ...)
    plugs in here: the loader yields :class:`Record`s and is responsible for any
    caching/adaptation. The harness itself never fetches anything.
    """
    _TASK_FAMILIES[name] = loader


def load_task_family(name: str) -> Iterator[Record]:
    """Return an iterator over a registered task family's :class:`Record`s.

    The name is validated **eagerly** (before any iteration), so an unknown
    ``name`` raises ``KeyError`` at call time rather than on first ``next()``.
    """
    if name not in _TASK_FAMILIES:
        raise KeyError(f"unknown task family {name!r}; registered: {sorted(_TASK_FAMILIES)}")
    return iter(_TASK_FAMILIES[name]())


# --- The default in-repo synthetic family: typed function-call / intent routing.
#
# Each record is unstructured context text + a tool/argument schema (enum tool id,
# bounded-int argument count, boolean confirmation flag, bounded-string query) +
# the gold typed call. Small, offline, and shaped exactly like the real target
# (BFCL-style function calling) so a real dataset swaps in behind the loader.
_INTENT_ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"enum": ["search", "calculate", "translate", "noop"]},
        "arg_count": {"type": "integer", "minimum": 0, "maximum": 4},
        "confirmed": {"type": "boolean"},
        "query": {"type": "string", "maxLength": 64},
    },
    "required": ["tool", "arg_count"],
}

_INTENT_ROUTING_ROWS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "Look up today's weather in Berlin.",
        {"tool": "search", "arg_count": 1, "confirmed": True, "query": "weather Berlin"},
    ),
    (
        "What is 17 times 23?",
        {"tool": "calculate", "arg_count": 2, "confirmed": True, "query": "17*23"},
    ),
    (
        "Translate 'good morning' into French.",
        {"tool": "translate", "arg_count": 2, "confirmed": False, "query": "good morning"},
    ),
    (
        "Never mind, ignore that.",
        {"tool": "noop", "arg_count": 0, "confirmed": False, "query": ""},
    ),
    (
        "Find open pull requests in the ascend repo.",
        {"tool": "search", "arg_count": 1, "confirmed": True, "query": "open PRs ascend"},
    ),
    (
        "Add 4 and 5 for me please.",
        {"tool": "calculate", "arg_count": 2, "confirmed": True, "query": "4+5"},
    ),
)


def _synthetic_intent_routing() -> Iterator[Record]:
    for i, (context, gold) in enumerate(_INTENT_ROUTING_ROWS):
        yield Record(
            context=f"[{i}] {context}",
            schema=copy.deepcopy(_INTENT_ROUTING_SCHEMA),
            gold=copy.deepcopy(gold),
        )


register_task_family("intent_routing", _synthetic_intent_routing)


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(
    runtime: Runtime,
    records: Sequence[Record],
    *,
    n_bins: int = DEFAULT_N_BINS,
    timer: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Score ``runtime`` on ``records``; return a JSON-serializable report.

    The report carries all four metric families::

        {
          "n_records": int,
          "latency":     {"median_s", "p95_s", "mean_s", "n"},
          "accuracy":    {"exact_match", "n_correct", "n_total", "per_field": {path: acc}},
          "validity":    {"valid_fraction", "n_valid", "n_total", "validator"},
          "calibration": {"ece", "n_bins", "n_predictions", "reliability_bins": [...]},
        }

    ``timer`` (default ``time.perf_counter``) is injectable so latency is
    deterministic under test. Calibration points are per **leaf field** per
    record: confidence vs whether that field matched gold.
    """
    clock = timer if timer is not None else time.perf_counter

    predictions: list[Prediction] = []
    durations: list[float] = []
    for rec in records:
        start = clock()
        pred = runtime.predict(rec.context, rec.schema)
        end = clock()
        durations.append(end - start)
        predictions.append(pred)

    # --- accuracy ---
    n_correct = sum(1 for pred, rec in zip(predictions, records) if exact_match(pred.value, rec.gold))
    n_total = len(records)
    per_field = per_field_accuracy(predictions, records)

    # --- validity + calibration (share the per-record IR compile) ---
    n_valid = 0
    conf_points: list[float] = []
    correct_points: list[bool] = []
    for pred, rec in zip(predictions, records):
        ir = compile_schema(rec.schema)
        valid, _offending = validate_value(ir, pred.value)
        if valid:
            n_valid += 1
        for descriptor in ir.validator_descriptors():
            path = descriptor["path"]
            conf = pred.confidences.get(path)
            if conf is None:
                continue  # runtime declared no confidence for this field
            _, gold_v = _lookup(rec.gold, path)
            _, pred_v = _lookup(pred.value, path)
            conf_points.append(float(conf))
            correct_points.append(pred_v == gold_v)

    ece = expected_calibration_error(conf_points, correct_points, n_bins=n_bins)
    bins = reliability_bins(conf_points, correct_points, n_bins=n_bins)

    return {
        "n_records": n_total,
        "latency": latency_stats(durations),
        "accuracy": {
            "exact_match": (n_correct / n_total) if n_total else 0.0,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_field": per_field,
        },
        "validity": {
            "valid_fraction": (n_valid / n_total) if n_total else 0.0,
            "n_valid": n_valid,
            "n_total": n_total,
            "validator": VALIDATOR_NAME,
        },
        "calibration": {
            "ece": ece,
            "n_bins": n_bins,
            "n_predictions": len(conf_points),
            "reliability_bins": bins,
        },
    }


# --------------------------------------------------------------------------- #
# CLI (thin wrapper around the importable API)
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark on a task family with the built-in StubRuntime.

    The CLI is intentionally minimal — the importable API is primary. A real
    runtime is scored by importing :func:`run_benchmark` and passing it in.
    """
    parser = argparse.ArgumentParser(description="System-One benchmark harness (T0.3)")
    parser.add_argument(
        "--task-family",
        default="intent_routing",
        help="registered task family to load (default: intent_routing)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help="calibration bins over [0, 1] (default: 10)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the JSON report here (default: stdout)",
    )
    args = parser.parse_args(argv)

    records = list(load_task_family(args.task_family))
    report = run_benchmark(StubRuntime(records), records, n_bins=args.n_bins)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
