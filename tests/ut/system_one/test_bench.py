# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the System-One benchmark harness (task T0.3).

The harness (``tools/system_one/bench.py``) scores a *pluggable* runtime on
``(context, schema, gold_value)`` records across four metric families
(PRD §7/§8):

* **latency** (median / p95 per-request wall time),
* **task accuracy** (exact-match per record + per-field accuracy),
* **schema-validity %** (fraction of predictions satisfying the schema — using
  T0.2 ``validate.py`` when importable, else an IR-driven fallback), and
* **calibration** (ECE + reliability-diagram bins from per-prediction confidence
  vs correctness).

These tests are the RED-first contract. The ECE/reliability math is checked
against **hand-computed** fixtures with a KNOWN answer; latency against a
controlled set of durations; accuracy/validity against a ``StubRuntime`` with a
known error pattern; and the in-repo synthetic task family must load into
well-formed records whose schema compiles via T0.1 and whose gold validates.

The module is loaded **directly by file path** (not via ``import ...``) so the
test stays pure-Python and host-side: no torch, no torch_npu, no triton.

Run: ``python3 -m pytest -q --noconftest tests/ut/system_one/test_bench.py``
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "tools" / "system_one" / "bench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("system_one_bench", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any dataclass forward-ref resolution can find the
    # module in ``sys.modules`` on 3.12+ (same discipline as the schema_ir test).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module()

expected_calibration_error = bench.expected_calibration_error
reliability_bins = bench.reliability_bins
latency_stats = bench.latency_stats
run_benchmark = bench.run_benchmark
StubRuntime = bench.StubRuntime
Record = bench.Record
Prediction = bench.Prediction
load_task_family = bench.load_task_family


# --------------------------------------------------------------------------- #
# Small helpers for building controlled records
# --------------------------------------------------------------------------- #
_FLAT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"enum": ["a", "b", "c"]},
        "count": {"type": "integer", "minimum": 0, "maximum": 9},
    },
    "required": ["tool"],
}


def _records(golds):
    return [Record(context=f"ctx-{i}", schema=_FLAT_SCHEMA, gold=g) for i, g in enumerate(golds)]


# --------------------------------------------------------------------------- #
# ECE math — hand-computed, KNOWN answers
# --------------------------------------------------------------------------- #
def test_ece_perfectly_calibrated_is_zero():
    # Two extreme groups: conf 1.0 all-correct, conf 0.0 all-incorrect.
    # Each bin has acc == avg_confidence, so ECE is exactly 0.
    confidences = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    correct = [True, True, True, False, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) == 0.0


def test_ece_single_bin_known_value():
    # 10 predictions, all confidence 0.9 (bin 9), exactly 5 correct.
    # acc = 0.5, conf = 0.9 -> |0.5 - 0.9| = 0.4, single bin weight 1.0 -> ECE 0.4.
    confidences = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.4)


def test_ece_two_bin_weighted_known_value():
    # Low bin: 4 preds conf 0.1, 1 correct -> acc 0.25, |0.1-0.25| = 0.15, w = 0.4.
    # High bin: 6 preds conf 0.9, 3 correct -> acc 0.50, |0.9-0.50| = 0.40, w = 0.6.
    # ECE = 0.4*0.15 + 0.6*0.40 = 0.06 + 0.24 = 0.30.
    confidences = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    correct = [True, False, False, False, True, True, True, False, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.30)


def test_ece_empty_is_zero():
    assert expected_calibration_error([], [], n_bins=10) == 0.0


# --------------------------------------------------------------------------- #
# Reliability bins — counts + per-bin accuracy/confidence
# --------------------------------------------------------------------------- #
def test_reliability_bins_counts_and_values():
    confidences = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    correct = [True, False, False, False, True, True, True, False, False, False]
    bins = reliability_bins(confidences, correct, n_bins=10)

    assert len(bins) == 10
    # Counts sum to N and land in exactly the two expected bins.
    assert sum(b["count"] for b in bins) == 10
    low = bins[1]
    high = bins[9]
    assert low["count"] == 4
    assert low["accuracy"] == pytest.approx(0.25)
    assert low["avg_confidence"] == pytest.approx(0.1)
    assert high["count"] == 6
    assert high["accuracy"] == pytest.approx(0.5)
    assert high["avg_confidence"] == pytest.approx(0.9)
    # Empty bins carry a count of 0 and no accuracy/confidence.
    assert bins[5]["count"] == 0
    assert bins[5]["accuracy"] is None
    assert bins[0]["lower"] == pytest.approx(0.0)
    assert bins[9]["upper"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Latency — median / p95 on a controlled set of durations
# --------------------------------------------------------------------------- #
def test_latency_stats_median_and_p95():
    durations = [float(x) for x in range(1, 21)]  # 1.0 .. 20.0
    stats = latency_stats(durations)
    assert stats["n"] == 20
    assert stats["median_s"] == pytest.approx(10.5)  # avg of 10th & 11th
    # nearest-rank p95: ceil(0.95 * 20) = 19 -> 19th value = 19.0
    assert stats["p95_s"] == pytest.approx(19.0)
    assert stats["mean_s"] == pytest.approx(10.5)


def test_latency_stats_single_value():
    stats = latency_stats([2.5])
    assert stats["median_s"] == pytest.approx(2.5)
    assert stats["p95_s"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Accuracy + per-field accuracy via a StubRuntime with a KNOWN error pattern
# --------------------------------------------------------------------------- #
def test_accuracy_and_per_field_with_known_errors():
    records = _records(
        [
            {"tool": "a", "count": 1},
            {"tool": "b", "count": 2},
            {"tool": "c", "count": 3},
            {"tool": "a", "count": 4},
        ]
    )
    # r1: count wrong (but valid); r2: tool wrong (but valid).
    runtime = StubRuntime(
        records,
        overrides={1: {"$.count": 9}, 2: {"$.tool": "a"}},
    )
    report = run_benchmark(runtime, records)

    acc = report["accuracy"]
    # exact-match: r0 ok, r1 wrong, r2 wrong, r3 ok -> 0.5
    assert acc["exact_match"] == pytest.approx(0.5)
    assert acc["per_field"]["$.tool"] == pytest.approx(0.75)
    assert acc["per_field"]["$.count"] == pytest.approx(0.75)
    # All emitted values are still schema-valid (9 in range, "a" a member).
    assert report["validity"]["valid_fraction"] == pytest.approx(1.0)


def test_perfect_stub_scores_all_metrics():
    records = _records([{"tool": "a", "count": 1}, {"tool": "b", "count": 2}])
    runtime = StubRuntime(records)  # no overrides -> emits gold, all correct
    report = run_benchmark(runtime, records)
    assert report["accuracy"]["exact_match"] == pytest.approx(1.0)
    assert report["validity"]["valid_fraction"] == pytest.approx(1.0)
    # calibration is still exercised (correct-but-not-1.0 confidence -> ECE > 0)
    assert report["calibration"]["ece"] > 0.0
    assert report["n_records"] == 2


# --------------------------------------------------------------------------- #
# Validity % — a StubRuntime that emits some invalid values
# --------------------------------------------------------------------------- #
def test_validity_fraction_with_invalid_values():
    records = _records(
        [
            {"tool": "a", "count": 1},
            {"tool": "b", "count": 2},
            {"tool": "c", "count": 3},
            {"tool": "a", "count": 4},
        ]
    )
    # r0: enum non-member; r1: out-of-range numeric. r2/r3 valid.
    runtime = StubRuntime(
        records,
        overrides={0: {"$.tool": "zzz"}, 1: {"$.count": 99}},
    )
    report = run_benchmark(runtime, records)
    assert report["validity"]["valid_fraction"] == pytest.approx(0.5)
    assert report["validity"]["n_valid"] == 2
    assert report["validity"]["n_total"] == 4
    # T0.2 may or may not exist at run time; either way the label is one of these.
    assert report["validity"]["validator"] in {"t0.2", "fallback"}


# --------------------------------------------------------------------------- #
# Latency in run_benchmark via an injected (fake) timer -> deterministic
# --------------------------------------------------------------------------- #
def test_run_benchmark_latency_with_injected_timer():
    records = _records([{"tool": "a", "count": 1}, {"tool": "b", "count": 2}])
    runtime = StubRuntime(records)
    # (t0, t1) pairs -> durations 1.0 and 3.0 -> median 2.0.
    ticks = iter([0.0, 1.0, 5.0, 8.0])
    report = run_benchmark(runtime, records, timer=lambda: next(ticks))
    assert report["latency"]["median_s"] == pytest.approx(2.0)
    assert report["latency"]["n"] == 2


# --------------------------------------------------------------------------- #
# Report shape + JSON serializability (downstream device gates consume this)
# --------------------------------------------------------------------------- #
def test_report_has_four_metric_families_and_is_json_serializable():
    records = _records([{"tool": "a", "count": 1}, {"tool": "b", "count": 2}])
    report = run_benchmark(StubRuntime(records), records)
    for key in ("latency", "accuracy", "validity", "calibration"):
        assert key in report, f"missing metric family {key!r}"
    calib = report["calibration"]
    assert "ece" in calib and "reliability_bins" in calib and "n_bins" in calib
    # one calibration point per leaf field per record
    assert calib["n_predictions"] == 2 * 2
    # Must round-trip through JSON unchanged in structure.
    dumped = json.dumps(report)
    assert json.loads(dumped)["accuracy"]["exact_match"] == report["accuracy"]["exact_match"]


# --------------------------------------------------------------------------- #
# The in-repo synthetic task family: loads, compiles via T0.1, gold validates
# --------------------------------------------------------------------------- #
def test_synthetic_task_family_loads_and_is_well_formed():
    records = list(load_task_family("intent_routing"))
    assert len(records) >= 4  # a non-trivial fixture

    # Import the T0.1 compiler by file path to validate each record's schema.
    ir_path = _REPO_ROOT / "vllm_ascend" / "system_one" / "schema_ir.py"
    spec = importlib.util.spec_from_file_location("t0_1_schema_ir", ir_path)
    ir_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ir_mod
    spec.loader.exec_module(ir_mod)

    for rec in records:
        assert isinstance(rec.context, str) and rec.context
        assert isinstance(rec.schema, dict)
        assert isinstance(rec.gold, dict)
        # schema compiles via T0.1 ...
        ir = ir_mod.compile_schema(rec.schema)
        # ... and the gold value validates against it (fallback oracle).
        valid, offending = bench.validate_value(ir, rec.gold)
        assert valid, f"gold for {rec.context!r} invalid at {offending}"


def test_task_family_unknown_name_raises():
    with pytest.raises((KeyError, ValueError)):
        load_task_family("does-not-exist")


def test_synthetic_family_runs_end_to_end_on_stub():
    records = list(load_task_family("intent_routing"))
    report = run_benchmark(StubRuntime(records), records)
    # A perfect stub over the synthetic family: valid + exact-match everywhere.
    assert report["validity"]["valid_fraction"] == pytest.approx(1.0)
    assert report["accuracy"]["exact_match"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Import hygiene: pure-Python, no NPU/Triton/torch on the import path
# --------------------------------------------------------------------------- #
def test_module_has_no_npu_or_triton_imports():
    """Grep-gate: no NPU/Triton/torch on the harness import path.

    Parsed via ``ast`` so it flags real ``import``/``from`` statements only, not
    module names appearing in docstrings/comments.
    """
    import ast

    forbidden_roots = {"triton", "torch", "torch_npu", "vllm"}
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
    offenders = forbidden_roots & imported_roots
    assert not offenders, f"bench.py must not import {sorted(offenders)}"
    for line in _MODULE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "triton" not in stripped and "torch_npu" not in stripped, f"forbidden import: {stripped!r}"
