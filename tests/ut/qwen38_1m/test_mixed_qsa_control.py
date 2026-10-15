# SPDX-License-Identifier: Apache-2.0
"""Audit wrapper-owned PLE without poisoning mutation admission on read errors."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_audit():
    path = Path(__file__).resolve().parents[3] / "artifacts/qfn-mixed-opt-20261015/final-audit-source.py"
    spec = importlib.util.spec_from_file_location("mixed_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrapper():
    worker = SimpleNamespace(
        rank=0, _qfn_mixed_qsa=dict(mode="score", failed=False, replacements=[]), _qfn_mixed_qsa_gate=dict(passed=True)
    )
    ple = SimpleNamespace(status=lambda: dict(enabled=True, verify=False, failed=False, checked=10))
    return SimpleNamespace(worker=worker, _qfn_native_ple_serving=dict(manager=ple))


def test_audit_reads_ple_from_wrapper_and_score_from_underlying_worker():
    audit = load_audit()
    audit.inspect_retained = lambda worker: dict(passed=True)
    result = audit.inspect_worker(wrapper())
    assert result["passed"] and result["native_ple"]["checked"] == 10
    assert result["score_mode"] == "score"


def test_worker_read_error_fails_preflight_before_mutation_phase():
    audit = load_audit()
    audit.inspect_retained = lambda worker: dict(passed=True)
    target = wrapper()
    del target._qfn_native_ple_serving
    schedule = lambda: None
    scheduler = SimpleNamespace(
        schedule=schedule,
        get_num_unfinished_requests=lambda: 0,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    engine = SimpleNamespace(
        _pause_complete=lambda: True,
        batch_queue=[],
        scheduler=scheduler,
        _qfn_mixed_budget=SimpleNamespace(max_prefill_tokens=512, wrapper=schedule),
        model_executor=SimpleNamespace(collective_rpc=lambda fn: [fn(target) for _ in range(4)]),
    )
    result = audit.Candidate().validate(engine)
    assert result["passed"] is False
    assert all("_qfn_native_ple_serving" in row["error"] for row in result["workers"])
