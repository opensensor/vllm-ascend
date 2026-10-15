# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import qsa_gather_staging as qsa
from tools.qwen4exp.resident_candidates import qfn_qsa_gather_staging as candidate

FIXTURE = Path(__file__).parent / "fixtures/qfn_pinned_qsa_gather.txt"


def request(action="preflight", rows=None):
    return dict(action=action, build="/immutable/build", rows=[1, 3, 64] if rows is None else rows, repeats=1)


def environment(monkeypatch):
    worker = SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=None), rank=2)
    fake = SimpleNamespace(
        npu=SimpleNamespace(
            mem_get_info=lambda: (2 << 30, 4 << 30),
            is_current_stream_capturing=lambda: False,
            current_device=lambda: 0,
            synchronize=lambda: None,
        ),
        device=lambda *a: "cpu",
        inference_mode=nullcontext,
    )
    monkeypatch.setattr(qsa, "verify_build", lambda _: {})
    monkeypatch.setattr(qsa, "load_resources", lambda _, owner: owner.update(gather=object(), eager_only=True))
    return worker, fake


def test_fixture_source_digest():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == candidate.QUALIFIED_SOURCE_SHA256


@pytest.mark.parametrize(
    "change",
    [
        {"action": "reset"},
        {"build": "relative"},
        {"rows": []},
        {"rows": [True]},
        {"rows": [65]},
        {"rows": [1, 1]},
        {"rows": [1.0]},
        {"repeats": False},
        {"repeats": 0},
        {"extra": 1},
    ],
)
def test_parse_strict_request(change):
    payload = request()
    payload.update(change)
    with pytest.raises(ValueError):
        candidate.parse_request(json.dumps(payload))


def test_parse_defaults():
    parsed = candidate.parse_request(json.dumps(dict(action="validate", build="/tmp/build")))
    assert parsed["rows"] == [1, 3, 64] and parsed["repeats"] == 3


def test_private_state_requires_preflight_load_validate(monkeypatch):
    worker, fake = environment(monkeypatch)
    with pytest.raises(RuntimeError, match="preflight"):
        candidate.run_action(worker, request("load"), fake)
    assert candidate.run_action(worker, request(), fake)["passed"]
    assert candidate.run_action(worker, request("load"), fake)["loaded"]
    with pytest.raises(RuntimeError, match="validate each"):
        candidate.run_action(worker, request("timing"), fake)
    receipt = dict(passed=True, cases=[], attention=[])
    monkeypatch.setattr(qsa, "validate", lambda *a, **kw: receipt)
    assert candidate.run_action(worker, request("validate"), fake) is receipt
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    assert state["validated_rows"] == {1, 3, 64} and state["validation_receipts"] == [receipt]
    # A passed scalar alone cannot admit model binding.
    with pytest.raises(RuntimeError, match="complete physical"):
        candidate.make_replacements(worker)


@pytest.mark.parametrize("blocker", ["active", "profile", "capture", "memory"])
def test_private_admission_blocks_before_allocation(monkeypatch, blocker):
    worker, fake = environment(monkeypatch)
    if blocker == "active":
        worker.model_runner.execute_model_state = object()
    if blocker == "profile":
        worker._qfn_native_profile_session = object()
    if blocker == "capture":
        fake.npu.is_current_stream_capturing = lambda: True
    if blocker == "memory":
        fake.npu.mem_get_info = lambda: (candidate.MEMORY_REQUIRED_BYTES - 1, 4 << 30)
    if blocker == "memory":
        assert not candidate.run_action(worker, request(), fake)["passed"]
    else:
        with pytest.raises(RuntimeError):
            candidate.run_action(worker, request(), fake)


def test_native_failure_keeps_worker_owner_and_earliest_traceback(monkeypatch):
    worker, fake = environment(monkeypatch)
    candidate.run_action(worker, request(), fake)

    def fail(_, owner):
        owner["kernel"] = object()
        raise RuntimeError("load failed")

    monkeypatch.setattr(qsa, "load_resources", fail)
    with pytest.raises(RuntimeError, match="load failed"):
        candidate.run_action(worker, request("load"), fake)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    error, trace = state["failure"], state["failure_traceback"]
    assert state["failed"] and state["resources"]["kernel"] is not None
    with pytest.raises(RuntimeError, match="retain resources"):
        candidate.run_action(worker, request("load"), fake)
    assert state["failure"] is error and state["failure_traceback"] is trace


def forward(monkeypatch):
    calls = []

    def original(*args, **kwargs):
        calls.append(("original", args, kwargs))
        return "original"

    class Resource:
        _configs = {}

        def __call__(self, *args, **kwargs):
            pytest.fail("model binding repeated public admission")

        def launch_admitted(self, *args, **kwargs):
            calls.append(("native", args, kwargs))
            return kwargs["output"]

    resource = Resource()
    monkeypatch.setattr(qsa, "_is_capturing", lambda: False)
    monkeypatch.setattr(qsa, "geometry", lambda *a: ("key",))
    state = dict(failed=False)
    function = candidate.build_forward(original, resource, source=FIXTURE.read_text(), state=state)
    return function, resource, calls, state


def test_binding_rejects_changed_source(monkeypatch):
    with pytest.raises(RuntimeError, match="exact qualified"):
        candidate.build_forward(lambda: None, object(), source=FIXTURE.read_text() + "\n")


@pytest.mark.parametrize("cached", [False, True])
def test_binding_capture_falls_back_even_cached_without_geometry(monkeypatch, cached):
    fn, resource, calls, _ = forward(monkeypatch)
    if cached:
        resource._configs = {("key",): object()}
    monkeypatch.setattr(qsa, "_is_capturing", lambda: True)
    monkeypatch.setattr(qsa, "geometry", lambda *a: pytest.fail("capture touched geometry"))
    with torch.inference_mode():
        assert fn(None, None, None, head_dim=256, transpose_output=True) == "original"
    assert calls[0][0] == "original"


def test_binding_preserves_provided_output_and_arguments(monkeypatch):
    fn, _, calls, _ = forward(monkeypatch)
    cache, selection, table, output = object(), object(), object(), object()
    with torch.inference_mode():
        assert fn(cache, selection, table, head_dim=256, transpose_output=True, output=output) is output
    assert calls == [("native", (("key",), cache, selection, table), dict(output=output, variant=8))]


def test_binding_capacity_and_shape_fallback(monkeypatch):
    fn, resource, calls, _ = forward(monkeypatch)
    resource._configs = {i: None for i in range(qsa.MAX_CONFIGS)}
    with torch.inference_mode():
        assert fn(None, None, None, head_dim=256, transpose_output=True) == "original"
    monkeypatch.setattr(qsa, "geometry", lambda *a: None)
    with torch.inference_mode():
        assert fn(None, None, None, head_dim=256, transpose_output=True) == "original"
    assert len(calls) == 2


def full_receipt():
    cases = [
        dict(
            rows=row,
            heads=head,
            table_width=width,
            transpose=transpose,
            step=step,
            variant=variant,
            exact=True,
            inputs_preserved=True,
            output_reused=True,
        )
        for row in (1, 3, 64)
        for head in (1, 2)
        for width in (2048, 4096)
        for transpose in (False, True)
        for step in range(9)
        for variant in (4, 8)
    ]
    attention = [
        dict(heads=head, parallel=parallel, variant=variant, exact=True, inputs_preserved=True)
        for head in (1, 2)
        for parallel in (False, True)
        for variant in (4, 8)
    ]
    return dict(passed=True, cases=cases, attention=attention)


@pytest.mark.parametrize("missing", ["none", "case", "attention", "preservation", "reuse", "overall"])
def test_full_qualification_receipt_guard(missing):
    receipt = full_receipt()
    if missing == "case":
        receipt["cases"].pop()
    if missing == "attention":
        receipt["attention"].pop()
    if missing == "preservation":
        receipt["attention"][0]["inputs_preserved"] = False
    if missing == "reuse":
        receipt["cases"][0]["output_reused"] = False
    if missing == "overall":
        receipt["passed"] = False
    assert candidate._qualified_receipts(dict(validation_receipts=[receipt])) is (missing == "none")


def test_rpc_keeps_non_json_original_and_returns_ranked_error():
    original = lambda self, output: ("original", output)
    rpc = candidate.make_rpc(original)
    assert rpc(SimpleNamespace(rank=1), "profile.json") == ("original", "profile.json")
    result = rpc(SimpleNamespace(rank=1), "{}")
    assert result["rank"] == 1 and result["passed"] is False


@pytest.mark.parametrize("failed", [False, True])
def test_value_gather_always_falls_back_before_tensor_or_resource_work(monkeypatch, failed):
    fn, resource, calls, state = forward(monkeypatch)
    state["failed"] = failed
    resource._configs = None

    def forbidden(*args):
        pytest.fail("V gather entered native admission")

    monkeypatch.setattr(qsa, "geometry", forbidden)
    monkeypatch.setattr(qsa, "_is_capturing", forbidden)
    monkeypatch.setattr(torch, "is_grad_enabled", forbidden)
    cache, selection, table, output = object(), object(), object(), object()
    assert fn(cache, selection, table, head_dim=256, transpose_output=False, output=output) == "original"
    assert calls == [("original", (cache, selection, table), dict(head_dim=256, transpose_output=False, output=output))]


@pytest.mark.parametrize("extra", [{"output": "relative"}, {"output": "/tmp/profile", "repeats": 4}, {}])
def test_profile_parse_rejects_unbounded_or_missing_output(extra):
    payload = request("profile")
    payload.update(extra)
    with pytest.raises(ValueError, match="absolute output"):
        candidate.parse_request(json.dumps(payload))


def test_profile_parse_and_nonprofile_output_rejection():
    payload = request("profile")
    payload["output"] = "/tmp/profile"
    payload["repeats"] = 3
    assert candidate.parse_request(json.dumps(payload))["output"] == "/tmp/profile"
    payload["action"] = "timing"
    with pytest.raises(ValueError, match="only for QSA profile"):
        candidate.parse_request(json.dumps(payload))


def test_profile_requires_qualified_rows_before_profiler_creation(monkeypatch):
    worker, fake = environment(monkeypatch)
    candidate.run_action(worker, request(), fake)
    candidate.run_action(worker, request("load"), fake)
    payload = request("profile")
    payload["output"] = "/tmp/profile"
    with pytest.raises(RuntimeError, match="validate each"):
        candidate.run_action(worker, payload, fake)
    assert not getattr(worker, candidate.STATE_ATTRIBUTE)["failed"]


def test_profile_failure_retains_worker_traceback_and_does_not_requalify(monkeypatch):
    from tools.qwen4exp import qsa_gather_diagnostic

    worker, fake = environment(monkeypatch)
    candidate.run_action(worker, request(), fake)
    candidate.run_action(worker, request("load"), fake)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    state["validated_rows"] = {64}

    def fail(worker, state, *args, **kwargs):
        state["profile_session"] = object()
        raise RuntimeError("profile failed")

    monkeypatch.setattr(qsa_gather_diagnostic, "profile", fail)
    payload = request("profile", [64])
    payload["output"] = "/tmp/profile"
    with pytest.raises(RuntimeError, match="profile failed"):
        candidate.run_action(worker, payload, fake)
    assert state["failed"] and state["failure_traceback"] is not None
    assert state["profile_session"] is not None and not state["validation_receipts"]


def test_completed_profile_cannot_retry_or_poison_healthy_model_owner(monkeypatch):
    worker, fake = environment(monkeypatch)
    candidate.run_action(worker, request(), fake)
    candidate.run_action(worker, request("load"), fake)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    state["validated_rows"] = {64}
    state["profile_session"] = object()
    payload = request("profile", [64])
    payload["output"] = "/tmp/profile"
    with pytest.raises(RuntimeError, match="one-shot"):
        candidate.run_action(worker, payload, fake)
    assert not state["failed"]


def test_admitted_timing_rpc_requires_qualification_and_explicit_routes(monkeypatch):
    worker, fake = environment(monkeypatch)
    candidate.run_action(worker, request(), fake)
    candidate.run_action(worker, request("load"), fake)
    with pytest.raises(RuntimeError, match="validate each"):
        candidate.run_action(worker, request("timing_admitted", [64]), fake)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    state["validated_rows"] = {64}
    calls = []
    monkeypatch.setattr(qsa, "timing", lambda *a, **kw: calls.append(kw) or dict(passed=True))
    assert candidate.run_action(worker, request("timing_admitted", [64]), fake)["passed"]
    assert calls == [dict(rows=[64], repeats=1, diagnostic_routes=True)]
