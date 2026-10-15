# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import hc_combine as hc
from tools.qwen4exp.resident_candidates import qfn_hc_combine as candidate


def source():
    return (Path(__file__).parent / "fixtures/qfn_pinned_hc_combine.txt").read_text()


def request(action, *, rows=(68,), build="/private/hc"):
    return candidate.parse_request(json.dumps(dict(action=action, rows=list(rows), build=build, repeats=1)))


def successful_receipt(rows):
    return dict(
        passed=True,
        scope="private test",
        cases=[dict(rows=count, step=step, exact=True, inputs_preserved=True) for count in rows for step in range(3)],
    )


def worker_setup(monkeypatch):
    worker = SimpleNamespace(rank=2, model_runner=SimpleNamespace(execute_model_state=None))
    fake_torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        device=lambda *args: "cpu",
        npu=SimpleNamespace(
            mem_get_info=lambda: (2**32, 2**35),
            current_device=lambda: 0,
            is_current_stream_capturing=lambda: False,
            synchronize=lambda: None,
        ),
    )
    monkeypatch.setattr(hc, "verify_build", lambda build: {})

    def load(build, *, owner):
        owner.update(eager_only=True, combine=lambda *args: None)

    monkeypatch.setattr(hc, "load_resources", load)
    monkeypatch.setattr(hc, "validate", lambda resource, device, *, rows, repeats: successful_receipt(rows))
    monkeypatch.setattr(hc, "timing", lambda *args, **kwargs: dict(passed=True, scope="component"))
    return worker, fake_torch


def test_private_lifecycle_qualifies_rows_only_after_final_drain(monkeypatch):
    worker, fake_torch = worker_setup(monkeypatch)
    rows = (68, 640, 2048)
    with pytest.raises(RuntimeError, match="preflight"):
        candidate.run_action(worker, request("load", rows=rows), fake_torch)
    assert candidate.run_action(worker, request("preflight", rows=rows), fake_torch)["passed"]
    assert candidate.run_action(worker, request("load", rows=rows), fake_torch)["passed"]
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    resource = state["resources"]["combine"]
    assert candidate.run_action(worker, request("load", rows=rows), fake_torch)["already_loaded"]
    with pytest.raises(RuntimeError, match="validate each"):
        candidate.run_action(worker, request("timing", rows=rows), fake_torch)

    def drain():
        assert state["validated_rows"] == set() and state["validation_receipts"] == []

    fake_torch.npu.synchronize = drain
    assert candidate.run_action(worker, request("validate", rows=rows), fake_torch)["passed"]
    fake_torch.npu.synchronize = lambda: None
    assert state["validated_rows"] == set(rows) and len(state["validation_receipts"]) == 1
    assert candidate.run_action(worker, request("timing", rows=rows), fake_torch)["passed"]
    assert state["resources"]["combine"] is resource
    with pytest.raises(RuntimeError, match="do not replace or unload"):
        candidate.run_action(worker, request("preflight", build="/replacement"), fake_torch)


@pytest.mark.parametrize("failure", ["partial_load", "native", "drain", "numerical"])
def test_private_failure_retains_resources_and_tensor_tracebacks(monkeypatch, failure):
    worker, fake_torch = worker_setup(monkeypatch)
    candidate.run_action(worker, request("preflight"), fake_torch)
    references = []

    def fail(*args, **kwargs):
        private_input = torch.ones(2)
        references.append(weakref.ref(private_input))
        if failure == "partial_load":
            kwargs["owner"]["kernel"] = object()
        raise RuntimeError("private HC native failure")

    if failure == "partial_load":
        monkeypatch.setattr(hc, "load_resources", fail)
    else:
        candidate.run_action(worker, request("load"), fake_torch)
        if failure == "native":
            monkeypatch.setattr(hc, "validate", fail)
        elif failure == "drain":
            fake_torch.npu.synchronize = fail
        else:
            monkeypatch.setattr(hc, "validate", lambda *args, **kwargs: dict(passed=False))
    action = "load" if failure == "partial_load" else "validate"
    if failure == "numerical":
        assert not candidate.run_action(worker, request(action), fake_torch)["passed"]
    else:
        with pytest.raises(RuntimeError, match="private HC native failure"):
            candidate.run_action(worker, request(action), fake_torch)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    assert state["failed"] and state["resources"] and state["validated_rows"] == set()
    first_failure, first_traceback = state.get("failure"), state.get("failure_traceback")
    gc.collect()
    assert all(reference() is not None for reference in references)
    with pytest.raises(RuntimeError, match="retain resources"):
        candidate.run_action(worker, request("load"), fake_torch)
    if failure != "numerical":
        assert state["failure"] is first_failure and state["failure_traceback"] is first_traceback


@pytest.mark.parametrize("case", ["pending", "profiler", "cpu_profile", "native_profile", "capture", "memory"])
def test_private_preflight_rejects_unsafe_or_insufficient_conditions(monkeypatch, case):
    worker, fake_torch = worker_setup(monkeypatch)
    if case == "pending":
        worker.model_runner.execute_model_state = object()
    elif case == "profiler":
        worker.profiler = object()
    elif case == "cpu_profile":
        worker._qfn_cpu_profile = object()
    elif case == "native_profile":
        worker._qfn_native_profile_session = object()
    elif case == "capture":
        fake_torch.npu.is_current_stream_capturing = lambda: True
    else:
        fake_torch.npu.mem_get_info = lambda: (0, 2**35)
        assert not candidate.run_action(worker, request("preflight"), fake_torch)["passed"]
        assert not getattr(worker, candidate.STATE_ATTRIBUTE)["resources"]
        return
    with pytest.raises(RuntimeError):
        candidate.run_action(worker, request("preflight"), fake_torch)
    assert not hasattr(worker, candidate.STATE_ATTRIBUTE)


@pytest.mark.parametrize(
    "updates",
    [
        dict(rows=[]),
        dict(rows=[True]),
        dict(rows=[[1]]),
        dict(rows=[68, 68]),
        dict(rows=[2561]),
        dict(repeats=0),
        dict(repeats=True),
        dict(build="relative"),
        dict(action="unload"),
        dict(unknown=1),
    ],
)
def test_rpc_contract_is_bounded(updates):
    with pytest.raises(ValueError):
        candidate.parse_request(json.dumps({"action": "preflight", "build": "/private", **updates}))


def test_memory_budget_covers_oracle_and_maximum_rows():
    fake = SimpleNamespace(npu=SimpleNamespace(mem_get_info=lambda: (2**32, 2**35)))
    assert candidate.memory_preflight(fake, [2048])["required_bytes"] == 768 << 20
    assert candidate.memory_preflight(fake, [2560])["required_bytes"] == 896 << 20


def model_setup(monkeypatch, *, rows=68):
    calls = []

    def linear(xn, weight, dtype):
        calls.append("projection")
        return xn[:, :4].half()

    namespace = {"torch": torch, "_linear": linear}
    exec(source(), namespace)
    original = namespace["combine"]
    model = SimpleNamespace(
        use_combine=True,
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        block_inject_weight=torch.empty(4, 10240, dtype=torch.float16),
    )
    generator = torch.Generator().manual_seed(22)
    hyper = torch.randn(rows, 10240, generator=generator, dtype=torch.float16)
    xn = torch.randn(rows, 10240, generator=generator)
    block = torch.randn(rows, 2560, generator=generator)
    captured = []

    def native(residual, update, gates):
        captured.append((residual, update, gates))
        calls.append("native")
        return hc.baseline(residual, update, gates)

    native._configs = {}
    monkeypatch.setattr(hc, "_on_npu", lambda device: True)
    monkeypatch.setattr(hc, "_is_capturing", lambda: False)
    return model, block, (hyper, xn), original, native, calls, captured


def test_exact_source_binding_preserves_fp16_injection_projection_and_sigmoid(monkeypatch):
    model, block, residuals, original, native, calls, captured = model_setup(monkeypatch)
    assert hashlib.sha256(source().encode()).hexdigest() == candidate.QUALIFIED_SOURCE_SHA256
    forward = candidate.build_forward(original, native, source=source())
    with torch.inference_mode():
        expected = original(model, block, residuals)
        calls.clear()
        actual = forward(model, block, residuals)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert calls == ["projection", "native"]
    residual, update, gates = captured[0]
    assert residual.dtype == torch.float16 and update.dtype == gates.dtype == torch.float32
    assert residual.data_ptr() == residuals[0].data_ptr() and update is block
    expected_gates = (2.0 * torch.sigmoid(residuals[1][:, :4].half() / 4)).float()
    assert torch.equal(gates, expected_gates)


@pytest.mark.parametrize("cached", [False, True])
def test_model_capture_fallback_precedes_descriptors_even_when_cached(monkeypatch, cached):
    model, block, residuals, original, native, calls, _ = model_setup(monkeypatch)
    if cached:
        native._configs[(68, block.device)] = object()
    snapshot = dict(native._configs)
    forward = candidate.build_forward(original, native, source=source())
    monkeypatch.setattr(hc, "_is_capturing", lambda: True)
    monkeypatch.setattr(candidate, "descriptors_available", lambda *args, **kwargs: pytest.fail("capture descriptors"))
    with torch.inference_mode():
        actual = forward(model, block, residuals)
    assert actual.shape == (68, 10240) and calls == ["projection"] and native._configs == snapshot


@pytest.mark.parametrize("rows", [1, 3, 6, 63])
def test_short_eager_decode_preserves_original_before_capture_or_descriptors(monkeypatch, rows):
    model, block, residuals, original, native, calls, _ = model_setup(monkeypatch, rows=rows)
    forward = candidate.build_forward(original, native, source=source())
    monkeypatch.setattr(hc, "_is_capturing", lambda: pytest.fail("short call capture query"))
    with torch.inference_mode():
        assert forward(model, block, residuals).shape == (rows, 10240)
    assert calls == ["projection"]


def test_capacity_fallback_and_dynamic_projection_global(monkeypatch):
    model, block, residuals, original, native, calls, captured = model_setup(monkeypatch)
    forward = candidate.build_forward(original, native, source=source())
    monkeypatch.setattr(hc, "MAX_CONFIGS", 1)
    native._configs[(99, block.device)] = object()
    with torch.inference_mode():
        forward(model, block, residuals)
    assert calls == ["projection"] and not captured
    native._configs.clear()

    def new_linear(xn, weight, dtype):
        calls.append("new_projection")
        return torch.zeros(xn.shape[0], 4, dtype=torch.float16)

    original.__globals__["_linear"] = new_linear
    with torch.inference_mode():
        forward(model, block, residuals)
    assert calls[-2:] == ["new_projection", "native"]


def test_model_native_failure_retains_state_and_will_not_retry(monkeypatch):
    model, block, residuals, original, native, _, _ = model_setup(monkeypatch)
    state = dict(failed=False)
    refs = []

    def fail(residual, update, gates):
        refs.append(weakref.ref(gates))
        raise RuntimeError("HC model launch failed")

    fail._configs = {}
    forward = candidate.build_forward(original, fail, source=source(), state=state)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="launch failed"):
        forward(model, block, residuals)
    gc.collect()
    assert state["failed"] and refs[0]() is not None
    traceback = state["failure_traceback"]
    with torch.inference_mode(), pytest.raises(RuntimeError, match="resource failed"):
        forward(model, block, residuals)
    assert state["failure_traceback"] is traceback


@pytest.mark.parametrize(
    "case", ["pending", "failed", "unloaded", "rows", "marker", "receipt", "inexact", "missing_step"]
)
def test_model_binding_requires_complete_healthy_numerical_receipts(monkeypatch, case):
    worker, fake_torch = worker_setup(monkeypatch)
    rows = tuple(candidate.REQUIRED_ROWS)
    for action in ("preflight", "load", "validate"):
        candidate.run_action(worker, request(action, rows=rows), fake_torch)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    if case == "pending":
        worker.model_runner.execute_model_state = object()
    elif case == "failed":
        state["failed"] = True
    elif case == "unloaded":
        state["loaded"] = False
    elif case == "rows":
        state["validated_rows"].remove(640)
    elif case == "marker":
        state["resources"]["eager_only"] = False
    elif case == "receipt":
        state["validation_receipts"] = []
    elif case == "inexact":
        state["validation_receipts"][0]["cases"][0]["exact"] = False
    else:
        state["validation_receipts"][0]["cases"].pop()
    monkeypatch.setattr(candidate, "_hc_type", lambda: pytest.fail("unqualified binding"))
    with pytest.raises(RuntimeError):
        candidate.make_replacements(worker)


def test_binding_factory_borrows_qualified_owner_and_rpc_keeps_profile_delegation(monkeypatch):
    worker, fake_torch = worker_setup(monkeypatch)
    for action in ("preflight", "load", "validate"):
        candidate.run_action(worker, request(action, rows=tuple(candidate.REQUIRED_ROWS)), fake_torch)
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    method = object()
    monkeypatch.setattr(candidate, "_hc_type", lambda: SimpleNamespace(combine=method))
    monkeypatch.setattr(candidate, "build_forward", lambda original, resource, *, state: (original, resource, state))
    result = candidate.make_replacements(worker)
    assert result[candidate.MODEL_TARGET] == (method, state["resources"]["combine"], state)
    calls = []
    rpc = candidate.make_rpc(lambda self, output: calls.append(output))
    rpc(worker, "/profile/path")
    assert calls == ["/profile/path"]
    monkeypatch.setattr(candidate, "run_action", lambda *args: dict(passed=True, scope="private"))
    receipt = rpc(worker, json.dumps(dict(action="preflight", build="/private")))
    assert receipt == dict(rank=2, action="preflight", passed=True, scope="private")


def test_source_drift_is_rejected():
    with pytest.raises(RuntimeError, match="exact qualified"):
        candidate.build_forward(lambda *args: None, None, source=source().replace("2.0 *", "1.0 *"))
