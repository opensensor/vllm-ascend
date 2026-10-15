# SPDX-License-Identifier: Apache-2.0
"""Independent regression gates for private numerical coverage and ownership."""

import contextlib
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import benchmark_route_finalize_310 as benchmark
from tools.qwen4exp import route_finalize
from tools.qwen4exp.resident_candidates import qfn_w4_prefill_probe as probe


def test_one_repeat_still_tests_nonzero_special_values_mixed_and_all_local():
    cases = []

    def resource(routed, inverse, weights):
        cases.append((routed.clone(), inverse.clone(), weights.clone()))
        return route_finalize.simulation(routed, inverse, weights)

    receipt = route_finalize.validate(resource, "cpu", rows=(10,), repeats=1)
    assert receipt["passed"]
    assert [case["local_rows"] for case in receipt["cases"]] == [0, 5, 10]
    assert len(cases) == 3
    assert torch.count_nonzero(cases[0][0]) == 0
    assert torch.count_nonzero(cases[1][0][5:]) == 0
    assert torch.count_nonzero(cases[2][0][-1]) > 0
    special = torch.tensor([-0.0, 2**-24, -(2**-24), 65504], dtype=torch.float16)
    for routed, _, _ in cases[1:]:
        assert torch.equal(routed[0, :4].view(torch.int16), special.view(torch.int16))
    assert not torch.equal(cases[1][1], cases[2][1])
    assert not torch.equal(cases[1][2], cases[2][2])


def test_all_zero_kernel_cannot_qualify_even_with_one_repeat():
    receipt = route_finalize.validate(
        lambda routed, inverse, weights: torch.zeros_like(routed, dtype=torch.float32),
        "cpu",
        rows=(10,),
        repeats=1,
    )
    assert not receipt["passed"]
    assert receipt["cases"][0]["weighted_exact"]
    assert not receipt["cases"][1]["weighted_exact"]


@pytest.mark.parametrize("rows,repeats", [((), 1), ((10,), 0), ((10,), True), ((11,), 1)])
def test_empty_or_invalid_validation_cannot_return_vacuous_success(rows, repeats):
    with pytest.raises(ValueError):
        route_finalize.validate(None, "cpu", rows=rows, repeats=repeats)


def loaded_worker(monkeypatch):
    state = {
        "build": "/private/frozen",
        "resources": {"finalize": object()},
        "failed": False,
        "loaded": True,
        "preflight": 650,
        "validated_rows": {650},
        "schedule_receipts": [],
    }
    worker = SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=None))
    setattr(worker, probe.STATE_ATTRIBUTE, state)
    fake_torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        npu=SimpleNamespace(
            mem_get_info=lambda: (2**32, 2**35),
            is_current_stream_capturing=lambda: False,
            synchronize=lambda: None,
        ),
    )
    request = dict(action="validate", build=state["build"], rows=[650], repeats=1, projection="down_proj")
    return worker, state, fake_torch, request


def test_native_failure_retains_private_tensor_frames_and_retry_cannot_replace_them(monkeypatch):
    worker, state, fake_torch, request = loaded_worker(monkeypatch)
    references = []

    def validation(*args):
        private_input = torch.ones(10)
        private_output = torch.empty_like(private_input)
        references.extend((weakref.ref(private_input), weakref.ref(private_output)))
        raise RuntimeError("uncertain native submission")

    monkeypatch.setattr(probe, "run_validation", validation)
    with pytest.raises(RuntimeError, match="uncertain native"):
        probe.run_action(worker, request, fake_torch)
    gc.collect()
    assert state["failed"] and all(reference() is not None for reference in references)
    first_traceback = state["failure_traceback"]
    first_failure = state["failure"]
    with pytest.raises(RuntimeError, match="retain resources"):
        probe.run_action(worker, request, fake_torch)
    gc.collect()
    assert state["failure_traceback"] is first_traceback and state["failure"] is first_failure
    assert all(reference() is not None for reference in references)


def test_warmup_output_survives_synchronization_failure(monkeypatch):
    worker, state, fake_torch, request = loaded_worker(monkeypatch)
    references = []

    def baseline(*args):
        output = torch.ones(10)
        references.append(weakref.ref(output))
        return output

    def fail_sync():
        assert references[0]() is not None
        raise RuntimeError("private stream failed")

    monkeypatch.setattr(benchmark, "baseline", baseline)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=fail_sync), raising=False)
    monkeypatch.setattr(
        probe,
        "run_validation",
        lambda *args: benchmark.time_finalization({}, None, None, None, None, repeats=1),
    )
    with pytest.raises(RuntimeError, match="private stream"):
        probe.run_action(worker, request, fake_torch)
    gc.collect()
    assert references[0]() is not None and state["failed"]


def test_active_native_profiler_rejected_before_any_gate_submission(monkeypatch):
    worker, state, fake_torch, request = loaded_worker(monkeypatch)
    worker._qfn_native_profile_session = object()
    with pytest.raises(RuntimeError, match="profilers stopped"):
        probe.run_action(worker, request, fake_torch)
    assert not state["failed"]


def test_schedule_receipt_published_only_after_successful_final_drain(monkeypatch):
    worker, state, fake_torch, request = loaded_worker(monkeypatch)
    request["action"] = "schedule"
    receipt = dict(passed=True, projection="down_proj", records=[dict(rows=650, local_rows=0, skew=False)])
    monkeypatch.setattr(probe, "run_validation", lambda *args: receipt)

    def drain():
        assert state["schedule_receipts"] == []

    fake_torch.npu.synchronize = drain
    assert probe.run_action(worker, request, fake_torch) is receipt
    assert state["schedule_receipts"] == [receipt] and not state["failed"]


def test_timing_inverts_integer_permutation_on_cpu_before_both_transfers(monkeypatch):
    operations = []

    class Tensor:
        def __init__(self, value, device="cpu"):
            self.value, self.device = value, device

        def to(self, device):
            operations.append(("transfer", self.value.dtype, self.device, device))
            return Tensor(self.value, device)

    def argsort(value):
        assert value.device == "cpu", "integer NPU argsort would trigger the AiCPU fallback"
        operations.append(("argsort", value.value.dtype, value.device))
        return Tensor(torch.argsort(value.value))

    fake_torch = SimpleNamespace(
        device=lambda kind, index: f"{kind}:{index}",
        npu=SimpleNamespace(current_device=lambda: 0),
        Generator=torch.Generator,
        float16=torch.float16,
        randn=lambda *args, **kwargs: Tensor(torch.randn(*args, **kwargs)),
        randperm=lambda *args, **kwargs: Tensor(torch.randperm(*args, **kwargs)),
        rand=lambda *args, **kwargs: Tensor(torch.rand(*args, **kwargs)),
        argsort=argsort,
    )

    def time(resources, routed, order, inverse, weights, *, repeats):
        assert all(value.device == "npu:0" for value in (routed, order, inverse, weights))
        assert torch.equal(order.value[inverse.value], torch.arange(order.value.numel()))
        return []

    monkeypatch.setattr(benchmark, "time_finalization", time)
    receipt = probe.run_validation(None, {}, dict(action="timing", rows=[10], repeats=1), fake_torch)
    assert receipt["passed"]
    integer_operations = [entry for entry in operations if entry[1] == torch.int64]
    assert integer_operations == [
        ("argsort", torch.int64, "cpu"),
        ("transfer", torch.int64, "cpu", "npu:0"),
        ("transfer", torch.int64, "cpu", "npu:0"),
    ]
