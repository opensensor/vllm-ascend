# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import w4_activation_pack as packing


def pack_reference(value):
    groups = value.float().reshape(value.shape[0], -1, 128)
    maximum = groups.abs().amax(-1)
    scale = torch.where(maximum == 0, 1, maximum / 127)
    quant = (groups / scale[..., None]).round().clamp(-127, 127)
    high = torch.floor(quant / 16)
    low = quant - high * 16 - 8

    def limbs(tensor):
        integer = tensor.to(torch.int32).reshape(value.shape)
        return ((integer[:, ::2] & 15) | ((integer[:, 1::2] & 15) << 4)).to(torch.int8)

    return (
        limbs(low),
        limbs(high),
        scale[..., None].expand(-1, -1, 8).contiguous(),
        quant.sum(-1)[..., None].expand(-1, -1, 8).contiguous(),
    )


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(packing, "MIN_ROWS", 2)
    monkeypatch.setattr(packing, "MAX_ROWS", 8)
    monkeypatch.setattr(packing.epilogue, "_on_npu", lambda _: True)
    monkeypatch.setattr(packing.epilogue, "_capturing", lambda: False)
    monkeypatch.setattr(packing.epilogue, "_dense_format", lambda _: True)
    launches, projections = [], []

    def launch(kernel, tensors, blocks):
        activated, projected, *outputs, descriptor = tensors
        assert descriptor.tolist() == [projected.shape[0]] and blocks == 8
        launches.append(tensors)
        reference = pack_reference((activated * projected[:, 640:].float()).half())
        for output, expected in zip(outputs, reference):
            output.copy_(expected)

    def native(prepared, ends):
        projections.append((prepared, ends))
        return prepared

    projection = SimpleNamespace(
        native_linear=native, grouped_linear=lambda value, ends: native(pack_reference(value), ends)
    )
    owner = packing.W4ActivationPack(object(), launch, packing.epilogue.reference, pack_reference)
    return owner, projection, launches, projections


@torch.no_grad()
def test_changed_inputs_reference_returning_shadow_then_candidate(setup):
    owner, projection, launches, projections = setup
    ends = torch.tensor([1, 3])
    for step in range(3):
        value = torch.randn(3, 1280).half()
        original = value.clone()
        expected = pack_reference(packing.epilogue.reference(value, torch.float32, torch.float16))
        owner.verify = step < 2
        actual = owner(value, projection, ends, torch.float32, torch.float16)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert torch.equal(value, original) and projections[-1][1] is ends
        assert (actual[0] is launches[-1][2]) == (step == 2)
    assert owner.checked == 2 and owner.mismatches == 0 and owner.calls == 3
    assert all(launch[-1] is launches[0][-1] for launch in launches)
    assert launches[0][2].data_ptr() != launches[1][2].data_ptr()


@pytest.mark.parametrize(
    "condition",
    ["small", "large", "dtype", "width", "strided", "capture", "cpu", "format", "off", "compute", "params", "grad"],
)
def test_fallback_uses_retained_activation_and_grouped_projection(setup, monkeypatch, condition):
    owner, projection, launches, _ = setup
    value = torch.randn(3, 1280).half()
    compute, params = torch.float32, torch.float16
    if condition == "small":
        value = value[:1]
    elif condition == "large":
        value = value.repeat(3, 1)
    elif condition == "dtype":
        value = value.float()
    elif condition == "width":
        value = value[:, :512].contiguous()
    elif condition == "strided":
        value = torch.randn(3, 2560).half()[:, ::2]
    elif condition == "capture":
        monkeypatch.setattr(packing.epilogue, "_capturing", lambda: True)
    elif condition == "cpu":
        monkeypatch.setattr(packing.epilogue, "_on_npu", lambda _: False)
    elif condition == "format":
        monkeypatch.setattr(packing.epilogue, "_dense_format", lambda _: False)
    elif condition == "off":
        owner.enabled = False
    elif condition == "compute":
        compute = torch.float16
    elif condition == "params":
        params = torch.float32
    with torch.set_grad_enabled(condition == "grad"):
        result = owner(value, projection, None, compute, params)
    expected = pack_reference(packing.epilogue.reference(value, compute, params))
    assert all(torch.equal(a, b) for a, b in zip(result, expected))
    assert not launches and not owner.configs and owner.fallbacks == 1


@pytest.mark.parametrize("output", range(4))
@torch.no_grad()
def test_any_packed_output_mismatch_disables_and_returns_reference(setup, output):
    owner, projection, _, _ = setup
    launch = owner.launch

    def corrupt(kernel, tensors, blocks):
        launch(kernel, tensors, blocks)
        tensors[output + 2].fill_(42)

    owner.launch = corrupt
    value = torch.randn(3, 1280).half()
    expected = pack_reference(packing.epilogue.reference(value, torch.float32, torch.float16))
    for _ in range(2):
        actual = owner(value, projection, None, torch.float32, torch.float16)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert not owner.enabled and owner.checked == owner.mismatches == owner.calls == owner.fallbacks == 1


@torch.no_grad()
def test_launch_failure_retains_owner_and_prevents_reuse(setup):
    owner, projection, _, _ = setup
    failure = RuntimeError("submission failed")

    def fail(*args):
        raise failure

    owner.launch = fail
    value = torch.ones(3, 1280).half()
    with pytest.raises(RuntimeError, match="submission failed"):
        owner(value, projection, None, torch.float32, torch.float16)
    assert owner.failed and owner.failure is failure and owner.configs
    with pytest.raises(RuntimeError, match="retain native resources"):
        owner(value, projection, None, torch.float32, torch.float16)


@pytest.fixture
def local_setup(setup):
    owner, projection, launches, projections = setup
    full_launch = owner.launch
    local_kernel = object()

    def launch(kernel, tensors, blocks):
        if kernel is not local_kernel:
            return full_launch(kernel, tensors, blocks)
        activated, projected, *outputs, ends, descriptor = tensors
        launches.append(tensors)
        active = int(ends[-1])
        expected = pack_reference((activated * projected[:, 640:].float()).half())
        for output, values in zip(outputs, expected):
            output.fill_(101 if output.dtype == torch.int8 else float("nan"))
            output[:active].copy_(values[:active])

    def native(prepared, ends):
        projections.append((prepared, ends))
        result = torch.zeros((prepared[0].shape[0], 320))
        active = int(ends[-1])
        result[:active] = prepared[0][:active].float()
        return result

    projection.native_linear = native
    owner.launch = launch
    owner.local_kernel = local_kernel
    owner.local_projection = lambda bank, prepared, ends: bank is projection
    owner.local_routes = True
    return owner, projection, launches, projections


@pytest.mark.parametrize("active", [0, 1, 3])
@torch.no_grad()
def test_local_prefix_poisoned_tail_and_reference_projection(local_setup, active, monkeypatch):
    owner, projection, launches, projections = local_setup
    value = torch.randn(3, 1280).half()
    ends = torch.tensor([(i + 1) * active // 128 for i in range(128)])
    expected = projection.native_linear(
        pack_reference(packing.epilogue.reference(value, torch.float32, torch.float16)), ends
    )
    actual = owner(value, projection, ends, torch.float32, torch.float16)
    assert torch.equal(actual, expected)
    assert (launches[-1][2][active:] == 101).all() and torch.isnan(launches[-1][4][active:]).all()
    assert owner.checked == owner.local_calls == 1 and owner.mismatches == 0
    assert owner.checked_capacity_rows == 3 and owner.checked_local_rows == active
    assert sum(owner.expert_row_histogram.values()) == 128
    assert actual is not launches[-1][2]
    before = len(projections)
    owner.verify = False

    def no_readback(*args, **kwargs):
        raise AssertionError("serving must not read device route counts on the host")

    monkeypatch.setattr(torch.Tensor, "cpu", no_readback)
    actual = owner(value, projection, ends, torch.float32, torch.float16)
    assert torch.equal(actual, expected) and len(projections) == before + 1
    assert owner.checked == 1 and owner.local_calls == 2


@pytest.mark.parametrize("case", ["off", "consumer", "dtype", "shape", "stride"])
@torch.no_grad()
def test_local_contract_failure_keeps_full_pack(local_setup, case):
    owner, projection, launches, _ = local_setup
    ends = torch.full((128,), 3, dtype=torch.int64)
    if case == "off":
        owner.local_routes = False
    elif case == "consumer":
        owner.local_projection = lambda *args: False
    elif case == "dtype":
        ends = ends.int()
    elif case == "shape":
        ends = ends[:64]
    else:
        ends = torch.full((256,), 3, dtype=torch.int64)[::2]
    owner.verify = False
    owner(torch.randn(3, 1280).half(), projection, ends, torch.float32, torch.float16)
    assert len(launches[-1]) == 7 and owner.local_calls == 0 and owner.calls == 1


@torch.no_grad()
def test_local_shadow_rejects_consumer_that_reads_poisoned_tail(local_setup):
    owner, projection, _, _ = local_setup
    projection.native_linear = lambda prepared, ends: prepared[2].clone()
    actual = owner(
        torch.randn(3, 1280).half(), projection, torch.ones(128, dtype=torch.int64), torch.float32, torch.float16
    )
    assert torch.isfinite(actual).all()
    assert owner.mismatches == 1 and not owner.enabled


def test_local_kernel_requires_explicit_consumer_contract():
    with pytest.raises(ValueError, match="qualified consumer"):
        packing.W4ActivationPack(None, None, None, None, local_kernel=object())


@pytest.mark.parametrize("condition", ["ready", "changed", "ineligible", "capture", "full", "cached"])
def test_local_projection_admission_matches_consumer_fallback(monkeypatch, condition):
    class Bank:
        def native_linear(self, prepared, ends):
            raise AssertionError("metadata admission must not execute projection")

    bank = Bank()
    bank.weight = torch.empty((128, 2560, 320), device="meta")
    consumer = Bank.native_linear
    prepared = (torch.empty((2560, 320)),)
    key = ((2560, 128, 2560, 640, 8, 0, 1), prepared[0].device)
    resource = SimpleNamespace(_configs={})
    monkeypatch.setattr(packing.qfn_prefill_schedule, "eligible", lambda *a: condition != "ineligible")
    monkeypatch.setattr(packing.native_prefill, "_capturing", lambda _: condition == "capture")
    monkeypatch.setattr(packing.native_prefill, "MAX_CONFIGS", 1)
    if condition == "changed":
        Bank.native_linear = lambda *a: None
    if condition == "full":
        resource._configs["another shape"] = object()
    if condition == "cached":
        resource._configs[key] = object()
    before = dict(resource._configs)
    guard = packing.local_projection_guard(consumer, resource, "native")
    assert guard(bank, prepared, None) == (condition in ("ready", "cached"))
    assert resource._configs == before
