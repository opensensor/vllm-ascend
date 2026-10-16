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
