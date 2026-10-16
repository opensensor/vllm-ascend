# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.qwen4exp import w4_activation_epilogue as activation


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(activation, "MIN_ROWS", 2)
    monkeypatch.setattr(activation, "MAX_ROWS", 8)
    monkeypatch.setattr(activation, "_on_npu", lambda _: True)
    monkeypatch.setattr(activation, "_capturing", lambda: False)
    monkeypatch.setattr(activation, "_dense_format", lambda _: True)
    calls = []

    def launch(kernel, tensors, blocks):
        activated, projected, output, config = tensors
        assert blocks == 8 and config.tolist() == [projected.shape[0]]
        calls.append(tensors)
        output.copy_((activated * projected[:, activation.WIDTH :].float()).half())

    return activation.W4ActivationEpilogue(object(), launch), calls


@torch.no_grad()
def test_exact_changed_inputs_distinct_storage_and_descriptor_reuse(owner):
    op, calls = owner
    generator = torch.Generator().manual_seed(14)
    outputs = []
    for _ in range(2):
        values = torch.randn(3, 1280, generator=generator).half()
        before = values.clone()
        expected = activation.reference(values, torch.float32, torch.float16)
        outputs.append(op(values, torch.float32, torch.float16))
        assert torch.equal(outputs[-1], expected) and torch.equal(values, before)
        assert outputs[-1].data_ptr() != values.data_ptr()
    assert calls[0][-1] is calls[1][-1]
    assert calls[0][2].data_ptr() != calls[1][2].data_ptr()
    assert op.checked == 2 and op.mismatches == 0
    op.verify = False
    result = op(values, torch.float32, torch.float16)
    assert result is calls[-1][2] and torch.equal(result, expected)
    assert op.checked == 2


@pytest.mark.parametrize(
    "condition", ["small", "large", "width", "dtype", "strided", "cpu", "capture", "format", "off"]
)
@torch.no_grad()
def test_unsupported_inputs_preserve_reference_without_launch(owner, monkeypatch, condition):
    op, calls = owner
    value = torch.randn(3, 1280).half()
    if condition == "small":
        value = value[:1]
    elif condition == "large":
        value = value.repeat(3, 1)
    elif condition == "width":
        value = value[:, :640].contiguous()
    elif condition == "dtype":
        value = value.float()
    elif condition == "strided":
        value = torch.randn(3, 2560).half()[:, ::2]
    elif condition == "cpu":
        monkeypatch.setattr(activation, "_on_npu", lambda _: False)
    elif condition == "capture":
        monkeypatch.setattr(activation, "_capturing", lambda: True)
    elif condition == "format":
        monkeypatch.setattr(activation, "_dense_format", lambda _: False)
    else:
        op.enabled = False
    expected = activation.reference(value, torch.float32, torch.float16)
    assert torch.equal(op(value, torch.float32, torch.float16), expected)
    assert not calls and not op.configs and op.fallbacks == 1


def test_autograd_uses_original_expression(owner):
    op, calls = owner
    value = torch.ones(3, 1280, dtype=torch.float16, requires_grad=True)
    with torch.enable_grad():
        output = op(value, torch.float32, torch.float16)
        output.sum().backward()
    assert value.grad is not None and not calls


@torch.no_grad()
def test_shadow_mismatch_returns_reference_and_disables_candidate(owner):
    op, calls = owner
    op.launch = lambda kernel, tensors, blocks: tensors[2].fill_(123)
    value = torch.ones(3, 1280).half()
    expected = activation.reference(value, torch.float32, torch.float16)
    assert torch.equal(op(value, torch.float32, torch.float16), expected)
    assert not op.enabled and op.mismatches == 1 and op.checked == 1
    assert torch.equal(op(value, torch.float32, torch.float16), expected)
    assert op.calls == 1 and op.fallbacks == 1


@torch.no_grad()
def test_failed_native_submission_poisoned_and_retained(owner):
    op, _ = owner
    failure = RuntimeError("submission failed")

    def fail(*args):
        raise failure

    op.launch = fail
    value = torch.ones(3, 1280).half()
    with pytest.raises(RuntimeError, match="submission failed"):
        op(value, torch.float32, torch.float16)
    assert op.failed and op.failure is failure and op.configs
    with pytest.raises(RuntimeError, match="retain native resources"):
        op(value, torch.float32, torch.float16)
