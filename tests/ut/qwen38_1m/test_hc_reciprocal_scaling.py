# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.resident_candidates import hc_reciprocal_scaling as candidate
from tools.qwen4exp.resident_candidates import padded_hc_injection as padded
from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@torch.no_grad()
def test_power_of_two_scaling_preserves_fp16_bits(monkeypatch, count):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    # Enumerate finite half values, including signed zero and subnormals.
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = values[torch.isfinite(values)]
    before = values.clone()
    expected = values / count
    actual = candidate.scale_projection(values, count)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(values.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize(
    "count,dtype,device,grad",
    [
        (3, torch.float16, True, False),
        (4, torch.float32, True, False),
        (4, torch.float16, False, False),
        (4, torch.float16, True, True),
    ],
)
def test_unsupported_scaling_keeps_original_division(monkeypatch, count, dtype, device, grad):
    monkeypatch.setattr(candidate, "_supports_device", lambda value: device)
    values = torch.tensor([-1.5, -0.0, 0.25, 3.0], dtype=dtype)
    with torch.set_grad_enabled(grad):
        expected = values / count
        actual = candidate.scale_projection(values, count)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


@pytest.mark.parametrize("tokens", [3, 6, 2560])
@torch.no_grad()
def test_mix_and_combine_preserve_inputs_and_weights(monkeypatch, tokens):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(padded, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    module = model._GatedResidual(
        hc_count=4, hidden_size=16, lowrank=8, eps=1e-6, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(235)
    for parameter in module.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator).half() / 16)
    hyper = torch.randn(tokens, 64, generator=generator).half()
    block = torch.randn(tokens, 16, generator=generator).half()
    saved = [value.clone() for value in (hyper, block, *module.parameters())]
    expected_mix, residuals = module.mix(hyper)
    mixed, actual_residuals = candidate.mix(module, hyper)
    assert torch.equal(expected_mix.view(torch.uint8), mixed.view(torch.uint8))
    expected = padded.combine_with_prefill_addcmul(module, block, residuals)
    actual = candidate.combine(module, block, actual_residuals)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert all(torch.equal(a, b) for a, b in zip((hyper, block, *module.parameters()), saved))


def test_mix_only_module_still_rejects_combine():
    module = model._GatedResidual(
        hc_count=4,
        hidden_size=16,
        lowrank=8,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=False,
    )
    with pytest.raises(RuntimeError, match="combine disabled"):
        candidate.combine(module, None, None)
