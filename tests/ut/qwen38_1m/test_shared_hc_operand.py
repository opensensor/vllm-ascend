# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Saved hyperconnection operands preserve projection rounding and fallback."""

import pytest
import torch

from tools.qwen4exp.resident_candidates import shared_hc_operand as candidate
from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("injection_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_shared_operand_matches_original(monkeypatch, combine, injection_dtype, grad_enabled):
    # Exercise the native operand policy using CPU tensors; NPU math is checked
    # by the standalone hardware benchmark with real projection dimensions.
    policy = lambda device, weight, compute: weight
    monkeypatch.setattr(model, "_linear_operand_dtype", policy)
    monkeypatch.setattr(candidate, "_linear_operand_dtype", policy)
    torch.manual_seed(1045)
    module = model._GatedResidual(
        hc_count=2,
        hidden_size=16,
        lowrank=8,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=combine,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        if combine:
            module.block_inject_weight.data = module.block_inject_weight.data.to(injection_dtype)
    values = torch.randn(3, 32).half()
    block = torch.randn(3, 16).half()
    with torch.set_grad_enabled(grad_enabled):
        expected, expected_residual = module.mix(values)
        actual, actual_residual = candidate.mix(module, values)
        assert torch.equal(actual, expected)
        assert actual_residual[0] is values
        saved_dtype = (
            torch.float16 if combine and not grad_enabled and injection_dtype == torch.float16 else torch.float32
        )
        assert actual_residual[1].dtype == saved_dtype
        if combine:
            assert torch.equal(module.combine(block, actual_residual), module.combine(block, expected_residual))
        if grad_enabled:
            expected.sum().backward()
            gradients = [parameter.grad.clone() for parameter in module.parameters() if parameter.grad is not None]
            module.zero_grad()
            actual.sum().backward()
            assert all(
                torch.equal(actual_grad, expected_grad)
                for actual_grad, expected_grad in zip(
                    [parameter.grad for parameter in module.parameters() if parameter.grad is not None], gradients
                )
            )


def test_replacements_only_names_mix():
    assert candidate.replacements() == {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": candidate.mix}
