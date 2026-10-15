# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row padding preserves the HC contract; device dispatch still needs a gate."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp.resident_candidates import hc_projection_layout as candidate
from vllm_ascend.models.qwen4_exp import model


def make_module():
    module = model._GatedResidual(
        hc_count=4, hidden_size=16, lowrank=16, eps=1e-6, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    return module


@pytest.mark.parametrize("rows", [1, 3, 6, 15])
@torch.no_grad()
def test_padding_refreshes_rows_without_mutating_inputs_or_weights(rows):
    torch.manual_seed(123)
    down, up = torch.randn(16, 64).half() / 16, torch.randn(64, 16).half() / 16
    before = (down.clone(), up.clone())
    for scale in (0.0, 0.1, 4.0):
        values = torch.randn(rows, 64) * scale
        saved = values.clone()
        expected = torch.sigmoid(F.linear(F.silu(F.linear(values.half(), down) / 4), up))
        actual = candidate.padded_gate(values, down, up, 4)
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        assert actual.shape == (rows, 64)
        assert torch.equal(values, saved)
        assert all(torch.equal(a, b) for a, b in zip(before, (down, up)))


@pytest.mark.parametrize("rows", [0, 16, 17])
def test_reject_invalid_padding_rows(rows):
    with pytest.raises(ValueError, match="1 through 15"):
        candidate.padded_gate(torch.zeros(rows, 64), torch.zeros(16, 64), torch.zeros(64, 16), 4)


@pytest.mark.parametrize("rows", [3, 6, 16])
@pytest.mark.parametrize("share", [False, True])
def test_cpu_reference_preserves_output_residual_and_gradient(rows, share):
    torch.manual_seed(124)
    module = make_module()
    module.share_projection_operand = share
    values = torch.randn(rows, 64).half().requires_grad_(True)
    expected, residual = module.mix(values)
    actual, actual_residual = candidate.mix(module, values)
    assert torch.equal(actual, expected)
    assert actual_residual[0] is values
    assert torch.equal(actual_residual[1], residual[1])
    expected.sum().backward()
    gradient = values.grad.clone()
    values.grad = None
    actual.sum().backward()
    assert torch.equal(values.grad, gradient)


@torch.no_grad()
def test_fast_path_keeps_normalization_and_injection_operands(monkeypatch):
    torch.manual_seed(125)
    module = make_module()
    # Validate composition on CPU with NPU's FP16 operand policy. Exact actual
    # 10240<->320 dimensions and CANN accumulation are gated on hardware.
    monkeypatch.setattr(candidate, "eligible", lambda module, normalized: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    for _ in range(3):
        values, block = torch.randn(3, 64).half(), torch.randn(3, 16).half()
        expected, residual = module.mix(values)
        actual, actual_residual = candidate.mix(module, values)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_residual[1], residual[1])
        assert actual_residual[1].dtype == torch.float32
        assert torch.equal(module.combine(block, actual_residual), module.combine(block, residual))


def test_only_mix_is_replaced():
    assert candidate.replacements() == {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": candidate.mix}


@pytest.mark.parametrize("rows", [1, 3, 6, 9, 16, 2560])
@torch.no_grad()
def test_only_qualified_geometry_is_eligible(monkeypatch, rows):
    normalized = torch.empty(rows, candidate.HC_WIDTH, dtype=torch.float32, device="meta")
    module = SimpleNamespace(
        hc_count=4,
        hidden_size=2560,
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        input_mix_weight_down=torch.empty(320, 10240, dtype=torch.float16, device="meta"),
        input_mix_weight_up=torch.empty(10240, 320, dtype=torch.float16, device="meta"),
    )
    assert not candidate.eligible(module, normalized)
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    assert candidate.eligible(module, normalized) == (rows in candidate.DECODE_ROWS)
    module.input_mix_weight_up = module.input_mix_weight_up.float()
    assert not candidate.eligible(module, normalized)
