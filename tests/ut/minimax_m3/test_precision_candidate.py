# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.minimax_m3.precision_candidate import fused_gemma_forward, gemma_forward, rotary_forward


def test_gemma_scale_is_not_rounded_before_normalization():
    value = torch.tensor([[0.5, 1.5]]).half()
    module = SimpleNamespace(weight=torch.full((2,), 2**-11).half(), variance_epsilon=1e-6)
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized * (1 + module.weight.float())).half()
    prematurely_rounded = (normalized * (1 + module.weight).float()).half()
    assert not torch.equal(expected, prematurely_rounded)
    torch.testing.assert_close(gemma_forward(module, value), expected, rtol=0, atol=0)
    output, residual = gemma_forward(module, value / 2, value / 2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(residual, value, rtol=0, atol=0)


def test_partial_rotary_preserves_unrotated_dimensions_and_head_order():
    module = SimpleNamespace(head_size=4, rotary_dim=2, cos_sin_cache=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).half())
    query = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]).half()
    key = query[:, :4].clone()
    q, k = rotary_forward(module, torch.tensor([1]), query, key, True)
    torch.testing.assert_close(q, torch.tensor([[-2.0, 1.0, 3.0, 4.0, -6.0, 5.0, 7.0, 8.0]]).half())
    torch.testing.assert_close(k, torch.tensor([[-2.0, 1.0, 3.0, 4.0]]).half())
    with pytest.raises(ValueError, match="NeoX"):
        rotary_forward(module, torch.tensor([1]), query, key, False)


def test_fused_gemma_normalizes_unrounded_sum_and_stores_rounded_residual():
    generator = torch.Generator().manual_seed(717)
    x = torch.randn(4, 128, generator=generator).half()
    residual = torch.randn(4, 128, generator=generator).half()
    original_x, original_residual = x.clone(), residual.clone()
    module = SimpleNamespace(weight=torch.randn(128, generator=generator).half(), variance_epsilon=1e-6)
    values = x.float() + residual.float()
    expected = (values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + module.weight.float())).half()
    actual, stored = fused_gemma_forward(module, x, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(stored, values.half(), rtol=0, atol=0)
    assert not torch.equal(actual, gemma_forward(module, x, residual)[0])
    assert torch.equal(x, original_x) and torch.equal(residual, original_residual)
    torch.testing.assert_close(fused_gemma_forward(module, x), gemma_forward(module, x), rtol=0, atol=0)
