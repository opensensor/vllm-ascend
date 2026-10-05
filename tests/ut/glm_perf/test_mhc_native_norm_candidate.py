# SPDX-License-Identifier: Apache-2.0
"""Preserve the mHC normalization precision and caller-owned tensors."""

import pytest
import torch

from tools.glm_perf.resident_candidates.mhc_native_norm import native_norm


@pytest.mark.parametrize("rows", [2, 8, 640])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_input_precision_and_final_dtype(rows, dtype):
    generator = torch.Generator().manual_seed(310)
    x = torch.randn(rows, 4096, generator=generator) * 100000
    weight = torch.randn(4096, generator=generator).to(dtype)
    saved_x, saved_weight = x.clone(), weight.clone()
    eps = 1e-5
    calls = []

    def operation(value, gamma, epsilon):
        calls.append((value, gamma, epsilon))
        rstd = torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
        return value * rstd * gamma, rstd

    result = native_norm(x, weight, eps, operation)
    expected = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert calls[0][0].data_ptr() == x.data_ptr()
    assert calls[0][0].dtype == calls[0][1].dtype == torch.float32
    assert calls[0][2] == eps
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, saved_weight, rtol=0, atol=0)
