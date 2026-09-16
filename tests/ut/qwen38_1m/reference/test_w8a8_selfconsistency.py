# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the W8A8 dynamic INT8 QDQ reference.

Acceptance (T0.6): W8A8 QDQ round-trip within the declared tolerance; correct
offset application order.
"""

import pytest
import torch

from tests.ut.qwen38_1m.reference.tolerances import (
    W8A8_GEMM_ATOL,
    W8A8_GEMM_RTOL,
    W8A8_LINEAR_REL,
    W8A8_ROUNDTRIP_EPS,
)
from tests.ut.qwen38_1m.reference.w8a8_reference import (
    INT8_MAX,
    INT8_MIN,
    compute_weight_qparams,
    dequantize_per_token,
    dequantize_weight,
    quantize_per_token_int8,
    quantize_weight_int8,
    w8a8_dynamic_linear,
)


def _rand(shape, seed, scale=3.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen) * scale


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(1, 8), (5, 64), (17, 256), (33, 128)])
def test_activation_roundtrip(seed, shape):
    x = _rand(shape, seed)
    q, scale = quantize_per_token_int8(x)
    assert q.dtype == torch.int8
    assert torch.all(q >= INT8_MIN) and torch.all(q <= INT8_MAX)
    x_rec = dequantize_per_token(q, scale)
    # Tight, correct bound: |err| <= scale/2 (half a grid step) per element.
    assert torch.all((x_rec - x).abs() <= scale / 2 + W8A8_ROUNDTRIP_EPS)


def test_activation_scale_is_per_token():
    """Each row is scaled independently by its own amax."""
    x = torch.tensor([[1.0, -2.0, 0.5], [100.0, -50.0, 25.0]])
    _, scale = quantize_per_token_int8(x)
    assert scale.shape == (2, 1)
    torch.testing.assert_close(scale[0, 0], torch.tensor(2.0 / 127.0))
    torch.testing.assert_close(scale[1, 0], torch.tensor(100.0 / 127.0))


def test_zero_row_is_safe():
    x = torch.zeros(3, 16)
    q, scale = quantize_per_token_int8(x)
    assert torch.all(q == 0)
    assert torch.all(dequantize_per_token(q, scale) == 0)


@pytest.mark.parametrize("seed", [10, 11, 12])
@pytest.mark.parametrize("shape", [(8, 32), (64, 128), (256, 64)])
def test_weight_roundtrip_offset_order(seed, shape):
    w = _rand(shape, seed)
    scale, offset = compute_weight_qparams(w)
    q = quantize_weight_int8(w, scale, offset)
    assert torch.all(q >= INT8_MIN) and torch.all(q <= INT8_MAX)
    w_rec = dequantize_weight(q, scale, offset)
    # Asymmetric int8 grid: error bounded by scale/2 per element.
    assert torch.all((w_rec - w).abs() <= scale / 2 + W8A8_ROUNDTRIP_EPS)


def test_wrong_offset_order_regresses():
    """Guard the offset order: the *wrong* order must NOT reconstruct.

    Using ``(q + offset) * scale`` (offset added instead of subtracted) is a
    plausible bug; it must produce a materially different result.
    """
    w = _rand((16, 48), seed=99)
    scale, offset = compute_weight_qparams(w)
    q = quantize_weight_int8(w, scale, offset)
    correct = dequantize_weight(q, scale, offset)
    wrong = (q.to(torch.float32) + offset) * scale
    # Where offset != 0 the two disagree well beyond quantization noise.
    nonzero_offset = offset.abs().squeeze(-1) > 0.5
    assert nonzero_offset.any()
    diff = (correct - wrong).abs()[nonzero_offset]
    assert diff.max() > scale.squeeze(-1)[nonzero_offset].max()


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_w8a8_dynamic_linear_is_consistent(seed):
    """The QDQ linear equals its definition: dequant(quant(x)) @ dequant(w).T."""
    x = _rand((12, 128), seed)
    w = _rand((64, 128), seed + 500, scale=0.5)
    scale, offset = compute_weight_qparams(w)
    q_w = quantize_weight_int8(w, scale, offset)

    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a)
    w_deq = dequantize_weight(q_w, scale, offset)
    y_ref = x_deq @ w_deq.t()

    y_qdq = w8a8_dynamic_linear(x, q_w, scale, offset)
    torch.testing.assert_close(y_qdq, y_ref, rtol=W8A8_GEMM_RTOL, atol=W8A8_GEMM_ATOL)


def test_w8a8_linear_close_to_true_fp():
    """End-to-end QDQ linear stays near the un-quantized FP GEMM."""
    x = _rand((8, 256), seed=31, scale=1.0)
    w = _rand((32, 256), seed=32, scale=0.3)
    scale, offset = compute_weight_qparams(w)
    q_w = quantize_weight_int8(w, scale, offset)

    y_true = x @ w.t()
    y_qdq = w8a8_dynamic_linear(x, q_w, scale, offset)
    rel = (y_qdq - y_true).norm() / y_true.norm()
    assert rel < W8A8_LINEAR_REL
