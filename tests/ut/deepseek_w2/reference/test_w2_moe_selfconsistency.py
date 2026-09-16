# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the W2 QDQ MoE reference (E0.4, priority 1).

Acceptance:
  * W2 pack -> unpack -> QDQ MoE round-trips within the declared bound.
  * grouped == per-token reference.
  * skew (all tokens -> one expert) holds.
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.tolerances import (
    W2_LINEAR_ATOL,
    W2_LINEAR_RTOL,
    W2_MOE_ATOL,
    W2_MOE_RTOL,
    W2_ROUNDTRIP_EPS,
)
from tests.ut.deepseek_w2.reference.w2_moe_reference import (
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODE_MAX,
    W2_CODE_MIN,
    W2Expert,
    _broadcast_block_scales,
    compute_w2_block_scales,
    dequantize_weight_w2,
    pack_w2_codes,
    quantize_weight_w2,
    route_topk,
    unpack_w2_codes,
    unpack_w2_to_int8,
    w2_moe_forward,
    w2_moe_forward_per_token,
    w2_qdq_linear,
)


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


# --- W2 pack / unpack / round-trip -----------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64), (96, 128), (128, 64)],
)
def test_pack_unpack_is_exact(seed, shape):
    """pack -> unpack recovers the signed int2 codes bit for bit."""
    w = _rand(shape, seed, scale=2.0)
    codes, _ = quantize_weight_w2(w)
    assert codes.dtype == torch.int8
    assert torch.all(codes >= W2_CODE_MIN) and torch.all(codes <= W2_CODE_MAX)
    packed = pack_w2_codes(codes)
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] == shape[-1] // 4
    restored = unpack_w2_codes(packed, shape[-1])
    assert torch.equal(codes, restored)


@pytest.mark.parametrize("seed", [4, 5, 6])
@pytest.mark.parametrize("shape", [(32, 32), (64, 96), (128, 128)])
def test_weight_roundtrip_within_half_step(seed, shape):
    """|w - dequant(quant(w))| <= scale/2 per element (half a grid step)."""
    w = _rand(shape, seed, scale=3.0)
    codes, block_scale = quantize_weight_w2(w)
    w_rec = dequantize_weight_w2(codes, block_scale)
    full_scale = _broadcast_block_scales(block_scale, shape[0], shape[1])
    assert torch.all((w_rec - w).abs() <= full_scale / 2 + W2_ROUNDTRIP_EPS)


@pytest.mark.parametrize("seed", [7, 8])
def test_unpack_to_int8_matches_dequant(seed):
    """The bit-unpack path equals the no-bit-ops dequant path exactly."""
    w = _rand((64, 96), seed, scale=1.5)
    codes, block_scale = quantize_weight_w2(w)
    packed = pack_w2_codes(codes)
    via_bits = unpack_w2_to_int8(packed, block_scale, 64, 96)
    via_dense = dequantize_weight_w2(codes, block_scale)
    assert torch.equal(via_bits, via_dense)


def test_block_scale_shape_and_positivity():
    w = _rand((64, 128), seed=9)
    scale = compute_w2_block_scales(w)
    assert scale.shape == (64 // W2_BLOCK_ROWS, 128 // W2_BLOCK_COLS)
    assert torch.all(scale > 0)


def test_zero_block_is_safe():
    w = torch.zeros(32, 32, dtype=torch.float64)
    codes, block_scale = quantize_weight_w2(w)
    assert torch.all(codes == 0)
    assert torch.all(dequantize_weight_w2(codes, block_scale) == 0)


def test_per_block_scaling_is_independent():
    """Two blocks with very different magnitudes get independent scales."""
    w = torch.zeros(32, 64, dtype=torch.float64)
    w[:, :32] = 0.1
    w[:, 32:] = 100.0
    scale = compute_w2_block_scales(w)
    assert scale[0, 0] < scale[0, 1]


# --- W2 QDQ linear ----------------------------------------------------------


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_w2_qdq_linear_matches_definition(seed):
    """w2_qdq_linear == dequant(quant(x)) @ dequant_w2(w).T."""
    from tests.ut.qwen38_1m.reference.w8a8_reference import (
        dequantize_per_token,
        quantize_per_token_int8,
    )

    x = _rand((12, 64), seed)
    w = _rand((96, 64), seed + 100, scale=0.5)
    codes, block_scale = quantize_weight_w2(w)
    packed = pack_w2_codes(codes)

    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a).double()
    w_deq = unpack_w2_to_int8(packed, block_scale, 96, 64)
    y_ref = x_deq @ w_deq.t()

    y = w2_qdq_linear(x, packed, block_scale, 96, 64)
    torch.testing.assert_close(y, y_ref, rtol=W2_LINEAR_RTOL, atol=W2_LINEAR_ATOL)


# --- MoE grouped vs per-token ----------------------------------------------

_HIDDEN = 64
_INTER = 96
_NUM_EXPERTS = 8
_TOP_K = 3


def _make_experts(num_experts, seed, hidden=_HIDDEN, inter=_INTER):
    experts = []
    for e in range(num_experts):
        gate = _rand((inter, hidden), seed + 10 * e + 1, scale=0.4)
        up = _rand((inter, hidden), seed + 10 * e + 2, scale=0.4)
        down = _rand((hidden, inter), seed + 10 * e + 3, scale=0.4)
        experts.append(W2Expert(gate, up, down))
    return experts


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_grouped_equals_per_token(seed):
    x = _rand((11, _HIDDEN), seed, scale=1.0)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 999)[0]
    router_logits = _rand((11, _NUM_EXPERTS), seed + 7)

    grouped = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    per_token = w2_moe_forward_per_token(x, experts, router_logits, _TOP_K, shared)
    torch.testing.assert_close(grouped, per_token, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


@pytest.mark.parametrize("seed", [40, 41])
def test_grouped_equals_per_token_no_shared(seed):
    x = _rand((9, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((9, _NUM_EXPERTS), seed + 3)
    grouped = w2_moe_forward(x, experts, router_logits, _TOP_K)
    per_token = w2_moe_forward_per_token(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(grouped, per_token, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_skew_all_tokens_to_one_expert():
    """All tokens routed to expert 0: grouped must still match per-token."""
    seed = 50
    x = _rand((13, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    # Router logits that make expert 0 dominate for every token.
    router_logits = torch.full((13, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    router_logits[:, 0] = 30.0
    # Give the remaining top-k slots deterministic small values.
    router_logits[:, 1] = 0.0
    router_logits[:, 2] = -1.0

    expert_ids, _ = route_topk(router_logits, _TOP_K)
    assert torch.all(expert_ids[:, 0] == 0)

    grouped = w2_moe_forward(x, experts, router_logits, _TOP_K)
    per_token = w2_moe_forward_per_token(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(grouped, per_token, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_unrouted_expert_is_skipped():
    """An expert that no token selects contributes nothing (grouped path)."""
    seed = 60
    x = _rand((6, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = torch.full((6, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    # Only experts 0,1,2 ever selected; expert 7 is never routed.
    router_logits[:, 0] = 5.0
    router_logits[:, 1] = 4.0
    router_logits[:, 2] = 3.0
    expert_ids, _ = route_topk(router_logits, _TOP_K)
    assert 7 not in set(expert_ids.flatten().tolist())
    grouped = w2_moe_forward(x, experts, router_logits, _TOP_K)
    per_token = w2_moe_forward_per_token(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(grouped, per_token, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_router_weights_normalized():
    router_logits = _rand((5, _NUM_EXPERTS), seed=70)
    _, weights = route_topk(router_logits, _TOP_K)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(5, dtype=torch.float64), rtol=0, atol=1e-12)
