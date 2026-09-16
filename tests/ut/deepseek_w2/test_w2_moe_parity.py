# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the E1.2 runtime W2->INT8 active-expert unpack + grouped MoE.

The unpack + grouped path (``vllm_ascend/models/deepseek_v41/w2_unpack.py``) is
validated against the frozen E0.4 golden reference
(``tests/ut/deepseek_w2/reference/w2_moe_reference.py``): the DeepSeek V4.1
top-``k`` + shared-expert W2 QDQ MoE forward.

Acceptance (task E1.2):
  * Parity: unpack+grouped == the reference forward within the declared W2_MOE
    tolerances (with and without the shared expert).
  * Skew (all tokens -> one expert) holds.
  * The unpack cache holds only the active experts (size == #active, not the
    full bank).
  * Router renorm has teeth (wrong renorm diverges from the reference).
  * Per-block-scale application order has teeth (folding the per-[32,32] block
    scale into a single per-output-channel factor diverges).

Run: ``python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_w2_moe_parity.py``
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.tolerances import (
    W2_MOE_ATOL,
    W2_MOE_RTOL,
)
from tests.ut.deepseek_w2.reference.w2_moe_reference import (
    W2Expert,
    unpack_w2_to_int8,
    w2_moe_forward,
)
from tests.ut.qwen38_1m.reference.w8a8_reference import (
    dequantize_per_token,
    quantize_per_token_int8,
)
from vllm_ascend.models.deepseek_v41.w2_unpack import (
    ActiveExpertWeights,
    route_topk_w2,
    unpack_active_experts,
    w2_active_moe_forward,
)

# --- reduced-scale DeepSeek geometry (384 experts / top-6 / shared) ----------
# The math is independent of the expert count; a modest bank keeps the CPU UT
# fast while exercising the production top_k=6 + shared-expert shape. Dims are
# multiples of the [32, 32] W2 block.
_HIDDEN = 64
_INTER = 96
_NUM_EXPERTS = 12
_TOP_K = 6


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


def _make_experts(num_experts, seed, hidden=_HIDDEN, inter=_INTER):
    experts = []
    for e in range(num_experts):
        gate = _rand((inter, hidden), seed + 10 * e + 1, scale=0.4)
        up = _rand((inter, hidden), seed + 10 * e + 2, scale=0.4)
        down = _rand((hidden, inter), seed + 10 * e + 3, scale=0.4)
        experts.append(W2Expert(gate, up, down))
    return experts


# --- parity: unpack+grouped == E0.4 reference forward ------------------------


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_parity_with_shared_expert(seed):
    x = _rand((11, _HIDDEN), seed, scale=1.0)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 9999)[0]
    router_logits = _rand((11, _NUM_EXPERTS), seed + 7)

    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K, shared)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


@pytest.mark.parametrize("seed", [110, 111])
def test_parity_without_shared_expert(seed):
    x = _rand((9, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((9, _NUM_EXPERTS), seed + 3)

    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_parity_single_token():
    seed = 120
    x = _rand((1, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 1)[0]
    router_logits = _rand((1, _NUM_EXPERTS), seed + 2)
    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K, shared)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


# --- skew: all tokens -> one expert ------------------------------------------


def test_skew_all_tokens_to_one_expert():
    seed = 200
    x = _rand((13, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    # Make expert 0 the top pick for every token; deterministic remaining slots.
    router_logits = torch.full((13, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    router_logits[:, 0] = 30.0
    for slot in range(1, _TOP_K):
        router_logits[:, slot] = -float(slot)

    expert_ids, _ = route_topk_w2(router_logits, _TOP_K)
    assert torch.all(expert_ids[:, 0] == 0)

    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


# --- bounded active-only unpack cache ----------------------------------------


def test_cache_holds_only_active_experts():
    """Only the routed experts are unpacked, never the full bank."""
    seed = 300
    experts = _make_experts(_NUM_EXPERTS, seed)
    # Concentrate all routing on experts {0..5} (exactly top_k of them) so the
    # active set is a strict subset of the 12-expert bank.
    num_tokens = 8
    router_logits = torch.full((num_tokens, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    for slot in range(_TOP_K):
        router_logits[:, slot] = 5.0 - slot

    expert_ids, _ = route_topk_w2(router_logits, _TOP_K)
    active = set(expert_ids.reshape(-1).tolist())
    assert active == set(range(_TOP_K))

    cache = unpack_active_experts(experts, expert_ids)
    # Cache is bounded to the active set -- not the 12-expert (production: 384)
    # bank.
    assert len(cache) == len(active)
    assert len(cache) < _NUM_EXPERTS
    assert set(cache.keys()) == active
    assert all(isinstance(v, ActiveExpertWeights) for v in cache.values())


def test_cache_size_is_one_under_full_skew():
    seed = 310
    experts = _make_experts(_NUM_EXPERTS, seed)
    # All tokens, all slots -> expert 3 only.
    router_logits = torch.full((5, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    router_logits[:, 3] = 30.0
    # Runner-up slots still land on other experts unless we starve them; force a
    # single active expert by giving only expert 3 a finite-competitive logit.
    expert_ids, _ = route_topk_w2(router_logits, top_k=1)
    cache = unpack_active_experts(experts, expert_ids)
    assert len(cache) == 1
    assert set(cache.keys()) == {3}


def test_forward_uses_active_cache_size():
    """The forward's internal cache matches the standalone active set."""
    seed = 320
    x = _rand((7, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = torch.full((7, _NUM_EXPERTS), -30.0, dtype=torch.float64)
    for slot in range(_TOP_K):
        router_logits[:, slot] = 4.0 - slot
    expert_ids, _ = route_topk_w2(router_logits, _TOP_K)
    cache = unpack_active_experts(experts, expert_ids)
    assert len(cache) == _TOP_K < _NUM_EXPERTS
    # Passing the pre-built (active-only) cache reproduces the reference exactly.
    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K, cache=cache)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


# --- unpack operand is bit-identical to the reference dequant ----------------


@pytest.mark.parametrize("seed", [400, 401])
def test_unpack_operand_matches_reference_dequant(seed):
    """codes * broadcast(block_scale) equals the reference unpack_w2_to_int8."""
    experts = _make_experts(1, seed)
    expert = experts[0]
    router_logits = torch.zeros(1, 1, dtype=torch.float64)  # active = {0}
    expert_ids = torch.zeros(1, 1, dtype=torch.int64)
    cache = unpack_active_experts([expert], expert_ids)
    w = cache[0]

    # w13 = [gate ; up], w2 = down. Compare each block against the reference
    # unpack (int8 codes widened * per-block scale).
    gate_ref = unpack_w2_to_int8(expert.gate_packed, expert.gate_scale, expert.inter, expert.hidden)
    up_ref = unpack_w2_to_int8(expert.up_packed, expert.up_scale, expert.inter, expert.hidden)
    down_ref = unpack_w2_to_int8(expert.down_packed, expert.down_scale, expert.hidden, expert.inter)

    assert torch.equal(w.w13_weight()[: expert.inter], gate_ref)
    assert torch.equal(w.w13_weight()[expert.inter :], up_ref)
    assert torch.equal(w.w2_weight(), down_ref)
    # codes are the signed int2 grid {-2,-1,0,1}.
    assert w.w13_codes.dtype == torch.int8
    assert torch.all(w.w13_codes >= -2) and torch.all(w.w13_codes <= 1)
    del router_logits


# --- guard: router renormalization has teeth ---------------------------------


def test_renorm_matches_reference_and_wrong_renorm_diverges():
    seed = 500
    x = _rand((10, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 1)[0]
    router_logits = _rand((10, _NUM_EXPERTS), seed + 2)

    # Correct (renormalized) path matches the reference.
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    got = w2_active_moe_forward(x, experts, router_logits, _TOP_K, shared, renormalize=True)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)

    # Skipping renormalization (weights sum < 1) must diverge -- the guard bites.
    no_renorm = w2_active_moe_forward(x, experts, router_logits, _TOP_K, shared, renormalize=False)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(no_renorm, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_router_weights_sum_to_one():
    router_logits = _rand((6, _NUM_EXPERTS), seed=510)
    _, weights = route_topk_w2(router_logits, _TOP_K)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(6, dtype=torch.float64), rtol=0, atol=1e-12)


# --- guard: per-block-scale application order has teeth -----------------------


def test_block_scale_order_has_teeth():
    """The per-[32,32] block scale must be applied element-wise into the weight.

    It varies along the input axis every 32 columns, so folding it into a single
    per-output-channel factor (applied after the int8 matmul) diverges from the
    correct per-block dequant. This proves the unpack's scale-application order
    is load-bearing, not cosmetically re-associable.
    """
    seed = 600
    x = _rand((5, _HIDDEN), seed)
    experts = _make_experts(1, seed)
    expert = experts[0]
    expert_ids = torch.zeros(1, 1, dtype=torch.int64)
    weight = unpack_active_experts([expert], expert_ids)[0]

    q_a, scale_a = quantize_per_token_int8(x)
    x_deq = dequantize_per_token(q_a, scale_a).double()

    # Correct: block scale multiplied into the codes before the matmul.
    correct = x_deq @ weight.w13_weight().t()

    # Wrong: collapse the per-block scale to one per-output-channel factor
    # (column 0 of each row) and apply it *after* an int8-code matmul.
    per_channel = weight.w13_scale[:, :1]  # [2*inter, 1]
    wrong = (x_deq @ weight.w13_codes.double().t()) * per_channel.t()

    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong, correct, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)
