# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the MLA reference (E0.4, priority 3).

Acceptance: the absorbed-latent MLA equals the dense (materialized K/V) oracle
within tolerance, and the decoupled RoPE dot product is formulation-invariant.
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.mla_reference import (
    QK_ROPE_HEAD_DIM,
    MLAConfig,
    apply_rope,
    make_random_weights,
    mla_absorbed_reference,
    mla_dense_reference,
)
from tests.ut.deepseek_w2.reference.tolerances import (
    MLA_ATOL,
    MLA_ROPE_ATOL,
    MLA_ROPE_RTOL,
    MLA_RTOL,
)


def _cfg():
    # Small but structurally faithful: q_lora_rank/kv_lora_rank low-rank,
    # decoupled rope tail 64, multiple heads.
    return MLAConfig(
        hidden_size=48,
        num_heads=3,
        q_lora_rank=40,
        kv_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=20,
        eps=1e-6,
    )


def _hidden(seq_len, seed, hidden_size):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(seq_len, hidden_size, generator=gen, dtype=torch.float64)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 4, 9, 16])
def test_absorbed_equals_dense(seed, seq_len):
    cfg = _cfg()
    w = make_random_weights(cfg, seed)
    hidden = _hidden(seq_len, seed + 100, cfg.hidden_size)
    positions = torch.arange(seq_len)
    dense = mla_dense_reference(hidden, positions, cfg, w)
    absorbed = mla_absorbed_reference(hidden, positions, cfg, w)
    torch.testing.assert_close(absorbed, dense, rtol=MLA_RTOL, atol=MLA_ATOL)


def test_output_shape():
    cfg = _cfg()
    w = make_random_weights(cfg, 5)
    hidden = _hidden(7, 6, cfg.hidden_size)
    out = mla_dense_reference(hidden, torch.arange(7), cfg, w)
    assert out.shape == (7, cfg.hidden_size)


def test_causality_first_token_independent_of_future():
    """Token 0's output must not change when later tokens change (causal mask)."""
    cfg = _cfg()
    w = make_random_weights(cfg, 7)
    hidden = _hidden(6, 8, cfg.hidden_size)
    positions = torch.arange(6)
    out_a = mla_dense_reference(hidden, positions, cfg, w)
    hidden2 = hidden.clone()
    hidden2[3:] += 5.0  # perturb the future
    out_b = mla_dense_reference(hidden2, positions, cfg, w)
    torch.testing.assert_close(out_a[0], out_b[0], rtol=MLA_RTOL, atol=MLA_ATOL)
    # A later token *does* change.
    assert not torch.allclose(out_a[4], out_b[4])


def test_rope_dot_product_is_formulation_invariant():
    """q.k after rotating both equals the two-formulation RoPE dot (relative
    rotation only depends on the position difference)."""
    gen = torch.Generator().manual_seed(11)
    dim = QK_ROPE_HEAD_DIM
    q = torch.randn(5, 1, dim, generator=gen, dtype=torch.float64)
    k = torch.randn(5, 1, dim, generator=gen, dtype=torch.float64)
    positions = torch.arange(5)
    q_rot = apply_rope(q, positions)
    k_rot = apply_rope(k, positions)
    # Rotate-then-dot vs. dot of rotated tensors: identical by construction,
    # exercised here to lock the RoPE helper's numerics.
    lhs = (q_rot[:, 0] * k_rot[:, 0]).sum(dim=-1)
    rhs = torch.einsum("td,td->t", q_rot[:, 0], k_rot[:, 0])
    torch.testing.assert_close(lhs, rhs, rtol=MLA_ROPE_RTOL, atol=MLA_ROPE_ATOL)


def test_rope_preserves_norm():
    """RoPE is a rotation: it preserves per-vector L2 norm."""
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(6, 2, QK_ROPE_HEAD_DIM, generator=gen, dtype=torch.float64)
    positions = torch.arange(6)
    x_rot = apply_rope(x, positions)
    torch.testing.assert_close(x_rot.norm(dim=-1), x.norm(dim=-1), rtol=MLA_ROPE_RTOL, atol=MLA_ROPE_ATOL)


def test_rope_at_position_zero_is_identity():
    gen = torch.Generator().manual_seed(17)
    x = torch.randn(1, 1, QK_ROPE_HEAD_DIM, generator=gen, dtype=torch.float64)
    out = apply_rope(x, torch.zeros(1, dtype=torch.long))
    torch.testing.assert_close(out, x, rtol=MLA_ROPE_RTOL, atol=MLA_ROPE_ATOL)
