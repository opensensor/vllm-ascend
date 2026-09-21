#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""The WY UT transform must keep matching the row-by-row substitution it replaced.

The original solved ``T = attn + attn @ T`` one row at a time -- correct, but
~5 NPU ops per row, ~315 per call, in 48 of 64 layers on every prefill step.
The blocked inverse of ``I - attn`` is the same value in ~53 ops.
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
    CHUNK_SIZE,
    _inv_unit_lower_triangular,
    _ut_transform,
)


def _forward_substitution(attn: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """The pre-optimization implementation, kept verbatim as the oracle."""
    attn = attn.clone()
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)


# [1, 12, 128, 64, 64] is the real prefill shape at TP4: 48/4 value heads and
# 8192 tokens at CHUNK_SIZE=64. Scales stay small because the real operand is
# -(k_beta @ key.T * decay) with L2-normalised keys and beta = sigmoid(.) in
# (0, 1); by scale 2.0 the inverse itself reaches ~1e16 and fp32 stops being
# meaningful for either algorithm (fp64 shows them still agreeing to 2e-15).
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 2, 3), (2, 4, 16), (1, 12, 128)])
@pytest.mark.parametrize("scale", [0.05, 0.1, 0.3])
def test_ut_transform_matches_forward_substitution(shape, scale):
    torch.manual_seed(0)
    attn = (torch.randn(*shape, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32) * scale).tril(-1)

    expected = _forward_substitution(attn, CHUNK_SIZE)
    actual = _ut_transform(attn, CHUNK_SIZE)

    # Compare against the magnitude of the result as a whole. Element-wise
    # relative error is the wrong bar: the inverse spans many orders of
    # magnitude, so a negligible absolute difference on a near-zero entry reads
    # as several percent while the matrix agrees to ~1e-7 overall.
    scale_ref = expected.abs().max().clamp_min(1e-9)
    assert ((expected - actual).abs().max() / scale_ref) < 1e-5


def test_ut_transform_inverts_i_minus_attn():
    """Independent of the oracle: (I - attn) @ result must be the identity.

    In float64, so this is an algebraic check rather than a precision one.
    """
    torch.manual_seed(1)
    attn = (torch.randn(2, 3, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float64) * 0.3).tril(-1)
    eye = torch.eye(CHUNK_SIZE, dtype=attn.dtype)

    product = (eye - attn) @ _ut_transform(attn, CHUNK_SIZE)

    torch.testing.assert_close(product, eye.expand_as(product), rtol=1e-9, atol=1e-9)


def test_inverse_result_is_unit_lower_triangular():
    torch.manual_seed(2)
    attn = (torch.randn(1, 2, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32) * 0.3).tril(-1)

    out = _ut_transform(attn, CHUNK_SIZE)

    idx = torch.arange(CHUNK_SIZE)
    torch.testing.assert_close(out[..., idx, idx], torch.ones_like(out[..., idx, idx]), rtol=0, atol=1e-6)
    assert out.triu(1).abs().max() == 0


@pytest.mark.parametrize("n", [8, 16, 64])
def test_blocked_inverse_handles_base_and_recursive_sizes(n):
    """n == block exercises the base case; larger values exercise the recursion."""
    torch.manual_seed(3)
    m = torch.eye(n, dtype=torch.float64) + (torch.randn(4, n, n, dtype=torch.float64) * 0.3).tril(-1)

    product = m @ _inv_unit_lower_triangular(m)

    torch.testing.assert_close(product, torch.eye(n, dtype=m.dtype).expand_as(product), rtol=1e-10, atol=1e-10)


def test_agrees_with_substitution_even_when_ill_conditioned():
    """At large scale the inverse reaches ~1e16 and fp32 is meaningless for
    either algorithm; in float64 the two must still agree."""
    torch.manual_seed(4)
    attn = (torch.randn(1, 2, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float64) * 2.0).tril(-1)

    expected = _forward_substitution(attn, CHUNK_SIZE)
    actual = _ut_transform(attn, CHUNK_SIZE)

    rel = (expected - actual).abs().max() / expected.abs().max()
    assert rel < 1e-12


def test_inverse_handles_every_rank_its_callers_produce():
    """Including 6D, which is what the grouped WY gram path hands it.

    The blocking inside the inverse adds two dimensions of its own, so a 6D
    `attn` used to produce 7D matmuls, and aclnnMatmul fails on those -- the
    server came up and then died on the first request with "the current working
    operator name is aclnnMatmul". A microbenchmark missed it by flattening
    `attn` to 5D before the call, which the live path does not do.
    """
    import torch

    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        _inv_unit_lower_recursive,
        _inv_unit_lower_triangular,
    )

    torch.manual_seed(0)
    for lead in [(4,), (3, 5), (1, 12, 128), (1, 4, 3, 128)]:
        strictly_lower = (torch.randn(*lead, 64, 64, dtype=torch.float64) * 0.15).tril(-1)
        mat = torch.eye(64, dtype=torch.float64) + strictly_lower

        got = _inv_unit_lower_triangular(mat)

        assert got.shape == mat.shape, lead
        assert torch.equal(got, _inv_unit_lower_recursive(mat, 8)), lead
