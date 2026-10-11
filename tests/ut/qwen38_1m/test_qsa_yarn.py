# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YaRN parity for the actual QSA attention/indexer rotary path."""

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_rope_utils import _compute_yarn_parameters

from vllm_ascend.models.qwen4_exp.model import _QSAAttention, _step_rope_cos_sin
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import _QSA_SELECTION_MAX_TILE_ELEMENTS
from vllm_ascend.models.qwen4_exp.qsa import (
    AscendQwen4ExpQSAAttention,
    _mrope_interleaved_dims,
    _reduce_rotary_angles,
    apply_partial_rope,
    partial_rope_cos_sin,
)
from vllm_ascend.models.qwen4_exp.rope_scaling import QSARopeScaling

PARAMETERS = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


@pytest.mark.parametrize("requests", [1, 2, 6])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("capacity", [8, 2048, 8192, 16384])
def test_exact_dense_attention_excludes_unused_million_context_table(requests, block_size, capacity):
    owner = SimpleNamespace(indexer=SimpleNamespace(token_topk=2048))
    table = torch.arange(requests * capacity, dtype=torch.int32).reshape(requests, capacity)
    compact = _QSAAttention._dense_block_tables(owner, table, block_size)
    width = min(capacity, (2048 + block_size - 1) // block_size)
    assert compact.shape == (requests, width)
    assert compact.is_contiguous()
    assert torch.equal(compact, table[:, :width])
    if requests == 1 or capacity <= width:
        assert compact.data_ptr() == table.data_ptr()


@pytest.mark.parametrize(
    "extra",
    [{}, {"factor": 2.0}, {"attention_factor": 1.2}, {"truncate": False}, {"mscale": 0.7, "mscale_all_dim": 0.3}],
)
def test_frequencies_and_scale_match_transformers(extra):
    parameters = {**PARAMETERS, **extra}
    cfg = SimpleNamespace(
        head_dim=256,
        hidden_size=6144,
        num_attention_heads=24,
        max_position_embeddings=262144,
        rope_parameters=parameters,
        standardize_rope_params=lambda: None,
    )
    reference, attention_factor = _compute_yarn_parameters(cfg, device=torch.device("cpu"))
    scaling = QSARopeScaling.from_parameters(parameters)
    torch.testing.assert_close(scaling.inverse_frequencies(64, 10000000.0), reference, rtol=0, atol=0)
    assert scaling.attention_factor == attention_factor


@pytest.mark.parametrize("kind", ["text", "image"])
def test_million_position_attention_and_indexer_share_exact_yarn_tables(kind):
    cfg = SimpleNamespace(
        num_attention_heads=4, num_key_value_heads=2, head_dim=256, rope_parameters=PARAMETERS, rms_norm_eps=1e-6
    )
    module = AscendQwen4ExpQSAAttention(config=cfg, layer_idx=0)
    assert module.rotary_dim == 64
    assert module.rope_inv_freq.numel() == 32
    assert "rope_inv_freq" not in module.state_dict()
    positions = torch.tensor([0, 262143, 262144, 524288, 1048575])
    if kind == "image":
        positions = torch.stack((positions, positions // 3, positions // 7))
    axes = torch.tensor(_mrope_interleaved_dims([11, 11, 10]))
    kwargs = dict(
        rotary_dim=64,
        base=module.rope_theta,
        dtype=torch.float32,
        mrope_section=module.mrope_section,
        mrope_interleaved=True,
        rope_scaling=module.rope_scaling,
        inv_freq=module.rope_inv_freq,
        frequency_axes=axes,
    )
    tables = partial_rope_cos_sin(positions, **kwargs)
    reference_positions = positions[:, None] if kind == "text" else positions.index_select(0, axes).t()
    angles = reference_positions * module.rope_inv_freq
    for actual, expected in zip(tables, (angles.cos(), angles.sin())):
        expected = torch.cat((expected, expected), dim=-1) * module.rope_scaling.attention_factor
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Both the 256-dimensional attention heads and 128-dimensional index heads
    # rotate the same leading 64 dimensions; the non-rotary tail stays exact.
    for heads, dim in [(4, 256), (2, 256), (4, 128), (1, 128)]:
        x = torch.randn(5, heads, dim)
        actual = apply_partial_rope(x, positions, 64, module.rope_theta, torch.float32, cos_sin=tables)
        direct = apply_partial_rope(
            x,
            positions,
            64,
            module.rope_theta,
            torch.float32,
            mrope_section=module.mrope_section,
            mrope_interleaved=True,
            rope_scaling=module.rope_scaling,
            inv_freq=module.rope_inv_freq,
        )
        torch.testing.assert_close(actual, direct, rtol=0, atol=0)
        assert torch.equal(actual[..., 64:], x[..., 64:])
    cache = {}
    first = _step_rope_cos_sin(positions, positions, cache, "query", **kwargs)
    assert _step_rope_cos_sin(positions, positions, cache, "query", **kwargs) is first
    native = _step_rope_cos_sin(
        positions, positions, cache, "query", **{**kwargs, "rope_scaling": None, "inv_freq": None}
    )
    assert native is not first
    positions.add_(1)
    replay = _step_rope_cos_sin(positions, positions, {}, "query", **kwargs)
    assert not torch.equal(replay[0], first[0])


@pytest.mark.parametrize(
    "change",
    [
        {"factor": 0},
        {"factor": float("nan")},
        {"original_max_position_embeddings": 0},
        {"beta_slow": 0},
        {"beta_fast": 0.5},
        {"attention_factor": float("inf")},
        {"truncate": "yes"},
        {"rope_type": "dynamic"},
    ],
)
def test_invalid_policy_fails_before_capture(change):
    with pytest.raises(ValueError):
        QSARopeScaling.from_parameters({**PARAMETERS, **change})


def test_yarn_does_not_transfer_host_frequencies_during_forward():
    scaling = QSARopeScaling.from_parameters(PARAMETERS)
    with pytest.raises(ValueError, match="resident"):
        partial_rope_cos_sin(torch.arange(3), rotary_dim=64, base=10000000.0, dtype=torch.float32, rope_scaling=scaling)


def test_three_part_angle_reduction_preserves_million_radian_phases():
    angles = torch.linspace(-1048576, 1048576, 10001)
    reduced = _reduce_rotary_angles(angles)
    # FP32 rounding can choose the adjacent turn near +/-pi; either phase
    # remains small and is equivalent modulo 2*pi.
    assert reduced.abs().max() <= 3.3
    torch.testing.assert_close(reduced.cos(), angles.cos(), rtol=0, atol=1e-6)
    torch.testing.assert_close(reduced.sin(), angles.sin(), rtol=0, atol=1e-6)


def test_million_context_selection_keeps_sixteen_row_score_tiles():
    assert _QSA_SELECTION_MAX_TILE_ELEMENTS // (1048576 // 4) == 16
