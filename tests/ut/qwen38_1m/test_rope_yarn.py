# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU unit tests for the Qwen4Exp 1M long-context RoPE/YaRN config (T7.1).

Self-contained: only imports the 310P host-side rope module and computes the
YaRN reference formula inline, so it runs on CPU without a server or NPU. Run
with ``pytest --noconftest`` because the shared ut conftest fails to import.
"""

import math

import pytest
import torch

from vllm_ascend._310p.worker.v2.rope import (
    MAX_EXTENDED_POSITION_EMBEDDINGS,
    NATIVE_MAX_POSITION_EMBEDDINGS,
    QWEN4EXP_ROPE_THETA,
    QWEN4EXP_YARN_BETA_FAST,
    QWEN4EXP_YARN_BETA_SLOW,
    QWEN4EXP_YARN_FACTOR,
    LongContextRopeConfigError,
    build_qwen4exp_1m_cos_sin_cache,
    validate_long_context_rope,
)

# Declared FP tolerance for the CPU cos/sin parity checks. Both the module and
# the reference compute ``position * inv_freq`` in float32, so the paths are
# bit-identical up to op ordering; 1e-4 gives ample margin even at position
# 1,048,575 where the rotary argument reaches ~1e6.
ATOL = 1e-4
RTOL = 1e-4

# Authoritative rotary dimension for Qwen4Exp attention: head_dim(256) * 0.25.
ROTARY_DIM = 64

REFERENCE_POSITIONS = [0, 262_143, 262_144, 524_288, 1_048_575]


def _ref_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _reference_inv_freq(rotary_dim, base, factor, original_max, beta_fast, beta_slow):
    """Independent transcription of the YaRN inverse-frequency formula."""
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low = math.floor(_ref_correction_dim(beta_fast, rotary_dim, base, original_max))
    high = math.ceil(_ref_correction_dim(beta_slow, rotary_dim, base, original_max))
    low = max(low, 0)
    high = min(high, rotary_dim - 1)
    if low == high:
        high += 0.001

    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low),
        0.0,
        1.0,
    )
    inv_freq_mask = 1.0 - ramp
    return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask


def _reference_mscale(factor):
    if factor <= 1.0:
        return 1.0
    return 0.1 * math.log(factor) + 1.0


def _reference_cos_sin_at(position, rotary_dim):
    inv_freq = _reference_inv_freq(
        rotary_dim,
        QWEN4EXP_ROPE_THETA,
        QWEN4EXP_YARN_FACTOR,
        NATIVE_MAX_POSITION_EMBEDDINGS,
        QWEN4EXP_YARN_BETA_FAST,
        QWEN4EXP_YARN_BETA_SLOW,
    )
    mscale = _reference_mscale(QWEN4EXP_YARN_FACTOR)
    angle = torch.tensor(float(position), dtype=torch.float32) * inv_freq
    return angle.cos() * mscale, angle.sin() * mscale


def test_cos_sin_cache_covers_1m_positions():
    cos, sin = build_qwen4exp_1m_cos_sin_cache(ROTARY_DIM)
    assert cos.shape == (MAX_EXTENDED_POSITION_EMBEDDINGS, ROTARY_DIM // 2)
    assert sin.shape == (MAX_EXTENDED_POSITION_EMBEDDINGS, ROTARY_DIM // 2)
    # Position 0 is the identity rotation.
    mscale = _reference_mscale(QWEN4EXP_YARN_FACTOR)
    assert torch.allclose(cos[0], torch.full_like(cos[0], mscale), atol=ATOL)
    assert torch.allclose(sin[0], torch.zeros_like(sin[0]), atol=ATOL)


@pytest.mark.parametrize("position", REFERENCE_POSITIONS)
def test_cos_sin_matches_reference_formula(position):
    cos, sin = build_qwen4exp_1m_cos_sin_cache(ROTARY_DIM)
    cos_ref, sin_ref = _reference_cos_sin_at(position, ROTARY_DIM)
    assert torch.allclose(cos[position], cos_ref, atol=ATOL, rtol=RTOL), f"cos mismatch at position {position}"
    assert torch.allclose(sin[position], sin_ref, atol=ATOL, rtol=RTOL), f"sin mismatch at position {position}"


def test_cos_sin_matches_vllm_canonical_yarn():
    """Cross-check against vLLM's canonical YaRNScalingRotaryEmbedding math."""
    from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )

    # Exercise the canonical cache math without the device-probing __init__.
    ref = object.__new__(YaRNScalingRotaryEmbedding)
    ref.base = QWEN4EXP_ROPE_THETA
    ref.rotary_dim = ROTARY_DIM
    ref.max_position_embeddings = NATIVE_MAX_POSITION_EMBEDDINGS
    ref.scaling_factor = QWEN4EXP_YARN_FACTOR
    ref.beta_fast = QWEN4EXP_YARN_BETA_FAST
    ref.beta_slow = QWEN4EXP_YARN_BETA_SLOW
    ref.truncate = True
    # extrapolation_factor=1 makes the canonical math identical to the module's.
    ref.extrapolation_factor = 1.0
    ref.attn_factor = 1.0
    ref.mscale = float(yarn_get_mscale(QWEN4EXP_YARN_FACTOR))

    cache = ref._compute_cos_sin_cache()
    half = ROTARY_DIM // 2
    cos_ref, sin_ref = cache[:, :half], cache[:, half:]

    cos, sin = build_qwen4exp_1m_cos_sin_cache(ROTARY_DIM)
    for position in REFERENCE_POSITIONS:
        assert torch.allclose(cos[position], cos_ref[position], atol=ATOL, rtol=RTOL)
        assert torch.allclose(sin[position], sin_ref[position], atol=ATOL, rtol=RTOL)


# --------------------------- startup-guard tests ---------------------------


def test_native_context_accepted_without_extension_config():
    cfg = validate_long_context_rope(NATIVE_MAX_POSITION_EMBEDDINGS, None)
    assert cfg.rope_type == "default"
    assert cfg.factor == 1.0
    assert not cfg.is_extended
    assert cfg.max_position_embeddings == NATIVE_MAX_POSITION_EMBEDDINGS


def test_bare_max_model_len_bump_is_rejected():
    with pytest.raises(LongContextRopeConfigError, match="no RoPE extension"):
        validate_long_context_rope(NATIVE_MAX_POSITION_EMBEDDINGS + 1, None)


def test_extension_beyond_native_accepted_with_valid_yarn_config():
    cfg = validate_long_context_rope(
        MAX_EXTENDED_POSITION_EMBEDDINGS,
        {
            "rope_type": "yarn",
            "factor": QWEN4EXP_YARN_FACTOR,
            "original_max_position_embeddings": NATIVE_MAX_POSITION_EMBEDDINGS,
        },
    )
    assert cfg.rope_type == "yarn"
    assert cfg.is_extended
    assert cfg.factor == QWEN4EXP_YARN_FACTOR
    assert cfg.max_position_embeddings == MAX_EXTENDED_POSITION_EMBEDDINGS


def test_extension_rejects_non_yarn_rope_type():
    with pytest.raises(LongContextRopeConfigError, match="rope_type='yarn'"):
        validate_long_context_rope(
            MAX_EXTENDED_POSITION_EMBEDDINGS,
            {"rope_type": "linear", "factor": 4.0},
        )


def test_extension_rejects_insufficient_factor():
    with pytest.raises(LongContextRopeConfigError, match="below the requested"):
        validate_long_context_rope(
            MAX_EXTENDED_POSITION_EMBEDDINGS,
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": NATIVE_MAX_POSITION_EMBEDDINGS,
            },
        )


def test_extension_rejects_factor_above_ceiling():
    with pytest.raises(LongContextRopeConfigError, match="above the supported ceiling"):
        validate_long_context_rope(
            MAX_EXTENDED_POSITION_EMBEDDINGS,
            {
                "rope_type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": NATIVE_MAX_POSITION_EMBEDDINGS,
            },
        )


def test_extension_rejects_wrong_original_window():
    with pytest.raises(LongContextRopeConfigError, match="must equal the native"):
        validate_long_context_rope(
            MAX_EXTENDED_POSITION_EMBEDDINGS,
            {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 131_072,
            },
        )
