# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from vllm.config import ModelConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

from vllm_ascend._310p.worker.v2.states import Ascend310PStagedWriteTensor


class Ascend310PRopeState:
    """Triton-free MRoPE state for 310P (Qwen3-VL / Qwen3.5)."""

    def __init__(
        self,
        num_dims: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ) -> None:
        self.num_dims = num_dims
        self.max_model_len = max_model_len
        self.device = device
        self.prefill_positions = Ascend310PStagedWriteTensor(
            (max_num_reqs * num_dims, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.prefill_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.positions_cpu = torch.zeros((num_dims, max_num_tokens + 1), dtype=torch.int64, device="cpu")
        self.positions = torch.zeros((num_dims, max_num_tokens + 1), dtype=torch.int64, device=device)

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        mrope_model = cast(SupportsMRoPE, model)
        # Qwen3-VL / Qwen3.5 return ``(Tensor[num_dims, seq], delta)``.
        prefill_positions, delta = mrope_model.get_mrope_input_positions(prefill_token_ids, mm_features)
        self.prefill_delta.np[req_idx] = delta

        for dim in range(self.num_dims):
            self.prefill_positions.stage_write(
                self.num_dims * req_idx + dim,
                0,
                prefill_positions[dim].tolist(),
            )

    def apply_staged_writes(self) -> None:
        self.prefill_positions.apply_write()
        self.prefill_delta.copy_to_uva()

    def prepare_positions_cpu(
        self,
        idx_mapping_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        prefill_lens_np: np.ndarray,
        num_computed_tokens_np: np.ndarray,
        num_tokens_after_padding: int,
    ) -> None:
        self.positions_cpu[:, :num_tokens_after_padding].zero_()
        for batch_idx, req_idx in enumerate(idx_mapping_np):
            query_start = int(query_start_loc_np[batch_idx])
            query_end = int(query_start_loc_np[batch_idx + 1])
            num_computed = int(num_computed_tokens_np[req_idx])
            query_len = query_end - query_start
            if num_computed < int(prefill_lens_np[req_idx]):
                row_start = self.num_dims * int(req_idx)
                positions = self.prefill_positions.cpu[
                    row_start : row_start + self.num_dims,
                    num_computed : num_computed + query_len,
                ]
                self.positions_cpu[:, query_start:query_end].copy_(positions)
            else:
                delta = int(self.prefill_delta.np[req_idx])
                decode_positions = torch.arange(
                    num_computed + delta,
                    num_computed + delta + query_len,
                    dtype=torch.int64,
                )
                self.positions_cpu[:, query_start:query_end] = decode_positions

        self.positions[:, :num_tokens_after_padding].copy_(
            self.positions_cpu[:, :num_tokens_after_padding], non_blocking=True
        )

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]


def get_310p_rope_state(
    model_config: ModelConfig,
    model: nn.Module,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    device: torch.device,
) -> Ascend310PRopeState | None:
    # 310P Qwen3-VL / Qwen3.5 use MRoPE only; XD-RoPE is out of scope.
    if model_config.uses_mrope:
        assert isinstance(model, SupportsMRoPE)
        return Ascend310PRopeState(3, max_num_reqs, max_num_tokens, max_model_len, device)
    return None


# ---------------------------------------------------------------------------
# Qwen4Exp (Qwen3.8-Flash-Next) long-context RoPE/YaRN — 310P host-side config
# ---------------------------------------------------------------------------
#
# AUTHORITATIVE 1M RoPE PARAMETER SET (resolves PRD open decision #5, T7.1).
# The native 262,144-token mode is left UNCHANGED; extension to 1,048,576 is
# only ever enabled through an EXPLICIT, VALIDATED deployment config. There is
# NO silent auto-scaling: a ``max_model_len`` beyond the native window without a
# matching extension config is REJECTED with a clear error.
#
#   rope_type / rope_scaling type ....... "yarn"
#   rope_theta (base) ................... 10_000_000.0
#   original_max_position_embeddings .... 262_144  (native window, unchanged)
#   factor (YaRN scaling_factor) ........ 4.0
#   -> extended max positions ........... 262_144 * 4 = 1_048_576
#   beta_fast ........................... 32
#   beta_slow ........................... 1
#   attention scaling (mscale) .......... yarn_get_mscale(4.0) = 0.1*ln(4)+1
#                                         ~= 1.1386294361119891
#   partial_rotary_factor ............... 0.25 (Qwen3-Next); the attention
#                                         rotary_dim is head_dim * 0.25
#
# The YaRN math below REUSES vLLM's canonical helpers
# (``yarn_find_correction_range`` / ``yarn_linear_ramp_mask`` /
# ``yarn_get_mscale``) so the 310P host-side tables match the reference
# ``YaRNScalingRotaryEmbedding`` bit-for-bit. All math runs on CPU in float32.

NATIVE_MAX_POSITION_EMBEDDINGS = 262_144
MAX_EXTENDED_POSITION_EMBEDDINGS = 1_048_576
QWEN4EXP_ROPE_THETA = 10_000_000.0
QWEN4EXP_YARN_FACTOR = 4.0
QWEN4EXP_YARN_BETA_FAST = 32
QWEN4EXP_YARN_BETA_SLOW = 1
QWEN4EXP_PARTIAL_ROTARY_FACTOR = 0.25

# The single valid extension rope_type for the 310P 1M path.
_SUPPORTED_YARN_TYPES = frozenset({"yarn"})


class LongContextRopeConfigError(ValueError):
    """Raised when a long-context request lacks a valid RoPE extension config."""


@dataclass(frozen=True)
class Qwen4ExpRopeConfig:
    """Validated RoPE configuration for a 310P Qwen4Exp deployment.

    ``factor`` is ``1.0`` (and ``rope_type`` ``"default"``) for the native
    262,144 window; a YaRN extension carries ``rope_type == "yarn"`` with a
    ``factor`` such that ``original_max_position_embeddings * factor`` covers the
    requested ``max_model_len``.
    """

    rope_type: str
    rope_theta: float
    original_max_position_embeddings: int
    factor: float
    beta_fast: int
    beta_slow: int

    @property
    def is_extended(self) -> bool:
        return self.factor > 1.0

    @property
    def max_position_embeddings(self) -> int:
        return int(self.original_max_position_embeddings * self.factor)


def validate_long_context_rope(
    max_model_len: int,
    rope_parameters: dict[str, Any] | None,
    *,
    native_max_position: int = NATIVE_MAX_POSITION_EMBEDDINGS,
    rope_theta: float = QWEN4EXP_ROPE_THETA,
) -> Qwen4ExpRopeConfig:
    """Validate the RoPE config for a requested ``max_model_len``.

    Native mode (``max_model_len <= native_max_position``) is accepted with or
    without an extension config and returns the native (factor 1.0) descriptor,
    leaving the 262,144 window UNCHANGED.

    Beyond the native window an explicit, valid YaRN extension config is
    REQUIRED. A bare ``max_model_len`` bump with no (or an invalid) extension
    config raises :class:`LongContextRopeConfigError` -- there is no silent
    auto-scaling.
    """
    if max_model_len <= 0:
        raise LongContextRopeConfigError(f"max_model_len must be positive, got {max_model_len}")

    params = rope_parameters or {}
    rope_type = str(params.get("rope_type") or params.get("type") or "default")
    theta = float(params.get("rope_theta", rope_theta))

    # Native window is unchanged: no extension config needed.
    if max_model_len <= native_max_position:
        return Qwen4ExpRopeConfig(
            rope_type="default",
            rope_theta=theta,
            original_max_position_embeddings=native_max_position,
            factor=1.0,
            beta_fast=QWEN4EXP_YARN_BETA_FAST,
            beta_slow=QWEN4EXP_YARN_BETA_SLOW,
        )

    # Beyond native: an explicit, valid YaRN extension config is mandatory.
    if not rope_parameters:
        raise LongContextRopeConfigError(
            f"max_model_len={max_model_len} exceeds the native "
            f"{native_max_position} context but no RoPE extension config was "
            "provided. Supply an explicit YaRN config (rope_type='yarn', "
            "factor, original_max_position_embeddings) via deployment metadata; "
            "the 310P path never auto-scales RoPE silently."
        )
    if rope_type not in _SUPPORTED_YARN_TYPES:
        raise LongContextRopeConfigError(
            f"max_model_len={max_model_len} exceeds the native "
            f"{native_max_position} context and requires rope_type='yarn', "
            f"got rope_type={rope_type!r}."
        )

    original = int(params.get("original_max_position_embeddings", native_max_position))
    if original != native_max_position:
        raise LongContextRopeConfigError(
            f"YaRN original_max_position_embeddings must equal the native window {native_max_position}, got {original}."
        )

    if "factor" not in params:
        raise LongContextRopeConfigError("YaRN extension config must declare a 'factor'.")
    factor = float(params["factor"])
    if factor <= 1.0:
        raise LongContextRopeConfigError(f"YaRN factor must be > 1.0 to extend context, got {factor}.")

    extended_max = int(original * factor)
    if extended_max < max_model_len:
        raise LongContextRopeConfigError(
            f"YaRN factor={factor} extends context to {extended_max}, which is "
            f"below the requested max_model_len={max_model_len}."
        )
    if extended_max > MAX_EXTENDED_POSITION_EMBEDDINGS:
        raise LongContextRopeConfigError(
            f"YaRN factor={factor} extends context to {extended_max}, above the "
            f"supported ceiling {MAX_EXTENDED_POSITION_EMBEDDINGS}."
        )

    beta_fast = int(params.get("beta_fast", QWEN4EXP_YARN_BETA_FAST))
    beta_slow = int(params.get("beta_slow", QWEN4EXP_YARN_BETA_SLOW))

    return Qwen4ExpRopeConfig(
        rope_type="yarn",
        rope_theta=theta,
        original_max_position_embeddings=original,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )


def compute_yarn_inv_freq(
    rotary_dim: int,
    base: float,
    scaling_factor: float,
    original_max_position: int,
    *,
    beta_fast: int = QWEN4EXP_YARN_BETA_FAST,
    beta_slow: int = QWEN4EXP_YARN_BETA_SLOW,
) -> torch.Tensor:
    """YaRN inverse frequencies (CPU/float32), matching vLLM's YaRN impl."""
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        rotary_dim,
        base,
        original_max_position,
        True,
    )
    inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float32)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq


def build_yarn_cos_sin_cache(
    rotary_dim: int,
    base: float,
    scaling_factor: float,
    original_max_position: int,
    *,
    beta_fast: int = QWEN4EXP_YARN_BETA_FAST,
    beta_slow: int = QWEN4EXP_YARN_BETA_SLOW,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` YaRN tables on CPU.

    Returns tables of shape ``[original_max_position * scaling_factor,
    rotary_dim // 2]``. For the authoritative 1M set this spans positions
    ``0 .. 1_048_575`` inclusive. The attention scaling (mscale) is folded into
    the tables, matching vLLM's ``YaRNScalingRotaryEmbedding``.
    """
    inv_freq = compute_yarn_inv_freq(
        rotary_dim,
        base,
        scaling_factor,
        original_max_position,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    mscale = float(yarn_get_mscale(scaling_factor))
    max_pos = int(original_max_position * scaling_factor)
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = (freqs.cos() * mscale).to(dtype)
    sin = (freqs.sin() * mscale).to(dtype)
    return cos, sin


def build_qwen4exp_1m_cos_sin_cache(
    rotary_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the authoritative Qwen4Exp 1,048,576-token YaRN tables (CPU)."""
    return build_yarn_cos_sin_cache(
        rotary_dim,
        QWEN4EXP_ROPE_THETA,
        QWEN4EXP_YARN_FACTOR,
        NATIVE_MAX_POSITION_EMBEDDINGS,
        beta_fast=QWEN4EXP_YARN_BETA_FAST,
        beta_slow=QWEN4EXP_YARN_BETA_SLOW,
        dtype=dtype,
    )
