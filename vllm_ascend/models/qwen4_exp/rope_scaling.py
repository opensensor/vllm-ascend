# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static YaRN policy for QSA's partial, interleaved rotary embeddings.

Only the small inverse-frequency vector is persistent. Tables are produced
from each forward's actual positions, including captured graph replay.
"""

import math
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)

_DEFAULT_BETA_FAST = 32.0
_DEFAULT_BETA_SLOW = 1.0


@dataclass(frozen=True)
class QSARopeScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    attention_factor: float
    truncate: bool

    @classmethod
    def from_parameters(cls, parameters: dict) -> "QSARopeScaling | None":
        rope_type = parameters.get("rope_type", parameters.get("type", "default"))
        if rope_type == "default":
            return None
        if rope_type != "yarn":
            raise ValueError(f"QSA supports default or static yarn RoPE, got {rope_type!r}")
        factor = float(parameters.get("factor", 0))
        original = parameters.get("original_max_position_embeddings", 0)
        beta_fast = float(parameters.get("beta_fast", _DEFAULT_BETA_FAST))
        beta_slow = float(parameters.get("beta_slow", _DEFAULT_BETA_SLOW))
        if not math.isfinite(factor) or factor < 1:
            raise ValueError("QSA YaRN factor must be finite and >= 1")
        if isinstance(original, bool) or not isinstance(original, int) or original <= 0:
            raise ValueError("QSA YaRN original_max_position_embeddings must be a positive integer")
        if not all(math.isfinite(beta) and beta > 0 for beta in (beta_fast, beta_slow)) or beta_fast < beta_slow:
            raise ValueError("QSA YaRN beta_fast must be >= beta_slow > 0, both finite")
        attention_factor = parameters.get("attention_factor")
        if attention_factor is None:
            mscale = parameters.get("mscale")
            mscale_all_dim = parameters.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                numerator = 1.0 if factor <= 1 else 0.1 * float(mscale) * math.log(factor) + 1.0
                denominator = 1.0 if factor <= 1 else 0.1 * float(mscale_all_dim) * math.log(factor) + 1.0
                attention_factor = numerator / denominator
            else:
                attention_factor = yarn_get_mscale(factor)
        attention_factor = float(attention_factor)
        if not math.isfinite(attention_factor) or attention_factor <= 0:
            raise ValueError("QSA YaRN attention_factor must be finite and positive")
        truncate = parameters.get("truncate", True)
        if not isinstance(truncate, bool):
            raise ValueError("QSA YaRN truncate must be boolean")
        return cls(factor, original, beta_fast, beta_slow, attention_factor, truncate)

    def inverse_frequencies(self, rotary_dim: int, base: float) -> torch.Tensor:
        """Build on CPU before loading/capture, using vLLM's canonical math."""
        if rotary_dim <= 0 or rotary_dim % 2 or not math.isfinite(base) or base <= 1:
            raise ValueError("QSA YaRN requires positive even rotary_dim and finite base > 1")
        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, rotary_dim, base, self.original_max_position_embeddings, self.truncate
        )
        # Model construction may run under torch.device("npu"). Keep this
        # initialization on CPU, then let the owner transfer its buffer once.
        with torch.device("cpu"):
            pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            extrapolated = 1.0 / pos_freqs
            interpolated = 1.0 / (self.factor * pos_freqs)
            mask = 1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float32)
            return interpolated * (1 - mask) + extrapolated * mask
