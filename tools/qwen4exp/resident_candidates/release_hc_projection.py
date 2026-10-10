# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore original mix math and release joined caches during drained warmup.

Apply through the resident harness after the fusion trial, then switch to
baseline. The harness clears old graphs before warmup reaches these modules.
"""

import torch
import torch.nn.functional as F

from tools.qwen4exp.resident_candidates.fused_hc_projection import (
    CACHE_ATTRIBUTE,
    _is_capturing,
    release_projection,
)
from vllm_ascend.models.qwen4_exp.model import _linear


def mix(self, hyper_input):
    if hasattr(self, CACHE_ATTRIBUTE):
        if _is_capturing(hyper_input.device):
            raise RuntimeError("release joined HC weights during warmup, after clearing old graphs")
        release_projection(self)
    num_tokens = hyper_input.shape[0]
    normalized = self._normalize(hyper_input)
    gate = F.silu(_linear(normalized, self.input_mix_weight_down, self.compute_dtype) / self.hc_count)
    gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
    gate = gate.view(num_tokens, self.hc_count, self.hidden_size)
    mixed = (gate * normalized.view(num_tokens, self.hc_count, self.hidden_size)).mean(dim=-2)
    return mixed.to(self.params_dtype), (hyper_input, normalized)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix}
