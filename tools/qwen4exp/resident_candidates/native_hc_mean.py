# SPDX-License-Identifier: Apache-2.0
"""Compose the qualified native residual with experimental gated-mean fusion."""

import torch
import torch.nn.functional as F

from tools.qwen4exp.direct_hc_residual import MAX_TOKENS, MAX_WIDTH, TILE_WIDTH
from tools.qwen4exp.resident_candidates.native_hc_residual import RESOURCE_NAME as RESIDUAL_RESOURCE
from tools.qwen4exp.resident_candidates.native_hc_residual import make_combine
from vllm_ascend.models.qwen4_exp.model import _linear

RESOURCE_NAME = "hc_mean_v1"
MIN_MEAN_PREFILL_TOKENS = 640


def _supports_device(device):
    return device.type == "npu"


def make_mix(op, *, prefill_only=False):
    def mix(self, hyper_input):
        tokens = hyper_input.shape[0]
        normalized = self._normalize(hyper_input)
        gate = F.silu(_linear(normalized, self.input_mix_weight_down, self.compute_dtype) / self.hc_count)
        gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
        if (
            not torch.is_grad_enabled()
            and _supports_device(hyper_input.device)
            and self.use_combine
            and self.hc_count == 4
            and self.params_dtype == torch.float16
            and normalized.dtype == torch.float32
            and gate.dtype == torch.float16
            and 0 < tokens <= MAX_TOKENS
            and (not prefill_only or tokens >= MIN_MEAN_PREFILL_TOKENS)
            and 0 < self.hidden_size <= MAX_WIDTH
            and self.hidden_size % TILE_WIDTH == 0
            and normalized.is_contiguous()
            and gate.is_contiguous()
        ):
            mixed = op(normalized, gate)
        else:
            gate = gate.view(tokens, self.hc_count, self.hidden_size)
            mixed = (gate * normalized.view(tokens, self.hc_count, self.hidden_size)).mean(dim=-2).to(self.params_dtype)
        return mixed, (hyper_input, normalized)

    return mix


def replacements(native_resources):
    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": make_mix(
            native_resources[RESOURCE_NAME], prefill_only=True
        ),
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": make_combine(native_resources[RESIDUAL_RESOURCE]),
    }
