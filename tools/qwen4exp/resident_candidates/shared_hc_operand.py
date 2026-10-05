# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse the normalized hyperconnection projection operand during inference.

This is a resident Python candidate, not a serving default. The FP32 normalized
values still feed the gated mean. Only the tensor saved for the later injection
projection changes, when its operand dtype matches the down projection.
"""

import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp.model import _linear, _linear_operand_dtype


def mix(self, hyper_input):
    num_tokens = hyper_input.shape[0]
    normalized = self._normalize(hyper_input)
    projection_input = normalized
    saved_input = normalized
    if self.use_combine and not torch.is_grad_enabled():
        down_dtype = _linear_operand_dtype(normalized.device.type, self.input_mix_weight_down.dtype, self.compute_dtype)
        injection_dtype = _linear_operand_dtype(
            normalized.device.type, self.block_inject_weight.dtype, self.compute_dtype
        )
        if down_dtype == injection_dtype:
            projection_input = normalized.to(down_dtype)
            saved_input = projection_input
    gate = F.silu(_linear(projection_input, self.input_mix_weight_down, self.compute_dtype) / self.hc_count)
    gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
    gate = gate.view(num_tokens, self.hc_count, self.hidden_size)
    mixed = (gate * normalized.view(num_tokens, self.hc_count, self.hidden_size)).mean(dim=-2)
    return mixed.to(self.params_dtype), (hyper_input, saved_input)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix}
