# SPDX-License-Identifier: Apache-2.0
"""Add exact power-of-two FP16 HC scaling to the padded prefill candidate."""

import torch
import torch.nn.functional as F

from tools.qwen4exp.resident_candidates.padded_hc_injection import prepare_injection
from tools.qwen4exp.resident_candidates.prefill_hc_addcmul import MIN_PREFILL_TOKENS
from vllm_ascend.models.qwen4_exp.model import _linear


def _supports_device(device):
    return device.type == "npu"


def scale_projection(projection, hc_count):
    # Restrict the optimization to the actual qualified operand dtype and
    # exact binary reciprocals. Other dtypes/devices keep the original division.
    if (
        hc_count > 0
        and hc_count & (hc_count - 1) == 0
        and projection.dtype == torch.float16
        and _supports_device(projection.device)
        and not torch.is_grad_enabled()
    ):
        return projection * (1.0 / hc_count)
    return projection / hc_count


def mix(self, hyper_input):
    tokens = hyper_input.shape[0]
    normalized = self._normalize(hyper_input)
    projection = _linear(normalized, self.input_mix_weight_down, self.compute_dtype)
    gate = F.silu(scale_projection(projection, self.hc_count))
    gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
    gate = gate.view(tokens, self.hc_count, self.hidden_size)
    mixed = (gate * normalized.view(tokens, self.hc_count, self.hidden_size)).mean(dim=-2)
    return mixed.to(self.params_dtype), (hyper_input, normalized)


def combine(self, block_output, residuals):
    if not self.use_combine:
        raise RuntimeError("combine disabled for this gated residual")
    hyper_input, normalized = residuals
    tokens = hyper_input.shape[0]
    residual = hyper_input.view(tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)
    padded = prepare_injection(self)
    if padded is None:
        projection = _linear(normalized, self.block_inject_weight, self.compute_dtype)
    else:
        projection = F.linear(normalized.to(padded.dtype), padded)[:, : self.hc_count]
    injection = 2.0 * torch.sigmoid(scale_projection(projection, self.hc_count))
    block = block_output.to(self.compute_dtype).unsqueeze(-2)
    if (
        tokens >= MIN_PREFILL_TOKENS
        and _supports_device(hyper_input.device)
        and not torch.is_grad_enabled()
        and hyper_input.dtype == self.params_dtype == torch.float16
        and self.compute_dtype == torch.float32
    ):
        output = residual.addcmul_(block, injection.unsqueeze(-1))
    else:
        output = residual + block * injection.unsqueeze(-1)
    return output.flatten(-2).to(self.params_dtype)


def replacements():
    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix,
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine,
    }
