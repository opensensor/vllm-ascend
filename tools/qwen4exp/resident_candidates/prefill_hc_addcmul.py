# SPDX-License-Identifier: Apache-2.0
"""Experimental prefill residual combination; reuse only a fresh cast buffer."""

import torch

from vllm_ascend.models.qwen4_exp.model import _linear

MIN_PREFILL_TOKENS = 2560


def _supports_device(device):
    return device.type == "npu"


def combine(self, block_output, residuals):
    if not self.use_combine:
        raise RuntimeError("combine disabled for this gated residual")
    hyper_input, normalized = residuals
    tokens = hyper_input.shape[0]
    residual = hyper_input.view(tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)
    injection = 2.0 * torch.sigmoid(_linear(normalized, self.block_inject_weight, self.compute_dtype) / self.hc_count)
    block = block_output.to(self.compute_dtype).unsqueeze(-2)
    eligible = (
        tokens >= MIN_PREFILL_TOKENS
        and _supports_device(hyper_input.device)
        and not torch.is_grad_enabled()
        and hyper_input.dtype == self.params_dtype == torch.float16
        and self.compute_dtype == torch.float32
    )
    if eligible:
        # FP16->FP32 always allocates: this cannot mutate the input residual or
        # checkpoint storage. Keep the FP16 injection until normal promotion.
        output = residual.addcmul_(block, injection.unsqueeze(-1))
    else:
        output = residual + block * injection.unsqueeze(-1)
    return output.flatten(-2).to(self.params_dtype)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine}
