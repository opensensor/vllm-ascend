# SPDX-License-Identifier: Apache-2.0
"""Cache the small HC injection projection in an aligned NZ weight matrix."""

import torch
import torch.nn.functional as F

from tools.qwen4exp.resident_candidates.fused_hc_projection import (
    CUBE_ALIGNMENT,
    _dense,
    _format_joined,
    _is_capturing,
    _weight_key,
)
from tools.qwen4exp.resident_candidates.prefill_hc_addcmul import MIN_PREFILL_TOKENS
from vllm_ascend.models.qwen4_exp.model import _linear

CACHE_ATTRIBUTE = "_qwen_hc_padded_injection_cache"


def _supports_device(device):
    return device.type == "npu"


def prepare_injection(self):
    weight = self.block_inject_weight
    if (
        torch.is_grad_enabled()
        or not self.use_combine
        or not _supports_device(weight.device)
        or weight.dtype != self.params_dtype
        or weight.dtype != torch.float16
        or self.compute_dtype != torch.float32
        or tuple(weight.shape) != (self.hc_count, self.hyper_hidden)
    ):
        return None
    key = _weight_key(weight)
    if key is None:
        return None
    cached = getattr(self, CACHE_ATTRIBUTE, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if _is_capturing(weight.device):
        raise RuntimeError("warm up padded HC injection after weight changes and before graph capture")
    padding = (-self.hc_count) % CUBE_ALIGNMENT
    dense = _dense(weight)
    joined = torch.cat((dense, weight.new_zeros(padding, self.hyper_hidden)), dim=0) if padding else dense
    padded = _format_joined(joined.contiguous())
    setattr(self, CACHE_ATTRIBUTE, (key, padded))
    return padded


def combine_impl(self, block_output, residuals, *, use_cache, fuse_prefill):
    if not self.use_combine:
        raise RuntimeError("combine disabled for this gated residual")
    hyper_input, normalized = residuals
    tokens = hyper_input.shape[0]
    residual = hyper_input.view(tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)
    padded = prepare_injection(self) if use_cache else None
    if padded is None:
        projection = _linear(normalized, self.block_inject_weight, self.compute_dtype)
    else:
        projection = F.linear(normalized.to(padded.dtype), padded)[:, : self.hc_count]
    injection = 2.0 * torch.sigmoid(projection / self.hc_count)
    block = block_output.to(self.compute_dtype).unsqueeze(-2)
    if (
        fuse_prefill
        and tokens >= MIN_PREFILL_TOKENS
        and _supports_device(hyper_input.device)
        and not torch.is_grad_enabled()
        and hyper_input.dtype == self.params_dtype == torch.float16
        and self.compute_dtype == torch.float32
    ):
        output = residual.addcmul_(block, injection.unsqueeze(-1))
    else:
        output = residual + block * injection.unsqueeze(-1)
    return output.flatten(-2).to(self.params_dtype)


def combine(self, block_output, residuals):
    return combine_impl(self, block_output, residuals, use_cache=True, fuse_prefill=False)


def combine_with_prefill_addcmul(self, block_output, residuals):
    return combine_impl(self, block_output, residuals, use_cache=True, fuse_prefill=True)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine}
