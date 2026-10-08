# SPDX-License-Identifier: Apache-2.0
"""Compose qualified direct residual math with the padded HC projection."""

import torch
import torch.nn.functional as F

from tools.qwen4exp.direct_hc_residual import MAX_TOKENS, MAX_WIDTH, TILE_WIDTH
from tools.qwen4exp.resident_candidates.padded_hc_injection import (
    combine_with_prefill_addcmul,
    prepare_injection,
)
from vllm_ascend.models.qwen4_exp.model import _linear

RESOURCE_NAME = "hc_residual_v1"


def _supports_device(device):
    return device.type == "npu"


def make_combine(op, *, decode_only=False):
    def combine(self, block_output, residuals):
        if not self.use_combine:
            raise RuntimeError("combine disabled for this gated residual")
        hyper_input, normalized = residuals
        if (
            torch.is_grad_enabled()
            or not _supports_device(hyper_input.device)
            or hyper_input.dtype != torch.float16
            or block_output.dtype != torch.float16
            or self.params_dtype != torch.float16
            or self.compute_dtype != torch.float32
            or self.hc_count != 4
            or self.hidden_size % TILE_WIDTH
            or self.hidden_size > MAX_WIDTH
            or not 0 < hyper_input.shape[0] <= MAX_TOKENS
            or not hyper_input.is_contiguous()
            or not block_output.is_contiguous()
            or (decode_only and hyper_input.shape[0] > 6)
        ):
            return combine_with_prefill_addcmul(self, block_output, residuals)
        padded = prepare_injection(self)
        if padded is None:
            projection = _linear(normalized, self.block_inject_weight, self.compute_dtype)
        else:
            projection = F.linear(normalized.to(padded.dtype), padded)[:, : self.hc_count]
        injection = 2.0 * torch.sigmoid(projection / self.hc_count)
        return op(hyper_input, block_output, injection)

    return combine


def replacements(native_resources):
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": make_combine(native_resources[RESOURCE_NAME])}
