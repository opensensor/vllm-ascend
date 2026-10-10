# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse HC residual buffers while retaining the original injection projection.

Padding or joining the four-row injection GEMM can change CANN's FP16 result.
This candidate leaves its shape, layout, and sigmoid rounding unchanged. Only
the following FP32 residual arithmetic uses the admitted native kernel.
"""

import torch

from tools.qwen4exp.direct_hc_residual import MAX_TOKENS, MAX_WIDTH, TILE_WIDTH
from tools.qwen4exp.resident_candidates import fused_hc_projection, release_hc_projection, shared_hc_operand
from vllm_ascend.models.qwen4_exp.model import _linear

MIN_FUSION_TOKENS = 128
RESOURCE_NAME = "hc_residual_v3"


def _supports_device(device):
    return device.type == "npu"


def make_combine(op):
    def combine(self, block_output, residuals):
        if not self.use_combine:
            raise RuntimeError("combine disabled for this gated residual")
        hyper, normalized = residuals
        if (
            torch.is_grad_enabled()
            or not _supports_device(hyper.device)
            or block_output.device != hyper.device
            or hyper.dtype != torch.float16
            or block_output.dtype != torch.float16
            or self.params_dtype != torch.float16
            or self.compute_dtype != torch.float32
            or self.hc_count != 4
            or not 0 < self.hidden_size <= MAX_WIDTH
            or self.hidden_size % TILE_WIDTH
            or not MIN_FUSION_TOKENS <= hyper.shape[0] <= MAX_TOKENS
            or not hyper.is_contiguous()
            or not block_output.is_contiguous()
        ):
            return fused_hc_projection.combine(self, block_output, residuals)
        # Preserve the exact original GEMM, divide, sigmoid, and FP16 multiply.
        injection = 2.0 * torch.sigmoid(
            _linear(normalized, self.block_inject_weight, self.compute_dtype) / self.hc_count
        )
        return op(hyper, block_output, injection)

    return combine


def mix(self, hyper_input):
    # Old diagnostic joined caches are redundant here. The resident harness
    # clears graphs before warmup, so release their storage before recapture.
    # Real serving trials improved cold prefill but did not improve C1 decode.
    # Preserve the original small-batch graph instead of extrapolating an
    # isolated operation's improvement to whole-model throughput.
    if hyper_input.shape[0] < MIN_FUSION_TOKENS:
        return release_hc_projection.mix(self, hyper_input)
    if hasattr(self, fused_hc_projection.CACHE_ATTRIBUTE):
        if fused_hc_projection._is_capturing(hyper_input.device):
            raise RuntimeError("release joined HC weights during warmup, after clearing old graphs")
        fused_hc_projection.release_projection(self)
    return shared_hc_operand.mix(self, hyper_input)


def replacements(native_resources):
    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix,
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": make_combine(native_resources[RESOURCE_NAME]),
    }
