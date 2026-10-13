# SPDX-License-Identifier: Apache-2.0
"""Fuse residual arithmetic while preserving the injection GEMM and rounding.

Unlike the padded-injection experiment, this leaves the projection's shape,
layout, divide and sigmoid unchanged. Load and validate the versioned native
resource on every worker before installing this resident candidate.
"""

import torch

from tools.qwen4exp.direct_hc_residual import MAX_TOKENS, MAX_WIDTH, TILE_WIDTH

RESOURCE_NAME = "hc_residual_v3"


def _supports_device(device):
    return device.type == "npu"


def make_combine(op, *, linear, fallback):
    def combine(self, block_output, residuals):
        hyper_input, normalized = residuals
        if (
            torch.is_grad_enabled()
            or not self.use_combine
            or not _supports_device(hyper_input.device)
            or block_output.device != hyper_input.device
            or hyper_input.dtype != torch.float16
            or block_output.dtype != torch.float16
            or self.params_dtype != torch.float16
            or self.compute_dtype != torch.float32
            or self.hc_count != 4
            or self.hidden_size % TILE_WIDTH
            or not 0 < self.hidden_size <= MAX_WIDTH
            or not 0 < hyper_input.shape[0] <= MAX_TOKENS
            or not hyper_input.is_contiguous()
            or not block_output.is_contiguous()
        ):
            return fallback(self, block_output, residuals)
        injection = 2.0 * torch.sigmoid(
            linear(normalized, self.block_inject_weight, self.compute_dtype) / self.hc_count
        )
        return op(hyper_input, block_output, injection)

    return combine


def replacements(native_resources):
    # Worker-only import: importing this helper for CPU admission/tests must
    # not initialize the model registry or any NPU context.
    from vllm_ascend.models.qwen4_exp.model import _GatedResidual, _linear

    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": make_combine(
            native_resources[RESOURCE_NAME], linear=_linear, fallback=_GatedResidual.combine
        )
    }
