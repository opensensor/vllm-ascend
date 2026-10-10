# SPDX-License-Identifier: Apache-2.0
"""Combine the joined HC projection with the admitted native residual kernel."""

import torch

from tools.qwen4exp.direct_hc_residual import MAX_TOKENS, MAX_WIDTH, TILE_WIDTH
from tools.qwen4exp.resident_candidates import fused_hc_projection, native_hc_residual


def _supports_device(device):
    return device.type == "npu"


def make_combine(op):
    separate = native_hc_residual.make_combine(op)

    def combine(self, block_output, residuals):
        if not isinstance(residuals, fused_hc_projection.ProjectionResidual):
            return separate(self, block_output, residuals)
        hyper = residuals.hyper_input
        injection = residuals.injection
        if (
            not torch.is_grad_enabled()
            and self.use_combine
            and _supports_device(hyper.device)
            and block_output.device == injection.device == hyper.device
            and hyper.dtype == block_output.dtype == self.params_dtype == torch.float16
            and injection.dtype in (torch.float16, torch.float32)
            and self.compute_dtype == torch.float32
            and self.hc_count == 4
            and 0 < self.hidden_size <= MAX_WIDTH
            and self.hidden_size % TILE_WIDTH == 0
            and 0 < hyper.shape[0] <= MAX_TOKENS
            and hyper.is_contiguous()
            and block_output.is_contiguous()
            and injection.is_contiguous()
        ):
            return op(hyper, block_output, injection)
        return fused_hc_projection.combine(self, block_output, residuals)

    return combine


def replacements(native_resources):
    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": fused_hc_projection.mix,
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": make_combine(
            native_resources[native_hc_residual.RESOURCE_NAME]
        ),
    }
