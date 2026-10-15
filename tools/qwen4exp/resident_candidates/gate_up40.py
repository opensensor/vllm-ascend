# SPDX-License-Identifier: Apache-2.0
"""Opt-in extended gate/up schedule; requires a passed native transaction."""

import torch

from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND

NATIVE_RESOURCE = "qwen_gate40_v1"
PROFILE = "stream40"


def replacements(native_resources=None):
    if not native_resources or NATIVE_RESOURCE not in native_resources:
        raise RuntimeError("load and qualify qwen_gate40_v1 before preparing this candidate")
    resource = native_resources[NATIVE_RESOURCE][PROFILE]

    def native_linear(self, prepared, group_ends):
        if self.backend != NATIVE_INT4_BACKEND:
            raise ValueError("packed activation inputs require native INT4")
        # Selected MTP2 c1 (30 routes), down projections, grouped prefill and
        # all larger concurrency shapes retain their ordinary qualified op.
        if self.weight.shape[1:] == (1280, 1280) and group_ends.dtype == torch.int32 and 30 < group_ends.numel() <= 40:
            return resource(*prepared, self.weight, self.weight_scale, self.weight_offset, self.weight_sum, group_ends)
        return torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
            *prepared, self.weight, self.weight_scale, self.weight_offset, self.weight_sum, group_ends
        )

    return {"vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear": native_linear}
