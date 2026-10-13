# SPDX-License-Identifier: Apache-2.0
"""Explicit prefill binding; keep existing decode and smaller schedules."""

import torch

RESOURCE_NAME = "qwen_prefill_v5"
LARGE_ROWS_PER_EXPERT = 256
PROJECTION_SHAPES = ((1280, 2560), (2560, 640))


def replacements(native_resources):
    from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank
    from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND

    resource = native_resources[RESOURCE_NAME]["projection"]
    original = PackedExpertBank.native_linear
    original = getattr(original, "_qwen_grouped_prefill_base", original)

    def native_linear(self, prepared, group_ends):
        experts, outputs, packed_inputs = self.weight.shape
        if (
            self.backend == NATIVE_INT4_BACKEND
            and group_ends.dtype == torch.int64
            and prepared[0].shape[0] > experts * LARGE_ROWS_PER_EXPERT
            and (outputs, packed_inputs * 2) in PROJECTION_SHAPES
        ):
            return resource(self, prepared, group_ends)
        return original(self, prepared, group_ends)

    native_linear._qwen_grouped_prefill_base = original
    return {"vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear": native_linear}
