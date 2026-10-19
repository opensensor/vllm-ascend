# SPDX-License-Identifier: Apache-2.0
"""Versioned resources for qualified 4/6/12-token native gate/up routes."""

import torch

from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    NATIVE_INT4_BACKEND,
    pack_activation_device,
    swiglu_pack_activation_device,
)


def make_gate_linear(original, gate):
    def linear(self, prepared, routes):
        if (
            self.backend == NATIVE_INT4_BACKEND
            and prepared[0].device.type == "npu"
            and routes.dtype == torch.int32
            and routes.ndim == 1
            and routes.numel() in (40, 60, 120)
            and prepared[0].shape[0] * 10 == routes.numel()
            and self.weight.ndim == 3
            and self.weight.shape[1:] == (1280, 1280)
        ):
            return gate(*prepared, self.weight, self.weight_scale, self.weight_offset, self.weight_sum, routes)
        return original(self, prepared, routes)

    return linear


def make_routed(original, down):
    def routed(self, inputs, weights, ids):
        if (
            not self.native_int4
            or not self.fused_native_down_reduce
            or inputs.device.type != "npu"
            or inputs.shape[0] not in (4, 6)
            or inputs.shape[1] != 2560
            or self.top_k != 10
        ):
            return original(self, inputs, weights, ids)
        routes = (ids - self.expert_offset).to(torch.int32).flatten().contiguous()
        projected = self.projections["gate_up_proj"].native_linear(pack_activation_device(inputs), routes)
        prepared = swiglu_pack_activation_device(projected)
        projection = self.projections["down_proj"]
        return down(
            *prepared,
            projection.weight,
            projection.weight_scale,
            projection.weight_offset,
            projection.weight_sum,
            routes,
            weights.contiguous(),
        )

    return routed
