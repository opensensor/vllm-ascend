#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import torch_npu

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import maybe_trans_nz

INT8_SYMMETRIC_MAX = 127
GDN_QUANTIZED_PROJECTION_SUFFIXES = (
    ".linear_attn.in_proj_qkvz",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
)


def is_gdn_quantized_projection(prefix: str) -> bool:
    """Return whether ``prefix`` names a GDN projection converted to W8A8."""
    return prefix.endswith(GDN_QUANTIZED_PROJECTION_SUFFIXES)


def quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize an ``[out, in]`` FLOAT weight to INT8.

    The scale is kept in fp32 and zero rows use a scale of one. The latter is
    exact for a zero row and avoids a device-side divide by zero without a
    CPU/NPU synchronization.
    """
    weight_fp32 = weight.to(torch.float32)
    absmax = weight_fp32.abs().amax(dim=1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / INT8_SYMMETRIC_MAX)
    quantized_weight = torch.round(weight_fp32 / scale.unsqueeze(1)).clamp(
        -INT8_SYMMETRIC_MAX,
        INT8_SYMMETRIC_MAX,
    )
    return quantized_weight.to(torch.int8), scale


class AscendGDNW8A8LinearMethod310(AscendUnquantizedLinearMethod):
    """Load FLOAT GDN projections, then use dynamic W8A8 at inference."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        quantized_weight, weight_scale = quantize_weight_per_output_channel(layer.weight.data)
        # Match the layout consumed by the native 310P W8A8_DYNAMIC method.
        layer.weight.data = maybe_trans_nz(quantized_weight).transpose(0, 1)
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(weight_scale, requires_grad=False),
        )

    @staticmethod
    def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        needs_unsqueeze = pertoken_scale.dim() == 2
        if needs_unsqueeze:
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        return quantized_x, pertoken_scale, needs_unsqueeze

    @staticmethod
    def apply_quantized(
        layer: torch.nn.Module,
        quantized_x: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        needs_unsqueeze: bool,
    ) -> torch.Tensor:
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight.data,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=None,
            output_dtype=output_dtype,
        )
        return output.unsqueeze(dim=1) if needs_unsqueeze else output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("GDN W8A8 projections do not support bias")
        quantized_x, pertoken_scale, needs_unsqueeze = self.quantize_activation(x)
        return self.apply_quantized(
            layer,
            quantized_x,
            pertoken_scale,
            x.dtype,
            needs_unsqueeze,
        )
