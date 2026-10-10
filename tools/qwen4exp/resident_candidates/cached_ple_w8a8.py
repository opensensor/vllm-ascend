# SPDX-License-Identifier: Apache-2.0
"""Opt-in PLE W8 comparison retaining original weights for resident rollback.

This changes projection precision to the existing dynamic-W8A8 policy. Its
six-chip graph/image gates passed, but its trial did not improve decode speed.
It remains an opt-in comparison, not a serving default.
"""

import torch

from tools.qwen4exp.resident_candidates.fused_hc_projection import _is_capturing, _weight_key
from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import (
    QUANT_MATMUL_ALIGNMENT,
    dynamic_w8a8_linear,
    quantize_linear_weight,
)

CACHE_ATTRIBUTE = "_qwen_ple_w8_shadow_cache"


def _supports_device(device):
    return device.type == "npu"


def prepare_weight(self):
    weight = self.kv_proj_weight
    if (
        torch.is_grad_enabled()
        or not _supports_device(weight.device)
        or self.projection_execution != "float16"
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or any(size % QUANT_MATMUL_ALIGNMENT for size in weight.shape)
    ):
        return None
    key = _weight_key(weight)
    if key is None:
        return None
    cached = getattr(self, CACHE_ATTRIBUTE, None)
    if cached is not None and cached[0] == key:
        return cached[1:]
    if _is_capturing(weight.device):
        raise RuntimeError("warm up the PLE W8 cache after weight changes and before graph capture")
    # Match the existing load-time PLE policy, including CPU weight rounding.
    # This transfer is preparation only; cache hits perform no host copies.
    packed, scales = quantize_linear_weight(weight.detach().cpu())
    if weight.device.type == "npu":
        from vllm_ascend.utils import maybe_trans_nz

        packed = maybe_trans_nz(packed.to(weight.device)).transpose(0, 1)
        scales = scales.to(weight.device)
    else:
        packed = packed.transpose(0, 1).contiguous()
    setattr(self, CACHE_ATTRIBUTE, (key, packed, scales))
    return packed, scales


def project(self, embeddings, *, fallback):
    cached = prepare_weight(self)
    if cached is None:
        return fallback(self, embeddings)
    return dynamic_w8a8_linear(embeddings, *cached)


def release_weight(self):
    if hasattr(self, CACHE_ATTRIBUTE):
        if _is_capturing(self.kv_proj_weight.device):
            raise RuntimeError("release the PLE W8 cache only after clearing old graphs")
        delattr(self, CACHE_ATTRIBUTE)


def replacements():
    from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer

    original = AscendQwen4ExpPLELayer.project_merged

    def project_merged(self, embeddings):
        return project(self, embeddings, fallback=original)

    return {"vllm_ascend.models.qwen4_exp.ple_layer:AscendQwen4ExpPLELayer.project_merged": project_merged}
