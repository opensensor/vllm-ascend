# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental joint down/injection projection for inference hyperconnections.

Only matching FP16 NPU projections are eligible. The checkpoint parameters
stay intact; an instance cache holds a padded NZ copy of the joined weights.
Warmup must prepare that copy before graph capture. This is not a default.
"""

import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp.model import _linear

CUBE_ALIGNMENT = 16
ACL_FORMAT_ND = 2
CACHE_ATTRIBUTE = "_qwen_hc_fused_projection_cache"


class ProjectionResidual:
    """Keep the tiny injection projection across the intervening model block."""

    __slots__ = ("hyper_input", "injection")

    def __init__(self, hyper_input, injection):
        self.hyper_input = hyper_input
        self.injection = injection


def _supports_device(device):
    return device.type == "npu"


def _is_capturing(device):
    return device.type == "npu" and torch.npu.is_current_stream_capturing()


def _weight_key(weight):
    # Parameters made inside inference_mode lack version counters; do not cache
    # them, because an in-place update could otherwise silently reuse old data.
    if weight.is_inference():
        return None
    return (
        id(weight),
        weight.data_ptr(),
        weight._version,
        weight.device,
        weight.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
    )


def _eligible(self):
    if torch.is_grad_enabled() or not self.use_combine:
        return False
    down = self.input_mix_weight_down
    injection = self.block_inject_weight
    up = self.input_mix_weight_up
    return (
        _supports_device(down.device)
        and injection.device == down.device == up.device
        and down.dtype == injection.dtype == up.dtype == self.params_dtype == torch.float16
        and self.compute_dtype == torch.float32
        and down.ndim == 2
        and down.shape[1] == self.hyper_hidden
        and injection.shape == (self.hc_count, self.hyper_hidden)
        and up.shape == (self.hyper_hidden, down.shape[0])
        and not down.is_inference()
        and not injection.is_inference()
    )


def _dense(weight):
    if weight.device.type != "npu":
        return weight
    # Lazy import keeps host preparation free of NPU package initialization.
    import torch_npu

    return torch_npu.npu_format_cast(weight, ACL_FORMAT_ND)


def _format_joined(weight):
    if weight.device.type != "npu":
        return weight
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


def prepare_projection(self):
    """Build/reuse the instance cache outside capture, without changing weights."""
    if not _eligible(self):
        return None
    down = self.input_mix_weight_down
    injection = self.block_inject_weight
    key = (_weight_key(down), _weight_key(injection))
    cached = getattr(self, CACHE_ATTRIBUTE, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if _is_capturing(down.device):
        raise RuntimeError("warm up fused hyperconnection projection after weight changes and before graph capture")
    rows = down.shape[0] + injection.shape[0]
    padding = (-rows) % CUBE_ALIGNMENT
    weights = [_dense(down), _dense(injection)]
    if padding:
        weights.append(down.new_zeros((padding, down.shape[1])))
    joined = _format_joined(torch.cat(weights, dim=0).contiguous())
    setattr(self, CACHE_ATTRIBUTE, (key, joined))
    return joined


def release_projection(self):
    """Release the extra cache only after draining requests and clearing graphs."""
    if hasattr(self, CACHE_ATTRIBUTE):
        delattr(self, CACHE_ATTRIBUTE)


def mix(self, hyper_input):
    num_tokens = hyper_input.shape[0]
    normalized = self._normalize(hyper_input)
    joined = prepare_projection(self)
    if joined is None:
        down = _linear(normalized, self.input_mix_weight_down, self.compute_dtype)
        residual = (hyper_input, normalized)
    else:
        projections = F.linear(normalized.to(joined.dtype), joined)
        lowrank = self.input_mix_weight_down.shape[0]
        down = projections[:, :lowrank]
        # The existing pointwise gate produces a compact tensor, releasing the
        # joined projection output without introducing a separate clone call.
        injection = 2.0 * torch.sigmoid(projections[:, lowrank : lowrank + self.hc_count] / self.hc_count)
        residual = ProjectionResidual(hyper_input, injection)
    gate = F.silu(down / self.hc_count)
    gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
    gate = gate.view(num_tokens, self.hc_count, self.hidden_size)
    mixed = (gate * normalized.view(num_tokens, self.hc_count, self.hidden_size)).mean(dim=-2)
    return mixed.to(self.params_dtype), residual


def combine(self, block_output, residuals):
    if not self.use_combine:
        raise RuntimeError("combine disabled for this gated residual")
    if isinstance(residuals, ProjectionResidual):
        hyper_input = residuals.hyper_input
        injection = residuals.injection
    else:
        hyper_input, normalized = residuals
        projection = _linear(normalized, self.block_inject_weight, self.compute_dtype)
        injection = 2.0 * torch.sigmoid(projection / self.hc_count)
    num_tokens = hyper_input.shape[0]
    residual = hyper_input.view(num_tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)
    block = block_output.to(self.compute_dtype).unsqueeze(-2)
    out = residual + block * injection.unsqueeze(-1)
    return out.flatten(-2).to(self.params_dtype)


def replacements():
    # Define fallback math above instead of retaining the currently installed
    # methods: preparation can run while a previous candidate is still active.
    return {
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix,
        "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine,
    }
