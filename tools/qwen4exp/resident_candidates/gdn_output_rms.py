# SPDX-License-Identifier: Apache-2.0
"""Decode-only native RMSNorm affine for GDN output; no precision policy change."""

import torch

from tools.qwen4exp.direct_gdn_output import HEAD_WIDTH, MAX_ROWS
from tools.qwen4exp.resident_candidates.fused_hc_projection import _is_capturing, _weight_key
from vllm_ascend.models.qwen4_exp.model import _linear, _rms_norm

# The six-chip MTP2 maximum-concurrency graph contains 18 scheduled tokens.
# A direct fused op still obeys its independent MAX_ROWS bound; native RMSNorm
# can cover this whole graph without changing FP32 accumulation or TP reduce.
MAX_DECODE_TOKENS = 18
CACHE_ATTRIBUTE = "_qwen_gdn_output_norm_weight"


def _supports_device(device):
    return device.type == "npu"


def prepare_norm_weight(self):
    """Retain a small FP32 gamma copy; verify host-side identity/version only."""
    weight = self.norm_weight
    key = _weight_key(weight)
    if key is None:
        return weight.to(self.compute_dtype)
    cached = getattr(self, CACHE_ATTRIBUTE, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if _is_capturing(weight.device):
        raise RuntimeError("warm up GDN norm weights after changes and before capture")
    value = weight.to(self.compute_dtype)
    setattr(self, CACHE_ATTRIBUTE, (key, value))
    return value


def _project_output(self, block_input, out, op=None, gate_probabilities=False):
    seq_len = out.shape[0]
    eligible = (
        not torch.is_grad_enabled()
        and _supports_device(out.device)
        and 0 < seq_len <= MAX_DECODE_TOKENS
        and out.dtype == self.compute_dtype == torch.float32
        and out.is_contiguous()
    )
    if eligible and op is not None and out.shape[-1] == HEAD_WIDTH and seq_len * self.num_v_heads <= MAX_ROWS:
        z = _linear(block_input, self.in_proj_z, self.compute_dtype).reshape(seq_len * self.num_v_heads, HEAD_WIDTH)
        gated = op(
            out.reshape(-1, HEAD_WIDTH),
            prepare_norm_weight(self),
            torch.sigmoid(z) if gate_probabilities else z,
        ).reshape(seq_len, self.value_dim)
        result = _linear(gated, self.out_proj, self.compute_dtype)
        if self.tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError("Qwen GDN TP requires all-reduce")
            result = self._tp_reduce(result)
        return result.to(self.params_dtype)
    if eligible:
        import torch_npu

        normed, _ = torch_npu.npu_rms_norm(out, prepare_norm_weight(self), self.rms_norm_eps)
    else:
        normed = _rms_norm(out, self.norm_weight, self.rms_norm_eps, self.compute_dtype)
    z = _linear(block_input, self.in_proj_z, self.compute_dtype).reshape(
        seq_len, self.num_v_heads, self.params.head_v_dim
    )
    gated = (normed * torch.sigmoid(z)).reshape(seq_len, self.value_dim)
    result = _linear(gated, self.out_proj, self.compute_dtype)
    if self.tp_size > 1:
        if self._tp_reduce is None:
            raise RuntimeError("Qwen GDN TP requires all-reduce")
        result = self._tp_reduce(result)
    return result.to(self.params_dtype)


def project_output(self, block_input, out):
    return _project_output(self, block_input, out)


def make_project_output(op, *, gate_probabilities=False):
    def project(self, block_input, out):
        return _project_output(self, block_input, out, op, gate_probabilities)

    return project


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GDNAttention._project_output": project_output}
