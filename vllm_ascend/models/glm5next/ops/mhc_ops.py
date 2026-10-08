# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection width helpers used by the GLM-5.3-Flash decoder layers.

Upstream keeps these next to the MHC ops in ``vllm.model_executor.layers.mhc``.
They are plain shape ops with no backend-specific behavior, so vLLM Ascend
carries its own copy while the GLM-5.3-Flash architecture lives downstream.
"""

import torch

# The isolated 310P benchmark qualified only large prefill shapes. Smaller
# calls (including decode graphs) retain the existing post mixer.
MHC_STREAMING_POST_MIN_TOKENS = 640


def mhc_post_streaming(x, residual, post_mix, comb_mix):
    """Mix streams in FP32 with bounded scratch instead of a general einsum.

    Inputs are never mutated. The accumulation/FMA order differs from einsum;
    serving dispatch is experimental and restricted to FP16-rounded state.
    """
    streams = residual.shape[-2]
    if streams < 1:
        raise ValueError("at least one residual stream is required")
    residual_fp32 = residual.float()
    comb_fp32 = comb_mix.float()
    output = residual_fp32[..., :1, :] * comb_fp32[..., 0, :, None]
    for stream in range(1, streams):
        output.addcmul_(residual_fp32[..., stream : stream + 1, :], comb_fp32[..., stream, :, None])
    output.add_(post_mix.float() * x.unsqueeze(-2).float())
    return output.to(residual.dtype)


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, hidden_size] -> [s, n * hidden_size] by replication."""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, n * hidden_size] -> [s, hidden_size] by averaging."""
    return x.mean(dim=1)
