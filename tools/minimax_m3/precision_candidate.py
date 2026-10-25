# SPDX-License-Identifier: Apache-2.0
"""Resident-only experiment aligning normalization/rotary with the FP16 oracle."""

import torch


def gemma_forward(self, x, residual=None):
    if residual is not None:
        residual_dtype = residual.dtype
        x = x + residual.to(x.dtype)
        residual = x.to(residual_dtype)
    values = x.float()
    inverse = torch.rsqrt((values * values).mean(-1, keepdim=True) + self.variance_epsilon)
    # Preserve 1 + weight in FP32 until the normalized output is rounded.
    output = (values * inverse * (1 + self.weight.float())).to(x.dtype)
    return output if residual is None else (output, residual)


def fused_gemma_forward(self, x, residual=None):
    """Normalize the FP32 residual sum, matching FlashInfer's fused convention.

    The stored residual is rounded separately; it must not become the input
    to normalization. Keep this distinct from the earlier FP16-oracle trial.
    """
    values = x.float()
    if residual is not None:
        values = values + residual.float()
        residual = values.to(residual.dtype)
    inverse = torch.rsqrt((values * values).mean(-1, keepdim=True) + self.variance_epsilon)
    output = (values * inverse * (1 + self.weight.float())).to(x.dtype)
    return output if residual is None else (output, residual)


def fused_replacements(native_resources):
    return {"vllm_ascend._310p.ops.layernorm:AscendGemmaRMSNorm310.forward": fused_gemma_forward}


def rotary_forward(self, positions, query, key, is_neox_style, offsets=None):
    if offsets is not None or not is_neox_style:
        raise ValueError("M3 precision candidate requires offset-free NeoX rotary")
    cache = self.cos_sin_cache.to(device=query.device, dtype=query.dtype)
    cosine, sine = cache.index_select(0, positions).chunk(2, -1)
    cosine, sine = cosine[:, None], sine[:, None]

    def rotate(value):
        shaped = value.view(len(positions), -1, self.head_size)
        half = self.rotary_dim // 2
        left, right = shaped[..., :half], shaped[..., half : self.rotary_dim]
        return torch.cat(
            (left * cosine - right * sine, right * cosine + left * sine, shaped[..., self.rotary_dim :]), -1
        ).reshape(value.shape)

    return rotate(query), rotate(key)


def replacements(native_resources):
    return {
        "vllm_ascend._310p.ops.layernorm:AscendGemmaRMSNorm310.forward": gemma_forward,
        "vllm_ascend._310p.ops.rotary_embedding:_rope_forward_oot": rotary_forward,
    }
