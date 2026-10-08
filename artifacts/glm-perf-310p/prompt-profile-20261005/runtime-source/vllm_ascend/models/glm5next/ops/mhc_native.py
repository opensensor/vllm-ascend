# SPDX-License-Identifier: Apache-2.0
"""Shape-only dispatch for the experimental four-stream mHC prefill kernel."""

import torch

MHC_NATIVE_MIN_TOKENS = 640
MHC_NATIVE_MAX_TOKENS = 32768
MHC_NATIVE_MAX_WIDTH = 16384
MHC_NATIVE_WIDTH_ALIGNMENT = 32
MHC_NATIVE_STREAMS = 4


def native_mhc_post_enabled(config):
    """Validate the explicit experiment selection at model construction."""
    enabled = bool(getattr(config, "ascend_glm_native_mhc_post", False))
    if enabled and not getattr(config, "ascend_glm_mhc_fp16_state", False):
        raise ValueError("ascend_glm_native_mhc_post requires ascend_glm_mhc_fp16_state")
    if enabled and getattr(config, "ascend_glm_prefill_mhc_post", False):
        raise ValueError("Select only one of native and streaming mHC post")
    return enabled


def use_native_mhc_post(op, x, residual, post_mix, comb_mix):
    """Keep decode, unsupported layouts, and BF16 state on the existing path.

    The selected op is deliberately required to exist: an old extension must
    fail visibly rather than silently contaminate a candidate timing run.
    No value reads, copies, device discovery, or synchronization occur here.
    """
    if not getattr(op, "use_310p_native_mhc_post", False) or not getattr(op, "use_310p_fp16_mhc_state", False):
        return False
    if x.ndim != 2 or x.dtype != torch.float16 or not x.is_contiguous():
        return False
    tokens, width = x.shape
    if not (
        MHC_NATIVE_MIN_TOKENS <= tokens <= MHC_NATIVE_MAX_TOKENS
        and 0 < width <= MHC_NATIVE_MAX_WIDTH
        and width % MHC_NATIVE_WIDTH_ALIGNMENT == 0
    ):
        return False
    shapes = (
        (tokens, MHC_NATIVE_STREAMS, width),
        (tokens, MHC_NATIVE_STREAMS, 1),
        (tokens, MHC_NATIVE_STREAMS, MHC_NATIVE_STREAMS),
    )
    return all(
        value.dtype == torch.float32 and value.shape == shape and value.device == x.device and value.is_contiguous()
        for value, shape in zip((residual, post_mix, comb_mix), shapes, strict=True)
    )
