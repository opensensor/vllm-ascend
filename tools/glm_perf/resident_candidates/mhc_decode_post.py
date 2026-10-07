# SPDX-License-Identifier: Apache-2.0
"""Opt-in native mHC post experiment for the qualified GLM decode shapes."""

import torch

MAX_DECODE_ROWS = 8
GLM_HIDDEN_SIZE = 4096
RESIDUAL_STREAMS = 4


def select_native_post(original, op, x, residual, post_mix, comb_mix):
    if original(op, x, residual, post_mix, comb_mix):
        return True
    if not getattr(op, "use_310p_native_mhc_post", False) or not getattr(op, "use_310p_fp16_mhc_state", False):
        return False
    if (
        x.ndim != 2
        or x.dtype != torch.float16
        or not x.is_contiguous()
        or not 1 <= x.shape[0] <= MAX_DECODE_ROWS
        or x.shape[1] != GLM_HIDDEN_SIZE
    ):
        return False
    rows = x.shape[0]
    shapes = (
        (rows, RESIDUAL_STREAMS, GLM_HIDDEN_SIZE),
        (rows, RESIDUAL_STREAMS, 1),
        (rows, RESIDUAL_STREAMS, RESIDUAL_STREAMS),
    )
    return all(
        value.dtype == torch.float32 and value.shape == shape and value.device == x.device and value.is_contiguous()
        for value, shape in zip((residual, post_mix, comb_mix), shapes, strict=True)
    )


def extend_replacements(changes):
    # Resolve the existing, separately qualified prefill implementation only
    # inside the worker. This experiment changes its small-shape selection.
    from vllm_ascend.patch.worker import patch_mhc_norm

    target = "vllm_ascend.patch.worker.patch_mhc_norm:use_native_mhc_post"
    original = changes.get(target, patch_mhc_norm.use_native_mhc_post)
    original = getattr(original, "__glm_decode_post_original__", original)

    def select(op, x, residual, post_mix, comb_mix):
        return select_native_post(original, op, x, residual, post_mix, comb_mix)

    select.__glm_decode_post_original__ = original
    result = dict(changes)
    result[target] = select
    return result
