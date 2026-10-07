# SPDX-License-Identifier: Apache-2.0
"""Unqualified row-bounded selection candidate; no change until applied."""

from functools import partial

from vllm_ascend.models.glm5next.kpool_ops import topk_pool_indices

PREFILL_TOPK_ROWS_PER_CALL = 32


def replacements():
    return {
        "vllm_ascend.models.glm5next.kpool_ops:topk_pool_indices": partial(
            topk_pool_indices, rows_per_call=PREFILL_TOPK_ROWS_PER_CALL
        ),
    }
