# SPDX-License-Identifier: Apache-2.0
"""Compose qualified padded injection and the prefill-only residual candidate."""

from tools.qwen4exp.resident_candidates.padded_hc_injection import combine_with_prefill_addcmul


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine_with_prefill_addcmul}
