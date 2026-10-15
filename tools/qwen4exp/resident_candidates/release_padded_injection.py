# SPDX-License-Identifier: Apache-2.0
"""Release padded injection caches during drained warmup after graph clearing."""

from tools.qwen4exp.resident_candidates.padded_hc_injection import CACHE_ATTRIBUTE, _is_capturing, combine_impl


def combine(self, block_output, residuals):
    if hasattr(self, CACHE_ATTRIBUTE):
        if _is_capturing(block_output.device):
            raise RuntimeError("release padded HC injection cache before graph capture")
        delattr(self, CACHE_ATTRIBUTE)
    return combine_impl(self, block_output, residuals, use_cache=False, fuse_prefill=False)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine": combine}
