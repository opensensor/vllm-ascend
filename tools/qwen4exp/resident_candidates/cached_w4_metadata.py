# SPDX-License-Identifier: Apache-2.0
"""Use per-output-tile scale/offset/sum caching only for grouped prefill."""

RESOURCE_NAME = "qwen_transfer_v1"


def replacements(native_resources):
    import torch

    from vllm_ascend.models.qwen4_exp.w4_moe import NATIVE_INT4_BACKEND, PackedExpertBank

    original = PackedExpertBank.native_linear
    original = getattr(original, "_qwen_packed_base", original)
    native = native_resources[RESOURCE_NAME]["cached_metadata"]

    def linear(self, prepared, group_ends):
        if group_ends.dtype != torch.int64 or prepared[0].shape[0] <= 128:
            return original(self, prepared, group_ends)
        if self.backend != NATIVE_INT4_BACKEND:
            raise ValueError("metadata caching requires native INT4")
        return native(self, prepared, group_ends)

    linear._qwen_packed_base = original
    return {"vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear": linear}
