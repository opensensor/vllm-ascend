# SPDX-License-Identifier: Apache-2.0
"""Explicit prefill-only WY replacement after native and recurrent-state gates."""

RESOURCE_NAME = "qwen_prefill_v2"


def replacements(native_resources):
    from vllm_ascend._310p.ops.fla import gdn_310

    native = native_resources[RESOURCE_NAME]["wy"]
    original = gdn_310.chunk_gated_delta_rule_310
    original = getattr(original, "_qwen_prefill_base", original)

    def chunk(*args, **kwargs):
        if kwargs.get("wy_prepare") is not None:
            raise ValueError("cannot stack multiple WY preparation candidates")
        kwargs["wy_prepare"] = native
        return original(*args, **kwargs)

    chunk._qwen_prefill_base = original
    return {"vllm_ascend._310p.ops.fla.gdn_310:chunk_gated_delta_rule_310": chunk}
