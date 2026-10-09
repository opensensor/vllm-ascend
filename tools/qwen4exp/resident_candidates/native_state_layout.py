# SPDX-License-Identifier: Apache-2.0
"""Reversible native FP32 state IO on the custom Qwen serving path."""

RESOURCE_NAME = "qwen_transfer_v1"


def replacements(native_resources):
    from vllm_ascend.models.qwen4_exp.model import _GDNAttention

    original = _GDNAttention._native_delta_rule
    original = getattr(original, "_qwen_delta_rule_base", original)
    native = native_resources[RESOURCE_NAME]["state_io"]

    def call(self, *args, **kwargs):
        existed = hasattr(self, "_gdn_state_io")
        previous = getattr(self, "_gdn_state_io", None)
        if previous is not None:
            raise ValueError("cannot stack native state IO resources")
        self._gdn_state_io = native
        try:
            return original(self, *args, **kwargs)
        finally:
            if existed:
                self._gdn_state_io = previous
            else:
                del self._gdn_state_io

    call._qwen_delta_rule_base = original
    return {"vllm_ascend.models.qwen4_exp.model:_GDNAttention._native_delta_rule": call}
