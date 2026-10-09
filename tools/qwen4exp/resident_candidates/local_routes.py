# SPDX-License-Identifier: Apache-2.0
"""Limit route preparation to device-counted local rows; reversible binding."""

from types import SimpleNamespace

RESOURCE_NAME = "qwen_prefill_v2"


def replacements(native_resources):
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

    resources = native_resources[RESOURCE_NAME]
    local = SimpleNamespace(gather=resources["route_gather"], swiglu=resources["local_swiglu"])
    original = W4SparseMoE._forward_grouped_chunk
    original = getattr(original, "_qwen_prefill_base", original)

    def forward(self, inputs, weights, ids):
        if not self.native_int4 or self.grouped_activation != "cann_swiglu_pack":
            raise ValueError("local routes requires native INT4 with cann_swiglu_pack; gate this precision separately")
        existed = hasattr(self, "_qwen_local_routes")
        previous = getattr(self, "_qwen_local_routes", None)
        if previous is not None:
            raise ValueError("cannot stack local-route resources")
        self._qwen_local_routes = local
        try:
            return original(self, inputs, weights, ids)
        finally:
            if existed:
                self._qwen_local_routes = previous
            else:
                del self._qwen_local_routes

    forward._qwen_prefill_base = original
    return {"vllm_ascend.models.qwen4_exp.w4_moe:W4SparseMoE._forward_grouped_chunk": forward}
