# SPDX-License-Identifier: Apache-2.0
"""Test dispatch selection for existing DeepSeek flags, using resident weights."""

DECODE_SELECTION = (True, True)


def select_decode_flags(original, selection):
    original = getattr(original, "__decode_flags_original__", original)

    def selected(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert):
        saved = experts.decode_swiglu, experts.decode_combine
        enabled = tuple(flag and not experts.offload_to_cpu for flag in selection)
        if saved == enabled:
            return original(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert)
        experts.decode_swiglu, experts.decode_combine = enabled
        try:
            return original(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert)
        finally:
            experts.decode_swiglu, experts.decode_combine = saved

    selected.__decode_flags_original__ = original
    return selected


def replacements(native_resources):
    from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310

    changes = qualified_replacements(native_resources)  # noqa: F821 - composed with qualified factory
    original = AscendW2DynamicFusedMoEMethod310._apply_device_grouped
    changes[
        "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
    ] = select_decode_flags(original, DECODE_SELECTION)
    target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    status = changes[target]

    def selected_status(self):
        receipt = status(self)
        receipt["decode_selection"] = {"swiglu": DECODE_SELECTION[0], "combine": DECODE_SELECTION[1]}
        return receipt

    changes[target] = selected_status
    return changes
