# SPDX-License-Identifier: Apache-2.0
"""Replace supported packed expert projections; retain surrounding fusions."""

METHOD_TARGET = (
    "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
)
STATUS_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
PERMANENT_METHOD_TARGET = "vllm_ascend.model_loader.glm_native_int4:NativeInt4MoEMethod._apply_device_grouped"


def wrap_projection(original, projection, audit=None, fused_pipeline=False):
    original = getattr(original, "__glm_reconstruction_original__", original)
    if fused_pipeline:
        original = getattr(original, "__decode_flags_original__", original)

    def reconstructed(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert):
        def selected(inputs, codes, scales, ends, *args):
            supported = projection.supports(inputs, codes, scales, ends)
            if audit is not None:
                key = "native_dispatches" if supported else "fallback_dispatches"
                audit[key] += 1
                if "bank_coverage" in audit and hasattr(inputs, "shape"):
                    bits = codes.shape[-1] * 8 // inputs.shape[-1]
                    signature = f"W{bits}:R{inputs.shape[0]}:E{codes.shape[0]}:N{codes.shape[1]}:K{inputs.shape[-1]}"
                    bank = audit["bank_coverage"].setdefault(signature, {"native": 0, "fallback": 0})
                    bank["native" if supported else "fallback"] += 1
            operation = projection if supported else grouped_op
            return operation(inputs, codes, scales, ends, *args)

        if not fused_pipeline:
            return original(self, selected, experts, x, topk_weights, topk_ids, shared_expert)
        # Select existing native SwiGLU and FP32 route combine for the complete
        # MoE path. Restore bank flags even if projection/capture fails.
        flags = ("decode_swiglu", "prefill_swiglu", "decode_combine", "fp32_route_combine")
        saved = {name: (hasattr(experts, name), getattr(experts, name, False)) for name in flags}
        try:
            for name in flags:
                setattr(experts, name, True)
            if audit is not None:
                audit["fused_pipeline_calls"] = audit.get("fused_pipeline_calls", 0) + 1
            return original(self, selected, experts, x, topk_weights, topk_ids, shared_expert)
        finally:
            for name, (existed, value) in saved.items():
                if existed:
                    setattr(experts, name, value)
                else:
                    delattr(experts, name)

    # Existing candidate factories use their own unwrap markers while preparing
    # the next generation. Preserve those markers so they cannot retain this
    # native wrapper inside a seemingly restored selector.
    reconstructed.__dict__.update(getattr(original, "__dict__", {}))
    reconstructed.__glm_reconstruction_original__ = original
    return reconstructed


def extend_replacements(changes, native_resources, profile, resource_name="reconstruction_v1"):
    # Worker-only import keeps host tools independent of plugin initialization.
    from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310
    from vllm_ascend._310p.worker_310p import NPUWorker310

    projection = native_resources[resource_name][profile]
    result = dict(changes)
    original = result.get(METHOD_TARGET, AscendW2DynamicFusedMoEMethod310._apply_device_grouped)
    audit = {"profile": profile, "native_dispatches": 0, "fallback_dispatches": 0, "bank_coverage": {}}
    result[METHOD_TARGET] = (
        wrap_fused_moe(original, projection, audit)
        if profile.startswith("fused_int4a")
        else wrap_projection(original, projection, audit, fused_pipeline=profile == "int4a8")
    )
    if profile.startswith("fused_int4a"):
        from vllm_ascend.model_loader.glm_native_int4 import NativeInt4MoEMethod

        result[PERMANENT_METHOD_TARGET] = wrap_fused_moe(NativeInt4MoEMethod._apply_device_grouped, projection, audit)
    original_status = result.get(STATUS_TARGET, NPUWorker310.resident_status)

    def selected_status(self):
        receipt = original_status(self)
        receipt["reconstruction"] = dict(audit)
        receipt["reconstruction"]["bank_coverage"] = {key: dict(value) for key, value in audit["bank_coverage"].items()}
        return receipt

    result[STATUS_TARGET] = selected_status
    if getattr(projection, "prepared_weight_layout", False):
        # Preparation runs only while paused, before graph capture; the apply
        # hook restores exact original bytes before baseline hooks are removed.
        result = native_resources[resource_name]["_weight_layout"].wrap_worker_hooks(result, NPUWorker310)
    return result


def wrap_fused_moe(original, native, audit):
    original = getattr(original, "__glm_reconstruction_original__", original)
    original = getattr(original, "__decode_flags_original__", original)

    def fused(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert):
        inputs, weights = x.to(dtype=native.input_dtype).contiguous(), topk_weights.float().contiguous()
        banks = (
            experts.gate_up_packed_bank,
            experts.gate_up_scale_bank,
            experts.down_packed_bank,
            experts.down_scale_bank,
        )
        ids = topk_ids.contiguous()
        try:
            geometry = native.geometry(inputs, *banks, weights, ids)
        except ValueError:
            if getattr(native, "prepared_weight_layout", False):
                raise ValueError("prepared weight banks require complete native geometry") from None
            audit["fallback_dispatches"] += 2
            return original(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert)
        audit["native_dispatches"] += 2
        audit["kernel_fused_calls"] = audit.get("kernel_fused_calls", 0) + 1
        audit["activation_bits"] = geometry.activation_bits
        for bits, n, k in (
            (geometry.gate_bits, 2 * geometry.intermediate, geometry.hidden),
            (geometry.down_bits, geometry.hidden, geometry.intermediate),
        ):
            key = f"W{bits}:R{geometry.tokens * geometry.top_k}:E{geometry.experts}:N{n}:K{k}"
            audit["bank_coverage"].setdefault(key, {"native": 0, "fallback": 0})["native"] += 1
        output = native(inputs, *banks, weights, ids, experts.local_expert_offset)
        if shared_expert is not None:
            output += shared_expert.forward(x).float()
        return output

    fused.__dict__.update(getattr(original, "__dict__", {}))
    fused.__glm_reconstruction_original__ = original
    return fused
