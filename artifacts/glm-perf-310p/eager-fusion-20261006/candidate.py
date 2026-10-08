# SPDX-License-Identifier: Apache-2.0
"""Replace eager MoE activation/combine and rounded mHC post at every GLM batch size."""

FUSION_SELECTION = (True, True, True)


def replacements(native_resources):
    import inspect
    import textwrap

    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp

    from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310

    fuse_swiglu, fuse_combine, fuse_post = FUSION_SELECTION
    changes = qualified_replacements(native_resources)  # noqa: F821
    operation = native_resources["glm_eager_fusions_v1"]

    def rewrite(original, substitutions, additions):
        source = textwrap.dedent(inspect.getsource(original))
        for old, new in substitutions:
            if source.count(old) != 1:
                raise ValueError("runtime function changed; refuse partial fusion")
            source = source.replace(old, new)
        namespace = dict(original.__globals__, **additions)
        exec(compile(source, "<glm-resident-eager-fusion>", "exec"), namespace)
        wrapped = namespace[original.__name__]
        wrapped.__glm_fusion_original__ = original
        return wrapped

    original = AscendW2DynamicFusedMoEMethod310._apply_device_grouped
    original = getattr(original, "__glm_fusion_original__", original)
    source = textwrap.dedent(inspect.getsource(original))
    begin = source.index("    fp32_combine = ")
    end = source.index("    if fp32_combine and fp32_combine_op is None:", begin)
    combine_block = source[begin:end]
    begin = source.index("    use_swiglu = ")
    end = source.index("    if use_swiglu and swiglu_op is None:", begin)
    swiglu_block = source[begin:end]
    substitutions = []
    if fuse_combine:
        substitutions.append((combine_block, "    fp32_combine = True\n    fp32_combine_op = _resident_combine\n"))
    if fuse_swiglu:
        substitutions.append((swiglu_block, "    use_swiglu = has_fused_gate_up\n    swiglu_op = _resident_swiglu\n"))
    if substitutions:
        grouped = rewrite(
            original, substitutions, {"_resident_combine": operation.combine, "_resident_swiglu": operation.swiglu}
        )
        changes[
            "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
        ] = grouped

    if fuse_post:
        original = MHCFusedPostPreOp.forward_oot
        original = getattr(original, "__glm_fusion_original__", original)
        source = textwrap.dedent(inspect.getsource(original))
        begin = source.index("    if use_native_mhc_post(")
        end = source.index("    pre_impl = ", begin)
        post_block = source[begin:end]
        replacement = """    if not use_fp16:
            raise RuntimeError("resident mHC post requires the qualified FP16 state policy")
        residual_cur = _resident_mhc_post(x, residual, post_layer_mix, comb_res_mix)
    """
        post = rewrite(original, [(post_block, replacement)], {"_resident_mhc_post": operation.mhc_post})
        for name in ("forward_oot", "forward_native"):
            changes[f"vllm.model_executor.layers.mhc:MHCFusedPostPreOp.{name}"] = post

    target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original_status = changes[target]

    def status(self):
        receipt = original_status(self)
        receipt["eager_fusions"] = {
            name: [{"geometry": geometry, "calls": count} for geometry, count in calls.items()]
            for name, calls in operation.calls.items()
        }
        return receipt

    changes[target] = status
    return changes
