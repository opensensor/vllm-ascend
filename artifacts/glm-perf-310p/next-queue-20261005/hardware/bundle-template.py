# SPDX-License-Identifier: Apache-2.0
"""Compose qualified queue candidates explicitly, including shared methods.

The experiment controller fills SELECTED only after individual serving results.
An empty template deliberately refuses to apply as an active candidate.
"""

import ast
import inspect

SELECTED = frozenset()


def replacements(native_resources=None):
    if not SELECTED:
        raise ValueError("select individually qualified candidates before applying this template")
    allowed = {
        "kda_batched_qk",
        "kda_gate_beta",
        "kpool_decode_epilogue",
        "moe_half_unpermute",
        "direct_route_tokens",
    }
    if not allowed >= SELECTED:
        raise ValueError("unknown or unqualified bundle member")
    result = {}
    if SELECTED & {"kda_batched_qk", "kda_gate_beta"}:
        from tools.glm_perf.resident_candidates import kda_input_preparation as preparation
        from vllm_ascend.models.glm5next_w2 import kda_310

        native = native_resources["kda_gate_beta_v1"] if "kda_gate_beta" in SELECTED else None
        prepare = preparation.make_preparer(
            kda_310._l2norm_310p,
            kda_310._safe_gate_for_layer,
            batch_qk="kda_batched_qk" in SELECTED,
            gate_beta=native,
        )
        result["vllm_ascend.models.glm5next_w2.kda_310:_run_recurrent"] = preparation.replace_input_prelude(
            kda_310._run_recurrent, prepare
        )
    if "kpool_decode_epilogue" in SELECTED:
        from tools.glm_perf.resident_candidates import kpool_decode_epilogue

        result.update(kpool_decode_epilogue.replacements())
    if SELECTED & {"moe_half_unpermute", "direct_route_tokens"}:
        from tools.glm_perf.resident_candidates import direct_route_tokens, moe_half_unpermute
        from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310
        from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

        active = AscendW2DynamicFusedMoEMethod310._apply_device_grouped
        original = getattr(active, "__glm_resident_original__", active)
        source = inspect.getsource(original)
        if "moe_half_unpermute" in SELECTED:
            source = ast.unparse(moe_half_unpermute.rewrite_combine(source))
        if "direct_route_tokens" in SELECTED:
            source = ast.unparse(direct_route_tokens.rewrite_grouped(source))
        scope = dict(
            original.__globals__,
            _candidate_combine_routes=moe_half_unpermute.combine_routes,
            _candidate_build_dispatch=direct_route_tokens.make_dispatch(build_grouped_expert_dispatch),
            _candidate_token_ids=direct_route_tokens.sorted_token_ids,
        )
        # Keep both transformations; calling the independent factories in
        # sequence would restore the original and discard the first change.
        exec(compile(source, inspect.getfile(original), "exec"), scope)
        combined = scope[original.__name__]
        combined.__glm_resident_original__ = original
        target = (
            "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
        )
        result[target] = combined
    return result
