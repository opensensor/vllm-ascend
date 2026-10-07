# SPDX-License-Identifier: Apache-2.0
"""GLM-only experiment: unpermute FP16 routes before FP32 weight/reduction.

Keep the original route order within each token and accumulation dtype. Native
FP32/fused combine branches are untouched. The resident harness owns recapture
and rollback; this module does not mutate the installed quantization method.
"""

import ast
import inspect
import textwrap

import torch


def combine_routes(routed, inverse_order, route_weights, num_tokens, top_k, hidden):
    restored = routed.index_select(0, inverse_order).to(torch.float32)
    restored.mul_(route_weights)
    return restored.reshape(num_tokens, top_k, hidden).sum(1)


def rewrite_combine(source):
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.parse(
        "routed = routed.to(torch.float32)\n"
        "routed *= dispatch.route_weights.index_select(0, dispatch.order)\n"
        "output = routed.index_select(0, dispatch.inverse_order).reshape(num_tokens, top_k, hidden).sum(1)\n"
    ).body
    expected_dump = [ast.dump(node) for node in expected]
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and [ast.dump(stmt) for stmt in node.orelse] == expected_dump:
            matches.append(node)
    if len(matches) != 1:
        raise ValueError("GLM fallback combine changed: re-audit before applying")
    matches[0].orelse = ast.parse(
        "output = _candidate_combine_routes(routed, dispatch.inverse_order, dispatch.route_weights, "
        "num_tokens, top_k, hidden)"
    ).body
    return ast.fix_missing_locations(tree)


def replace_combine(original):
    original = getattr(original, "__glm_resident_original__", original)
    tree = rewrite_combine(inspect.getsource(original))
    scope = dict(original.__globals__, _candidate_combine_routes=combine_routes)
    exec(compile(tree, inspect.getfile(original), "exec"), scope)
    candidate = scope[original.__name__]
    candidate.__glm_resident_original__ = original
    return candidate


def replacements(native_resources=None):
    # Worker-only import. Do not patch the dispatch helper shared with Qwen.
    from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310

    candidate = replace_combine(AscendW2DynamicFusedMoEMethod310._apply_device_grouped)
    target = "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
    return {target: candidate}
