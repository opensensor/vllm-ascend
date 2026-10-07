# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM route metadata without expanded token IDs or their gather.

Clone the current dispatch builder privately; never patch the shared Qwen
module. Preserve its routing, stable expert sort, counts and inverse sort.
"""

import ast
import inspect
import textwrap

import torch

MAX_INT32_ROUTES = 1 << 31


class RouteDispatch:
    __slots__ = ("order", "inverse_order", "route_weights", "counts")

    def __init__(self, order, inverse_order, route_weights, counts):
        self.order, self.inverse_order = order, inverse_order
        self.route_weights, self.counts = route_weights, counts

    @property
    def group_list(self):
        return self.counts.cumsum(dim=0)


def sorted_token_ids(order, top_k):
    if top_k <= 0 or order.ndim != 1 or order.dtype not in (torch.int32, torch.int64):
        raise ValueError("requires a route permutation and positive top-k")
    # The input is an argsort permutation in [0, num_routes), not arbitrary IDs.
    # Shape bounds are available on host; no scalar device value is read.
    indices = order.to(torch.int32) if order.numel() <= MAX_INT32_ROUTES else order
    # The scalar right-shift path copies a scalar synchronously on this 310P
    # runtime and fails full graph capture. Floor division stays capturable.
    return torch.div(indices, top_k, rounding_mode="floor")


def rewrite_dispatch(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    expanded = ast.parse(
        "token_indices = torch.arange(num_tokens, device=topk_ids.device).unsqueeze(1).expand(-1, top_k).reshape(-1)"
    ).body[0]
    returned = ast.parse(
        "return GroupedExpertDispatch(order, inverse_order, token_indices, route_weights, counts)"
    ).body[0]
    expansion_positions = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(expanded)]
    return_positions = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(returned)]
    if len(expansion_positions) != 1 or len(return_positions) != 1:
        raise ValueError("dispatch token metadata changed: re-audit before applying")
    function.body[return_positions[0]] = ast.parse(
        "return _candidate_dispatch_type(order, inverse_order, route_weights, counts)"
    ).body[0]
    del function.body[expansion_positions[0]]
    if any(isinstance(node, ast.Name) and node.id == "token_indices" for node in ast.walk(function)):
        raise ValueError("dispatch has another token_indices consumer: re-audit before applying")
    function.returns = None
    return ast.fix_missing_locations(tree)


def make_dispatch(original):
    scope = dict(original.__globals__, _candidate_dispatch_type=RouteDispatch)
    exec(compile(rewrite_dispatch(inspect.getsource(original)), inspect.getfile(original), "exec"), scope)
    return scope[original.__name__]


def rewrite_grouped(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    imported = ast.parse(
        "from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch"
    ).body[0]
    gathered = ast.parse("sorted_tokens = dispatch.token_indices.index_select(0, dispatch.order)").body[0]
    imports = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(imported)]
    gathers = [i for i, node in enumerate(function.body) if ast.dump(node) == ast.dump(gathered)]
    if len(imports) != 1 or len(gathers) != 1:
        raise ValueError("GLM grouped routing changed: re-audit before applying")
    function.body[imports[0]] = ast.parse("build_grouped_expert_dispatch = _candidate_build_dispatch").body[0]
    function.body[gathers[0]] = ast.parse("sorted_tokens = _candidate_token_ids(dispatch.order, top_k)").body[0]
    if any(isinstance(node, ast.Attribute) and node.attr == "token_indices" for node in ast.walk(function)):
        raise ValueError("GLM has another token_indices consumer: re-audit before applying")
    return ast.fix_missing_locations(tree)


def replace_grouped(original, builder):
    original = getattr(original, "__glm_resident_original__", original)
    scope = dict(original.__globals__, _candidate_build_dispatch=builder, _candidate_token_ids=sorted_token_ids)
    exec(compile(rewrite_grouped(inspect.getsource(original)), inspect.getfile(original), "exec"), scope)
    candidate = scope[original.__name__]
    candidate.__glm_resident_original__ = original
    return candidate


def replacements(native_resources=None):
    from vllm_ascend._310p.quantization.methods.w2_dynamic import AscendW2DynamicFusedMoEMethod310
    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

    builder = make_dispatch(build_grouped_expert_dispatch)
    candidate = replace_grouped(AscendW2DynamicFusedMoEMethod310._apply_device_grouped, builder)
    target = "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped"
    return {target: candidate}
