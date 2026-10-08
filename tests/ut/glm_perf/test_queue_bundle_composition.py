# SPDX-License-Identifier: Apache-2.0
"""Check that composing the route candidates retains both transformations."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates.direct_route_tokens import rewrite_grouped, sorted_token_ids
from tools.glm_perf.resident_candidates.moe_half_unpermute import combine_routes, rewrite_combine


def grouped(x, topk_ids, topk_weights):
    from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

    num_tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    dispatch = build_grouped_expert_dispatch(topk_ids, topk_weights)
    sorted_tokens = dispatch.token_indices.index_select(0, dispatch.order)
    routed = x.index_select(0, sorted_tokens)
    routed = routed * topk_ids.flatten().index_select(0, dispatch.order).half().unsqueeze(1)
    fused_combine = False
    if fused_combine:
        output = None
    else:
        routed = routed.to(torch.float32)
        routed *= dispatch.route_weights.index_select(0, dispatch.order)
        output = routed.index_select(0, dispatch.inverse_order).reshape(num_tokens, top_k, hidden).sum(1)
    return output


@pytest.mark.parametrize("tokens", [1, 2, 8, 640])
def test_combined_metadata_and_half_unpermute_preserve_route_reduction(tokens):
    tree = rewrite_grouped(ast.unparse(rewrite_combine(inspect.getsource(grouped))))
    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    assert names.count("_candidate_combine_routes") == names.count("_candidate_token_ids") == 1

    def builder(ids, weights):
        order = ids.flatten().argsort(stable=True)
        return SimpleNamespace(order=order, inverse_order=order.argsort(), route_weights=weights.reshape(-1, 1))

    scope = dict(
        grouped.__globals__,
        _candidate_build_dispatch=builder,
        _candidate_token_ids=sorted_token_ids,
        _candidate_combine_routes=combine_routes,
    )
    exec(compile(tree, "<combined_routes>", "exec"), scope)
    generator = torch.Generator().manual_seed(310)
    x = torch.randn(tokens, 128, generator=generator).half()
    ids = torch.randint(0, 8, (tokens, 8), generator=generator)
    weights = torch.randn(tokens, 8, generator=generator)
    weights[ids >= 4] = 0
    expected = ((x[:, None, :] * ids.half()[:, :, None]).float() * weights[:, :, None]).sum(1)
    actual = scope["grouped"](x, ids, weights)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
