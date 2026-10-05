# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates.moe_half_unpermute import combine_routes, rewrite_combine

ROOT = Path(__file__).resolve().parents[3]


def reference(routed, inverse, weights, order, tokens, top_k, hidden):
    routed = routed.to(torch.float32)
    routed *= weights.index_select(0, order)
    return routed.index_select(0, inverse).reshape(tokens, top_k, hidden).sum(1)


@pytest.mark.parametrize("tokens", [1, 2, 8, 640, 1280])
@pytest.mark.parametrize("peer_routes", [False, True])
def test_exact_route_weight_and_reduction_order(tokens, peer_routes):
    generator = torch.Generator().manual_seed(310)
    top_k, hidden = 8, 128
    routes = tokens * top_k
    order = torch.randperm(routes, generator=generator)
    inverse = torch.argsort(order)
    routed = torch.randn(routes, hidden * 2, generator=generator).half()[:, ::2]
    weights = torch.randn(routes, 1, generator=generator)
    if peer_routes:
        weights[::2] = 0
        # The baseline fallback's peer rows are zero-initialized by grouped op.
        routed[weights.index_select(0, order).flatten() == 0] = 0
    before = routed.clone()
    expected = reference(routed, inverse, weights, order, tokens, top_k, hidden)
    actual = combine_routes(routed, inverse, weights, tokens, top_k, hidden)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(routed, before)


def test_extreme_values_nan_inf_and_signed_zero():
    routed = torch.tensor([0.0, -0.0, 65504, -65504, torch.inf, -torch.inf, torch.nan, 2**-24]).half().reshape(8, 1)
    order = torch.tensor([7, 4, 1, 6, 0, 5, 2, 3])
    inverse = torch.argsort(order)
    weights = torch.tensor([1.0, -1.0, 0.0, 1e-30, 1e30, -1.0, 1.0, 0.0]).reshape(8, 1)
    # top_k=1 exposes individual values instead of hiding them in a NaN sum.
    expected = reference(routed, inverse, weights, order, 8, 1, 1)
    actual = combine_routes(routed, inverse, weights, 8, 1, 1)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite_or_inf = ~torch.isnan(expected)
    assert torch.equal(actual[finite_or_inf].view(torch.int32), expected[finite_or_inf].view(torch.int32))


def test_reorders_half_and_never_gathers_weights(monkeypatch):
    observed = []
    original = torch.Tensor.index_select

    def spy(self, dim, indices):
        observed.append((self.dtype, tuple(self.shape)))
        return original(self, dim, indices)

    monkeypatch.setattr(torch.Tensor, "index_select", spy)
    combine_routes(torch.ones(16, 128).half(), torch.arange(16), torch.ones(16, 1), 2, 8, 128)
    assert observed == [(torch.float16, (16, 128))]


def test_actual_grouped_method_changes_only_combine_fallback():
    path = ROOT / "vllm_ascend/_310p/quantization/methods/w2_dynamic.py"
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendW2DynamicFusedMoEMethod310"
    )
    function = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_device_grouped"
    )
    original = ast.parse(ast.get_source_segment(source, function))
    updated = rewrite_combine(ast.get_source_segment(source, function))
    replacement = next(
        node
        for node in ast.walk(updated)
        if isinstance(node, ast.If)
        and len(node.orelse) == 1
        and isinstance(node.orelse[0], ast.Assign)
        and isinstance(node.orelse[0].value, ast.Call)
        and isinstance(node.orelse[0].value.func, ast.Name)
        and node.orelse[0].value.func.id == "_candidate_combine_routes"
    )
    baseline = next(
        node
        for node in ast.walk(original)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "fused_combine"
    )
    replacement.orelse = baseline.orelse
    assert ast.dump(updated) == ast.dump(original)


def test_changed_combine_rejected():
    with pytest.raises(ValueError, match="re-audit"):
        rewrite_combine("def changed():\n    return None\n")


@pytest.mark.parametrize("native,fused", [(False, False), (False, True), (True, False)])
def test_native_branches_are_not_replaced(native, fused):
    source = """
def combine(routed, dispatch, num_tokens, top_k, hidden, fp32_combine, fused_combine):
    if fp32_combine:
        output = "native"
    elif fused_combine:
        output = "fused"
    else:
        routed = routed.to(torch.float32)
        routed *= dispatch.route_weights.index_select(0, dispatch.order)
        output = routed.index_select(0, dispatch.inverse_order).reshape(num_tokens, top_k, hidden).sum(1)
    return output
"""
    scope = {"torch": torch, "_candidate_combine_routes": combine_routes}
    exec(compile(rewrite_combine(source), "<candidate>", "exec"), scope)
    dispatch = SimpleNamespace(inverse_order=torch.arange(8), route_weights=torch.ones(8, 1))
    result = scope["combine"](torch.ones(8, 16).half(), dispatch, 1, 8, 16, native, fused)
    if native or fused:
        assert result == ("native" if native else "fused")
    else:
        assert torch.equal(result, torch.full((1, 16), 8.0))
