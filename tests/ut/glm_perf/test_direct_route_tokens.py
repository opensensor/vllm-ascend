# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from tools.glm_perf.resident_candidates import direct_route_tokens as candidate

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def dispatch_module(monkeypatch):
    name = "glm_queue_dispatch_fixture"
    path = ROOT / "vllm_ascend/models/qwen4_exp/grouped_expert_dispatch.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens,top_k", [(0, 8), (1, 8), (8, 8), (640, 8), (1280, 8), (7, 3), (8, 1)])
@pytest.mark.parametrize("count_mode", ["compare", "histogram"])
@pytest.mark.parametrize("peer_only", [False, True])
def test_actual_dispatch_metadata_and_token_gather_match(dispatch_module, tokens, top_k, count_mode, peer_only):
    generator = torch.Generator().manual_seed(310)
    # Noncontiguous inputs, including duplicate expert selections and ties.
    ids = torch.randint(0, 288, (tokens, top_k * 2), generator=generator)[:, ::2]
    weights = torch.rand(tokens, top_k * 2, generator=generator)[:, ::2]
    if peer_only:
        ids.fill_(288)
        weights.zero_()
    before_ids, before_weights = ids.clone(), weights.clone()
    original = dispatch_module.build_grouped_expert_dispatch
    private = candidate.make_dispatch(original)
    options = dict(num_local_experts=72, expert_offset=72, weight_dtype=torch.float32, count_mode=count_mode)
    expected, actual = original(weights, ids, **options), private(weights, ids, **options)
    assert dispatch_module.build_grouped_expert_dispatch is original
    assert not hasattr(actual, "token_indices")
    for field in ("order", "inverse_order", "route_weights", "counts", "group_list"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))
    sorted_tokens = candidate.sorted_token_ids(actual.order, top_k)
    assert sorted_tokens.dtype == torch.int32
    old_tokens = expected.token_indices.index_select(0, expected.order)
    assert torch.equal(sorted_tokens.long(), old_tokens)
    x = torch.randn(tokens, 128, generator=generator).half()
    assert torch.equal(x.index_select(0, sorted_tokens), x.index_select(0, old_tokens))
    assert torch.equal(ids, before_ids) and torch.equal(weights, before_weights)


def test_token_expansion_really_removed(dispatch_module, monkeypatch):
    private = candidate.make_dispatch(dispatch_module.build_grouped_expert_dispatch)
    arange_calls = []
    original_arange = torch.arange

    def arange(*args, **kwargs):
        arange_calls.append(args)
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", arange)
    private(
        torch.ones(17, 8),
        torch.zeros(17, 8, dtype=torch.long),
        num_local_experts=7,
        expert_offset=0,
        weight_dtype=torch.float32,
        count_mode="compare",
    )
    assert arange_calls == [(7,)]  # Only expert IDs for counting; no repeated token IDs.


def test_integer_bound_retains_int64_path(monkeypatch):
    # Exercise both sides without allocating billions of routes.
    monkeypatch.setattr(candidate, "MAX_INT32_ROUTES", 16)
    assert candidate.sorted_token_ids(torch.arange(16), 8).dtype == torch.int32
    order = torch.arange(17)
    actual = candidate.sorted_token_ids(order, 8)
    assert actual.dtype == torch.int64 and torch.equal(actual, order // 8)


@pytest.mark.parametrize("top_k", [0, -1])
def test_bad_top_k_rejected(top_k):
    with pytest.raises(ValueError, match="positive top-k"):
        candidate.sorted_token_ids(torch.arange(8), top_k)


def test_dispatch_rewrite_leaves_sort_counts_and_inverse_untouched(dispatch_module):
    source = inspect.getsource(dispatch_module.build_grouped_expert_dispatch)
    original = ast.parse(source)
    updated = candidate.rewrite_dispatch(source)
    old_function, new_function = original.body[0], updated.body[0]
    expansion = next(
        (i, node)
        for i, node in enumerate(old_function.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "token_indices"
    )
    new_function.body.insert(*expansion)
    new_function.body[-1] = old_function.body[-1]
    new_function.returns = old_function.returns
    assert ast.dump(original) == ast.dump(updated)


def grouped_source():
    source = (ROOT / "vllm_ascend/_310p/quantization/methods/w2_dynamic.py").read_text()
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendW2DynamicFusedMoEMethod310"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_device_grouped"
    )
    return ast.get_source_segment(source, method)


def test_grouped_rewrite_preserves_peer_mask_experts_and_combine():
    original = ast.parse(grouped_source())
    updated = candidate.rewrite_grouped(grouped_source())
    for i, node in enumerate(original.body[0].body):
        if isinstance(node, ast.ImportFrom) or (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "sorted_tokens"
        ):
            updated.body[0].body[i] = node
    assert ast.dump(original) == ast.dump(updated)


def test_changed_sources_fail_closed(dispatch_module):
    source = inspect.getsource(dispatch_module.build_grouped_expert_dispatch)
    with pytest.raises(ValueError, match="re-audit"):
        candidate.rewrite_dispatch(source.replace("token_indices = torch.arange", "token_indices = torch.ones"))
    with pytest.raises(ValueError, match="re-audit"):
        candidate.rewrite_grouped(
            grouped_source().replace("sorted_tokens = dispatch.token_indices", "tokens = dispatch.token_indices")
        )


def test_private_method_repeat_preparation(dispatch_module):
    def grouped(weights, ids, x):
        from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

        top_k = ids.shape[1]  # noqa: F841 - consumed by the rewritten candidate
        dispatch = build_grouped_expert_dispatch(
            weights, ids, num_local_experts=3, expert_offset=0, weight_dtype=torch.float32
        )
        sorted_tokens = dispatch.token_indices.index_select(0, dispatch.order)
        return x.index_select(0, sorted_tokens)

    builder = candidate.make_dispatch(dispatch_module.build_grouped_expert_dispatch)
    first = candidate.replace_grouped(grouped, builder)
    second = candidate.replace_grouped(first, builder)
    assert second.__glm_resident_original__ is grouped
    ids = torch.tensor([[1, 2], [0, 1]])
    result = second(torch.ones(2, 2), ids, torch.tensor([[10.0], [20.0]]))
    assert result.flatten().tolist() == [20.0, 10.0, 20.0, 10.0]
