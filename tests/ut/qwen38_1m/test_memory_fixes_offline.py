# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for the Qwen memory audit's opt-in candidates."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend._310p.host_staging import PinnedHostStaging
from vllm_ascend.models.qwen4_exp import model
from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
from vllm_ascend.models.qwen4_exp.ops.qsa_group_major_attention_310 import (
    _group_major_attention,
    build_qsa_fixed_group_major_plan,
    build_qsa_group_major_plan,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer
from vllm_ascend.models.qwen4_exp.route_workspace import grouped_route_chunk_tokens

ROOT = Path(__file__).resolve().parents[3]


def _method(path, class_name, method_name, **scope):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    ns = {"torch": torch, "F": F, **scope}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / path), "exec"), ns)
    return ns[method_name]


@pytest.mark.parametrize("seed", range(12))
def test_fixed_union_exact_masks_and_attention_without_dynamic_outputs(monkeypatch, seed):
    torch.manual_seed(seed)
    queries, width, groups = 4, 5, 12
    selection = QSAGroupSelection(
        torch.stack([torch.randperm(groups)[:width].int() for _ in range(queries)]),
        torch.randint(1, width + 1, (queries,), dtype=torch.int32),
        4 * torch.randint(0, groups, (queries,), dtype=torch.int32),
        torch.randint(0, 5, (queries,), dtype=torch.int32),
    )
    reference = build_qsa_group_major_plan(selection)
    for name in ("unique", "nonzero", "masked_select"):
        monkeypatch.setattr(torch, name, lambda *a, **kw: pytest.fail("dynamic output forbidden"))
    fixed = build_qsa_fixed_group_major_plan(selection)
    valid_groups = int(fixed.device_group_count[0])  # CPU test only.
    assert fixed.group_indices.shape == (queries * (width + 1),)
    assert valid_groups == reference.group_indices.numel()
    assert torch.equal(fixed.group_indices[:valid_groups], reference.group_indices)
    assert torch.equal(fixed.token_mask[:, : valid_groups * 4], reference.token_mask)
    assert not fixed.token_mask[:, valid_groups * 4 :].any()
    query = torch.randn(queries, 4, 16).half()
    key = torch.randn(groups * 4, 2, 16).half()
    value = torch.randn_like(key)

    def attention(plan):
        rows = (plan.group_indices.long()[:, None] * 4 + torch.arange(4)).flatten()
        return _group_major_attention(
            query, key[rows].permute(1, 2, 0)[None], value[rows].permute(1, 0, 2)[None], plan.token_mask, scale=0.25
        )

    torch.testing.assert_close(attention(fixed), attention(reference), rtol=0.002, atol=0.001)


def test_fixed_union_empty_tail_and_full_overlap():
    selection = QSAGroupSelection(
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        torch.tensor([2, 2]),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    plan = build_qsa_fixed_group_major_plan(selection)
    assert plan.device_group_count.tolist() == [2]
    assert torch.equal(plan.token_mask[0], plan.token_mask[1])
    assert not plan.token_mask[:, 8:].any()


@pytest.mark.parametrize("shapes", [(0, 4), (4, 0)])
def test_fixed_union_rejects_empty_selection(shapes):
    selection = QSAGroupSelection(
        torch.zeros(shapes, dtype=torch.int32), *(torch.zeros(shapes[0], dtype=torch.int32) for _ in range(3))
    )
    with pytest.raises(ValueError):
        build_qsa_fixed_group_major_plan(selection)


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 9])
def test_mtp_grouped_capture_bypasses_host_callback_only_for_fixed_route_shape(rows):
    capture = SimpleNamespace(current=Mock(return_value=None))
    forward = _method(
        "vllm_ascend/models/qwen4_exp/mtp.py",
        "_MTPFP16MoE",
        "forward",
        BreakableCUDAGraphCapture=capture,
        _FUSED_ROUTING_MAX_TOKENS=8,
    )
    bank = SimpleNamespace(
        grouped_graph=True, _grouped_weights_prepared=True, _forward_eager=Mock(return_value="output")
    )
    x = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(rows, 2560))
    assert forward(bank, x) == "output"
    assert capture.current.call_count == (0 if rows <= 8 else 1)
    bank._grouped_weights_prepared = False
    if rows <= 8:
        with pytest.raises(RuntimeError):
            forward(bank, x)


@pytest.mark.parametrize("metadata", [{"mtp_grouped_graph": 1}, {"mtp_grouped_graph": True}])
def test_mtp_graph_config_requires_explicit_quantized_grouped_weights(metadata):
    tree = ast.parse((ROOT / "vllm_ascend/models/qwen4_exp/mtp.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MTPFP16MoE")
    ns = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<MTP bank>", "exec"), ns)
    config = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=8,
        moe_intermediate_size=4,
        ascend_expert_quantization=metadata,
    )
    with pytest.raises(ValueError):
        ns["_MTPFP16MoE"](config, SimpleNamespace(main_dtype=torch.float16), (0, 1))


@pytest.mark.parametrize("budget", [1, 32, 64, 128, 512])
def test_route_workspace_bound_is_monotone_and_respects_requested_limit(budget):
    args = dict(top_k=10, hidden=2560, intermediate=640, local_experts=128, histogram_counts=True)
    chunk = grouped_route_chunk_tokens(2560, scratch_mib=budget, **args)
    bigger = grouped_route_chunk_tokens(2560, scratch_mib=budget * 2, **args)
    assert 1 <= chunk <= bigger <= 2560
    assert grouped_route_chunk_tokens(chunk, scratch_mib=budget, **args) == chunk
    assert grouped_route_chunk_tokens(2560, scratch_mib=0, **args) == 2560


@pytest.mark.parametrize("budget", [-1, True, "128", 0.1])
def test_route_workspace_rejects_invalid_budget(budget):
    with pytest.raises(ValueError):
        grouped_route_chunk_tokens(
            2560, top_k=10, hidden=2560, intermediate=640, local_experts=128, scratch_mib=budget, histogram_counts=True
        )


@pytest.mark.parametrize("count_mode", ["compare", "histogram"])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_bounded_grouped_routes_preserve_peer_zeros_and_token_order(count_mode, chunk):
    forward = _method(
        "vllm_ascend/models/qwen4_exp/w4_moe.py",
        "W4SparseMoE",
        "_forward_grouped",
        build_grouped_expert_dispatch=build_grouped_expert_dispatch,
    )
    torch.manual_seed(454)
    x = torch.randn(7, 16).half()
    ids = torch.stack([torch.randperm(8)[:3] for _ in range(7)])
    weights = torch.softmax(torch.randn(7, 3), -1)
    gate_weights, down_weights = torch.randn(4, 8, 16).half(), torch.randn(4, 16, 4).half()

    class Bank:
        def __init__(self, bank):
            self.bank = bank

        def grouped_linear(self, inputs, group_ends):
            result = torch.zeros(inputs.shape[0], self.bank.shape[1], dtype=torch.float16)
            start = 0
            for expert, stop in enumerate(group_ends.tolist()):
                result[start:stop] = F.linear(inputs[start:stop], self.bank[expert])
                start = stop
            return result

    bank = SimpleNamespace(
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        grouped_chunk_tokens=chunk,
        num_local_experts=4,
        expert_offset=4,
        top_k=3,
        native_int4=False,
        grouped_activation="torch",
        grouped_finalize="torch",
        grouped_route_count_mode=count_mode,
        projections={"gate_up_proj": Bank(gate_weights), "down_proj": Bank(down_weights)},
    )
    chunk_forward = _method(
        "vllm_ascend/models/qwen4_exp/w4_moe.py",
        "W4SparseMoE",
        "_forward_grouped_chunk",
        build_grouped_expert_dispatch=build_grouped_expert_dispatch,
    )
    bank._forward_grouped_chunk = MethodType(chunk_forward, bank)
    expected = torch.zeros(7, 16)
    for token in range(7):
        for slot in range(3):
            local = int(ids[token, slot]) - 4
            if 0 <= local < 4:
                gate, up = F.linear(x[token], gate_weights[local]).float().chunk(2)
                activated = (F.silu(gate) * up).half()
                expected[token] += F.linear(activated, down_weights[local]).float() * weights[token, slot]
    torch.testing.assert_close(forward(bank, x, weights, ids), expected, rtol=0.002, atol=0.001)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("injection_dtype", [torch.float16, torch.float32])
def test_core_hc_operand_reuse_retains_gate_precision(monkeypatch, enabled, injection_dtype):
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    torch.manual_seed(224)
    module = model._GatedResidual(
        hc_count=2,
        hidden_size=16,
        lowrank=8,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        share_projection_operand=enabled,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        module.block_inject_weight.data = module.block_inject_weight.data.to(injection_dtype)
        value, block = torch.randn(3, 32).half(), torch.randn(3, 16).half()
        module.share_projection_operand = False
        expected, baseline_residual = module.mix(value)
        module.share_projection_operand = enabled
        actual, residual = module.mix(value)
        assert torch.equal(actual, expected)
        assert torch.equal(module.combine(block, residual), module.combine(block, baseline_residual))
        assert residual[1].dtype == (torch.float16 if enabled and injection_dtype == torch.float16 else torch.float32)


@pytest.mark.parametrize("tokens", [1, 2])
def test_ple_staging_reuses_host_storage_and_dequantizes_after_transfer(monkeypatch, tokens):
    original_empty = torch.empty
    event = Mock()
    event.query.return_value = True

    def allocate(*args, **kwargs):
        device = kwargs.get("device")
        if getattr(device, "type", None) == "npu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    make_stage = lambda shape, dtype: PinnedHostStaging(shape, dtype, pin_memory=False, event_factory=lambda: event)
    gather = _method(
        "vllm_ascend/models/qwen4_exp/ple_layer.py",
        "AscendQwen4ExpPLELayer",
        "gather_embeddings",
        PinnedHostStaging=make_stage,
    )
    ids = torch.arange(tokens * 4).reshape(tokens, 4)
    rows = torch.arange(tokens * 4 * 8).reshape(tokens * 4, 8).half()
    method = SimpleNamespace(gather_rows=lambda ids: rows, dequantize=lambda values, dtype: values.to(dtype))
    layer = SimpleNamespace(
        ple_method=method, num_ngram_heads=4, per_head_dim=8, host_staging_tokens=2, _row_host_stage=None
    )
    destination = SimpleNamespace(type="npu")
    expected = rows.reshape(tokens, 32).float()
    actual = gather(layer, ids, torch.float32, destination)
    pointer = layer._row_host_stage.host.data_ptr()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gather(layer, ids, torch.float32, destination), expected)
    assert layer._row_host_stage.host.data_ptr() == pointer
    assert event.record.call_count == 2
    layer.host_staging_tokens = 0
    cpu = AscendQwen4ExpPLELayer.gather_embeddings(layer, ids, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(cpu, expected)
    layer.host_staging_tokens = 1
    if tokens == 2:
        with pytest.raises(ValueError):
            gather(layer, ids, torch.float32, destination)


def test_qsa_config_keeps_default_and_exposes_fixed_union_candidate():
    assert model._qsa_prefill_policy(SimpleNamespace())[0] == "batched_gather"
    for backend in ("batched_gather", "fixed_group_major_union"):
        cfg = SimpleNamespace(ascend_qsa_prefill={"backend": backend, "query_tile": 8, "parallel_gather": False})
        assert model._qsa_prefill_policy(cfg) == (backend, 8, False)


def test_fixed_union_inactive_query_has_zero_attention_output():
    selection = QSAGroupSelection(
        torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32),
        torch.tensor([1, 0]),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    plan = build_qsa_fixed_group_major_plan(selection)
    query = torch.randn(2, 4, 16).half()
    selected_keys = torch.randn(1, 2, 16, plan.token_mask.shape[1]).half()
    selected_values = torch.randn(1, 2, plan.token_mask.shape[1], 16).half()
    result = _group_major_attention(query, selected_keys, selected_values, plan.token_mask, scale=0.25)
    assert torch.isfinite(result).all()
    assert not result[1].any()
    with pytest.raises(RuntimeError):
        _ = plan.unique_group_reads
