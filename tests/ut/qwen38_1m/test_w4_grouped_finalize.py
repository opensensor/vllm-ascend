# SPDX-License-Identifier: Apache-2.0
"""Host gates for the opt-in Qwen W4 grouped MoE finalizer."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.test_w4_moe import config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, finalize_grouped_routes, w4_config
from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND


def _native_config():
    cfg = config(num_layers=1, num_experts=8, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(
        group_size=128,
        backend=NATIVE_INT4_BACKEND,
        activation_quantization="int8_per_group",
    )
    return cfg


def test_finalizer_is_opt_in_and_restricted_to_native_int4():
    cfg = _native_config()
    assert W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy()).grouped_finalize == "torch"
    cfg.ascend_expert_quantization["grouped_finalize"] = "cann_v2"
    assert W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy()).grouped_finalize == "cann_v2"
    cfg.ascend_expert_quantization["backend"] = "cube_310_grouped"
    with pytest.raises(ValueError, match="requires native INT4"):
        w4_config(cfg)
    cfg.ascend_expert_quantization["grouped_finalize"] = "unknown"
    with pytest.raises(ValueError, match="grouped_finalize"):
        w4_config(cfg)


@pytest.mark.parametrize("all_peer", [False, True])
def test_cann_finalizer_receives_inverse_order_and_fp16_weights(all_peer):
    token_count, top_k, hidden = 4, 3, 32
    ids = torch.tensor([[2, 7, 3], [1, 4, 6], [3, 0, 5], [7, 2, 4]])
    if all_peer:
        ids.fill_(0)
    weights = torch.tensor([[0.125, 0.25, 0.625]]).expand(token_count, -1).clone()
    dispatch = build_grouped_expert_dispatch(
        weights, ids, num_local_experts=3, expert_offset=2, weight_dtype=torch.float32
    )
    original = torch.arange(token_count * top_k * hidden, dtype=torch.float32).reshape(-1, hidden).half() / 100
    sorted_ids = ids.flatten().index_select(0, dispatch.order)
    local = (sorted_ids >= 2) & (sorted_ids < 5)
    sorted_rows = torch.where(local[:, None], original.index_select(0, dispatch.order), 0)
    rounded_weights = weights.half().float()
    rounded_dispatch = build_grouped_expert_dispatch(
        rounded_weights, ids, num_local_experts=3, expert_offset=2, weight_dtype=torch.float32
    )
    reference = finalize_grouped_routes(sorted_rows, rounded_dispatch, rounded_weights, torch.float32)
    observed = []

    def fake_finalizer(rows, skip1, skip2, bias, scales, inverse, expert_ids, mode):
        assert rows is sorted_rows
        assert skip1 is skip2 is bias is expert_ids is None
        assert mode == 2
        assert scales.dtype == torch.float16 and scales.shape == (token_count, top_k)
        torch.testing.assert_close(scales, weights.half(), rtol=0, atol=0)
        assert inverse.dtype == torch.int32
        torch.testing.assert_close(inverse, dispatch.inverse_order.to(torch.int32), rtol=0, atol=0)
        observed.append(True)
        return (
            (rows.float().index_select(0, inverse.long()).reshape(token_count, top_k, hidden) * scales.unsqueeze(-1))
            .sum(1)
            .half()
        )

    with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu_moe_finalize_routing=fake_finalizer)}):
        candidate = finalize_grouped_routes(sorted_rows, dispatch, weights, torch.float32, "cann_v2")
    assert observed
    torch.testing.assert_close(candidate, reference.half().float(), rtol=0, atol=0)
    if all_peer:
        assert torch.count_nonzero(candidate) == 0


def test_cann_finalizer_rejects_non_fp16_rows():
    weights = torch.ones(1, 1)
    ids = torch.zeros(1, 1, dtype=torch.int64)
    dispatch = build_grouped_expert_dispatch(
        weights, ids, num_local_experts=1, expert_offset=0, weight_dtype=torch.float32
    )
    with pytest.raises(ValueError, match="requires FP16"):
        finalize_grouped_routes(torch.ones(1, 8), dispatch, weights, torch.float32, "cann_v2")


def test_finalizer_rejects_mismatched_route_shape():
    weights = torch.ones(2, 3)
    ids = torch.zeros(2, 3, dtype=torch.int64)
    dispatch = build_grouped_expert_dispatch(
        weights, ids, num_local_experts=1, expert_offset=0, weight_dtype=torch.float32
    )
    with pytest.raises(ValueError, match="do not match"):
        finalize_grouped_routes(torch.zeros(5, 32).half(), dispatch, weights, torch.float32)


def test_native_grouped_prefill_wires_opt_in_finalizer():
    cfg = _native_config()
    cfg.ascend_expert_quantization["grouped_finalize"] = "cann_v2"
    cfg.ascend_expert_quantization["grouped_activation"] = "torch"
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    tokens = 129  # Above the 128-route decode kernel budget.
    inputs = torch.zeros(tokens, cfg.hidden_size).half()
    weights = torch.full((tokens, 3), 1 / 3, dtype=torch.float32)
    ids = torch.zeros(tokens, 3, dtype=torch.int64)
    expected = torch.zeros(tokens, cfg.hidden_size, dtype=torch.float32)

    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.pack_activation_device", side_effect=lambda x: (x,)),
        patch.object(
            layer.projections["gate_up_proj"],
            "native_linear",
            return_value=torch.zeros(tokens * 3, cfg.hidden_size * 2).half(),
        ),
        patch.object(
            layer.projections["down_proj"],
            "grouped_linear",
            return_value=torch.zeros(tokens * 3, cfg.hidden_size).half(),
        ),
        patch("vllm_ascend.models.qwen4_exp.w4_moe.finalize_grouped_routes", return_value=expected) as finalize,
    ):
        actual = layer._forward_grouped(inputs, weights, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert finalize.call_count == 1
    assert finalize.call_args.args[-1] == "cann_v2"
