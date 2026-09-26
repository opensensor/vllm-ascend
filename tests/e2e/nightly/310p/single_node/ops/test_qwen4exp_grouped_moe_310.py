# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real 310P execution of Qwen4Exp's device-resident grouped expert path."""

import pytest
import torch
import torch_npu
from torch import nn

from vllm_ascend.models.qwen4_exp.model import _Qwen4ExpW8A8PostLoadMethod
from vllm_ascend.models.qwen4_exp.moe import w8a8_grouped_experts


def test_expert_bank_is_packed_once_in_nz_format() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    original = torch.randint(-4, 5, (2, 64, 32), device="npu:0", dtype=torch.int8)
    layer = nn.Module()
    layer.num_local_experts = original.shape[0]
    layer.w13_weight = nn.ParameterList(
        nn.Parameter(weight.t().contiguous(), requires_grad=False) for weight in original
    )

    _Qwen4ExpW8A8PostLoadMethod.pack_expert_weight_bank(layer, "w13_weight")

    assert torch_npu.get_npu_format(layer.w13_weight_grouped) == 29
    torch.testing.assert_close(layer.w13_weight_grouped.cpu(), original.cpu())
    for expert, weight in enumerate(layer.w13_weight):
        assert weight.untyped_storage().data_ptr() == layer.w13_weight_grouped.untyped_storage().data_ptr()
        torch.testing.assert_close(weight.cpu(), original[expert].t().cpu())


def test_grouped_moe_matches_qdq_and_masks_peer_routes() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    torch.manual_seed(419)
    hidden = 64
    intermediate = 32
    experts = 2
    x = torch.randn(4, hidden, device="npu:0", dtype=torch.float16) * 0.2
    ids = torch.tensor([[0, 2], [1, 3], [3, 2], [1, 0]], device="npu:0")
    route_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8]], device="npu:0")
    w13_nd = torch.randint(-4, 5, (experts, 2 * intermediate, hidden), device="npu:0", dtype=torch.int8)
    w2_nd = torch.randint(-4, 5, (experts, hidden, intermediate), device="npu:0", dtype=torch.int8)
    w13 = torch_npu.npu_format_cast(w13_nd, 29)
    w2 = torch_npu.npu_format_cast(w2_nd, 29)
    s13 = torch.full((experts, 2 * intermediate), 0.04, device="npu:0")
    s2 = torch.full((experts, hidden), 0.04, device="npu:0")
    o13 = torch.zeros_like(s13)
    o2 = torch.zeros_like(s2)

    actual = w8a8_grouped_experts(x, route_weights, ids, w13, s13, o13, w2, s2, o2, use_packed_grouped=True)
    expected = w8a8_grouped_experts(
        x.cpu(),
        route_weights.cpu(),
        ids.cpu(),
        w13_nd.cpu(),
        s13.cpu().unsqueeze(-1),
        o13.cpu().unsqueeze(-1),
        w2_nd.cpu(),
        s2.cpu().unsqueeze(-1),
        o2.cpu().unsqueeze(-1),
        num_global_experts=4,
    )
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.02, rtol=0.08)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[2].cpu(), torch.zeros(hidden, dtype=torch.float16))

    all_peer = w8a8_grouped_experts(
        x, route_weights, ids, w13, s13, o13, w2, s2, o2, expert_offset=4, use_packed_grouped=True
    )
    assert torch.equal(all_peer.cpu(), torch.zeros_like(x.cpu()))
