# SPDX-License-Identifier: Apache-2.0
"""310P regression tests for Qwen4Exp routed-expert output restoration."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.moe import _w8a16_linear_npu, w8a8_grouped_experts_npu


def _legacy_scatter_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    s1: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    top_k = ids.shape[1]
    local_experts = w1.shape[0]
    pair_expert = ids.reshape(-1)
    in_local = pair_expert < local_experts
    pair_expert = torch.where(in_local, pair_expert, local_experts)
    order = torch.argsort(pair_expert.float(), stable=True)
    # Keep the legacy reference's histogram on CPU: its NPU bincount can size
    # the output from a bad async value and request an exabyte allocation.
    counts = torch.bincount(pair_expert.cpu(), minlength=local_experts + 1)[:local_experts].tolist()
    local_order = order[: sum(counts)]
    token_ids = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)
    sorted_x = x[token_ids[local_order]]
    sorted_weights = weights.reshape(-1, 1)[local_order].to(x.dtype)
    output = torch.zeros(num_tokens * top_k, hidden, dtype=torch.float32, device=x.device)
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        stop = start + count
        gate_up = _w8a16_linear_npu(sorted_x[start:stop], w1[expert], s1[expert])
        activated = torch_npu.npu_swiglu(gate_up)
        routed = _w8a16_linear_npu(activated, w2[expert], s2[expert])
        output.index_copy_(0, local_order[start:stop], (routed * sorted_weights[start:stop]).float())
        start = stop
    return output.view(num_tokens, top_k, hidden).sum(1).to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 64, 2048])
def test_moe_inverse_gather_matches_legacy_scatter(num_tokens: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.manual_seed(437 + num_tokens)
    device = "npu:0"
    hidden = 64
    intermediate = 64
    local_experts = 4
    top_k = 3
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device=device) * 0.02
    ids = torch.randint(local_experts * 2, (num_tokens, top_k), device=device)
    ids[:, 0] = torch.randint(local_experts, (num_tokens,), device=device)
    weights = torch.rand(num_tokens, top_k, device=device)
    weights = weights / weights.sum(-1, keepdim=True)
    w1 = torch.randint(-50, 51, (local_experts, hidden, 2 * intermediate), dtype=torch.int8, device=device)
    w2 = torch.randint(-50, 51, (local_experts, intermediate, hidden), dtype=torch.int8, device=device)
    s1 = torch.full((local_experts, 2 * intermediate), 0.0002, dtype=torch.float16, device=device)
    s2 = torch.full((local_experts, hidden), 0.0002, dtype=torch.float16, device=device)

    actual = w8a8_grouped_experts_npu(x, weights, ids, w1, s1, w2, s2)
    expected = _legacy_scatter_reference(x, weights, ids, w1, s1, w2, s2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
