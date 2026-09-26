# SPDX-License-Identifier: Apache-2.0
"""Parity for fused small-batch Qwen4Exp routing on 310P."""

import pytest
import torch
import torch_npu  # noqa: F401 - register the NPU device extension

from vllm_ascend.models.qwen4_exp import moe

LOCAL_EXPERTS = 128
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 32


@pytest.mark.parametrize("expert_offset", [0, 128, 384])
@pytest.mark.parametrize("all_peer", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 2])
def test_fused_small_moe_routes_match_sort_dispatch(
    monkeypatch, expert_offset: int, all_peer: bool, num_tokens: int
) -> None:
    generator = torch.Generator().manual_seed(419)
    x = torch.randn((2, HIDDEN_SIZE), generator=generator, dtype=torch.float16, device="cpu").npu()
    ids = torch.tensor(
        [[0, 128, 130, 255, 256, 511, 129, 300, 3, 200], [131, 132, 1, 128, 255, 129, 130, 200, 201, 202]],
        dtype=torch.int64,
        device="npu",
    )
    if all_peer:
        ids = torch.full_like(ids, 511 if expert_offset == 0 else 0)
    weights = torch.rand((2, 10), generator=generator).npu()
    weights /= weights.sum(dim=1, keepdim=True)
    x, ids, weights = x[:num_tokens], ids[:num_tokens], weights[:num_tokens]
    w13 = torch.randint(-8, 8, (LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.int8, device="npu")
    w2 = torch.randint(-8, 8, (LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.int8, device="npu")
    s13 = torch.full((LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE), 0.01, dtype=torch.float32, device="npu")
    s2 = torch.full((LOCAL_EXPERTS, HIDDEN_SIZE), 0.01, dtype=torch.float32, device="npu")

    def run() -> torch.Tensor:
        return moe._w8a8_packed_grouped_experts_npu(x, weights, ids, w13, s13, w2, s2, expert_offset)

    fused = run()
    with monkeypatch.context() as sorted_dispatch:
        sorted_dispatch.setattr(moe, "_FUSED_ROUTING_MAX_TOKENS", 0)
        reference = run()
    torch.testing.assert_close(fused, reference, atol=2e-3, rtol=2e-3)
    if all_peer:
        torch.testing.assert_close(fused, torch.zeros_like(fused))
