# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for device-resident Qwen4Exp expert routing."""

import sys
import types

import pytest
import torch

from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch
from vllm_ascend.models.qwen4_exp.moe import _w8a8_packed_grouped_experts_npu, w8a8_grouped_experts
from vllm_ascend.models.qwen4_exp.weight_mapping import local_expert_range


@pytest.mark.parametrize("num_experts,world_size", [(8, 1), (8, 4), (512, 6)])
def test_dispatch_matches_global_topk_and_uneven_shards(num_experts: int, world_size: int) -> None:
    torch.manual_seed(419)
    num_tokens, top_k = 7, 3
    ids = torch.randint(num_experts, (num_tokens, top_k))
    weights = torch.rand(num_tokens, top_k)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    for rank in range(world_size):
        start, stop = local_expert_range(num_experts, world_size, rank)
        dispatch = build_grouped_expert_dispatch(
            weights,
            ids,
            num_local_experts=stop - start,
            expert_offset=start,
            weight_dtype=torch.float16,
        )
        flat_ids = ids.reshape(-1)
        local_mask = (flat_ids >= start) & (flat_ids < stop)
        expected_counts = torch.bincount(flat_ids[local_mask] - start, minlength=stop - start)
        torch.testing.assert_close(dispatch.counts, expected_counts)
        assert dispatch.group_list.tolist() == expected_counts.cumsum(0).tolist()

        # The sorted local portion contains only owned experts, in stable
        # expert order. Peer-owned routes occupy the trailing sentinel bin.
        local_count = int(expected_counts.sum())
        sorted_ids = flat_ids[dispatch.order]
        assert sorted_ids[:local_count].tolist() == sorted(flat_ids[local_mask].tolist())
        assert not (((sorted_ids[local_count:] >= start) & (sorted_ids[local_count:] < stop)).any())
        torch.testing.assert_close(dispatch.order[dispatch.inverse_order], torch.arange(num_tokens * top_k))
        torch.testing.assert_close(
            dispatch.route_weights[dispatch.order][dispatch.inverse_order].reshape(num_tokens, top_k),
            weights.to(torch.float16),
        )
        torch.testing.assert_close(
            dispatch.token_indices[dispatch.order][dispatch.inverse_order],
            torch.arange(num_tokens).repeat_interleave(top_k),
        )


def test_dispatch_skew_and_all_peer_owned_routes() -> None:
    ids = torch.full((4, 2), 5, dtype=torch.int64)
    weights = torch.full((4, 2), 0.5)
    owned = build_grouped_expert_dispatch(
        weights, ids, num_local_experts=2, expert_offset=4, weight_dtype=torch.float16
    )
    peer = build_grouped_expert_dispatch(weights, ids, num_local_experts=2, expert_offset=0, weight_dtype=torch.float16)
    assert owned.counts.tolist() == [0, 8]
    assert owned.group_list.tolist() == [0, 8]
    assert peer.counts.tolist() == [0, 0]
    assert peer.group_list.tolist() == [0, 0]
    torch.testing.assert_close(peer.order, torch.arange(8))


def test_packed_grouped_dispatch_omits_peer_rows(monkeypatch) -> None:
    """Prefill never passes peer-owned rows to grouped matmul."""

    grouped_input_rows = []

    def grouped_matmul(x, weight, scale, group_list):
        grouped_input_rows.append(x.shape[0])
        result = torch.full((x.shape[0], weight.shape[1]), torch.nan, dtype=x.dtype)
        start = 0
        group_ends = group_list.tolist() if isinstance(group_list, torch.Tensor) else group_list
        for expert, stop in enumerate(group_ends):
            if stop > start:
                result[start:stop] = (x[start:stop].float() @ (weight[expert].float() * scale[expert, :, None]).t()).to(
                    x.dtype
                )
            start = stop
        return result

    fake_npu = types.SimpleNamespace(
        npu_quant_grouped_matmul_dequant=grouped_matmul,
        npu_swiglu=lambda x: torch.nn.functional.silu(x[:, : x.shape[1] // 2]) * x[:, x.shape[1] // 2 :],
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    hidden, intermediate = 8, 4
    x = torch.randn(128, hidden, dtype=torch.float16)
    ids = torch.tensor([[0, 2], [2, 1], [2, 2]]).repeat(43, 1)[:128]
    weights = torch.full((128, 2), 0.5)
    w13 = torch.randint(-3, 4, (2, 2 * intermediate, hidden), dtype=torch.int8)
    w2 = torch.randint(-3, 4, (2, hidden, intermediate), dtype=torch.int8)
    s13 = torch.full((2, 2 * intermediate), 0.01)
    s2 = torch.full((2, hidden), 0.01)

    partial = _w8a8_packed_grouped_experts_npu(x, weights, ids, w13, s13, w2, s2, 0)
    assert torch.isfinite(partial).all()
    all_peer = _w8a8_packed_grouped_experts_npu(x, weights, ids, w13, s13, w2, s2, 3)
    torch.testing.assert_close(all_peer, torch.zeros_like(x))
    local_count = int((ids < 2).sum())
    assert grouped_input_rows == [local_count, local_count]

    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = ids[token, slot].item()
            if expert >= 2:
                continue
            gate_up = grouped_matmul(x[token : token + 1], w13[expert : expert + 1], s13[expert : expert + 1], [1])
            hidden_act = fake_npu.npu_swiglu(gate_up)
            output = grouped_matmul(hidden_act, w2[expert : expert + 1], s2[expert : expert + 1], [1])
            expected[token] += (output[0].float() * weights[token, slot]).to(expected.dtype)
    torch.testing.assert_close(partial, expected, atol=5e-4, rtol=5e-3)


def test_packed_small_batch_keeps_device_only_dispatch(monkeypatch) -> None:
    grouped_input_rows = []

    def grouped_matmul(x, weight, scale, group_list):
        grouped_input_rows.append(x.shape[0])
        return torch.ones((x.shape[0], weight.shape[1]), dtype=x.dtype)

    fake_npu = types.SimpleNamespace(
        npu_quant_grouped_matmul_dequant=grouped_matmul,
        npu_swiglu=lambda x: torch.nn.functional.silu(x[:, : x.shape[1] // 2]) * x[:, x.shape[1] // 2 :],
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    x = torch.ones((2, 8), dtype=torch.float16)
    ids = torch.tensor([[0, 2], [2, 1]])
    weights = torch.full((2, 2), 0.5)
    with monkeypatch.context() as no_sync:
        no_sync.setattr(torch.Tensor, "item", lambda self: pytest.fail("small batches must not synchronize"))
        output = _w8a8_packed_grouped_experts_npu(
            x,
            weights,
            ids,
            torch.zeros((2, 8, 8), dtype=torch.int8),
            torch.ones((2, 8)),
            torch.zeros((2, 8, 4), dtype=torch.int8),
            torch.ones((2, 8)),
            0,
        )
    assert torch.isfinite(output).all()
    assert grouped_input_rows == [4, 4]


def test_uneven_shard_partials_sum_to_full_moe_output() -> None:
    generator = torch.Generator().manual_seed(419)
    num_experts, hidden, intermediate, world_size = 6, 8, 4, 4
    x = torch.randn(5, hidden, generator=generator)
    ids = torch.tensor([[0, 5], [1, 2], [3, 4], [5, 0], [2, 4]])
    weights = torch.rand(5, 2, generator=generator)
    weights /= weights.sum(dim=-1, keepdim=True)
    w13 = torch.randint(-16, 17, (num_experts, 2 * intermediate, hidden), generator=generator, dtype=torch.int8)
    w2 = torch.randint(-16, 17, (num_experts, hidden, intermediate), generator=generator, dtype=torch.int8)
    s13 = torch.full((num_experts, 2 * intermediate, 1), 0.01)
    s2 = torch.full((num_experts, hidden, 1), 0.01)
    o13 = torch.zeros_like(s13)
    o2 = torch.zeros_like(s2)

    full = w8a8_grouped_experts(x, weights, ids, w13, s13, o13, w2, s2, o2)
    partials = []
    for rank in range(world_size):
        start, stop = local_expert_range(num_experts, world_size, rank)
        partials.append(
            w8a8_grouped_experts(
                x,
                weights,
                ids,
                w13[start:stop],
                s13[start:stop],
                o13[start:stop],
                w2[start:stop],
                s2[start:stop],
                o2[start:stop],
                expert_offset=start,
                num_global_experts=num_experts,
            )
        )
    torch.testing.assert_close(sum(partials), full, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "weights,ids,count,offset",
    [
        (torch.ones(2), torch.ones(2, dtype=torch.int64), 2, 0),
        (torch.ones(2, 2), torch.ones(2, 3, dtype=torch.int64), 2, 0),
        (torch.ones(2, 2), torch.ones(2, 2, dtype=torch.float32), 2, 0),
        (torch.ones(2, 2), torch.ones(2, 2, dtype=torch.int64), 0, 0),
        (torch.ones(2, 2), torch.ones(2, 2, dtype=torch.int64), 2, -1),
    ],
)
def test_dispatch_rejects_invalid_metadata(weights, ids, count, offset) -> None:
    with pytest.raises(ValueError):
        build_grouped_expert_dispatch(
            weights, ids, num_local_experts=count, expert_offset=offset, weight_dtype=torch.float16
        )
