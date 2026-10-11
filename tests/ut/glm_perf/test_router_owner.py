# SPDX-License-Identifier: Apache-2.0
"""Mixed-dtype ownership, pending lifetimes, and composed expert reduction."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.router_owner import RouterExchange, bind_moe, read_options


class Gather:
    def __init__(self, owner):
        self.owner = owner
        self.handles = []

    def all_gather_into_tensor(self, out, source, *, group, async_op=False):
        packet = self.owner.send[: source.numel()]
        for rank in range(self.owner.size):
            out[rank * source.numel() : (rank + 1) * source.numel()].copy_(packet)
        handle = SimpleNamespace(wait=lambda: self.handles.append("wait")) if async_op else None
        self.handles.append(async_op)
        return handle


def exchange(rank=0, size=2, owner=0, rows=640, top_k=8, budget=10**7, backend=None):
    return RouterExchange(
        group="local-stage",
        rank=rank,
        size=size,
        owner_rank=owner,
        max_tokens=rows,
        top_k=top_k,
        device="cpu",
        budget_bytes=budget,
        backend=backend,
    )


@pytest.mark.parametrize("rows", [1, 4, 37, 640])
@pytest.mark.parametrize("size,owner", [(2, 1), (4, 3)])
def test_compact_packet_changed_rows_exact_large_ids_and_fp32_weights(rows, size, owner):
    sender = exchange(rank=owner, size=size, owner=owner)
    gather = Gather(sender)
    sender.backend = gather
    peers = [exchange(rank=rank, size=size, owner=owner, backend=gather) for rank in range(size)]
    for step in (0, 1):
        ids = torch.arange(rows * 8, dtype=torch.int64).reshape(rows, 8) + 16_777_217 + step
        weights = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) / 8192 + step
        sender.post(rows, (ids, weights), overlap=True)
        for peer in peers:
            peer.post(rows, (ids, weights) if peer.rank == owner else None, overlap=True)
            received = peer.finish()
            assert torch.equal(received[0], ids)
            assert torch.equal(received[1], weights)
            assert received[0].is_contiguous() and received[1].is_contiguous()
        sender.finish()
    assert sender.required_bytes == (size + 1) * 640 * 8 * 12


def test_admission_and_pending_reuse_fail_before_transfer():
    with pytest.raises(ValueError, match="reservation"):
        exchange(budget=1)
    with pytest.raises(ValueError, match="geometry"):
        exchange(rank=True)
    sender = exchange(rows=4)
    sender.backend = Gather(sender)
    ids, weights = torch.zeros(4, 8, dtype=torch.int64), torch.ones(4, 8)
    with pytest.raises(ValueError, match="INT64"):
        sender.post(4, (ids.int(), weights))
    sender.post(4, (ids, weights), overlap=True)
    with pytest.raises(RuntimeError, match="scratch reuse"):
        sender.post(4, (ids, weights))
    sender.finish()
    with pytest.raises(RuntimeError, match="not been posted"):
        sender.finish()


@pytest.mark.parametrize(
    "options",
    [
        {"router_owner_rank": 0},
        {"router_overlap_prefill": True},
        {"router_owner_rank": 0, "router_exchange_budget_bytes": False},
    ],
)
def test_options_require_explicit_memory(options):
    with pytest.raises(ValueError):
        read_options(options)
    assert read_options({}) is None


@pytest.mark.parametrize("rows,overlap", [(4, False), (37, True)])
def test_owner_bypasses_peer_gate_topk_and_retains_shared_partial_single_reduction(monkeypatch, rows, overlap):
    events = []
    helper = SimpleNamespace(
        _ep_rank_size=lambda: (0, 2),
        _all_reduce_routed=lambda value: events.append("reduce") or value + 100,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.models.glm5next_w2.moe", helper)
    sender = exchange(rows=rows)
    gather = Gather(sender)
    sender.backend = gather
    x = torch.ones(rows, 2, dtype=torch.float16)
    ids, weights = torch.zeros(rows, 8, dtype=torch.int64), torch.ones(rows, 8)
    sender.post(rows, (ids, weights))
    sender.finish()
    peer = exchange(rank=1, rows=rows, backend=gather)
    delegate = SimpleNamespace(
        shared_expert=lambda value: events.append("shared") or value + 2,
        route=lambda logits: pytest.fail("peer ran top-k"),
        routed_experts_forward=lambda value, ids, weights, reduce_results: (
            events.append("routed") or value.float() + 3
        ),
        combine=lambda routed, shared: routed + shared,
    )
    moe = SimpleNamespace(
        experts=SimpleNamespace(_w2_delegate=delegate),
        is_sequence_parallel=False,
        gate=lambda value: pytest.fail("peer ran gate"),
    )
    bind_moe(moe, peer, overlap_prefill=overlap)
    assert torch.equal(moe.forward(x), torch.full_like(x, 107))
    assert events == (["shared", "routed", "reduce"] if overlap else ["routed", "shared", "reduce"])
    with pytest.raises(ValueError, match="cold-load"):
        bind_moe(moe, peer)
