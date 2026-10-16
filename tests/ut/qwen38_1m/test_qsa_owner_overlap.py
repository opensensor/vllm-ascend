# SPDX-License-Identifier: Apache-2.0
"""Offline ordering/admission checks; no claim of HCCL overlap qualification."""

from contextlib import contextmanager

import pytest
import torch

from tools.qwen4exp.qsa_owner_overlap import OwnerSelectionPolicy, QSAOwnerOverlap, selection_packet
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection


class FakeStream:
    def __init__(self, trace, name):
        self.trace, self.name = trace, name

    def wait_event(self, event):
        self.trace.append(("wait", self.name, event.name))


class FakeEvent:
    def __init__(self, trace, name):
        self.trace, self.name = trace, name

    def record(self, stream):
        self.trace.append(("record", stream.name, self.name))


class FakeNpu:
    def __init__(self, trace):
        self.trace, self.events = trace, 0
        self.main = FakeStream(trace, "main")

    def Stream(self):
        return FakeStream(self.trace, "selection")

    def Event(self):
        self.events += 1
        return FakeEvent(self.trace, f"event{self.events}")

    def current_stream(self, device):
        assert device == torch.device("cpu")
        return self.main

    @contextmanager
    def stream(self, stream):
        self.trace.append(("enter", stream.name))
        yield
        self.trace.append(("exit", stream.name))


def test_packet_has_one_allocation_with_contiguous_attention_views():
    packet, selection = selection_packet(3, 5, torch.device("cpu"))
    operands = (selection.group_indices, selection.group_counts, selection.tail_starts, selection.tail_counts)
    for value, tensor in enumerate(operands):
        assert tensor.dtype == torch.int32 and tensor.is_contiguous()
        assert tensor.untyped_storage().data_ptr() == packet.untyped_storage().data_ptr()
        tensor.fill_(value)
    assert packet.tolist() == [0] * 15 + [1] * 3 + [2] * 3 + [3] * 3


@pytest.mark.parametrize("tokens,groups", [(0, 5), (3, 0), (-1, 4)])
def test_invalid_packet(tokens, groups):
    with pytest.raises(ValueError):
        selection_packet(tokens, groups, torch.device("cpu"))


@pytest.mark.parametrize("rank,size,owner,minimum", [(0, 0, 0, 128), (6, 6, 0, 128), (0, 6, 6, 128), (0, 6, 0, 0)])
def test_invalid_owner_policy(rank, size, owner, minimum):
    with pytest.raises(ValueError):
        OwnerSelectionPolicy(rank, size, owner, minimum)


@pytest.mark.parametrize(
    "tokens,captured,prefill,size",
    [(3, False, False, 6), (640, True, True, 6), (640, False, False, 6), (640, False, True, 1)],
)
def test_decode_graph_mixed_batch_and_single_rank_keep_original_order(tokens, captured, prefill, size):
    trace = []
    resource = QSAOwnerOverlap(
        OwnerSelectionPolicy(0, size, 0), lambda *args: pytest.fail("collective in fallback"), npu=FakeNpu(trace)
    )
    expected = object()
    result = resource.run(
        tokens=tokens,
        groups=5,
        device=torch.device("cpu"),
        captured=captured,
        pure_prefill=prefill,
        indexer_inputs=[],
        select=lambda: trace.append("select") or expected,
        prepare_local=lambda: trace.append("local") or "qk",
    )
    assert result == ("qk", expected) and trace == ["local", "select"]


def test_owner_broadcast_and_peer_preprocessing_join_without_host_reads(monkeypatch):
    wire = []
    retained = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: retained.append((tensor, stream.name)))
    monkeypatch.setattr(torch.Tensor, "item", lambda tensor: pytest.fail("host scalar read"))
    selected = QSAGroupSelection(
        torch.tensor([[4, 0, 1], [2, 1, 0]], dtype=torch.int32),
        torch.tensor([3, 2], dtype=torch.int32),
        torch.tensor([32, 44], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
    )
    outputs = []
    # Owner first models receipt of its payload; this is a protocol unit test,
    # not a simulation of asynchronous device completion or HCCL latency.
    for rank in (2, 0, 1, 3, 4, 5):
        trace = []
        npu = FakeNpu(trace)

        def broadcast(packet, owner, stream, rank=rank, trace=trace):
            assert owner == 2 and stream.name == "selection"
            trace.append("broadcast")
            if rank == owner:
                wire.append(packet.clone())
            else:
                packet.copy_(wire[0])

        def select(rank=rank, trace=trace):
            assert rank == 2, "peer repeated indexer scoring"
            trace.append("select")
            return selected

        source = torch.ones(2, 128)
        resource = QSAOwnerOverlap(OwnerSelectionPolicy(rank, 6, 2, min_tokens=2), broadcast, npu=npu)
        local, result = resource.run(
            tokens=2,
            groups=5,
            device=source.device,
            captured=False,
            pure_prefill=True,
            indexer_inputs=[source],
            select=select,
            prepare_local=lambda trace=trace: trace.append("local") or "qk",
        )
        assert local == "qk"
        assert trace.index(("wait", "selection", "event1")) < trace.index("broadcast")
        assert trace.index("broadcast") < trace.index(("record", "selection", "event2"))
        assert trace.index("local") < trace.index(("wait", "main", "event2"))
        assert trace[-1] == ("wait", "main", "event2")
        assert any(tensor is source and stream == "selection" for tensor, stream in retained)
        outputs.append(result)
    for output in outputs:
        assert output.group_indices.tolist() == [[4, 0, 1, -1, -1], [2, 1, 0, -1, -1]]
        assert torch.equal(output.group_counts, selected.group_counts)
        assert torch.equal(output.tail_starts, selected.tail_starts)
        assert torch.equal(output.tail_counts, selected.tail_counts)


def test_wrong_device_fails_before_collective_submission():
    resource = QSAOwnerOverlap(
        OwnerSelectionPolicy(0, 6, 0), lambda *args: pytest.fail("invalid broadcast"), npu=FakeNpu([])
    )
    with pytest.raises(ValueError, match="local selection device"):
        resource.run(
            tokens=640,
            groups=5,
            device=torch.device("cpu"),
            captured=False,
            pure_prefill=True,
            indexer_inputs=[torch.empty(1, device="meta")],
            select=lambda: pytest.fail("invalid select"),
            prepare_local=lambda: pytest.fail("invalid preprocessing"),
        )
