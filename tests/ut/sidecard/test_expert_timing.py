# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.expert_actor import ExpertCoordinator
from tools.sidecard.expert_profile import ProfileCoordinator, ProfileDispatch, ProfileMiddle
from tools.sidecard.expert_timing import TimedPeer, WorkerTimers


def test_socket_observation_preserves_fragment_sizes_and_private_header():
    clock = iter([1.0, 1.2, 1.3, 1.6])
    sent = []
    peer = TimedPeer(
        SimpleNamespace(sendall=sent.append, recv_into=lambda data: 2),
        1,
        clock=lambda: next(clock),
        header_size=3,
        header_decoder=bytes,
    )
    records = []
    peer.record = records
    peer.sendall(b"tag")
    assert peer.recv_into(memoryview(bytearray(10))) == 2
    assert sent == [b"tag"]
    assert records[0]["header"] == b"tag"
    assert records[1]["bytes"] == 2
    assert records[0]["begin"] == 1 and records[1]["end"] == 1.6


def test_worker_events_read_after_existing_completion_without_adding_sync(tmp_path):
    calls = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.index = len(calls)
            calls.append("allocate")

        def record(self):
            calls.append("record")

        def elapsed_time(self, end):
            assert calls[-1] == "existing completion" or calls[-1] == "query"
            calls.append("query")
            return end.index - self.index

    profile = WorkerTimers(tmp_path / "events.jsonl", 1, SimpleNamespace(Event=Event))
    profile.begin(14, 19, 32)
    for phase in ("upload", "native", "download", "complete"):
        profile.mark(phase)
    calls.append("existing completion")
    profile.finish()
    profile.close()
    assert profile.records[0]["stream_ms"] == {"upload": 2, "native": 2, "download": 2}
    assert profile.records[0]["rank"] == 1
    assert calls.count("query") == 3
    assert '"sequence": 14' in (tmp_path / "events.jsonl").read_text()


def test_worker_rejects_overlapping_or_missing_boundaries(tmp_path):
    profile = WorkerTimers(tmp_path / "events.jsonl", 0, None)
    profile.begin(0, 18, 1)
    with pytest.raises(RuntimeError, match="overlapping"):
        profile.begin(1, 18, 1)
    with pytest.raises(ValueError, match="incomplete"):
        profile.finish()


def test_profile_middle_preserves_captured_routes_and_keeps_dsa_eager():
    calls, amounts, pairs = [], {}, []
    state = tuple(torch.ones(1, 2) for _ in range(4))
    captured = (*state, torch.tensor([[4, 7]]), torch.tensor([[0.3, 0.7]]))
    buffers = tuple(torch.zeros_like(value) for value in state)

    def finish(*values):
        calls.append(values)
        return values[:4]

    kda = SimpleNamespace(index=18, finish_forward=finish)
    dsa = SimpleNamespace(
        index=19, prepare_forward=lambda *values: values, route=lambda x: captured[4:], finish_forward=finish
    )
    stage = SimpleNamespace(layers=(kda, dsa), graphs={18: (SimpleNamespace(replay=lambda: None), buffers, captured)})

    def add(name, seconds):
        amounts[name] = amounts.get(name, 0) + seconds

    owner = SimpleNamespace(
        enabled=True,
        event=lambda: None,
        add=add,
        pair=lambda *args: pairs.append(args),
        frame={"host_seconds": amounts},
    )
    result = ProfileMiddle(stage, owner).forward(state)
    assert all(a is b for a, b in zip(result, state, strict=True))
    assert len(calls) == 2 and all(len(values) == 6 for values in calls)
    assert all(torch.equal(a, b) for a, b in zip(buffers, state, strict=True))
    assert [layer for _, _, layer in pairs] == [18, 19]
    assert amounts["middle_backbone_host_submit"] >= 0 and amounts["middle_finish_host_submit"] >= 0


def test_incompatible_profile_closes_loaded_workers_before_rejecting(monkeypatch):
    closed = []

    def initialize(owner, config):
        owner.prefix_graphs = owner.middle_graphs = owner.suffix_graphs = None
        owner.close = lambda: closed.append(True)

    monkeypatch.setattr(ExpertCoordinator, "__init__", initialize)
    with pytest.raises(ValueError, match="qualified CUDA graph"):
        ProfileCoordinator("unused")
    assert closed == [True]


def test_dma_profile_uses_control_observation_without_host_payload_or_extra_copy(monkeypatch):
    clock = iter([1.0, 6.0])
    monkeypatch.setattr("tools.sidecard.expert_profile.time.perf_counter", lambda: next(clock))
    frame = dict(host_seconds={}, dispatches=[])
    peer = SimpleNamespace(record=None)

    def call(*args):
        peer.record.extend(
            [
                dict(direction="send", begin=2, end=3),
                dict(direction="receive", begin=4, end=5),
            ]
        )
        return "result"

    owner = SimpleNamespace(
        enabled=True,
        spec={"expert_transport": "direct-dma"},
        frame=frame,
        event=lambda: None,
        pair=lambda *args: None,
        add=lambda key, value: frame["host_seconds"].update({key: value}),
    )
    dispatch = SimpleNamespace(connections=[peer], with_shared=call)
    assert (
        ProfileDispatch(dispatch, owner).with_shared(SimpleNamespace(index=18), torch.ones(1, 2), None, None)
        == "result"
    )
    assert frame["dispatches"][0]["cold_routes"] is None
    assert sum(frame["host_seconds"].values()) == 5
    assert peer.record is None


def test_timed_peer_preserves_transport_close_timeout():
    calls = []
    TimedPeer(SimpleNamespace(settimeout=calls.append), 0).settimeout(15)
    assert calls == [15]
