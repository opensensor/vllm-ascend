# SPDX-License-Identifier: Apache-2.0
"""Direct worker protocol, GPU slot reuse, and fail-closed ownership behavior."""

import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dma_dispatch import DmaDispatch
from tools.sidecard.dma_protocol import (
    CAPACITY,
    CLOSE_TAG,
    COMMAND,
    INPUT_ROW_BYTES,
    LENGTH,
    MAX_CONTROL_BYTES,
    OUTPUT_ROW_BYTES,
    PROTOCOL,
    AdmittedCudaSource,
    CudaSourceEvent,
    receive_control,
    send_control,
    validate_command,
    validate_descriptor,
    validate_identity,
)


def test_private_control_roundtrip_and_budget():
    a, b = socket.socketpair()
    with a, b:
        send_control(a, {"nonce": "abc", "ready": True})
        assert receive_control(b) == {"nonce": "abc", "ready": True}
        with pytest.raises(ValueError, match="budget"):
            send_control(a, {"padding": "x" * MAX_CONTROL_BYTES})


@pytest.mark.parametrize("length", [0, MAX_CONTROL_BYTES + 1, 2**32 - 1])
def test_untrusted_control_size_is_rejected_before_body_read(length):
    a, b = socket.socketpair()
    with a, b:
        a.sendall(LENGTH.pack(length))
        with pytest.raises(ValueError, match="size"):
            receive_control(b)


@pytest.mark.parametrize(
    "field", ["protocol", "nonce", "rank", "physical_chip", "checkpoint_sha256", "placement_sha256", "capacity"]
)
def test_worker_ownership_and_session_authentication(field):
    spec = {"checkpoint_sha256": "c" * 64, "expert_placement_sha256": "p" * 64}
    hello = dict(
        protocol=PROTOCOL,
        nonce="n",
        rank=0,
        physical_chip=2,
        checkpoint_sha256=spec["checkpoint_sha256"],
        placement_sha256=spec["expert_placement_sha256"],
        capacity=CAPACITY,
    )
    validate_identity(hello, spec, "n", 0)
    hello[field] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate_identity(hello, spec, "n", 0)


@pytest.mark.parametrize(
    "change",
    [
        {"size": True},
        {"size": CAPACITY - 1},
        {"offset": 1},
        {"offset": False},
        {"backend": "ascend"},
        {"handle": "z" * 128},
        {"handle": "aa"},
    ],
)
def test_cuda_descriptor_rejects_suballocation_and_invalid_handle(change):
    descriptor = dict(backend="cuda", size=CAPACITY, offset=0, handle="00" * 64)
    validate_descriptor(descriptor)
    descriptor.update(change)
    with pytest.raises(ValueError, match="IPC"):
        validate_descriptor(descriptor)


@pytest.mark.parametrize("values", [(1, 18, 1), (0, 17, 1), (0, 18, 0), (0, 18, 33)])
def test_commands_reject_stale_layer_and_capacity(values):
    with pytest.raises(ValueError, match="identity"):
        validate_command(COMMAND.pack(*values), 0)


class Event:
    def __init__(self, events, rank):
        self.events, self.rank = events, rank

    def record(self, stream):
        self.events.append(("source_ready", self.rank))

    def wait(self):
        self.events.append(("source_complete", self.rank))


class Peer:
    def __init__(self, owner, rank, events):
        self.owner, self.rank, self.events = owner, rank, events
        self.tag = None
        self.corrupt = False

    def sendall(self, tag):
        self.events.append(("control_send", self.rank))
        self.tag = tag
        _, _, rows = COMMAND.unpack(tag)
        slot = self.owner.inputs[self.rank]
        xbytes, ibytes = rows * 4096 * 2, rows * 8 * 4
        x = slot[:xbytes].view(torch.float16).reshape(rows, 4096).float()
        ids = slot[xbytes : xbytes + ibytes].view(torch.int32).reshape(rows, 8)
        weights = slot[xbytes + ibytes : rows * INPUT_ROW_BYTES].view(torch.float32).reshape(rows, 8)
        active = (ids >= self.rank * 144) & (ids < (self.rank + 1) * 144)
        partial = x * torch.where(active, weights * (ids + 1), 0).sum(-1, keepdim=True)
        self.owner.outputs[self.rank][: rows * OUTPUT_ROW_BYTES].view(torch.float32).reshape(rows, 4096).copy_(partial)

    def recv_into(self, target):
        self.events.append(("control_receive", self.rank))
        tag = self.tag if not self.corrupt else COMMAND.pack(999, 18, 1)
        target[:] = tag
        return len(tag)


def dispatcher():
    value = object.__new__(DmaDispatch)
    events = []
    hot = torch.tensor([1, 31, 76, 125, 144, 210, 223, 284])
    lookup = torch.full((288,), -1, dtype=torch.int32)
    lookup[hot] = torch.arange(8, dtype=torch.int32)
    value.lookups = {18: lookup}
    value.torch = SimpleNamespace(
        **{name: getattr(torch, name) for name in ("where", "stack", "float16", "float32", "int32")},
        cuda=SimpleNamespace(current_stream=lambda: "stream"),
    )
    value.inputs = [torch.empty(CAPACITY, dtype=torch.uint8) for _ in range(2)]
    value.outputs = [torch.full((CAPACITY,), 255, dtype=torch.uint8) for _ in range(2)]
    value.events = [Event(events, rank) for rank in range(2)]
    value.connections = [Peer(value, rank, events) for rank in range(2)]
    value.sequences = [0, 0]
    value.failed = value.closed = False
    value.lock = threading.RLock()
    value.driver = SimpleNamespace(flush=lambda: events.append(("write_visibility", None)))

    def cuda(layer, x, ids, weights):
        events.append(("cuda_hot", None))
        return x.float() * ((hot[ids.long()] + 1) * weights).sum(-1, keepdim=True)

    value.cuda = cuda
    return value, events


@pytest.mark.parametrize("rows", [1, 4, 17, 32])
@pytest.mark.parametrize(
    "selected",
    [[1, 31, 76, 125, 144, 210, 223, 284], list(range(2, 10)), list(range(145, 153)), [1, 2, 3, 4, 144, 145, 146, 210]],
)
def test_dma_slots_preserve_route_math_and_never_consume_inactive_outputs(rows, selected):
    actor, events = dispatcher()
    for amplitude in (0.5, 3.0):
        hidden = torch.full((rows, 4096), amplitude).half()
        ids = torch.tensor(selected).expand(rows, -1)
        weights = torch.arange(1, 9).float().expand(rows, -1) / 36
        actual = actor(18, hidden, ids, weights)
        expected = hidden.float() * ((ids + 1) * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert torch.isfinite(actual).all()
    assert sum(kind == "control_send" for kind, _ in events) == sum(actor.sequences)
    assert all(n == COMMAND.size for n in [len(peer.tag) for peer in actor.connections if peer.tag])


def test_source_events_precede_control_and_shared_overlaps_both_workers():
    actor, events = dispatcher()
    x = torch.ones(1, 4096).half()
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]])
    actor.compute(18, x, ids, torch.ones(1, 8), lambda: events.append(("shared", None)))
    assert events[:6] == [
        ("source_ready", 0),
        ("source_ready", 1),
        ("source_complete", 0),
        ("control_send", 0),
        ("source_complete", 1),
        ("control_send", 1),
    ]
    assert events.index(("shared", None)) < events.index(("control_receive", 0))
    assert events.index(("write_visibility", None)) > events.index(("control_receive", 1))


def test_corrupted_completion_permanently_fails_dispatch():
    actor, _ = dispatcher()
    actor.connections[0].corrupt = True
    args = (18, torch.ones(1, 4096).half(), torch.arange(2, 10)[None], torch.ones(1, 8))
    with pytest.raises(ValueError, match="another invocation"):
        actor(*args)
    assert actor.failed
    with pytest.raises(RuntimeError, match="failed"):
        actor(*args)


def test_protocol_rejects_non_object_control():
    a, b = socket.socketpair()
    with a, b:
        body = json.dumps([]).encode()
        a.sendall(LENGTH.pack(len(body)) + body)
        with pytest.raises(ValueError, match="object"):
            receive_control(b)


def test_failed_owner_cleanup_retains_slow_workers_without_killing(tmp_path):
    actor, events = dispatcher()
    actor.failed = True
    actor.output = tmp_path
    actor.inputs, actor.outputs, actor.events, actor.imports, actor.connections = [], [], [], [], []
    actor.logs = []
    actor.torch.cuda.synchronize = lambda: events.append(("drain_cuda", None))

    def wait(timeout):
        raise subprocess.TimeoutExpired("fake-worker", timeout)

    actor.processes = [SimpleNamespace(pid=123, wait=wait)]
    actor.close()
    receipt = json.loads((tmp_path / "dma-retained-workers.json").read_text())
    assert receipt["pids"] == [123] and receipt["automatic_kill"] is False
    assert actor.closed


def test_clean_close_drains_consumers_releases_imports_then_releases_exporter(tmp_path):
    actor, events = dispatcher()
    actor.output = tmp_path
    actor.torch.cuda.synchronize = lambda: events.append(("drain_cuda", None))
    actor.imports = [SimpleNamespace(close=lambda: events.append(("unmap_import", None)))]
    actor.events = [SimpleNamespace(close=lambda: events.append(("close_event", None)))]
    actor.logs = []
    actor.processes = [SimpleNamespace(wait=lambda timeout: 0)]

    class ClosingPeer:
        def settimeout(self, timeout):
            pass

        def sendall(self, body):
            events.append(("close_command" if body == CLOSE_TAG else "release_export", None))

        def recv_into(self, target):
            target[:] = CLOSE_TAG
            events.append(("native_quiescent", None))
            return len(CLOSE_TAG)

        def close(self):
            events.append(("socket_close", None))

    actor.connections = [ClosingPeer(), ClosingPeer()]
    actor.close()
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "drain_cuda"
    assert kinds.index("unmap_import") > max(i for i, kind in enumerate(kinds) if kind == "native_quiescent")
    assert kinds.index("release_export") > kinds.index("unmap_import")
    assert actor.closed


def test_quarantine_keeps_original_endpoint_owners_after_coordinator_loss(tmp_path, monkeypatch):
    from tools.sidecard import dma_worker

    signals = []
    monkeypatch.setattr(dma_worker.signal, "signal", lambda *args: signals.append(args))

    def stop_test_loop(seconds):
        raise InterruptedError("unit test stops keeper loop")

    monkeypatch.setattr(dma_worker.time, "sleep", stop_test_loop)
    slots = SimpleNamespace(peer=SimpleNamespace(failed=False))
    with pytest.raises(InterruptedError, match="keeper"):
        dma_worker.quarantine(slots, tmp_path, 0, RuntimeError("owner lost"))
    receipt = json.loads((tmp_path / "dma-quarantine-rank0.json").read_text())
    assert receipt["automatic_kill"] is False and receipt["gpu_and_hbm_owners_alive"] is True
    assert receipt["native_dma_uncertain"] is False and len(signals) == 2


def test_cancelled_prefill_stops_between_complete_chunks_without_poisoning_engine():
    from tools.sidecard.expert_actor import ExpertCoordinator

    engine = object.__new__(ExpertCoordinator)
    cancel = threading.Event()
    seen = []

    def forward(tokens):
        seen.append(list(tokens))
        cancel.set()
        return "first chunk"

    engine.forward = forward
    with pytest.raises(InterruptedError, match="cancelled"):
        engine.prefill(list(range(100)), cancelled=cancel)
    assert seen == [list(range(32))]


def test_receiver_never_treats_unadmitted_cuda_source_as_ready():
    seen = []
    source = AdmittedCudaSource(SimpleNamespace(check_owner=lambda: seen.append("owner")))
    with pytest.raises(RuntimeError, match="admitted"):
        source.wait()
    source.admitted = True
    source.wait()
    assert source.leases == 0 and seen == ["owner", "owner"]


def test_process_owned_source_event_orders_asgi_stream_before_socket_admission():
    import os

    event = object.__new__(CudaSourceEvent)
    event.pid = os.getpid()
    calls = []
    event.event = SimpleNamespace(pointer=123)
    event.driver = SimpleNamespace(call=lambda *args: calls.append(args))
    event.record(SimpleNamespace(cuda_stream=456))
    event.wait()
    assert calls == [("cuEventRecord", 123, 456), ("cuEventSynchronize", 123)]
    event.pid += 1
    with pytest.raises(RuntimeError, match="another process"):
        event.record(SimpleNamespace(cuda_stream=456))


def test_torch_hbm_endpoint_retains_storage_until_dma_lease_is_released():
    from tools.sidecard.dma_worker import TensorAllocation

    runtime = SimpleNamespace(kind="ascend", device=0, check_owner=lambda: None)
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="npu", index=0),
        is_contiguous=lambda: True,
        numel=lambda: 4096,
        element_size=lambda: 4,
        data_ptr=lambda: 12345,
    )
    allocation = TensorAllocation(runtime, tensor)
    assert allocation.pointer == 12345 and allocation.size == 16384 and allocation.tensor is tensor
    allocation.leases = 1
    with pytest.raises(RuntimeError, match="retained"):
        allocation.close()
    allocation.leases = 0
    allocation.close()
    assert allocation.tensor is None
    with pytest.raises(RuntimeError, match="closed"):
        _ = allocation.pointer
    tensor.device.index = 1
    with pytest.raises(ValueError, match="NPU"):
        TensorAllocation(runtime, tensor)
