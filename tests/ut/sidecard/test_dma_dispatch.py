# SPDX-License-Identifier: Apache-2.0
"""Direct worker protocol, GPU slot reuse, and fail-closed ownership behavior."""

import json
import socket
import struct
import subprocess
import threading
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dma_dispatch import DmaDispatch
from tools.sidecard.dma_pack_graph import prepare_payloads
from tools.sidecard.dma_protocol import (
    CAPACITY,
    CLOSE_TAG,
    COMMAND,
    DIAGNOSTIC_TAG,
    INPUT_ROW_BYTES,
    LENGTH,
    MAX_CONTROL_BYTES,
    OUTPUT_ROW_BYTES,
    PROTOCOL,
    AdmittedCudaSource,
    CudaSourceEvent,
    dma_capacity,
    receive_control,
    send_control,
    validate_command,
    validate_descriptor,
    validate_identity,
)
from tools.sidecard.dma_worker import WorkerSlots, diagnostic_control, validate_routes
from tools.sidecard.pcie.peer import ASCEND_TO_CUDA
from tools.sidecard.single_route_graph import SingleRouteDecodeGraphs


@pytest.mark.parametrize("rows", [1, 4, 17, 32])
def test_worker_metadata_accepts_cold_routes_and_zero_weight_hot_routes(rows):
    count = rows * 8
    ids = [1, 2, 145, 287, 0, 3, 144, 286] * rows
    weights = [0, 0.5, 1, 2, 0, 0.25, 0.125, 0] * rows
    metadata = struct.pack(f"<{count}i{count}f", *ids, *weights)
    assert validate_routes(metadata, rows, frozenset([1])) == (2 * rows, 3 * rows)


@pytest.mark.parametrize(
    "expert,weight", [(-1, 0), (288, 0), (2, -0.1), (2, float("nan")), (2, float("inf")), (1, 0.01)]
)
def test_worker_metadata_rejects_bad_route_before_native_launch(expert, weight):
    metadata = struct.pack("<8i8f", *([expert] + [2] * 7), *([weight] + [0.5] * 7))
    with pytest.raises(ValueError, match="route"):
        validate_routes(metadata, 1, frozenset([1]))


@pytest.mark.parametrize("rows", [0, 33, True])
def test_worker_metadata_rejects_invalid_capacity(rows):
    with pytest.raises(ValueError, match="row capacity"):
        validate_routes(b"", rows, frozenset())


def test_worker_metadata_does_not_read_past_buffer():
    with pytest.raises(struct.error):
        validate_routes(b"\0" * 63, 1, frozenset())


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("route_count", [0, 1, 2, 8])
@pytest.mark.parametrize("observing", [False, True])
def test_native_completion_event_is_fresh_and_fast_selection_counts_owned_contributions(
    monkeypatch, rank, route_count, observing
):
    slots = object.__new__(WorkerSlots)
    slots.max_chunk = 32
    slots.rank = rank
    slots.single_route_calls = 0
    completions = []
    slots.torch = SimpleNamespace(
        float16=torch.float16,
        int32=torch.int32,
        float32=torch.float32,
        npu=SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None)),
    )
    slots.ascend = SimpleNamespace(kind="ascend", device=0, copy=lambda *args: None, check_owner=lambda: None)
    slots.packed = torch.zeros(CAPACITY, dtype=torch.uint8)
    slots.npu_input = SimpleNamespace(pointer=1)
    slots.metadata_host = SimpleNamespace(pointer=1)
    # Repeated IDs are separate contributions, rather than one distinct expert.
    slots.metadata = struct.pack("<8i8f", *([2] * 8), *([0.5] * route_count + [0.0] * (8 - route_count)))
    slots.hot_ids = {18: frozenset([1])}
    slots.event = SimpleNamespace(record=lambda stream: None, wait=lambda: None)
    slots.stream = slots.guard = slots.input_export = slots.output_export = None
    slots.source_event = SimpleNamespace(admitted=True)
    slots.driver = SimpleNamespace(flush=lambda **kwargs: None)
    slots.transfer_calls = 0
    slots.calls = slots.upload_bytes = slots.download_bytes = slots.mapping_creations = slots.mapping_hits = 0
    slots.profile = []

    def copy(operation, *args, source_ready):
        if operation == ASCEND_TO_CUDA:
            completions.append(source_ready.event)
        return {"gpu_mapping_reused": 0}

    slots.peer = SimpleNamespace(copy=copy)
    output = SimpleNamespace(
        device=SimpleNamespace(type="npu", index=0),
        is_contiguous=lambda: True,
        numel=lambda: 4096,
        element_size=lambda: 4,
        data_ptr=lambda: 12345,
    )
    selections = []

    def execute(self, *args, **kwargs):
        selections.append(kwargs.get("owned_routes"))
        return output

    monkeypatch.setattr(SingleRouteDecodeGraphs, "__call__", execute)
    actor = object.__new__(SingleRouteDecodeGraphs)
    observed = []
    diagnostic = SimpleNamespace(writer=True, observe_compute=lambda *args, **kwargs: observed.append((args, kwargs)))
    for sequence in range(2):
        slots.compute(actor, sequence, 18, 1, diagnostic=diagnostic if observing else None)
    assert len(completions) == 2 and completions[0] is not completions[1]
    single = rank == 0 and route_count == 1
    assert selections == ([1, 1] if single else [None, None])
    assert slots.single_route_calls == (2 if single else 0)
    if observing:
        assert [values[0][0] for values in observed] == [0, 1]
        assert all(values[0][-1] is output for values in observed)
        assert all(
            values[1]["execution"] == ("single-route-graph" if single else "decode-graph") for values in observed
        )
    else:
        assert observed == []


def test_expert_gate_control_uses_reserved_frame_and_owner_bound_roundtrip():
    coordinator, native = socket.socketpair()
    seen, errors = [], []
    observer = SimpleNamespace(handle_command=lambda command: seen.append(command) or {"passed": True})
    command = {"action": "phase", "nonce": "fixture", "plan_sha256": "p" * 64}

    def serve():
        try:
            assert native.recv(COMMAND.size) == DIAGNOSTIC_TAG
            diagnostic_control(native, observer, "exact-owner")
        except BaseException as error:
            errors.append(error)

    dispatch = object.__new__(DmaDispatch)
    dispatch.lock = threading.RLock()
    dispatch.closed = dispatch.failed = False
    dispatch.nonce = "exact-owner"
    dispatch.connections = [coordinator, None]
    dispatch.execution_diagnostics = object()
    with coordinator, native:
        coordinator.settimeout(2)
        native.settimeout(2)
        worker = threading.Thread(target=serve)
        worker.start()
        assert dispatch.diagnostic_command(0, command) == {"passed": True}
        worker.join(timeout=2)
        assert not worker.is_alive() and not errors
    assert seen == [command]


@pytest.mark.parametrize("enabled,nonce", [(False, "owner"), (True, "stale-owner")])
def test_expert_gate_control_rejects_disabled_or_other_owner(monkeypatch, enabled, nonce):
    seen = []
    observer = SimpleNamespace(handle_command=lambda command: seen.append(command)) if enabled else None
    monkeypatch.setattr(
        "tools.sidecard.dma_worker.receive_control", lambda connection: {"execution_diagnostic": {}, "nonce": nonce}
    )
    with pytest.raises(ValueError, match="unauthenticated"):
        diagnostic_control(object(), observer, "owner")
    assert not seen


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


def dispatcher(rows=32):
    value = object.__new__(DmaDispatch)
    value.max_chunk = rows
    value.capacity = dma_capacity(dict(expert_transport="direct-dma", prefill_window_size=rows))
    events = []
    hot = torch.tensor([1, 31, 76, 125, 144, 210, 223, 284])
    lookup = torch.full((288,), -1, dtype=torch.int32)
    lookup[hot] = torch.arange(8, dtype=torch.int32)
    value.lookups = {18: lookup}
    value.torch = SimpleNamespace(
        **{name: getattr(torch, name) for name in ("where", "stack", "float16", "float32", "int32")},
        cuda=SimpleNamespace(current_stream=lambda: "stream"),
    )
    value.inputs = [torch.empty(value.capacity, dtype=torch.uint8) for _ in range(2)]
    value.outputs = [torch.full((value.capacity,), 255, dtype=torch.uint8) for _ in range(2)]
    value.events = [Event(events, rank) for rank in range(2)]
    value.connections = [Peer(value, rank, events) for rank in range(2)]
    value.sequences = [0, 0]
    value.pack_graphs = value.hot_graphs = None
    value.failed = value.closed = False
    value.lock = threading.RLock()
    value.driver = SimpleNamespace(flush=lambda: events.append(("write_visibility", None)))

    def cuda(layer, x, ids, weights):
        events.append(("cuda_hot", None))
        return x.float() * ((hot[ids.long()] + 1) * weights).sum(-1, keepdim=True)

    value.cuda = cuda
    return value, events


@pytest.mark.parametrize("rows", [1, 4, 17, 32])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    "selected",
    [[1, 31, 76, 125, 144, 210, 223, 284], list(range(2, 10)), list(range(145, 153)), [1, 2, 3, 4, 144, 145, 146, 210]],
)
def test_dma_slots_preserve_route_math_and_never_consume_inactive_outputs(rows, selected, captured):
    actor, events = dispatcher()
    if captured:
        actor.pack_graphs = SimpleNamespace(
            prepare=lambda layer, x, ids, weights: prepare_payloads(
                torch, actor.lookups[layer], actor.inputs, x, ids, weights
            )
        )
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


@pytest.mark.parametrize("rows", [64, 128, 256, 512])
def test_larger_prefill_batches_share_slots_and_keep_both_rank_results(rows):
    actor, events = dispatcher(rows)
    actor.max_chunk = rows
    x = torch.ones(rows, 4096).half()
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]]).expand(rows, -1)
    weights = torch.ones(rows, 8)
    actual = actor(18, x, ids, weights)
    torch.testing.assert_close(actual, x.float() * (ids + 1).sum(-1, keepdim=True))
    assert actor.sequences == [1, 1]
    assert events[-1] == ("write_visibility", None)


@pytest.mark.parametrize("tail", [1, 7, 32])
@pytest.mark.parametrize("chunks", [2, 4, 8, 16])
def test_cold_batch_uses_union_admission_and_keeps_each_cuda_chunk(tail, chunks):
    actor, events = dispatcher(chunks * 32)
    actor.max_chunk = chunks * 32
    actor.cuda.max_chunk = 32
    actor.torch.cat = torch.cat
    parts = []
    # Alternating chunks visit different ranks, including a partial final chunk.
    for index in range(chunks):
        rows = tail if index == chunks - 1 else 32
        start, amplitude = (2, 0.5) if index % 2 == 0 else (145, 3.0)
        x = torch.full((rows, 4096), amplitude).half()
        ids = torch.arange(start, start + 8).expand(rows, -1)
        weights = torch.ones(rows, 8)
        parts.append((x, None, None, None, ids, weights))
    layer = SimpleNamespace(index=18, dense_mlp=lambda x, name: x.float() * 2)
    outputs = list(actor.batch_with_shared(layer, parts))
    for value, (routed, shared) in zip(parts, outputs, strict=True):
        torch.testing.assert_close(routed, value[0].float() * (value[4] + 1).sum(-1, keepdim=True))
        assert torch.equal(shared, value[0].float() * 2)
    assert actor.sequences == [1, 1]
    assert events.count(("cuda_hot", None)) == chunks
    assert [COMMAND.unpack(peer.tag)[2] for peer in actor.connections] == [32 * (chunks - 1) + tail] * 2


@pytest.mark.parametrize("rows,shared", [(1, True), (4, True), (1, False)])
def test_hot_capture_runs_after_admission_and_prefill_keeps_eager_math(rows, shared):
    actor, events = dispatcher()
    graph_calls = []

    def captured(layer, x, ids, weights):
        graph_calls.append(layer)
        events.append(("hot_graph", None))
        return actor.cuda(layer, x, ids, weights), x.float() * 2

    actor.hot_graphs = SimpleNamespace(compute=captured)
    x = torch.ones(rows, 4096).half()
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]]).expand(rows, -1)
    weights = torch.ones(rows, 8)
    hot, shared_result = actor.compute(18, x, ids, weights, (lambda: x.float() * 2) if shared else None)
    torch.testing.assert_close(hot, x.float() * (ids + 1).sum(-1, keepdim=True))
    assert graph_calls == ([18] if rows == 1 and shared else [])
    if graph_calls:
        assert events.index(("control_send", 1)) < events.index(("hot_graph", None))
        assert events.index(("hot_graph", None)) < events.index(("control_receive", 0))
    assert shared_result is None if not shared else torch.equal(shared_result, x.float() * 2)


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
    actor.pack_graphs = SimpleNamespace(close=lambda: events.append(("release_graph_scratch", None)))
    actor.hot_graphs = SimpleNamespace(close=lambda: events.append(("release_hot_scratch", None)))
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
    assert kinds.index("release_hot_scratch") < kinds.index("unmap_import")
    assert kinds.index("release_graph_scratch") < kinds.index("unmap_import")
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
    engine.prefill_chunk_size = 32
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


@pytest.mark.parametrize("rows,capacity", [(32, 2**21), (128, 2**21), (256, 2**22), (512, 2**23), (1024, 2**24)])
def test_slot_capacity_is_derived_and_bound_to_identity_and_descriptors(rows, capacity):
    spec = dict(
        expert_transport="direct-dma",
        prefill_window_size=rows,
        checkpoint_sha256="c" * 64,
        expert_placement_sha256="p" * 64,
    )
    assert dma_capacity(spec) == capacity
    assert rows * INPUT_ROW_BYTES + 64 < capacity
    descriptor = dict(backend="cuda", size=capacity, offset=0, handle="00" * 64)
    validate_descriptor(descriptor, capacity=capacity)
    hello = dict(
        protocol=PROTOCOL,
        nonce="n",
        rank=0,
        physical_chip=2,
        checkpoint_sha256=spec["checkpoint_sha256"],
        placement_sha256=spec["expert_placement_sha256"],
        capacity=capacity,
    )
    validate_identity(hello, spec, "n", 0)
    hello["capacity"] = capacity // 2
    with pytest.raises(ValueError, match="identity"):
        validate_identity(hello, spec, "n", 0)
    descriptor["size"] = capacity // 2
    with pytest.raises(ValueError, match="descriptor"):
        validate_descriptor(descriptor, capacity=capacity)
