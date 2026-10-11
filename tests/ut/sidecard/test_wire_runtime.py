# SPDX-License-Identifier: Apache-2.0
"""Binary corruption, bounded sockets, real process IPC, and fail-closed gates."""

import hashlib
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import replace

import pytest

from tools.sidecard.placement import ExpertPlacement
from tools.sidecard.reference import ExpertReply, ExpertRequest, Route
from tools.sidecard.runtime import LocalWorkerPool, SocketPeer
from tools.sidecard.smoke import smoke_fixture
from tools.sidecard.wire import (
    MAGIC,
    METADATA_LIMIT,
    PREFIX,
    PacketLimits,
    WireIdentity,
    decode_packet,
    encode_packet,
    receive_packet,
)
from tools.sidecard.worker import DenseExpert, DenseExpertBank, ExpertWorker


@pytest.fixture
def packet_context():
    placement = ExpertPlacement(("cuda:0", "npu:0"), (("cuda:0", "npu:0", "npu:0"),))
    limits = PacketLimits(hidden_width=2, max_tokens=2, max_routes=4, experts=3, top_k=2)
    identity = WireIdentity.for_placement(placement, hashlib.sha256(b"tiny-source-weights").hexdigest())
    request = ExpertRequest("pass:1", 3, "npu:0", (Route(1, 0, 2), Route(1, 1, 1)), (1,), ((0.1, -0.0),))
    return placement, limits, identity, request


def changed_metadata(frame, **updates):
    _, metadata_bytes, _ = PREFIX.unpack(frame[: PREFIX.size])
    metadata = json.loads(frame[PREFIX.size : PREFIX.size + metadata_bytes])
    metadata.update(updates)
    data = json.dumps(metadata).encode()
    body = frame[PREFIX.size + metadata_bytes :]
    return PREFIX.pack(MAGIC, len(data), len(body)) + data + body


def test_request_round_trip_preserves_routes_and_fp16_rounding(packet_context):
    _, limits, identity, request = packet_context
    frame = encode_packet(request, identity, limits)
    decoded = decode_packet(frame, identity, limits)
    assert decoded == replace(request, hidden=((struct.unpack("<e", struct.pack("<e", 0.1))[0], -0.0),))
    assert struct.pack("<e", decoded.hidden[0][1]) == b"\x00\x80"
    assert len(frame) <= PREFIX.size + METADATA_LIMIT + limits.max_body_bytes


def test_reply_round_trip_uses_fp32(packet_context):
    _, limits, identity, request = packet_context
    reply = ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, ((0.1, 1.0), (2.0, 3.0)))
    decoded = decode_packet(encode_packet(reply, identity, limits), identity, limits)
    assert decoded.outputs[0][0] == struct.unpack("<f", struct.pack("<f", 0.1))[0]
    assert decoded.routes == request.routes


def test_duplicate_metadata_fields_are_rejected(packet_context):
    _, limits, identity, request = packet_context
    frame = encode_packet(request, identity, limits)
    _, metadata_bytes, body_bytes = PREFIX.unpack(frame[: PREFIX.size])
    metadata = frame[PREFIX.size : PREFIX.size + metadata_bytes]
    metadata = metadata[:-1] + b',"owner":"npu:0"}'
    frame = PREFIX.pack(MAGIC, len(metadata), body_bytes) + metadata + frame[PREFIX.size + metadata_bytes :]
    with pytest.raises(ValueError):
        decode_packet(frame, identity, limits)


@pytest.mark.parametrize(
    "failure",
    [
        "magic",
        "truncated",
        "trailing",
        "shape",
        "checkpoint",
        "placement",
        "kind",
        "bool_rows",
        "unknown_field",
        "body_checksum",
    ],
)
def test_corrupted_frame_is_rejected(packet_context, failure):
    _, limits, identity, request = packet_context
    frame = encode_packet(request, identity, limits)
    if failure == "magic":
        frame = b"BADMAGIC" + frame[8:]
    elif failure == "truncated":
        frame = frame[:-1]
    elif failure == "trailing":
        frame += b"x"
    elif failure == "body_checksum":
        frame = frame[:-1] + bytes([frame[-1] ^ 1])
    else:
        updates = {
            "shape": {"rows": 2},
            "checkpoint": {"checkpoint_sha256": "0" * 64},
            "placement": {"placement_sha256": "0" * 64},
            "kind": {"kind": "error"},
            "bool_rows": {"rows": True},
            "unknown_field": {"extra": 1},
        }[failure]
        frame = changed_metadata(frame, **updates)
    with pytest.raises(ValueError):
        decode_packet(frame, identity, limits)


@pytest.mark.parametrize(
    "failure", ["duplicate_slot", "foreign_token", "nan", "half_overflow", "negative_layer", "wide_row"]
)
def test_bad_packet_is_rejected_before_send(packet_context, failure):
    _, limits, identity, request = packet_context
    if failure == "duplicate_slot":
        request = replace(request, routes=(Route(1, 0, 2), Route(1, 0, 1)))
    elif failure == "foreign_token":
        request = replace(request, token_ids=(0,))
    elif failure == "nan":
        request = replace(request, hidden=((float("nan"), 1.0),))
    elif failure == "half_overflow":
        request = replace(request, hidden=((1e10, 1.0),))
    elif failure == "negative_layer":
        request = replace(request, layer_id=-1)
    else:
        request = replace(request, hidden=((1.0, 2.0, 3.0),))
    with pytest.raises(ValueError):
        encode_packet(request, identity, limits)


def test_oversized_socket_header_is_rejected_without_waiting_for_body(packet_context):
    _, limits, identity, _ = packet_context
    left, right = socket.socketpair()
    try:
        left.sendall(PREFIX.pack(MAGIC, 1, limits.max_body_bytes + 1))
        with pytest.raises(ValueError, match="oversized"):
            receive_packet(right, identity, limits, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_partial_socket_reads_and_retained_packet_storage(packet_context):
    _, limits, identity, request = packet_context
    left, right = socket.socketpair()
    frames = [
        encode_packet(request, identity, limits),
        encode_packet(replace(request, hidden=((2.0, 3.0),)), identity, limits),
    ]

    def sender():
        for frame in frames:
            for start in range(0, len(frame), 3):
                left.sendall(frame[start : start + 3])
        left.close()

    thread = threading.Thread(target=sender)
    thread.start()
    try:
        first = receive_packet(right, identity, limits, time.monotonic() + 2)
        second = receive_packet(right, identity, limits, time.monotonic() + 2)
        assert first.hidden != second.hidden
        assert first.hidden[0][0] == struct.unpack("<e", struct.pack("<e", 0.1))[0]
    finally:
        right.close()
        thread.join(2)


def test_timeout_closes_peer_and_prevents_stream_reuse(packet_context):
    _, limits, identity, request = packet_context
    left, right = socket.socketpair()
    peer = SocketPeer(left, identity, limits, 0.02)
    try:
        with pytest.raises(TimeoutError):
            peer.exchange(request)
        assert left.fileno() == -1
        with pytest.raises(OSError):
            peer.exchange(request)
    finally:
        peer.close()
        right.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected_without_allocating_workers(packet_context, timeout):
    _, limits, identity, _ = packet_context
    left, right = socket.socketpair()
    try:
        with pytest.raises(ValueError):
            SocketPeer(left, identity, limits, timeout)
    finally:
        left.close()
        right.close()


def test_wrong_reply_identity_closes_peer(packet_context):
    _, limits, identity, request = packet_context
    left, right = socket.socketpair()
    reply = ExpertReply("old-pass", 3, "npu:0", request.routes, ((0.0, 1.0),) * 2)
    right.sendall(encode_packet(reply, identity, limits))
    peer = SocketPeer(left, identity, limits, 1)
    try:
        with pytest.raises(ValueError, match="identity"):
            peer.exchange(request)
        assert left.fileno() == -1
    finally:
        peer.close()
        right.close()


def test_pool_rejects_wrong_identity_and_missing_owner_before_spawn():
    placement, limits, identity, backends, _ = smoke_fixture()
    with pytest.raises(ValueError, match="placement"):
        LocalWorkerPool(placement, limits, replace(identity, placement_sha256="0" * 64), backends)
    with pytest.raises(ValueError, match="backend"):
        LocalWorkerPool(placement, limits, identity, {})


def test_worker_rejects_nonowned_routes_and_missing_weights(packet_context):
    placement, limits, _, request = packet_context
    worker = ExpertWorker("npu:0", placement, limits, DenseExpertBank(()))
    with pytest.raises(ValueError, match="not resident"):
        worker.execute(request)
    with pytest.raises(ValueError, match="unowned"):
        worker.execute(replace(request, routes=(Route(1, 0, 0),)))


def test_tiny_swiglu_has_known_result_and_bank_uses_global_ids():
    # gate=0 gives exactly zero; no sigmoid approximation affects this result.
    expert = DenseExpert(((0.0, 0.0),), ((1.0, 2.0),), ((3.0,), (4.0,)))
    bank = DenseExpertBank(((3, 19, expert),))
    assert bank(3, 19, (5.0, 7.0)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        DenseExpertBank(((3, 19, expert), (3, 19, expert)))


def test_worker_passes_one_complete_batch_to_backend(packet_context):
    placement, limits, _, request = packet_context
    calls = []

    class BatchBackend:
        def compute(self, packet):
            calls.append(packet)
            return ((1.0, 2.0),) * len(packet.routes)

    worker = ExpertWorker("npu:0", placement, limits, BatchBackend())
    reply = worker.execute(request)
    assert calls == [request]
    assert reply.routes == request.routes
    assert reply.outputs == ((1.0, 2.0),) * 2


def test_cli_executes_five_persistent_processes_and_matches_serial():
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard", "smoke"], capture_output=True, text=True, check=True, timeout=15
    )
    report = json.loads(result.stdout)
    assert report["matches_serial_fp32"] is True
    assert report["hardware_validated"] is False
    assert report["real_glm_weights"] is False
    assert report["steps"] == 3
    assert len(set(report["worker_pids"].values())) == 5


def test_killed_worker_fails_entire_pool_and_children_are_reaped():
    placement, limits, identity, backends, _ = smoke_fixture()
    pool = LocalWorkerPool(placement, limits, identity, backends)
    try:
        worker = pool.processes["npu:0"]
        worker.terminate()
        worker.join(1)
        with pytest.raises((EOFError, OSError)):
            pool.execute(3, ((0.0,) * 4,), ((1,),), ((1.0,),), ((0.0,) * 4,), 2.5)
        with pytest.raises(RuntimeError, match="closed"):
            pool.execute(3, ((0.0,) * 4,), ((0,),), ((1.0,),), ((0.0,) * 4,), 2.5)
        assert all(not process.is_alive() for process in pool.processes.values())
    finally:
        pool.close()
