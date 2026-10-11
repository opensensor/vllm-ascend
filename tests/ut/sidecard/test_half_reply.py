# SPDX-License-Identifier: Apache-2.0
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ut.sidecard.test_wire_runtime import wire_fixture
from tools.sidecard.ordered_tensor import combine_compact
from tools.sidecard.readiness import receive_ready
from tools.sidecard.reference import ExpertReply, ExpertRequest, Route
from tools.sidecard.tensor_wire import TensorPacket, TensorSocketPeer, decode_tensor, encode_tensor, serve_tensor_worker
from tools.sidecard.wire import decode_packet


def test_half_reply_requires_explicit_precision_and_preserves_half_values():
    _, limits, identity, request = wire_fixture()
    reply = ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, ())
    values = np.array([[0.1, -0.0], [65504, -65504]], dtype="<f2")
    packet = TensorPacket(reply, values.tobytes(), reply_fp16=True)
    frame = encode_tensor(packet, identity, limits)
    assert frame[:8] == b"SCARD002"
    assert decode_tensor(frame, identity, limits, reply_fp16=True) == packet
    with pytest.raises(ValueError):
        decode_packet(frame, identity, limits)
    with pytest.raises(ValueError):
        decode_tensor(frame, identity, limits)
    legacy = TensorPacket(reply, values.astype("<f4").tobytes())
    old = encode_tensor(legacy, identity, limits)
    assert len(old) - len(frame) == values.size * 2
    with pytest.raises(ValueError, match="precision"):
        decode_tensor(old, identity, limits, reply_fp16=True)
    broken = bytearray(frame)
    broken[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_tensor(bytes(broken), identity, limits, reply_fp16=True)


def test_half_reply_rejects_nonfinite_and_mislabelled_requests():
    _, limits, identity, request = wire_fixture()
    reply = ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, ())
    with pytest.raises(ValueError, match="nonfinite"):
        encode_tensor(
            TensorPacket(reply, np.array([[float("inf"), 0], [0, 0]], dtype="<f2").tobytes(), True), identity, limits
        )
    with pytest.raises(ValueError, match="request"):
        encode_tensor(TensorPacket(request, b"1234", True), identity, limits)


@pytest.mark.parametrize("rows", [1, 32, 640])
def test_half_reply_combination_matches_expanded_fp32_bitwise(rows):
    generator = np.random.default_rng(53)
    routes = tuple(Route(token, slot, slot) for token in range(rows) for slot in (0, 3, 7))
    outputs = generator.normal(size=(len(routes), 32)).astype("<f2")
    weights = generator.random((rows, 8), dtype=np.float32)
    expected = combine_compact(routes, outputs.astype("<f4").tobytes(), weights, 32)
    actual = combine_compact(routes, outputs.tobytes(), weights, 32, reply_fp16=True)
    assert actual.tobytes() == expected.tobytes()


def test_half_reply_real_socket_round_trip():
    placement, limits, identity, oldrequest = wire_fixture()
    request = ExpertRequest(
        oldrequest.execution_id, oldrequest.layer_id, oldrequest.owner, oldrequest.routes, oldrequest.token_ids, ()
    )

    def compute(packet):
        r = packet.control
        return TensorPacket(
            ExpertReply(r.execution_id, r.layer_id, r.owner, r.routes, ()),
            np.array([[0.25, -0.0], [0.5, 1]], dtype="<f2").tobytes(),
            True,
        )

    worker = SimpleNamespace(
        owner=request.owner, placement=placement, limits=limits, backend=SimpleNamespace(compute_tensor=compute)
    )
    client, server = socket.socketpair()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve_tensor_worker, server, worker, identity)
        receive_ready(client, request.owner, identity, time.monotonic() + 5)
        peer = TensorSocketPeer(client, identity, limits, 5, reply_fp16=True)
        result = peer.exchange(TensorPacket(request, struct.pack("<2e", 0.1, -0.0)))
        assert result.reply_fp16
        assert result.control.routes == request.routes
        peer.close()
        assert future.result(timeout=5)
