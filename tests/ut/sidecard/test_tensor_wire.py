# SPDX-License-Identifier: Apache-2.0
"""Bulk tensor framing must retain every scalar protocol safety check."""

import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ut.sidecard.test_wire_runtime import changed_metadata
from tests.ut.sidecard.test_wire_runtime import packet_context as packet_context
from tools.sidecard.readiness import receive_ready
from tools.sidecard.reference import ExpertReply, ExpertRequest, Route
from tools.sidecard.resident import CudaResidentBackend
from tools.sidecard.runtime import serve_worker
from tools.sidecard.tensor_wire import TensorPacket, TensorSocketPeer, decode_tensor, encode_tensor
from tools.sidecard.wire import decode_packet, encode_packet
from tools.sidecard.worker import DenseExpert, DenseExpertBank, ExpertWorker


@pytest.mark.parametrize("reply", [False, True])
def test_tensor_frames_are_identical_to_reference_wire_bytes(packet_context, reply):
    _, limits, identity, request = packet_context
    packet = (
        ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, ((0.1, -0.0), (2.0, 3.0)))
        if reply
        else request
    )
    rows = packet.outputs if reply else packet.hidden
    code = "f" if reply else "e"
    body = b"".join(struct.pack("<" + code * limits.hidden_width, *row) for row in rows)
    control = replace(packet, **({"outputs": ()} if reply else {"hidden": ()}))
    tensor = TensorPacket(control, body)
    frame = encode_tensor(tensor, identity, limits)
    assert frame == encode_packet(packet, identity, limits)
    assert decode_tensor(frame, identity, limits) == tensor
    assert decode_packet(frame, identity, limits).routes == packet.routes


@pytest.mark.parametrize(
    "change",
    [
        dict(rows=True),
        dict(width=3),
        dict(owner=""),
        dict(checkpoint_sha256="b" * 64),
        dict(execution_id=""),
        dict(kind="unknown"),
    ],
)
def test_raw_path_retains_metadata_and_identity_rejections(packet_context, change):
    _, limits, identity, request = packet_context
    frame = changed_metadata(encode_packet(request, identity, limits), **change)
    with pytest.raises(ValueError):
        decode_tensor(frame, identity, limits)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_payload_is_rejected_before_transmission(packet_context, value):
    _, limits, identity, request = packet_context
    packet = TensorPacket(replace(request, hidden=()), struct.pack("<2e", value, 0))
    with pytest.raises(ValueError, match="nonfinite"):
        encode_tensor(packet, identity, limits)


def test_tensor_peer_can_exchange_with_reference_worker(packet_context):
    placement, limits, identity, request = packet_context
    expert = DenseExpert(((1.0, 0.5),), ((-0.25, 1.0),), ((0.75,), (-1.0,)))
    bank = DenseExpertBank(((3, 1, expert), (3, 2, expert)))
    worker = ExpertWorker("npu:0", placement, limits, bank)
    left, right = socket.socketpair()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve_worker, right, worker, identity)
        receive_ready(left, "npu:0", identity, time.monotonic() + 5)
        peer = TensorSocketPeer(left, identity, limits, 5)
        frame = encode_packet(request, identity, limits)
        raw_request = decode_tensor(frame, identity, limits)
        reply = peer.exchange(raw_request)
        actual = decode_packet(encode_tensor(reply, identity, limits), identity, limits)
        expected = worker.execute(decode_packet(frame, identity, limits))
        assert actual == expected
        peer.close()
        assert future.result(timeout=5)


def test_resident_raw_routes_gather_by_token_map_and_arbitrary_global_experts():
    backend = CudaResidentBackend.__new__(CudaResidentBackend)
    backend.plan = SimpleNamespace(by_layer={3: (0, 71)}, manifest={"activation_bits": 4})
    backend.metrics = {"requests": 0, "routes": 0, "compute_total_ms": 0.0, "compute_max_ms": 0.0}
    received = []
    output = np.asarray([[0.25] * 4096, [-0.0] * 4096, [0.75] * 4096], dtype="<f2").tobytes()

    def compute_bytes(layer, body, experts, bits):
        received.append((layer, body, experts, bits))
        return output

    backend.cuda = SimpleNamespace(compute_bytes=compute_bytes)
    control = ExpertRequest("raw", 3, "cuda:0", (Route(4, 2, 71), Route(1, 7, 0), Route(4, 0, 0)), (4, 1), ())
    hidden = np.asarray([[0.5] * 4096, [-1.0] * 4096], dtype="<f2")
    actual = backend.compute_tensor(TensorPacket(control, hidden.tobytes()))
    assert received == [(3, hidden[[0, 1, 0]].tobytes(), (1, 0, 0), 4)]
    assert actual.control.routes == control.routes
    assert actual.tensor == np.frombuffer(output, dtype="<f2").astype("<f4").tobytes()
    assert backend.metrics["requests"] == 1
    assert backend.metrics["routes"] == 3
