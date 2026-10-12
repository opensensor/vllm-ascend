# SPDX-License-Identifier: Apache-2.0
"""Moved experts retain their original TP input, even for shared token IDs."""

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ut.sidecard.test_wire_runtime import wire_fixture
from tools.sidecard.device_backend import CUDA_COMPACT_PLANES
from tools.sidecard.one_card_partition import PRECISION_PARTITION
from tools.sidecard.readiness import receive_ready
from tools.sidecard.reference import ExpertReply, ExpertRequest, Route
from tools.sidecard.resident import CudaResidentBackend
from tools.sidecard.service_prepare import prepare
from tools.sidecard.tensor_wire import TensorPacket, TensorSocketPeer, decode_tensor, encode_tensor, serve_tensor_worker
from tools.sidecard.wire import PREFIX, TP_INPUT_MAGIC, frame_lengths


def tp_fixture():
    placement, limits, identity, request = wire_fixture()
    packet = TensorPacket(
        replace(request, hidden=()),
        np.asarray([[1, 2], [1.001, 2.002]], dtype="<f2").tobytes(),
        partial_reduce=True,
        route_weights=np.asarray([0.25, 0.75], dtype="<f4").tobytes(),
        tp_rank_inputs=True,
    )
    return placement, limits, identity, packet


def test_two_inputs_round_trip_with_explicit_protocol_admission():
    _, limits, identity, packet = tp_fixture()
    frame = encode_tensor(packet, identity, limits)
    assert PREFIX.unpack(frame[: PREFIX.size])[0] == TP_INPUT_MAGIC
    assert decode_tensor(frame, identity, limits, partial_reduce=True, tp_rank_inputs=True) == packet
    with pytest.raises(ValueError, match="unsupported frame"):
        decode_tensor(frame, identity, limits, partial_reduce=True)


@pytest.mark.parametrize("change", (dict(tensor_transport=False), dict(partial_reduce=False), dict(reply_fp16=True)))
def test_precision_preparation_rejects_transport_that_cannot_preserve_tp_inputs(change):
    options = dict(
        one_card=True,
        cuda_partition=PRECISION_PARTITION,
        weight_layout=CUDA_COMPACT_PLANES,
        tensor_transport=True,
        partial_reduce=True,
        reply_fp16=False,
    )
    options.update(change)
    with pytest.raises(ValueError, match="TP inputs"):
        prepare(None, None, None, None, None, None, None, **options)


@pytest.mark.parametrize(
    "change", (dict(tensor=b""), dict(tp_rank_inputs=1), dict(partial_reduce=False), dict(reply_fp16=True))
)
def test_two_inputs_reject_invalid_geometry_and_modes(change):
    _, limits, identity, packet = tp_fixture()
    with pytest.raises(ValueError):
        encode_tensor(replace(packet, **change), identity, limits)


def test_two_input_allocation_is_bounded_before_receiving_payload():
    _, limits, _, _ = tp_fixture()
    maximum = limits.max_tokens * (limits.hidden_width * 4 + 4) + limits.max_routes * 16
    with pytest.raises(ValueError, match="oversized allocation"):
        frame_lengths(
            PREFIX.pack(TP_INPUT_MAGIC, 1, maximum + 1), limits, allow_partial_reduce=True, allow_tp_rank_inputs=True
        )


def test_resident_selects_rank_per_expert_with_arbitrary_token_order():
    backend = CudaResidentBackend.__new__(CudaResidentBackend)
    backend.plan = SimpleNamespace(by_layer={3: (0, 144, 216, 231)}, manifest={"activation_bits": 4})
    backend.specification = {"tp_rank_inputs": True}
    backend.metrics = {"requests": 0, "routes": 0, "compute_total_ms": 0.0, "compute_max_ms": 0.0}
    received = []

    def compute_partial(layer, body, experts, bits, row_map, weights, tokens):
        received.append((np.frombuffer(body, dtype="<f2").reshape(-1, 4096), experts))
        return np.zeros((tokens, 4096), dtype="<f4").tobytes()

    backend.cuda = SimpleNamespace(compute_partial_bytes=compute_partial)
    routes = (Route(4, 0, 231), Route(1, 0, 0), Route(4, 1, 144), Route(1, 1, 216))
    control = ExpertRequest("rank-inputs", 3, "cuda:0", routes, (4, 1), ())
    inputs = np.stack([np.full((2, 4096), [1, 2][rank], dtype="<f2") for rank in range(2)])
    inputs[0, 1] = 3
    inputs[1, 1] = 4
    packet = TensorPacket(
        control,
        inputs.tobytes(),
        partial_reduce=True,
        route_weights=np.ones(4, dtype="<f4").tobytes(),
        tp_rank_inputs=True,
    )
    reply = backend.compute_tensor(packet)
    assert received[0][1] == (0, 1, 2, 3)
    assert received[0][0][:, 0].tolist() == [3, 1, 4, 2]
    assert reply.tp_rank_inputs is False
    with pytest.raises(ValueError, match="TP input mode"):
        backend.compute_tensor(replace(packet, tp_rank_inputs=False))


def test_two_input_socket_request_keeps_single_input_fp32_reply():
    placement, limits, identity, packet = tp_fixture()
    output = np.asarray([[3, 4]], dtype="<f4").tobytes()

    def compute(value):
        assert value == packet
        q = value.control
        return TensorPacket(ExpertReply(q.execution_id, q.layer_id, q.owner, q.routes, ()), output, partial_reduce=True)

    worker = SimpleNamespace(
        owner="npu:0",
        placement=placement,
        limits=limits,
        backend=SimpleNamespace(specification={"partial_reduce": True, "tp_rank_inputs": True}, compute_tensor=compute),
    )
    left, right = socket.socketpair()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(serve_tensor_worker, right, worker, identity)
        receive_ready(left, "npu:0", identity, time.monotonic() + 5)
        peer = TensorSocketPeer(left, identity, limits, 5, partial_reduce=True, tp_rank_inputs=True)
        assert peer.exchange(packet).tensor == output
        peer.close()
        assert future.result(timeout=5)
