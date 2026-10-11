# SPDX-License-Identifier: Apache-2.0
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ut.sidecard.test_wire_runtime import wire_fixture
from tools.sidecard.reference import ExpertReply, ExpertRequest, Route
from tools.sidecard.resident import CudaResidentBackend
from tools.sidecard.tensor_wire import TensorPacket, decode_tensor, encode_tensor
from tools.sidecard.wire import decode_packet


def test_weighted_request_and_partial_reply_are_explicitly_versioned():
    _, limits, identity, old = wire_fixture()
    request = ExpertRequest(old.execution_id, old.layer_id, old.owner, old.routes, old.token_ids, ())
    packet = TensorPacket(
        request, struct.pack("<2e", 0.1, -0.0), partial_reduce=True, route_weights=struct.pack("<2f", 0.25, 0.5)
    )
    frame = encode_tensor(packet, identity, limits)
    assert frame[:8] == b"SCARD003"
    assert decode_tensor(frame, identity, limits, partial_reduce=True) == packet
    with pytest.raises(ValueError):
        decode_packet(frame, identity, limits)
    with pytest.raises(ValueError):
        decode_tensor(frame, identity, limits)
    reply = TensorPacket(
        ExpertReply(old.execution_id, old.layer_id, old.owner, old.routes, ()),
        struct.pack("<2f", 0.25, -0.5),
        partial_reduce=True,
    )
    frame = encode_tensor(reply, identity, limits)
    assert decode_tensor(frame, identity, limits, partial_reduce=True) == reply
    assert len(reply.tensor) == len(old.token_ids) * limits.hidden_width * 4
    broken = bytearray(frame)
    broken[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_tensor(bytes(broken), identity, limits, partial_reduce=True)


@pytest.mark.parametrize("weights", [b"", struct.pack("<2f", float("nan"), 1), struct.pack("<2f", -1, 1), bytearray(8)])
def test_partial_request_rejects_invalid_weights(weights):
    _, limits, identity, old = wire_fixture()
    request = ExpertRequest(old.execution_id, old.layer_id, old.owner, old.routes, old.token_ids, ())
    with pytest.raises(ValueError):
        encode_tensor(
            TensorPacket(request, struct.pack("<2e", 1, 1), partial_reduce=True, route_weights=weights),
            identity,
            limits,
        )


def test_partial_request_maps_original_tokens_slots_and_sorted_experts():
    backend = CudaResidentBackend.__new__(CudaResidentBackend)
    backend.plan = SimpleNamespace(by_layer={3: (0, 71)}, manifest={"activation_bits": 4})
    backend.specification = {"partial_reduce": True}
    backend.metrics = dict(requests=0, routes=0, compute_total_ms=0.0, compute_max_ms=0.0)
    calls = []
    output = np.array([[0.25] * 4096, [0.5] * 4096], dtype="<f4").tobytes()

    def partial(layer, body, experts, bits, row_map, weights, rows):
        calls.append(
            (
                layer,
                body,
                experts,
                bits,
                np.frombuffer(row_map, dtype="<i4").reshape(rows, 8),
                np.frombuffer(weights, dtype="<f4").reshape(rows, 8),
            )
        )
        return output

    backend.cuda = SimpleNamespace(compute_partial_bytes=partial)
    control = ExpertRequest("partial", 3, "cuda:0", (Route(4, 2, 71), Route(1, 7, 0), Route(4, 0, 0)), (4, 1), ())
    hidden = np.array([[0.5] * 4096, [-1] * 4096], dtype="<f2")
    packet = TensorPacket(
        control, hidden.tobytes(), partial_reduce=True, route_weights=struct.pack("<3f", 0.125, 0.25, 0.5)
    )
    reply = backend.compute_tensor(packet)
    layer, body, experts, bits, row_map, weights = calls[0]
    assert (layer, experts, bits) == (3, (0, 0, 1), 4)
    assert body == hidden[[1, 0, 0]].tobytes()
    expected = np.full((2, 8), -1, dtype=np.int32)
    expected[0, 7] = 0
    expected[1, 0] = 1
    expected[1, 2] = 2
    assert np.array_equal(row_map, expected)
    expected = np.zeros((2, 8), dtype=np.float32)
    expected[0, 7] = 0.25
    expected[1, 0] = 0.5
    expected[1, 2] = 0.125
    assert np.array_equal(weights, expected)
    assert reply.control.routes == control.routes
    assert reply.partial_reduce and reply.tensor == output
