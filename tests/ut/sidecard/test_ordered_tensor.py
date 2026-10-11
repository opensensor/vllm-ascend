# SPDX-License-Identifier: Apache-2.0
import struct

import numpy as np
import pytest

from tools.sidecard.ordered_tensor import combine_compact
from tools.sidecard.reference import Route, fp32
from tools.sidecard.service_adapter import local_routes
from tools.sidecard.service_prepare import prepare


@pytest.mark.parametrize("rows", [1, 32, 128, 640])
def test_compact_combination_preserves_fp32_slot_order(rows):
    generator = np.random.default_rng(53)
    weights = generator.random((rows, 8), dtype=np.float32)
    # Alternating large opposing values expose incorrect addition/reordering.
    outputs = generator.normal(size=(rows * 3, 4)).astype(np.float32)
    outputs[::3, 0] = 1e8
    outputs[1::3, 0] = -1e8
    routes = tuple(Route(token, slot, slot) for token in range(rows) for slot in (0, 3, 7))
    expected = np.zeros((rows, 4), dtype=np.float32)
    for index, route in enumerate(routes):
        for channel in range(4):
            product = fp32(float(outputs[index, channel]) * float(weights[route.token, route.slot]))
            expected[route.token, channel] = fp32(float(expected[route.token, channel]) + product)
    order = generator.permutation(len(routes))
    actual = combine_compact(tuple(routes[i] for i in order), outputs[order].tobytes(), weights, 4)
    assert actual.tobytes() == expected.tobytes()


def test_empty_routes_preserve_zero_output():
    result = combine_compact((), b"", [[0] * 8] * 3, 4)
    assert result.tobytes() == bytes(3 * 4 * 4)


@pytest.mark.parametrize(
    "routes,output,weights",
    [
        ((Route(0, 0, 0),), struct.pack("<f", float("nan")), [[1]]),
        ((Route(0, 0, 0),), b"", [[1]]),
        ((Route(2, 0, 0),), struct.pack("<f", 1), [[1]]),
        ((Route(0, 2, 0),), struct.pack("<f", 1), [[1]]),
        ((Route(0, 0, 0), Route(0, 0, 0)), struct.pack("<2f", 1, 2), [[1]]),
        ((Route(0, 0, 0),), struct.pack("<f", 1), [[-1]]),
    ],
)
def test_invalid_compact_combination(routes, output, weights):
    with pytest.raises(ValueError):
        combine_compact(routes, output, weights, 1)


def test_large_route_batch_is_explicitly_bounded():
    ids, weights = [[0] * 8] * 640, [[0] * 8] * 640
    assert local_routes(ids, weights, max_tokens=640) == ((), ())
    with pytest.raises(ValueError):
        local_routes(ids, weights)


@pytest.mark.parametrize("packet_tokens,tensor", [(0, True), (641, True), (True, True), (640, False)])
def test_invalid_packet_preparation_fails_before_filesystem(packet_tokens, tensor):
    with pytest.raises(ValueError):
        prepare(None, None, None, None, None, None, None, packet_tokens=packet_tokens, tensor_transport=tensor)
