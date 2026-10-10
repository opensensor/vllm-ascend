# SPDX-License-Identifier: Apache-2.0
"""Queued raw parity/changed-input graph gates; never executed by default."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.qwen38_1m.test_streaming_projection import fixture


def inputs(rows, outputs, k, ends, seed):
    arguments, expected, _ = fixture(rows, outputs, k, ends, seed)
    values = [torch.from_numpy(np.ascontiguousarray(value)).npu() for value in arguments[:9]]
    experts = len(ends)
    bank = SimpleNamespace(weight=values[4].reshape(experts, outputs, k // 2))
    for name, value in zip(("weight_scale", "weight_offset", "weight_sum"), values[5:8]):
        setattr(bank, name, value.reshape(experts, outputs, k // 128))
    return bank, tuple(values[:4]), values[8], expected


@pytest.mark.parametrize(
    "rows,outputs,k,ends",
    [(1, 160, 128, [1]), (11, 1280, 2560, [0, 11]), (33, 160, 640, [0] * 84 + [33]), (33, 160, 640, [0] * 85 + [17])],
)
def test_raw_parity_and_peer_zero(projection, rows, outputs, k, ends):
    for seed in (89, 91):
        bank, prepared, boundaries, expected = inputs(rows, outputs, k, ends, seed)
        output = projection(bank, prepared, boundaries)
        np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.parametrize("first", [0, 8])
def test_down_windows(projection, first):
    bank, prepared, ends, expected = inputs(33, 2560, 640, [0, 33], 89)
    output = projection.columns(bank, prepared, ends, first, 8)
    np.testing.assert_array_equal(output.cpu().numpy(), expected[:, first * 160 : (first + 8) * 160])


def test_changed_inputs_graph_replays_current_operand_metadata_weights_and_ends(projection):
    bank, prepared, ends, _ = inputs(33, 160, 640, [0, 33], 89)
    projection(bank, prepared, ends)  # Prewarm configuration addresses before capture.
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = projection(bank, prepared, ends)
    targets = (*prepared, bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum, ends)
    for seed, active in ((91, 33), (93, 0), (95, 17), (97, 33)):
        current, packed, boundaries, expected = inputs(33, 160, 640, [0, active], seed)
        sources = (*packed, current.weight, current.weight_scale, current.weight_offset, current.weight_sum, boundaries)
        for target, source in zip(targets, sources):
            target.copy_(source)
        graph.replay()
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
