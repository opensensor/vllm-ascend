# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dma_pack_graph import DmaPackGraphs, prepare_payloads
from tools.sidecard.dma_protocol import CAPACITY, INPUT_ROW_BYTES


def test_capture_keeps_remote_control_outside_graph_and_releases_scratch():
    replays = []

    def zeros(*args, **kwargs):
        return torch.zeros(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=zeros,
        float16=torch.float16,
        float32=torch.float32,
        int64=torch.int64,
        int32=torch.int32,
        where=torch.where,
        stack=torch.stack,
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            CUDAGraph=lambda: SimpleNamespace(replay=lambda: replays.append("replay")),
            graph=lambda graph: nullcontext(),
        ),
    )
    lookup = torch.full((288,), -1, dtype=torch.int32)
    lookup[[1, 145]] = torch.arange(2, dtype=torch.int32)
    slots = [torch.empty(CAPACITY, dtype=torch.uint8) for _ in range(2)]
    graphs = DmaPackGraphs(fake, {18: lookup, 19: lookup}, slots)
    x, ids, weights = torch.full((1, 4096), 7.0).half(), torch.arange(8)[None], torch.ones(1, 8)
    outputs = graphs.prepare(18, x, ids, weights)
    assert replays == ["replay"]
    assert outputs is graphs.graphs[18][2]
    assert all(torch.equal(a, b) for a, b in zip(graphs.graphs[18][1], (x, ids, weights), strict=True))
    with pytest.raises(ValueError, match="one decode row"):
        graphs.prepare(18, x.expand(2, -1), ids.expand(2, -1), weights.expand(2, -1))
    graphs.close()
    assert graphs.graphs == {}


@pytest.mark.parametrize(
    "selected,active",
    [
        ([1] * 8, [False, False]),
        ([2] * 8, [True, False]),
        ([146] * 8, [False, True]),
        ([1, 2, 145, 146, 0, 287, 3, 4], [True, True]),
    ],
)
def test_packing_masks_hot_weights_and_retains_original_cold_ids(selected, active):
    lookup = torch.full((288,), -1, dtype=torch.int32)
    lookup[[1, 145]] = torch.arange(2, dtype=torch.int32)
    slots = [torch.full((CAPACITY,), 0xA5, dtype=torch.uint8) for _ in range(2)]
    x, ids, weights = torch.arange(4096)[None].half(), torch.tensor([selected]), torch.ones(1, 8)
    hot_ids, hot_weights, flags = prepare_payloads(torch, lookup, slots, x, ids, weights)
    assert flags.tolist() == active
    expected = torch.where((ids == 1) | (ids == 145), 0, weights)
    for slot in slots:
        assert torch.equal(slot[:8192].view(torch.float16).reshape(1, 4096), x)
        assert torch.equal(slot[8192:8224].view(torch.int32).reshape(1, 8), ids)
        assert torch.equal(slot[8224:INPUT_ROW_BYTES].view(torch.float32).reshape(1, 8), expected)
        assert bool((slot[INPUT_ROW_BYTES:] == 0xA5).all())
    assert torch.equal(hot_weights + expected, weights)
    assert bool(((hot_ids >= 0) & (hot_ids < 2)).all())
