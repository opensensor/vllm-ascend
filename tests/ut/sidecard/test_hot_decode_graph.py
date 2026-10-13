# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.hot_decode_graph import HotDecodeGraphs


def test_capture_uses_original_local_kernels_and_replay_copies_dynamic_routes():
    calls = []

    def zeros(*args, **kwargs):
        return torch.zeros(*args, **dict(kwargs, device="cpu"))

    fake = SimpleNamespace(
        zeros=zeros,
        float16=torch.float16,
        float32=torch.float32,
        int32=torch.int32,
        cuda=SimpleNamespace(
            synchronize=lambda: calls.append("sync"),
            CUDAGraph=lambda: SimpleNamespace(replay=lambda: calls.append("replay")),
            graph=lambda _: nullcontext(),
        ),
    )

    class Local:
        use_stream = True

        def __call__(self, layer, x, ids, weights):
            calls.append(("hot", layer))
            return x.float() * weights.sum()

    layer = SimpleNamespace(
        index=18,
        dense_mlp=lambda x, prefix: calls.append(prefix) or x.float(),
    )
    graphs = HotDecodeGraphs(Local(), [layer], fake)
    assert calls.count(("hot", 18)) == 3
    assert calls.count("mlp.shared_experts") == 3
    inputs = (torch.ones(1, 4096).half(), torch.arange(8)[None].int(), torch.ones(1, 8))
    assert graphs.compute(18, *inputs) is graphs.graphs[18][2]
    assert calls[-1] == "replay"
    assert all(torch.equal(a, b) for a, b in zip(graphs.graphs[18][1], inputs, strict=True))
    with pytest.raises(ValueError, match="one decode row"):
        graphs.compute(18, *(value.expand(2, -1) for value in inputs))
    graphs.close()
    assert graphs.graphs == {}


def test_capture_rejects_legacy_synchronizing_kernel_abi():
    with pytest.raises(ValueError, match="stream-aware"):
        HotDecodeGraphs(SimpleNamespace(use_stream=False), [], None)
