# SPDX-License-Identifier: Apache-2.0
"""Worker-local Ascend replay of deterministic one-row cold expert execution."""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_weights import TOP_K, WIDTH


class ExpertDecodeGraphs:
    """Capture native fixed-slot reduction while keeping external DMA outside."""

    def __init__(self, actor, torch):
        self.actor = actor
        self.graphs = {}
        device = f"npu:{actor.rank}"
        for layer in actor.banks:
            inputs = (
                torch.zeros((1, WIDTH), dtype=torch.float16, device=device),
                torch.full((1, TOP_K), actor.rank * 144, dtype=torch.int32, device=device),
                torch.zeros((1, TOP_K), dtype=torch.float32, device=device),
            )
            for _ in range(CAPTURE_WARMUP_STEPS):
                actor(layer, *inputs)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = actor(layer, *inputs)
            torch.npu.synchronize()
            self.graphs[layer] = (graph, inputs, output)

    def __call__(self, layer, x, ids, weights):
        if x.shape[0] != 1:
            return self.actor(layer, x, ids, weights)
        graph, inputs, output = self.graphs[layer]
        for target, source in zip(inputs, (x, ids, weights), strict=True):
            target.copy_(source)
        graph.replay()
        return output

    def close(self):
        self.graphs.clear()
