# SPDX-License-Identifier: Apache-2.0
"""Capture local hot/shared experts without capturing external DMA control."""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_weights import TOP_K, WIDTH


def local_experts(cuda, layer, inputs):
    x, ids, weights = inputs
    return cuda(layer.index, x, ids, weights), layer.dense_mlp(x, "mlp.shared_experts")


class HotDecodeGraphs:
    """Use the original kernels and shared scratch on the caller's CUDA stream."""

    def __init__(self, cuda, layers, torch):
        if not cuda.use_stream:
            raise ValueError("hot expert capture requires stream-aware CUDA kernels")
        self.graphs = {}
        for layer in layers:
            inputs = (
                torch.zeros((1, WIDTH), dtype=torch.float16, device="cuda"),
                torch.zeros((1, TOP_K), dtype=torch.int32, device="cuda"),
                torch.zeros((1, TOP_K), dtype=torch.float32, device="cuda"),
            )
            for _ in range(CAPTURE_WARMUP_STEPS):
                local_experts(cuda, layer, inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = local_experts(cuda, layer, inputs)
            torch.cuda.synchronize()
            self.graphs[layer.index] = (graph, inputs, outputs)

    def compute(self, layer, x, ids, weights):
        if x.shape[0] != 1:
            raise ValueError("hot expert graph requires one decode row")
        graph, inputs, outputs = self.graphs[layer]
        for target, source in zip(inputs, (x, ids, weights), strict=True):
            target.copy_(source)
        graph.replay()
        return outputs

    def close(self):
        self.graphs.clear()
