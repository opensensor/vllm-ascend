# SPDX-License-Identifier: Apache-2.0
"""Capture CUDA attention/routing while keeping cold expert IPC outside graphs."""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_math import HIDDEN, STREAMS


def prepare_routes(layer, state):
    prepared = layer.prepare_forward(*state)
    return (*prepared, *layer.route(prepared[0]))


class ColdDecodeGraphs:
    def __init__(self, layers, torch):
        self.layers = tuple(layers)
        self.graphs = {}
        for layer in self.layers:
            if not layer.linear:
                continue
            inputs = (
                torch.zeros((1, HIDDEN), dtype=torch.float16, device="cuda"),
                torch.zeros((1, STREAMS, HIDDEN), dtype=torch.float32, device="cuda"),
                torch.ones((1, STREAMS, 1), dtype=torch.float32, device="cuda"),
                torch.eye(STREAMS, dtype=torch.float32, device="cuda")[None].contiguous(),
            )
            for _ in range(CAPTURE_WARMUP_STEPS):
                prepare_routes(layer, inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = prepare_routes(layer, inputs)
            torch.cuda.synchronize()
            self.graphs[layer.index] = (graph, inputs, outputs)
        for layer in self.layers:
            layer.reset()
        torch.cuda.synchronize()

    def forward(self, state):
        for layer in self.layers:
            captured = self.graphs.get(layer.index)
            if captured is None or state[0].shape[0] != 1:
                state = layer.forward(*state)
            else:
                graph, inputs, outputs = captured
                for target, source in zip(inputs, state, strict=True):
                    target.copy_(source)
                graph.replay()
                # Socket calls and both native cold experts execute here,
                # after capture; GPU hot/shared work overlaps those calls.
                state = layer.finish_forward(*outputs)
        return state
