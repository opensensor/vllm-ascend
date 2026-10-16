# SPDX-License-Identifier: Apache-2.0
"""Capture CUDA attention/routing while keeping cold expert IPC outside graphs."""

from .dsa_decode_graph import DsaDecodeGraph
from .layer_decode_graph import CAPTURE_WARMUP_STEPS, diagnostic_owner
from .layer_math import HIDDEN, STREAMS


def prepare_routes(layer, state):
    prepared = layer.prepare_forward(*state)
    return (*prepared, *layer.route(prepared[0]))


class ColdDecodeGraphs:
    def __init__(self, layers, torch, *, dsa_graphs=False):
        self.layers = tuple(layers)
        self.diagnostic = diagnostic_owner(self.layers)
        self.graphs = {}
        self.dsa_graphs = {}
        for layer in self.layers:
            if not layer.linear:
                if dsa_graphs:
                    self.dsa_graphs[layer.index] = DsaDecodeGraph(layer, torch, cold=True)
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
            key = f"cold-layer:{layer.index}"
            if self.diagnostic is not None:
                self.diagnostic.capture_begin(key)
            try:
                with torch.cuda.graph(graph):
                    outputs = prepare_routes(layer, inputs)
            finally:
                if self.diagnostic is not None:
                    self.diagnostic.capture_end(key)
            torch.cuda.synchronize()
            self.graphs[layer.index] = (graph, inputs, outputs)
        for layer in self.layers:
            layer.reset()
        torch.cuda.synchronize()

    def forward(self, state):
        for layer in self.layers:
            captured = self.graphs.get(layer.index)
            if captured is None or state[0].shape[0] != 1:
                dsa = self.dsa_graphs.get(layer.index) if state[0].shape[0] == 1 else None
                state = layer.finish_forward(*dsa.forward(state)) if dsa is not None else layer.forward(*state)
            else:
                graph, inputs, outputs = captured
                for target, source in zip(inputs, state, strict=True):
                    target.copy_(source)
                graph.replay()
                diagnostic = getattr(self, "diagnostic", None)
                if diagnostic is not None:
                    diagnostic.replayed(f"cold-layer:{layer.index}")
                # Socket calls and both native cold experts execute here,
                # after capture; GPU hot/shared work overlaps those calls.
                state = layer.finish_forward(*outputs)
        return state

    def close(self):
        self.graphs.clear()
        self.dsa_graphs.clear()
