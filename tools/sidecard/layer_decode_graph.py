# SPDX-License-Identifier: Apache-2.0
"""Owner-local replay of contiguous KDA spans for one-token steps.

DSA cache lengths and sparse selection stay eager. Each actor owns its graph
buffers, recurrent caches and the captured TP/EP collectives on its stream.
"""

from .layer_math import HIDDEN, STREAMS

CAPTURE_WARMUP_STEPS = 2


def linear_spans(layers):
    spans = []
    pending = []
    for layer in layers:
        if not layer.linear:
            if pending:
                spans.append(tuple(pending))
                pending = []
            spans.append((layer,))
        else:
            pending.append(layer)
    if pending:
        spans.append(tuple(pending))
    return tuple(spans)


def forward_span(layers, state):
    for layer in layers:
        state = layer.forward(*state)
    return state


class DecodeGraphs:
    def __init__(self, layers, torch, device, *, dsa_graphs=False):
        self.spans = linear_spans(layers)
        self.graphs = {}
        self.dsa_graphs = {}
        backend = torch.cuda if device == "cuda" else torch.npu
        graph_type = backend.CUDAGraph if device == "cuda" else backend.NPUGraph
        for span in self.spans:
            if not span[0].linear:
                if dsa_graphs:
                    # Load the CUDA-only helper after graph constants are initialized.
                    from .dsa_decode_graph import DsaDecodeGraph

                    self.dsa_graphs[span[0].index] = DsaDecodeGraph(span[0], torch, cold=False)
                continue
            inputs = (
                torch.zeros((1, HIDDEN), dtype=torch.float16, device=device),
                torch.zeros((1, STREAMS, HIDDEN), dtype=torch.float32, device=device),
                torch.ones((1, STREAMS, 1), dtype=torch.float32, device=device),
                torch.eye(STREAMS, dtype=torch.float32, device=device)[None].contiguous(),
            )
            if span[0].index == 0:
                inputs = (inputs[0], None, None, None)
            for _ in range(CAPTURE_WARMUP_STEPS):
                forward_span(span, inputs)
            backend.synchronize()
            graph = graph_type()
            with backend.graph(graph):
                outputs = forward_span(span, inputs)
            backend.synchronize()
            self.graphs[span[0].index] = (graph, inputs, outputs)
        for layer in layers:
            layer.reset()
        backend.synchronize()

    def forward(self, state):
        if state[0].shape[0] != 1:
            for span in self.spans:
                state = forward_span(span, state)
            return state
        for span in self.spans:
            captured = self.graphs.get(span[0].index)
            if captured is None:
                dsa = getattr(self, "dsa_graphs", {}).get(span[0].index)
                state = dsa.forward(state) if dsa is not None else forward_span(span, state)
            else:
                graph, inputs, outputs = captured
                for target, source in zip(inputs, state, strict=True):
                    if target is None:
                        if source is not None:
                            raise ValueError("initial-layer graph requires empty deferred state")
                    else:
                        target.copy_(source)
                graph.replay()
                state = outputs
        return state

    def close(self):
        self.graphs.clear()
        self.dsa_graphs.clear()
