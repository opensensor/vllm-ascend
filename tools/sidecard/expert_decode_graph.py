# SPDX-License-Identifier: Apache-2.0
"""Worker-local Ascend replay of fixed-shape cold expert execution."""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_weights import MAX_CHUNK, TOP_K, WIDTH, prefill_chunk_size


def prefill_graph_rows(spec):
    options = spec.get("direct_dma", {})
    enabled = options.get("npu_prefill_graphs", False)
    if type(enabled) is not bool:
        raise ValueError("NPU prefill graph selection must be boolean")
    if not enabled:
        return None
    if not options.get("npu_expert_graphs", False) or prefill_chunk_size(spec) != MAX_CHUNK:
        raise ValueError("NPU prefill graphs require decode graphs and 32-row chunks")
    return MAX_CHUNK


class ExpertDecodeGraphs:
    """Capture native fixed-slot reduction while keeping external DMA outside."""

    def __init__(self, actor, torch, *, prefill_rows=None):
        if prefill_rows is not None and (type(prefill_rows) is not int or prefill_rows != MAX_CHUNK):
            raise ValueError("NPU prefill capture requires exactly 32 rows")
        self.actor = actor
        self.graphs, self.prefill_graphs = {}, {}
        self.prefill_rows = prefill_rows
        for layer in actor.banks:
            self.graphs[layer] = self.capture(actor, torch, layer, 1)
            if prefill_rows is not None:
                self.prefill_graphs[layer] = self.capture(actor, torch, layer, prefill_rows)

    @staticmethod
    def capture(actor, torch, layer, rows):
        device = f"npu:{actor.rank}"
        inputs = (
            torch.zeros((rows, WIDTH), dtype=torch.float16, device=device),
            torch.full((rows, TOP_K), actor.rank * 144, dtype=torch.int32, device=device),
            torch.zeros((rows, TOP_K), dtype=torch.float32, device=device),
        )
        for _ in range(CAPTURE_WARMUP_STEPS):
            actor(layer, *inputs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = actor(layer, *inputs)
        torch.npu.synchronize()
        return graph, inputs, output

    def __call__(self, layer, x, ids, weights):
        rows = x.shape[0]
        if rows == 1:
            graph, inputs, output = self.graphs[layer]
        elif rows == self.prefill_rows:
            graph, inputs, output = self.prefill_graphs[layer]
        else:
            return self.actor(layer, x, ids, weights)
        for target, source in zip(inputs, (x, ids, weights), strict=True):
            target.copy_(source)
        graph.replay()
        return output

    def close(self):
        self.graphs.clear()
        self.prefill_graphs.clear()
