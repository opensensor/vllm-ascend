# SPDX-License-Identifier: Apache-2.0
"""Capture one-row routing and packing into private DMA input slots."""

from .dma_protocol import INPUT_ROW_BYTES
from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_weights import TOP_K, WIDTH


def prepare_payloads(torch, lookup, slots, x, ids, weights):
    """Keep route arithmetic on CUDA; return only bounded admission flags."""
    local_ids = lookup[ids.long()]
    owned = local_ids >= 0
    cold_weights = torch.where(owned, 0, weights).float()
    active = torch.stack(
        [((ids >= rank * 144) & (ids < (rank + 1) * 144) & (cold_weights > 0)).any() for rank in range(2)]
    )
    rows = x.shape[0]
    x_bytes, id_bytes = rows * WIDTH * 2, rows * TOP_K * 4
    for slot in slots:
        slot[:x_bytes].view(torch.float16).reshape(rows, WIDTH).copy_(x)
        slot[x_bytes : x_bytes + id_bytes].view(torch.int32).reshape(rows, TOP_K).copy_(ids)
        slot[x_bytes + id_bytes : rows * INPUT_ROW_BYTES].view(torch.float32).reshape(rows, TOP_K).copy_(cold_weights)
    return torch.where(owned, local_ids, 0), torch.where(owned, weights, 0), active


class DmaPackGraphs:
    """Own graph scratch until all imported CUDA slot mappings are released."""

    def __init__(self, torch, lookups, slots):
        self.graphs = {}
        for layer, lookup in lookups.items():
            inputs = (
                torch.zeros((1, WIDTH), dtype=torch.float16, device="cuda"),
                torch.zeros((1, TOP_K), dtype=torch.int64, device="cuda"),
                torch.zeros((1, TOP_K), dtype=torch.float32, device="cuda"),
            )
            for _ in range(CAPTURE_WARMUP_STEPS):
                prepare_payloads(torch, lookup, slots, *inputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = prepare_payloads(torch, lookup, slots, *inputs)
            torch.cuda.synchronize()
            self.graphs[layer] = (graph, inputs, outputs)

    def prepare(self, layer, x, ids, weights):
        if x.shape[0] != 1:
            raise ValueError("DMA packing graph requires one decode row")
        graph, inputs, outputs = self.graphs[layer]
        for target, source in zip(inputs, (x, ids, weights), strict=True):
            target.copy_(source)
        graph.replay()
        return outputs

    def close(self):
        self.graphs.clear()
