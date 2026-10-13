# SPDX-License-Identifier: Apache-2.0
"""Unpadded Ascend replay only for one validated owned contribution."""

from .expert_decode_graph import ExpertDecodeGraphs
from .layer_weights import TOP_K


def single_route_enabled(spec):
    options = spec.get("direct_dma", {})
    enabled = options.get("npu_single_route_graphs", False)
    if type(enabled) is not bool or (enabled and not options.get("npu_expert_graphs", False)):
        raise ValueError("single-route replay requires boolean selection and qualified NPU decode graphs")
    return enabled


class SingleRouteDecodeGraphs(ExpertDecodeGraphs):
    """Keep fixed-slot graphs for every multi-contribution reduction."""

    def __init__(self, actor, torch, *, prefill_rows=None):
        super().__init__(actor, torch, prefill_rows=prefill_rows)
        self.single_route_graphs = {}
        original = actor.stable_rows
        try:
            actor.stable_rows = 1
            for layer in actor.banks:
                self.single_route_graphs[layer] = self.capture(actor, torch, layer, 1)
        finally:
            actor.stable_rows = original

    def __call__(self, layer, x, ids, weights, *, owned_routes=None):
        # This count comes from already-authenticated CPU routing metadata.
        # Tensor values must never be read back just to choose a graph.
        if owned_routes is not None and (type(owned_routes) is not int or not 0 <= owned_routes <= x.shape[0] * TOP_K):
            raise ValueError("single-route selection requires a validated owned route count")
        if x.shape[0] != 1 or owned_routes != 1:
            return super().__call__(layer, x, ids, weights)
        graph, inputs, output = self.single_route_graphs[layer]
        for target, source in zip(inputs, (x, ids, weights), strict=True):
            target.copy_(source)
        graph.replay()
        return output

    def close(self):
        self.single_route_graphs.clear()
        super().close()
