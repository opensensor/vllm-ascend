# SPDX-License-Identifier: Apache-2.0
"""Opt-in synchronized attribution of the real Ascend middle stage.

This diagnostic deliberately serializes component completion. Its timings are
not production latency or a prediction of an expert-only architecture. Routing
records retain every selected expert; cold experts cannot be omitted.
"""

import json
import time
from pathlib import Path


def forward_traced(layers, graphs, state, trace):
    """Observe graph boundaries without recapture or graph-internal readback.

    Match DecodeGraphs.forward, including input copies and eager DSA control.
    Owner-local events surround whole replay spans; collective decomposition
    inside those spans requires a separately qualified native profiler.
    """
    if not trace.enabled:
        if graphs is not None:
            return graphs.forward(state)
        for layer in layers:
            state = layer.forward(*state)
        return state
    if graphs is None or state[0].shape[0] != 1:
        for layer in layers:
            with trace.span("eager_layer", device=True, layer=layer.index, attribution="collectives included"):
                state = layer.forward(*state)
        return state
    for span in graphs.spans:
        captured = graphs.graphs.get(span[0].index)
        indices = [layer.index for layer in span]
        if captured is None:
            dsa = getattr(graphs, "dsa_graphs", {}).get(span[0].index)
            with trace.span("dsa_graph" if dsa else "eager_span", device=True, layers=indices):
                if dsa is not None:
                    state = dsa.forward(state)
                else:
                    for layer in span:
                        state = layer.forward(*state)
        else:
            graph, inputs, outputs = captured
            with trace.span("graph_inputs", device=True, layers=indices):
                for target, source in zip(inputs, state, strict=True):
                    if target is None:
                        if source is not None:
                            raise ValueError("initial-layer graph requires empty deferred state")
                    else:
                        target.copy_(source)
            with trace.span("kda_graph", device=True, layers=indices, attribution="internal collectives unattributed"):
                graph.replay()
            if hasattr(graphs, "observe_replay"):
                graphs.observe_replay(span[0].index)
            state = outputs
    return state


class StageProfile:
    def __init__(self, output, rank, synchronize):
        self.output = Path(output)
        self.rank = rank
        self.synchronize = synchronize
        self.packet = None

    def begin(self, sequence, position, rows):
        self.packet = dict(sequence=sequence, position=position, rows=rows, rank=self.rank, transport_ms={}, layers=[])

    def measure(self, record, name, operation):
        self.synchronize()
        start = time.perf_counter()
        result = operation()
        self.synchronize()
        record[name] = (time.perf_counter() - start) * 1000
        return result

    def forward(self, layer, state):
        """Same operations and reduction boundaries as TextLayer.forward."""
        if not 18 <= layer.index < 33:
            raise ValueError("stage attribution requires a middle MoE layer")
        t = layer.torch
        hidden, residual, post, comb = state
        if post is None or residual is None or comb is None:
            raise ValueError("profile requires complete deferred mHC state")
        record = dict(layer=layer.index, attention="kda" if layer.linear else "dsa", components_ms={})
        timing = record["components_ms"]
        residual = self.measure(timing, "deferred_mhc_post", lambda: layer.mix_post(hidden, residual, post, comb))
        post, comb, x = self.measure(
            timing,
            "attention_mhc_pre",
            lambda: layer.mix_pre(residual, "attn", "input_layernorm.weight"),
        )
        x = self.measure(timing, "attention", lambda: layer.kda(x) if layer.linear else layer.dsa(x))

        def reduce_attention():
            value = x.float()
            t.distributed.all_reduce(value)
            return value.to(t.float16)

        if layer.tp_rank is not None:
            x = self.measure(timing, "attention_sum", reduce_attention)
        residual = self.measure(timing, "attention_mhc_post", lambda: layer.mix_post(x, residual, post, comb))
        post, comb, x = self.measure(
            timing,
            "ffn_mhc_pre",
            lambda: layer.mix_pre(residual, "ffn", "post_attention_layernorm.weight"),
        )

        ids, weights = self.measure(timing, "router", lambda: layer.route(x))
        routed = self.measure(timing, "routed_experts", lambda: layer.experts(layer.index, x, ids, weights))
        shared = self.measure(timing, "shared_mlp", lambda: layer.dense_mlp(x, "mlp.shared_experts"))

        def reduce_mlp():
            combined = routed.float() + shared.float()
            if layer.tp_rank is not None:
                t.distributed.all_reduce(combined)
            return combined.to(x.dtype)

        x = self.measure(timing, "mlp_sum", reduce_mlp)
        # CPU observation is outside the timed components and diagnostic only.
        record["expert_ids"] = ids.cpu().tolist()
        record["route_weights"] = weights.cpu().tolist()
        self.packet["layers"].append(record)
        return x, residual, post, comb

    def finish(self):
        with self.output.open("a") as handle:
            handle.write(json.dumps(self.packet) + "\n")
        self.packet = None
