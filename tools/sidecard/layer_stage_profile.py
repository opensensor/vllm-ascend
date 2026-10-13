# SPDX-License-Identifier: Apache-2.0
"""Opt-in synchronized attribution of the real Ascend middle stage.

This diagnostic deliberately serializes component completion. Its timings are
not production latency or a prediction of an expert-only architecture. Routing
records retain every selected expert; cold experts cannot be omitted.
"""

import json
import time
from pathlib import Path

from .layer_math import TOP_K, mhc_post, mhc_pre


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
        t, w, c = layer.torch, layer.weights, layer.config
        hidden, residual, post, comb = state
        if post is None or residual is None or comb is None:
            raise ValueError("profile requires complete deferred mHC state")
        record = dict(layer=layer.index, attention="kda" if layer.linear else "dsa", components_ms={})
        timing = record["components_ms"]
        residual = self.measure(
            timing, "deferred_mhc_post", lambda: mhc_post(hidden, residual, post, comb, t, layer.native_round)
        )
        post, comb, x = self.measure(
            timing,
            "attention_mhc_pre",
            lambda: mhc_pre(
                residual, w, "attn", "input_layernorm.weight", c, t, layer.native_sinkhorn, layer.native_round
            ),
        )
        x = self.measure(timing, "attention", lambda: layer.kda(x) if layer.linear else layer.dsa(x))

        def reduce_attention():
            value = x.float()
            t.distributed.all_reduce(value)
            return value.to(t.float16)

        if layer.tp_rank is not None:
            x = self.measure(timing, "attention_sum", reduce_attention)
        residual = self.measure(
            timing, "attention_mhc_post", lambda: mhc_post(x, residual, post, comb, t, layer.native_round)
        )
        post, comb, x = self.measure(
            timing,
            "ffn_mhc_pre",
            lambda: mhc_pre(
                residual, w, "ffn", "post_attention_layernorm.weight", c, t, layer.native_sinkhorn, layer.native_round
            ),
        )

        def route():
            gate = w["mlp.gate.weight"]
            logits = (
                (x.float()[:, None, :] * gate[None, :, :]).sum(-1)
                if layer.device.startswith("npu")
                else x.float() @ gate.t()
            )
            scores = t.sigmoid(logits)
            ids = t.topk(scores + w["mlp.gate.e_score_correction_bias"], TOP_K, -1).indices
            weights = scores.gather(-1, ids)
            weights = weights / weights.sum(-1, keepdim=True) * c["routed_scaling_factor"]
            return ids, weights

        ids, weights = self.measure(timing, "router", route)
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
