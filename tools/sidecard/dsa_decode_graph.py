# SPDX-License-Identifier: Apache-2.0
"""Capture fixed-shape DSA math around eager cache and sparse attention updates."""

from .layer_decode_graph import CAPTURE_WARMUP_STEPS
from .layer_math import HIDDEN, STREAMS


class DsaDecodeGraph:
    def __init__(self, layer, torch, *, cold):
        if layer.linear or layer.device != "cuda" or layer.tp_rank is not None:
            raise ValueError("DSA capture requires an unsharded CUDA DSA layer")
        self.layer, self.cold = layer, cold
        self.diagnostic = getattr(layer, "execution_diagnostic", None)
        inputs = (
            torch.zeros((1, HIDDEN), dtype=torch.float16, device="cuda"),
            torch.zeros((1, STREAMS, HIDDEN), dtype=torch.float32, device="cuda"),
            torch.ones((1, STREAMS, 1), dtype=torch.float32, device="cuda"),
            torch.eye(STREAMS, dtype=torch.float32, device="cuda")[None].contiguous(),
        )
        self.before_key = f"cuda-dsa:{layer.index}:before" if self.diagnostic is not None else None
        self.after_key = f"cuda-dsa:{layer.index}:after" if self.diagnostic is not None else None
        self.before = self.capture(torch, self.prepare, inputs, diagnostic=self.diagnostic, key=self.before_key)
        inputs = (
            torch.zeros(
                (1, layer.config["num_attention_heads"], layer.config["kv_lora_rank"]),
                dtype=torch.float16,
                device="cuda",
            ),
            *inputs[1:],
        )
        self.after = self.capture(torch, self.finish, inputs, diagnostic=self.diagnostic, key=self.after_key)

    @staticmethod
    def capture(torch, operation, inputs, *, diagnostic=None, key=None):
        for _ in range(CAPTURE_WARMUP_STEPS):
            operation(inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        if diagnostic is not None:
            diagnostic.capture_begin(key)
        try:
            with torch.cuda.graph(graph):
                outputs = operation(inputs)
        finally:
            if diagnostic is not None:
                diagnostic.capture_end(key)
        torch.cuda.synchronize()
        return graph, inputs, outputs

    def prepare(self, inputs):
        prepared = self.layer.prepare_attention(*inputs)
        return (*prepared, *self.layer.dsa_prepare(prepared[0]))

    def finish(self, inputs):
        latent, residual, post, comb = inputs
        prepared = self.layer.finish_attention(self.layer.dsa_project(latent), residual, post, comb)
        if self.cold:
            return (*prepared, *self.layer.route(prepared[0]))
        return self.layer.finish_forward(*prepared)

    @staticmethod
    def replay(captured, sources):
        graph, inputs, outputs = captured
        for target, source in zip(inputs, sources, strict=True):
            target.copy_(source)
        graph.replay()
        return outputs

    def forward(self, state):
        if state[0].shape[0] != 1:
            raise ValueError("DSA graph requires one decode row")
        x, residual, post, comb, qlatent, qr, kv = self.replay(self.before, state)
        diagnostic = getattr(self, "diagnostic", None)
        if diagnostic is not None:
            diagnostic.replayed(self.before_key)
        # Cache append, sparse indexer, causal masking and length mutation stay eager.
        latent = self.layer.dsa_attend(x, qlatent, qr, kv)
        output = self.replay(self.after, (latent, residual, post, comb))
        if diagnostic is not None:
            diagnostic.replayed(self.after_key)
        return output
