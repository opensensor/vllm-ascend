# SPDX-License-Identifier: Apache-2.0
"""Replay fixed-shape indexer projections while leaving sparse state eager."""

from .layer_indexer import HEAD_DIM

CAPTURE_WARMUP_STEPS = 2
LAYER_NORM_EPS = 1e-6


def enabled(spec):
    value = spec.get("cuda_indexer_projection_graphs", False)
    if type(value) is not bool or (value and not spec.get("cuda_indexer_hadamard", False)):
        raise ValueError("CUDA indexer projection graphs require a boolean and fused CUDA rotation")
    return value


def project(indexer, hidden, qr):
    t, w = indexer.torch, indexer.fp32
    xf = hidden.float()
    keys = t.nn.functional.linear(xf, w["wk.weight"])
    keys = t.nn.functional.layer_norm(keys, (HEAD_DIM,), w["k_norm.weight"], w["k_norm.bias"], LAYER_NORM_EPS)
    gates = t.nn.functional.linear(xf, w["index_kpool_compress_gate"])
    query = t.nn.functional.linear(qr, indexer.weights["self_attn.indexer.wq_b.weight"])
    query = query.view(-1, indexer.heads, HEAD_DIM).to(t.bfloat16).float()
    query = indexer.rotate(query).float()
    weights = t.nn.functional.linear(xf, w["weights_proj.weight"])
    weights = weights * (HEAD_DIM**-0.5 * indexer.heads**-0.5)
    return keys, gates, query, weights


class IndexerProjectionGraph:
    """Private owner buffers; no cache positions or tail lengths in capture."""

    def __init__(self, indexer):
        if not indexer.device.startswith("cuda") or not indexer.fused_cuda_rotation:
            raise ValueError("indexer projection capture requires qualified CUDA rotation")
        t = indexer.torch
        inputs = (
            t.zeros((1, indexer.fp32["wk.weight"].shape[1]), dtype=t.float16, device=indexer.device),
            t.zeros(
                (1, indexer.weights["self_attn.indexer.wq_b.weight"].shape[1]), dtype=t.float16, device=indexer.device
            ),
        )
        for _ in range(CAPTURE_WARMUP_STEPS):
            project(indexer, *inputs)
        t.cuda.synchronize()
        self.graph = t.cuda.CUDAGraph()
        with t.cuda.graph(self.graph):
            self.outputs = project(indexer, *inputs)
        t.cuda.synchronize()
        self.inputs = inputs

    def __call__(self, hidden, qr):
        if hidden.shape != self.inputs[0].shape or qr.shape != self.inputs[1].shape:
            raise ValueError("indexer projection graph requires one complete decode row")
        for target, source in zip(self.inputs, (hidden, qr), strict=True):
            if source.dtype != target.dtype or source.device != target.device:
                raise ValueError("indexer projection graph input owner or dtype differs")
            target.copy_(source)
        self.graph.replay()
        return self.outputs

    def close(self):
        self.graph = self.inputs = self.outputs = None
