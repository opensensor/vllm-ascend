# SPDX-License-Identifier: Apache-2.0
"""Small, shape-realistic math alternatives; none change serving defaults."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

HIDDEN_SIZE = 2560
HC_STREAMS = 4
TOP_K = 10
NUM_EXPERTS = 512
EPSILON = 1e-6
CASE_NAMES = (
    "swiglu_native_fp32",
    "swiglu_native_fp16",
    "swiglu_contiguous_halves",
    "hc_mix_sum_scale",
    "hc_mix_bmm",
    "hc_residual_addcmul",
    "hc_residual_addcmul_inplace",
    "hc_injection_reciprocal_inplace",
    "hc_down_reciprocal",
    "hc_injection_fp16_reciprocal_inplace",
    "hc_down_fp16_reciprocal",
    "route_reduce_bmm",
    "route_reduce_einsum",
    "route_weight_inplace",
    "router_cancel_normalizer",
    "router_unsorted_then_sort",
    "rms_native_fp32",
    "residual_rms_native_fp32",
    "residual_rms_native_fp16",
)
PATTERNS = ("random", "zeros", "large", "ties", "near_ties")


@dataclass(frozen=True)
class MathCase:
    name: str
    inputs: tuple[torch.Tensor, ...]
    baseline: Callable
    candidate: Callable
    description: str


def changed_inputs(inputs, pattern, seed):
    """Prepare changes on the host, outside timed/captured execution."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown input pattern: {pattern}")
    generator = torch.Generator().manual_seed(seed)
    result = []
    for tensor in inputs:
        if pattern == "zeros":
            value = torch.zeros(tensor.shape, dtype=tensor.dtype)
        elif pattern == "ties":
            value = torch.full(tensor.shape, 0.125, dtype=tensor.dtype)
        else:
            scale = 8.0 if pattern == "large" else 1e-7 if pattern == "near_ties" else 0.1
            value = (torch.randn(tensor.shape, generator=generator) * scale).to(tensor.dtype)
        result.append(value.to(tensor.device))
    return tuple(result)


def build_case(name, tokens, device="cpu", native=None):
    if name not in CASE_NAMES or tokens < 1:
        raise ValueError("choose a known case and positive token count")

    def values(*shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=device)

    def rms(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPSILON)

    if name.startswith("swiglu_"):
        operands = (values(tokens * TOP_K, 1280, dtype=torch.float16),)

        def baseline(x):
            gate, up = x.float().chunk(2, dim=-1)
            return (F.silu(gate) * up).half()

        if name == "swiglu_native_fp32":
            candidate = lambda x: native.npu_swiglu(x.float(), dim=-1).half()
        elif name == "swiglu_native_fp16":
            candidate = lambda x: native.npu_swiglu(x, dim=-1)
        else:

            def candidate(x):
                gate, up = x.float().chunk(2, dim=-1)
                return (F.silu(gate.contiguous()) * up.contiguous()).half()

        description = "FP32 SwiGLU followed by FP16 rounding; dynamic copies included"
    elif name.startswith("hc_mix_"):
        operands = (values(tokens, HC_STREAMS, HIDDEN_SIZE), values(tokens, HC_STREAMS, HIDDEN_SIZE))
        baseline = lambda x, gate: (x * gate).mean(dim=1).half()
        if name == "hc_mix_sum_scale":
            candidate = lambda x, gate: ((x * gate).sum(dim=1) * (1.0 / HC_STREAMS)).half()
        else:

            def candidate(x, gate):
                # Gates vary by channel. Both dynamic transposes/copies count.
                left = x.transpose(1, 2).reshape(-1, 1, HC_STREAMS)
                right = gate.transpose(1, 2).reshape(-1, HC_STREAMS, 1)
                return (torch.bmm(left, right).reshape(tokens, HIDDEN_SIZE) / HC_STREAMS).half()

        description = "four-stream channel-wise gated mean with final FP16 rounding"
    elif name.startswith("hc_residual_"):
        operands = (
            values(tokens, HC_STREAMS, HIDDEN_SIZE, dtype=torch.float16),
            values(tokens, 1, HIDDEN_SIZE),
            values(tokens, HC_STREAMS, 1),
        )
        baseline = lambda residual, block, gate: (residual.float() + block * gate).half()
        if name == "hc_residual_addcmul":
            candidate = lambda residual, block, gate: torch.addcmul(residual.float(), block, gate).half()
        else:
            candidate = lambda residual, block, gate: residual.float().addcmul_(block, gate).half()
        description = "FP16 residual cast included; FP32 FMA may change rounding; reuse only the fresh cast buffer"
    elif name == "hc_injection_reciprocal_inplace":
        operands = (values(tokens, HC_STREAMS, dtype=torch.float16),)
        baseline = lambda x: 2.0 * torch.sigmoid(x.float() / HC_STREAMS)
        candidate = lambda x: torch.sigmoid(x.float().mul_(1.0 / HC_STREAMS)).mul_(2.0)
        description = "power-of-two scaling and reuse of fresh FP32 injection buffers"
    elif name == "hc_down_reciprocal":
        operands = (values(tokens, 320, dtype=torch.float16),)
        baseline = lambda x: F.silu(x.float() / HC_STREAMS)
        candidate = lambda x: F.silu(x.float().mul_(1.0 / HC_STREAMS))
        description = "power-of-two down scaling; input parameter storage stays intact"
    elif name == "hc_injection_fp16_reciprocal_inplace":
        operands = (values(tokens, HC_STREAMS, dtype=torch.float16),)
        baseline = lambda x: 2.0 * torch.sigmoid(x / HC_STREAMS)
        candidate = lambda x: torch.sigmoid(x * (1.0 / HC_STREAMS)).mul_(2.0)
        description = "actual NPU HC FP16 projection output; FP16 sigmoid and compact gate scaling"
    elif name == "hc_down_fp16_reciprocal":
        operands = (values(tokens, 320, dtype=torch.float16),)
        baseline = lambda x: F.silu(x / HC_STREAMS)
        candidate = lambda x: F.silu(x * (1.0 / HC_STREAMS))
        description = "actual NPU HC FP16 down output and FP16 SwiGLU gate; power-of-two multiply"
    elif name.startswith("route_reduce_"):
        operands = (values(tokens, TOP_K, HIDDEN_SIZE, dtype=torch.float16), values(tokens, TOP_K))
        baseline = lambda rows, weights: (rows.float() * weights.unsqueeze(-1)).sum(1).half()
        if name == "route_reduce_bmm":
            candidate = lambda rows, weights: torch.bmm(weights.unsqueeze(1), rows.float()).squeeze(1).half()
        else:
            candidate = lambda rows, weights: torch.einsum("tr,trh->th", weights, rows.float()).half()
        description = "FP32 route-weighted reduction in ten-slot order versus matrix reduction"
    elif name == "route_weight_inplace":
        operands = (values(tokens * TOP_K, HIDDEN_SIZE, dtype=torch.float16), values(tokens * TOP_K, 1))
        baseline = lambda rows, weights: rows.float() * weights
        candidate = lambda rows, weights: rows.float().mul_(weights)
        description = "reuse the fresh FP32 cast buffer for route weighting"
    elif name.startswith("router_"):
        operands = (values(tokens, NUM_EXPERTS, dtype=torch.float16),)

        def baseline(logits):
            weights, ids = logits.float().softmax(-1).topk(TOP_K, dim=-1)
            return weights / weights.sum(-1, keepdim=True), ids

        if name == "router_cancel_normalizer":

            def candidate(logits):
                weights, ids = logits.float().topk(TOP_K, dim=-1)
                return weights.softmax(-1), ids

        else:

            def candidate(logits):
                weights, ids = logits.float().softmax(-1).topk(TOP_K, dim=-1, sorted=False)
                weights, order = weights.sort(dim=-1, descending=True)
                return weights / weights.sum(-1, keepdim=True), ids.gather(-1, order)

        description = "renormalized router only; exact IDs, boundary ties and FP32 weights are required"
    else:
        dtype = torch.float16 if name.endswith("fp16") else torch.float32
        # Four contiguous groups reflect HC RMSNorm geometry. Unit gamma is
        # static and prepared outside timing/capture, just as in the runtime.
        operands = (values(tokens * HC_STREAMS, HIDDEN_SIZE, dtype=dtype),)
        unit = torch.ones(HIDDEN_SIZE, dtype=dtype, device=device)
        if name == "rms_native_fp32":
            baseline = rms
            candidate = lambda x: native.npu_rms_norm(x, unit, EPSILON)[0]
        else:
            operands += (values(tokens * HC_STREAMS, HIDDEN_SIZE, dtype=dtype),)

            def baseline(x, residual):
                total = x + residual
                return rms(total.float()).to(dtype), total

            def candidate(x, residual):
                normalized, _, total = native.npu_add_rms_norm(x, residual, unit, EPSILON)
                return normalized, total

        description = "grouped RMSNorm calibration or residual-add plus RMSNorm fusion"
    return MathCase(name, operands, baseline, candidate, description)
