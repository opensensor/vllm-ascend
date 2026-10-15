# SPDX-License-Identifier: Apache-2.0
"""Remaining stock 310P math probes; accuracy-changing paths are screening only."""

import torch
import torch.nn.functional as F

from tools.qwen4exp.math_probe_cases import MathCase

CASE_NAMES = ("glu_fp32", "glu_fp16", "glu_mixed", "gdn_rms_affine_gate", "router_fused_fp16", "rms_quant_static")
EPSILON = 1e-6


def build_case(name, tokens, device="cpu", native=None):
    if name not in CASE_NAMES or tokens < 1:
        raise ValueError("choose a known case and positive token count")
    if name.startswith("glu_"):
        dtype = torch.float16 if name == "glu_fp16" else torch.float32
        gate_dtype = torch.float16 if name == "glu_mixed" else dtype
        inputs = (
            torch.zeros(tokens, 2560, device=device, dtype=dtype),
            torch.zeros(tokens, 2560, device=device, dtype=gate_dtype),
        )
        baseline = lambda x, gate: x * torch.sigmoid(gate)
        candidate = lambda x, gate: F.glu(torch.cat((x, gate.to(x.dtype)), -1), dim=-1)
        description = "GLU sigmoid-multiply; concatenation/casting costs included"
    elif name == "gdn_rms_affine_gate":
        inputs = (
            torch.zeros(tokens * 8, 128, device=device, dtype=torch.float32),
            torch.zeros(tokens * 8, 128, device=device, dtype=torch.float16),
        )
        # Fixed learned-like gamma is prepared outside timing; runtime accumulates in FP32.
        gamma = torch.linspace(0.8, 1.2, 128, device=device, dtype=torch.float32)

        def baseline(x, gate):
            normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPSILON)
            return (normalized * gamma * torch.sigmoid(gate)).half()

        candidate = lambda x, gate: (native.npu_rms_norm(x, gamma, EPSILON)[0] * torch.sigmoid(gate)).half()
        description = "GDN output RMSNorm affine plus sigmoid gating; native norm only, full fusion follows"
    elif name == "router_fused_fp16":
        inputs = (torch.zeros(tokens, 512, device=device, dtype=torch.float16),)

        def baseline(x):
            weights, indices = x.float().softmax(-1).topk(10, -1)
            return weights / weights.sum(-1, keepdim=True), indices

        def candidate(x):
            weights, indices, _ = native.npu_moe_gating_top_k_softmax(x, k=10)
            weights = weights.float()
            return weights / weights.sum(-1, keepdim=True), indices.long()

        description = "FP16 fused router vs current FP32 probabilities; not an assumed exact replacement"
    else:
        inputs = (torch.zeros(tokens, 2560, device=device, dtype=torch.float16),)
        gamma = torch.ones(2560, device=device, dtype=torch.float16)
        beta = torch.zeros_like(gamma)
        scale = torch.full_like(gamma, 16.0)
        offset = torch.zeros(2560, device=device, dtype=torch.int8)

        def baseline(x):
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPSILON)
            return torch.round(normalized * scale.float()).clamp(-128, 127).to(torch.int8)

        candidate = lambda x: native.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=EPSILON)
        description = "static norm+INT8 quantization calibration probe; not native-W4 dynamic group packing"
    return MathCase(name, inputs, baseline, candidate, description)
