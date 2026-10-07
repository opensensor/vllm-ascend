# SPDX-License-Identifier: Apache-2.0
"""Versioned resident resource for the staged KDA preparation kernel.

Construct only in paused workers after loading a qualified native library.
No constructor or device operation is invoked by importing this module.
"""

import math

import torch


class GateBeta:
    def __init__(self, binary, *, heads, lower_bound, device, namespace="glm_kda_prepare_v1"):
        if heads <= 0 or heads % 16 or not math.isfinite(lower_bound) or lower_bound >= 0:
            raise ValueError("requires groups of 16 heads and a finite negative gate lower bound")
        self.heads = heads
        self.lower_bound = lower_bound
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, "glm_kda_gate_beta_v1")
        self.launch = getattr(torch.ops, namespace).launch
        self.lower = torch.tensor([lower_bound], dtype=torch.float32, device=device)
        self.tiling = {
            (rows, gate_dtype, beta_dtype): torch.tensor(
                [rows, heads, int(gate_dtype == torch.float32), int(beta_dtype == torch.float32)],
                dtype=torch.int64,
                device=device,
            )
            for rows in range(1, 9)
            for gate_dtype in (torch.float16, torch.float32)
            for beta_dtype in (torch.float16, torch.float32)
        }

    def supports(self, raw_gate, beta_raw, scale, bias, lower_bound):
        return (
            raw_gate.ndim == 4
            and raw_gate.shape[0] == 1
            and raw_gate.shape[2:] == (self.heads, 128)
            and beta_raw.shape == raw_gate.shape[:3]
            and (raw_gate.shape[1], raw_gate.dtype, beta_raw.dtype) in self.tiling
            and scale.dtype == bias.dtype == torch.float32
            and scale.shape == (1, 1, self.heads, 1)
            and bias.shape == (1, 1, self.heads, 128)
            and lower_bound == self.lower_bound
            and all(t.is_contiguous() and t.device == self.lower.device for t in (raw_gate, beta_raw, scale, bias))
        )

    def __call__(self, raw_gate, beta_raw, scale, bias, lower_bound):
        if not self.supports(raw_gate, beta_raw, scale, bias, lower_bound):
            raise ValueError("unsupported KDA gate/beta inputs; use the reference fallback")
        gate = torch.empty(raw_gate.shape[1:], dtype=torch.float32, device=raw_gate.device)
        beta = torch.empty(beta_raw.shape[1:], dtype=torch.float16, device=raw_gate.device)
        self.launch(
            self.kernel,
            [
                raw_gate,
                beta_raw,
                scale,
                bias,
                self.lower,
                gate,
                beta,
                self.tiling[raw_gate.shape[1], raw_gate.dtype, beta_raw.dtype],
            ],
            8,
        )
        return gate, beta
