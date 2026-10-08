# SPDX-License-Identifier: Apache-2.0
"""Experimental normalization-only Sinkhorn kernel for resident decode probes.

The qualified softmax remains outside the kernel. Load through the resident
validation gate before enabling serving; importing this module enables nothing.
"""

import math
import struct

import torch

MAX_DECODE_ROWS = 8
MATRIX_WIDTH = 4
MAX_ITERATIONS = 64


def normalization_config(rows, iterations, epsilon, order):
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (rows, iterations, order)):
        raise ValueError("normalization rows, iterations, and order must be integers")
    if not 1 <= rows <= MAX_DECODE_ROWS:
        raise ValueError("normalization supports one through eight decode rows")
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError("normalization iterations must be in [1, 64]")
    if not math.isfinite(epsilon) or epsilon < 0 or epsilon > 1:
        raise ValueError("normalization epsilon must be finite and in [0, 1]")
    if order not in (0, 1, 2):
        raise ValueError("normalization reduction order must be pairwise, sequential, or mixed")
    # Native layout is int64 rows, int64 iterations, float32 epsilon with
    # four padding bytes, then int64 order. Avoid numeric casts of bit fields.
    epsilon_bits = struct.unpack("<I", struct.pack("<f", epsilon))[0]
    return (rows, iterations, epsilon_bits, order)


def reference_normalize(mix, iterations, epsilon):
    """Same iteration order as upstream, accepting softmax(logits) + epsilon."""
    mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + epsilon)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    return mix


class SinkhornNormalize:
    def __init__(self, binary, iterations=20, epsilon=1e-6, order=2):
        # Call only from the paused native loader, before any graph capture.
        configs = [normalization_config(rows, iterations, epsilon, order) for rows in range(1, MAX_DECODE_ROWS + 1)]
        self.iterations = iterations
        self.epsilon = epsilon
        self.kernel = torch.classes.glm_sinkhorn_v1.Kernel(binary, "glm_sinkhorn_normalize_v1")
        self.launch = torch.ops.glm_sinkhorn_v1.launch
        self.configs = {rows: torch.tensor(config, dtype=torch.int64).npu() for rows, config in enumerate(configs, 1)}

    def __call__(self, mix):
        if (
            mix.dtype != torch.float32
            or mix.ndim != 3
            or mix.shape[1:] != (MATRIX_WIDTH, MATRIX_WIDTH)
            or mix.shape[0] not in self.configs
            or not mix.is_contiguous()
            or mix.device.type != "npu"
        ):
            raise ValueError("normalization requires contiguous NPU float32 [1..8,4,4]")
        output = torch.empty_like(mix)
        self.launch(self.kernel, [mix, output, self.configs[mix.shape[0]]], mix.shape[0])
        return output
