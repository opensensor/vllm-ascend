# SPDX-License-Identifier: Apache-2.0
"""Explicitly prepared tiled normalization; qualified torch softmax stays intact."""

import math
import struct
from pathlib import Path

import torch

MATRIX_WIDTH = 4
MATRIX_ELEMENTS = MATRIX_WIDTH * MATRIX_WIDTH
TILE_ROWS = 16
VECTOR_CORES = 8
MAX_ROWS = 640
MAX_ITERATIONS = 64
SEQUENTIAL_ROW_SUM_MIN_ROWS = 128


def reduction_order(rows):
    """Match the deployed ReduceSum axis order; columns remain sequential."""
    return 1 if rows >= SEQUENTIAL_ROW_SUM_MIN_ROWS else 2


def gather_indices():
    """Each gather is confined to its own 4x4 matrix, including the final tile."""
    elements = TILE_ROWS * MATRIX_ELEMENTS
    row = [[((i // MATRIX_WIDTH) * MATRIX_WIDTH + j) * 4 for i in range(elements)] for j in range(MATRIX_WIDTH)]
    col = [
        [((i // MATRIX_ELEMENTS) * MATRIX_ELEMENTS + j * MATRIX_WIDTH + i % MATRIX_WIDTH) * 4 for i in range(elements)]
        for j in range(MATRIX_WIDTH)
    ]
    return torch.tensor(row + col, dtype=torch.int32)


def descriptor(rows, iterations, epsilon, order):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("normalization rows must be in [1, 640]")
    if type(iterations) is not int or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError("normalization iterations must be in [1, 64]")
    if type(order) is not int or order not in (0, 1, 2):
        raise ValueError("normalization reduction order must be pairwise, sequential or mixed")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("normalization epsilon must be finite in [0, 1]")
    bits = struct.unpack("<I", struct.pack("<f", epsilon))[0]
    return rows, iterations, bits, order


class SinkhornTiled:
    def __init__(self, build, namespace, *, iterations=20, epsilon=1e-6, order=None, kernel_factory=None, launch=None):
        descriptor(1, iterations, epsilon, reduction_order(1) if order is None else order)
        self.device = torch.device("npu", torch.npu.current_device())
        factory = kernel_factory or getattr(torch.classes, namespace).Kernel
        self.kernel = factory(str(Path(build) / "glm_sinkhorn_tiled.bin"), "glm_sinkhorn_tiled_v1")
        self.launch = launch or getattr(torch.ops, namespace).launch
        self.iterations, self.epsilon, self.order = iterations, epsilon, order
        self.indices = gather_indices().to(self.device)
        self.configs = {}
        self.calls = 0

    def prepare_counts(self, counts):
        counts = list(counts)
        for rows in counts:
            descriptor(rows, self.iterations, self.epsilon, reduction_order(rows) if self.order is None else self.order)
        missing = [rows for rows in dict.fromkeys(counts) if rows not in self.configs]
        values = [
            descriptor(rows, self.iterations, self.epsilon, reduction_order(rows) if self.order is None else self.order)
            for rows in missing
        ]
        if values:
            backing = torch.tensor(values, dtype=torch.int64, device=self.device)
            self.configs.update((rows, backing[index]) for index, rows in enumerate(missing))

    def __call__(self, mix):
        if (
            mix.dtype != torch.float32
            or mix.device != self.device
            or mix.ndim != 3
            or mix.shape[1:] != (MATRIX_WIDTH, MATRIX_WIDTH)
            or mix.shape[0] not in self.configs
            or not mix.is_contiguous()
        ):
            raise ValueError("normalization requires prepared same-device contiguous FP32 [rows,4,4]")
        output = torch.empty_like(mix)
        cores = min(VECTOR_CORES, (mix.shape[0] + TILE_ROWS - 1) // TILE_ROWS)
        self.launch(self.kernel, [mix, output, self.configs[mix.shape[0]], self.indices], cores)
        self.calls += 1
        return output
