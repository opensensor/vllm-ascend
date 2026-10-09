# SPDX-License-Identifier: Apache-2.0
"""Experimental tile-local mHC projection; importing changes no serving methods."""

import json

import torch

STREAMS = 4
WIDTH = 4096
INPUT_WIDTH = STREAMS * WIDTH
OUTPUT_WIDTH = 24
PADDED_OUTPUT_WIDTH = 32
TILE_ROWS = 16
TILE_K = 256
MAX_ROWS = 32768
VECTOR_CORES = 8
# UB: FP32 input, two FP16 inputs, byte gather indices, two FP32 outputs.
UB_BYTES = TILE_ROWS * TILE_K * (4 + 2 + 2 + 4) + TILE_ROWS * PADDED_OUTPUT_WIDTH * (4 + 8)


def offsets(tile_k=TILE_K):
    if type(tile_k) is not int or tile_k not in (256, 512, 1024):
        raise ValueError("projection K tile must be 256, 512 or 1024")
    columns = torch.arange(tile_k).reshape(tile_k // TILE_ROWS, TILE_ROWS)
    rows = torch.arange(TILE_ROWS)
    input_offsets = (rows[None, :, None] * tile_k + columns[:, None, :]).reshape(-1) * 2
    output_offsets = (
        torch.arange(PADDED_OUTPUT_WIDTH)[None, :] // TILE_ROWS * TILE_ROWS * TILE_ROWS
        + rows[:, None] * TILE_ROWS
        + torch.arange(PADDED_OUTPUT_WIDTH)[None, :] % TILE_ROWS
    ).reshape(-1) * 4
    return torch.cat((input_offsets, output_offsets)).to(torch.int32)


def pack_weight(fn):
    if fn.dtype != torch.float32 or tuple(fn.shape) != (OUTPUT_WIDTH, INPUT_WIDTH):
        raise ValueError("mHC projection requires FP32 [24,16384] weights")
    # Preparation runs while idle. Qualified state already has FP16-rounded
    # values; the kernel retains FP32 dot accumulation and output storage.
    half = fn.detach().half()
    padded = torch.zeros((PADDED_OUTPUT_WIDTH, INPUT_WIDTH), dtype=torch.float16, device=fn.device)
    padded[:OUTPUT_WIDTH].copy_(half)
    return padded.reshape(PADDED_OUTPUT_WIDTH, INPUT_WIDTH // TILE_ROWS, TILE_ROWS).permute(1, 0, 2).contiguous()


class NativeMhcProjection:
    @staticmethod
    def weight_key(fn):
        # Serving weights are immutable. Ordinary tensors additionally expose
        # a version counter; worker-loaded inference tensors do not have one.
        return fn.data_ptr(), None if fn.is_inference() else fn._version

    def __init__(self, build, namespace, *, device, kernel_factory=None, launch=None):
        self.device = device
        factory = kernel_factory or getattr(torch.classes, namespace).Kernel
        self.kernel = factory(str(build / "glm_mhc_projection.bin"), "glm_mhc_projection_fp32_v1")
        self.launch = launch or getattr(torch.ops, namespace).launch
        provenance = build / "provenance.json"
        options = json.loads(provenance.read_text())["_build"] if provenance.exists() else {}
        self.tile_k = options.get("mhc_projection_tile_k", TILE_K)
        self.indices = offsets(self.tile_k).to(device)
        self.configs = {}
        self.weights = {}

    def prepare(self, weights, row_counts):
        for fn in weights:
            if fn.device != self.device:
                raise ValueError("projection weights must belong to this device")
            key = self.weight_key(fn)
            if key not in self.weights:
                self.weights[key] = pack_weight(fn)
        for rows in row_counts:
            if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
                raise ValueError("projection rows must be a bounded integer")
            if rows not in self.configs:
                self.configs[rows] = torch.tensor([rows], dtype=torch.int64, device=self.device)

    def __call__(self, residual, fn):
        rows = residual.shape[0]
        if (
            residual.dtype != torch.float32
            or residual.device != self.device
            or residual.shape != (rows, STREAMS, WIDTH)
            or not residual.is_contiguous()
            or rows not in self.configs
            or self.weight_key(fn) not in self.weights
        ):
            raise ValueError("projection needs prepared weights, rows and contiguous FP32 state")
        output = torch.empty((rows, PADDED_OUTPUT_WIDTH), dtype=torch.float32, device=self.device)
        self.launch(
            self.kernel,
            [residual, self.weights[self.weight_key(fn)], output, self.configs[rows], self.indices],
            VECTOR_CORES,
        )
        return output[:, :OUTPUT_WIDTH].contiguous()
