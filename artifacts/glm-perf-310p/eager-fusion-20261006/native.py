# SPDX-License-Identifier: Apache-2.0
"""Shape-aware resident fusion resources for decode and prefill; no import side effects."""

from pathlib import Path

import torch


class Fusions:
    MAX_ROWS = 32768
    MAX_WIDTH = 16384
    ALIGNMENT = 32
    BLOCKS = 8

    def __init__(self, root):
        root = Path(root)
        self.kernels = {
            name: torch.classes.glm_eager_fusions_v1.Kernel(str(root / f"{name}.bin"), f"glm_resident_{name}_v1")
            for name in ("swiglu", "combine", "mhc_post")
        }
        self.configs = {}
        self.launch = torch.ops.glm_eager_fusions_v1.launch
        # Preallocate every graph decode shape. Larger eager prefill shapes
        # acquire one immutable per-shape config before their first launch.
        for tokens in range(1, 9):
            self.config((tokens * 8, 2048))
            self.config((tokens, 4096, 8, 64))
            self.config((tokens, 4096))
        self.calls = {name: {} for name in self.kernels}

    def config(self, geometry):
        if geometry not in self.configs:
            self.configs[geometry] = torch.tensor(geometry, dtype=torch.int64).npu()
        return self.configs[geometry]

    @staticmethod
    def tensor(value, dtype, shape=None):
        if value.device.type != "npu" or value.dtype != dtype or not value.is_contiguous():
            raise ValueError("fusion requires contiguous NPU tensors with the qualified dtype")
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError("fusion input shape mismatch")

    def record(self, name, geometry):
        self.calls[name][geometry] = self.calls[name].get(geometry, 0) + 1

    def swiglu(self, gate_up):
        self.tensor(gate_up, torch.float16)
        if gate_up.ndim != 2:
            raise ValueError("SwiGLU expects [routes, 2*width]")
        rows, twice_width = gate_up.shape
        if not (
            0 < rows <= self.MAX_ROWS
            and 0 < twice_width <= 2 * self.MAX_WIDTH
            and twice_width % (2 * self.ALIGNMENT) == 0
        ):
            raise ValueError("unsupported SwiGLU geometry")
        geometry = (rows, twice_width // 2)
        output = torch.empty(geometry, dtype=gate_up.dtype, device=gate_up.device)
        self.launch(self.kernels["swiglu"], [gate_up, output, self.config(geometry)], self.BLOCKS)
        self.record("swiglu", geometry)
        return output

    def combine(self, routed, inverse, weights, ends):
        self.tensor(routed, torch.float16)
        self.tensor(inverse, torch.int64)
        self.tensor(weights, torch.float32)
        self.tensor(ends, torch.int64)
        if routed.ndim != 2 or weights.ndim != 2 or inverse.ndim != 1 or ends.ndim != 1:
            raise ValueError("unsupported route metadata rank")
        rows, hidden = routed.shape
        tokens, top_k = weights.shape
        if not (
            rows == tokens * top_k == inverse.numel()
            and 0 < rows <= self.MAX_ROWS
            and 0 < hidden <= self.MAX_WIDTH
            and hidden % self.ALIGNMENT == 0
            and 0 < top_k <= 32
            and 0 < ends.numel() <= 1024
        ):
            raise ValueError("unsupported route combine geometry")
        if any(value.device != routed.device for value in (inverse, weights, ends)):
            raise ValueError("fusion inputs must be on the same device")
        geometry = (tokens, hidden, top_k, ends.numel())
        output = torch.empty((tokens, hidden), dtype=torch.float32, device=routed.device)
        self.launch(
            self.kernels["combine"], [routed, inverse, weights, ends, output, self.config(geometry)], self.BLOCKS
        )
        self.record("combine", geometry)
        return output

    def mhc_post(self, x, residual, post, comb):
        self.tensor(x, torch.float16)
        if x.ndim != 2:
            raise ValueError("mHC post expects [tokens,width]")
        rows, width = x.shape
        if not (0 < rows <= self.MAX_ROWS and 0 < width <= self.MAX_WIDTH and width % self.ALIGNMENT == 0):
            raise ValueError("unsupported mHC post geometry")
        for value, shape in ((residual, (rows, 4, width)), (post, (rows, 4, 1)), (comb, (rows, 4, 4))):
            self.tensor(value, torch.float32, shape)
            if value.device != x.device:
                raise ValueError("fusion inputs must be on the same device")
        geometry = (rows, width)
        output = torch.empty_like(residual)
        self.launch(self.kernels["mhc_post"], [x, residual, post, comb, output, self.config(geometry)], self.BLOCKS)
        self.record("mhc_post", geometry)
        return output
