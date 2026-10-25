# SPDX-License-Identifier: Apache-2.0
"""Direct device-only clipped SwiGLU; preserve the FP32 activation contract."""

import torch

from vllm_ascend._310p.ops.minimax_m3 import clipped_swiglu

TILE = 256
BLOCKS = 8


def _dense_format(value):
    # Lazy dependency: CPU fallback and admission tests do not import torch_npu.
    from torch_npu import get_npu_format

    return get_npu_format(value) == 2


class ClippedSwiGLUKernel:
    def __init__(self, namespace, binary):
        self.kernel = getattr(torch.classes, namespace).Kernel(binary, "minimax_clipped_swiglu")
        self.launch = getattr(torch.ops, namespace).launch
        self.configs = {}

    @staticmethod
    def accepts(gate, up, alpha, beta, limit):
        return (
            not torch.is_grad_enabled()
            and gate.device.type == "npu"
            and gate.device == up.device
            and gate.dtype == up.dtype == torch.float16
            and gate.shape == up.shape
            and gate.is_contiguous()
            and up.is_contiguous()
            and gate.numel() > 0
            and gate.numel() % TILE == 0
            and (alpha, beta, limit) == (1.702, 1.0, 7.0)
            and _dense_format(gate)
            and _dense_format(up)
        )

    def __call__(self, gate, up, alpha=1.702, beta=1.0, limit=7.0):
        if not self.accepts(gate, up, alpha, beta, limit):
            return clipped_swiglu(gate, up, alpha, beta, limit)
        key = (gate.numel(), gate.device)
        if key not in self.configs:
            # Immutable descriptors are uploaded once per geometry, not per token.
            self.configs[key] = torch.tensor([key[0]], dtype=torch.int64, device=gate.device)
        output = torch.empty_like(gate, dtype=torch.float32)
        self.launch(self.kernel, [gate, up, output, self.configs[key]], min(BLOCKS, gate.numel() // TILE))
        return output


def validate_kernel(kernel):
    generator = torch.Generator().manual_seed(43)
    reports = []
    with torch.inference_mode():
        for rows, width in ((1, 3072), (4, 3072), (64, 3072), (1, 65536)):
            gate = (torch.randn(rows, width, generator=generator) * 15).half()
            if width == 65536:
                gate = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(1, -1)
                gate = gate.nan_to_num()
            gate = gate.npu()
            up = (torch.randn(rows, width, generator=generator) * 15).half().npu()
            expected = clipped_swiglu(gate, up)
            actual = kernel(gate, up)
            reports.append({"rows": rows, "width": width, "exact": torch.equal(actual, expected)})
    return {"passed": all(case["exact"] for case in reports), "cases": reports}
