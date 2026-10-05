# SPDX-License-Identifier: Apache-2.0
"""Compare NPU SwiGLU formulas at prefill size without changing the model."""

import json

import torch
import torch.nn.functional as F
import torch_npu


def main() -> None:
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    rows = 15360
    generator = torch.Generator().manual_seed(1000 + rows)
    gate_up = (torch.randn(rows, 1280, generator=generator) * 1.5).half().npu()
    gate, up = gate_up.float().chunk(2, -1)
    reference = (F.silu(gate) * up).half()
    denominator = 1 + torch.exp(-gate)
    variants = {
        "division": (gate / denominator * up).half(),
        "reciprocal_mul": (gate * denominator.reciprocal() * up).half(),
        "sigmoid_mul": (gate * torch.sigmoid(gate) * up).half(),
    }
    result = {}
    for name, value in variants.items():
        difference = (value != reference).nonzero()
        result[name] = {
            "mismatches": difference.shape[0],
            "first": difference[:3].tolist(),
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
