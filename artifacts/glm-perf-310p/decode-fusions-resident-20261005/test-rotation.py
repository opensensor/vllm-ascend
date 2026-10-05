# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import torch
import torch_npu
from probe import measure
from rotation import Rotation

from vllm_ascend.models.glm5next.kpool_ops import hadamard128

root = Path(__file__).resolve().parent
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.ops.load_library(str(root / "glm_rotation_bridge_v1.so"))
op = Rotation(str(root / "rotation-v1.bin"))
generator = torch.Generator().manual_seed(310)
records = []
for rows in (1, 2, 4, 8):
    for kind in ("normal", "bf16", "small", "large", "impulse", "zero"):
        x = torch.randn(rows, 32, 128, generator=generator).half()
        if kind == "bf16":
            x = x.bfloat16()
        elif kind == "small":
            x *= 0.00001
        elif kind == "large":
            x = (x * 10000).clamp(-65504, 65504)
        elif kind == "impulse":
            x.zero_()
            x[..., 37] = 1
        elif kind == "zero":
            x.zero_()
        x = x.npu()
        expected = hadamard128(x).bfloat16().half()
        actual = op(x)
        difference = int((expected != actual).sum().item())
        records.append(
            {"rows": rows, "kind": kind, "mismatches": difference, "max_abs": (expected - actual).abs().max().item()}
        )
        print(json.dumps(records[-1]), flush=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


timings = []
for rows in (2, 8):
    x = torch.randn(rows, 32, 128, generator=generator).bfloat16().npu()
    timings.append(
        {
            "rows": rows,
            "timing": measure(
                {
                    "reference": lambda x=x: hadamard128(x).bfloat16().half(),
                    "native": lambda x=x: op(x),
                }
            ),
        }
    )
(root / "rotation-results.json").write_text(json.dumps({"parity": records, "timings": timings}, indent=2))
print(json.dumps(timings), flush=True)
