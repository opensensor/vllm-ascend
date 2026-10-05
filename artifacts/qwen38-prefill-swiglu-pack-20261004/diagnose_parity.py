# SPDX-License-Identifier: Apache-2.0
"""Inspect the first mismatches in prefill SwiGLU packing on one 310P."""

import json

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, swiglu_pack_activation_device
from vllm_ascend.utils import enable_custom_op


def main() -> None:
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    for rows in (15360, 20480):
        generator = torch.Generator().manual_seed(1000 + rows)
        gate_up = (torch.randn(rows, 1280, generator=generator) * 1.5).half().npu()
        gate, up = gate_up.float().chunk(2, -1)
        torch_activation = (F.silu(gate) * up).half()
        explicit_activation = ((gate / (1 + torch.exp(-gate))) * up).half()
        expected = [tensor.cpu() for tensor in pack_activation_device(torch_activation)]
        explicit = [tensor.cpu() for tensor in pack_activation_device(explicit_activation)]
        actual = [tensor.cpu() for tensor in swiglu_pack_activation_device(gate_up)]
        report = {"rows": rows, "outputs": []}
        for name, reference, formula, candidate in zip(("low", "high", "scale", "sum"), expected, explicit, actual):
            mismatch = (candidate != reference).nonzero()
            formula_mismatch = (candidate != formula).nonzero()
            first = []
            for index in mismatch[:4].tolist():
                row, column = index[:2]
                position = tuple(index)
                item = {
                    "index": index,
                    "torch": float(reference[position]),
                    "explicit": float(formula[position]),
                    "fused": float(candidate[position]),
                }
                if name in ("low", "high"):
                    offset = column * 2
                    item["activation_window"] = torch_activation[row, offset : offset + 2].cpu().tolist()
                    item["explicit_activation_window"] = explicit_activation[row, offset : offset + 2].cpu().tolist()
                    item["gate_window"] = gate[row, offset : offset + 2].cpu().tolist()
                    item["up_window"] = up[row, offset : offset + 2].cpu().tolist()
                first.append(item)
            report["outputs"].append(
                {
                    "name": name,
                    "mismatch_count": mismatch.shape[0],
                    "mismatch_vs_explicit_count": formula_mismatch.shape[0],
                    "first": first,
                }
            )
        activation_mismatch = (torch_activation.cpu() != explicit_activation.cpu()).nonzero()
        report["activation_formula_mismatch_count"] = activation_mismatch.shape[0]
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
