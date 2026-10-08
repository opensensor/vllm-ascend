# SPDX-License-Identifier: Apache-2.0
"""Real-weight W3 unpack/dequant/matmul parity probe on one free Ascend NPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from safetensors import safe_open

from tools.deepseek_w2.w2_format import unpack_codes
from vllm_ascend._310p.quantization.methods.w2_dynamic import _w2_dequant_fp32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    stem = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}.gate_proj"
    code_name, scale_name = f"{stem}_codes", f"{stem}_scale"
    with safe_open(str(args.checkpoint / weight_map[code_name]), framework="pt") as reader:
        codes_cpu = reader.get_tensor(code_name)
        scale_cpu = reader.get_tensor(scale_name)
    out_features = codes_cpu.shape[0]
    in_features = codes_cpu.shape[1] * 8 // 3
    device = torch.device("npu:0")
    codes_npu = codes_cpu.to(device)
    scale_npu = scale_cpu.to(device)

    unpacked_cpu = unpack_codes(codes_cpu, in_features, 3)
    unpacked_npu = unpack_codes(codes_npu, in_features, 3).cpu()
    if not torch.equal(unpacked_npu, unpacked_cpu):
        raise AssertionError("W3 NPU unpack differs from CPU")

    weight_cpu = _w2_dequant_fp32(codes_cpu, scale_cpu, out_features, in_features)
    weight_npu = _w2_dequant_fp32(codes_npu, scale_npu, out_features, in_features)
    torch.testing.assert_close(weight_npu.cpu(), weight_cpu, atol=0, rtol=0)
    x_cpu = torch.randn(2, in_features, generator=torch.Generator().manual_seed(53), dtype=torch.float32) * 0.1
    y_cpu = x_cpu @ weight_cpu.T
    y_npu = (x_cpu.to(device) @ weight_npu.T).cpu()
    torch.testing.assert_close(y_npu, y_cpu, atol=5e-3, rtol=5e-3)
    print(
        json.dumps(
            {
                "status": "pass",
                "layer": args.layer,
                "expert": args.expert,
                "packed_shape": list(codes_cpu.shape),
                "max_matmul_abs_error": float((y_npu - y_cpu).abs().max()),
                "npu_peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
            }
        )
    )


if __name__ == "__main__":
    main()
