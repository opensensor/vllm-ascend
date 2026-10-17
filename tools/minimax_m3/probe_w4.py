# SPDX-License-Identifier: Apache-2.0
"""Measure CPU W4 reconstruction on selected real M3 expert matrices.

This is a bounded format experiment, not a calibrated export or model-quality
gate. It uses the existing 310P signed-code packing contract, but does not
claim that M3 routing or SwiGLU-OAI is supported by that expert kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from tools.deepseek_w2.w2_format import pack_codes, quantize_weight, unpack_codes
from tools.minimax_m3.audit_checkpoint import EXPERT_BLOCK_SIZE, INDEX_FILE

W4_BITS = 4
FP16_MIN_SUBNORMAL = 2**-24


def quantize_chunk(
    weight: torch.Tensor, method: str, *, mse_backend: str = "torch"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed codes, stored FP16 scales, and FP32 reconstructed values."""
    if weight.device.type != "cpu" or not torch.isfinite(weight).all():
        raise ValueError("Expected finite CPU weights")
    if mse_backend not in ("torch", "numba"):
        raise ValueError(f"Unknown CPU MSE backend: {mse_backend}")
    if mse_backend == "numba" and method == "mse":
        # Lazy import: Numba is optional and only needed for CPU conversion.
        from tools.minimax_m3.mse_cpu import quantize_mse_cpu

        codes, scales = quantize_mse_cpu(weight)
    else:
        codes, scales = quantize_weight(weight, n_bits=W4_BITS, method=method)
    stored_scales = scales.clamp_min(FP16_MIN_SUBNORMAL).to(torch.float16)
    if not torch.isfinite(stored_scales).all():
        raise ValueError("Block scales exceed FP16 storage range")
    packed = pack_codes(codes, W4_BITS)
    unpacked = unpack_codes(packed, weight.shape[1], W4_BITS)
    if not torch.equal(unpacked, codes):
        raise AssertionError("Signed W4 code roundtrip failed")
    expanded_scales = stored_scales.float().repeat_interleave(EXPERT_BLOCK_SIZE, 0)
    expanded_scales = expanded_scales.repeat_interleave(EXPERT_BLOCK_SIZE, 1)
    return packed, stored_scales, unpacked.float() * expanded_scales


def probe_tensor(model_path: Path, weight_map: dict, name: str, rows: int, method: str) -> dict:
    if rows <= 0 or rows % EXPERT_BLOCK_SIZE:
        raise ValueError("Chunk rows must be a positive multiple of 32")
    filename = weight_map[name]
    if Path(filename).name != filename:
        raise ValueError("Expected shard basename")
    source_hash = hashlib.sha256()
    packed_hash = hashlib.sha256()
    scales_hash = hashlib.sha256()
    source_norm = reconstructed_norm = dot = squared_error = 0.0
    max_error = 0.0
    packed_bytes = scale_bytes = 0
    with safe_open(model_path / filename, framework="pt", device="cpu") as handle:
        source = handle.get_slice(name)
        shape = source.get_shape()
        if len(shape) != 2 or any(dim % EXPERT_BLOCK_SIZE for dim in shape):
            raise ValueError(f"Expected a 32x32-tiled matrix: {name}")
        for start in range(0, shape[0], rows):
            weight = source[start : start + rows, :].contiguous()
            source_hash.update(weight.view(torch.uint8).numpy().tobytes())
            packed, scales, reconstructed = quantize_chunk(weight, method)
            packed_hash.update(packed.numpy().tobytes())
            scales_hash.update(scales.view(torch.uint8).numpy().tobytes())
            reference = weight.double()
            actual = reconstructed.double()
            error = actual - reference
            source_norm += reference.square().sum().item()
            reconstructed_norm += actual.square().sum().item()
            dot += (reference * actual).sum().item()
            squared_error += error.square().sum().item()
            max_error = max(max_error, error.abs().max().item())
            packed_bytes += packed.numel()
            scale_bytes += scales.numel() * scales.element_size()
    return {
        "tensor": name,
        "shard": filename,
        "shape": shape,
        "source_tensor_sha256": source_hash.hexdigest(),
        "packed_codes_sha256": packed_hash.hexdigest(),
        "fp16_scales_sha256": scales_hash.hexdigest(),
        "packed_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "code_roundtrip_exact": True,
        "weight_nrmse": math.sqrt(squared_error / source_norm) if source_norm else 0.0,
        "weight_cosine": dot / math.sqrt(source_norm * reconstructed_norm)
        if source_norm * reconstructed_norm
        else None,
        "max_abs_weight_error": max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 31, 59])
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--method", choices=("minmax", "mse"), default="minmax")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    index_path = args.model_path / INDEX_FILE
    weight_map = json.loads(index_path.read_text())["weight_map"]
    results = []
    for layer in args.layers:
        for projection in ("w1", "w2", "w3"):
            name = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{args.expert}.{projection}.weight"
            result = probe_tensor(args.model_path, weight_map, name, args.chunk_rows, args.method)
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "validation": "CPU weight reconstruction only; no activation calibration, model accuracy, or NPU execution",
        "model_path": str(args.model_path.resolve()),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "method": args.method,
        "format": "signed W4, 32x32 blocks, codes selected before FP16 scale rounding",
        "chunk_rows": args.chunk_rows,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
