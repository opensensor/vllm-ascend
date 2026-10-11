# SPDX-License-Identifier: Apache-2.0
"""Startup-only conversion from qualified packed W4 to per-channel INT8."""

import hashlib
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.qwen4exp.benchmark_int8_requant_projection_310 import requantize_weight
from tools.qwen4exp.native_channel_int8 import tile_channel_weights


def channel_bank(bank, *, progress=None, cache_path=None, target_device=None):
    target_device = bank.weight.device if target_device is None else target_device
    experts, outputs, half_inputs = bank.weight.shape
    groups = half_inputs // 64
    packed = bank.weight.cpu().reshape(experts, outputs // 16, groups, 2, 16, 32)
    packed = packed.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(experts * outputs, half_inputs)
    fields = []
    for field in (bank.weight_scale, bank.weight_offset):
        decoded = field.cpu().reshape(experts, outputs // 16, groups, 16)
        fields.append(decoded.permute(0, 1, 3, 2).contiguous().reshape(experts * outputs, groups))
    offsets = fields[1].to(torch.int8)
    if not torch.equal(offsets.to(fields[1].dtype), fields[1]):
        raise ValueError("packed channel recipe requires integer offsets in INT8 range")
    source_digest = hashlib.sha256()
    for tensor in (packed, *fields):
        source_digest.update(tensor.numpy().tobytes())
    source_sha256 = source_digest.hexdigest()
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        with safe_open(cache_path, framework="pt", device="cpu") as saved:
            metadata = saved.metadata() or {}
        if metadata.get("source_sha256") != source_sha256 or metadata.get("layout") != "channel_int8_n16_k32_v1":
            raise ValueError("cached INT8 bank source or layout differs")
        tensors = load_file(cache_path)
        if set(tensors) != {"weight", "scale"}:
            raise ValueError("cached INT8 tensor inventory differs")
        quantized, scales = tensors["weight"], tensors["scale"]
        if quantized.shape != (experts, outputs, half_inputs * 2) or quantized.dtype != torch.int8:
            raise ValueError("cached INT8 weight geometry differs")
        if scales.shape != (experts, outputs) or scales.dtype != torch.float32:
            raise ValueError("cached INT8 scale geometry differs")
        for name, tensor in tensors.items():
            if hashlib.sha256(tensor.numpy().tobytes()).hexdigest() != metadata.get(name + "_sha256"):
                raise ValueError("cached INT8 tensor digest differs")
        return quantized.to(target_device), scales.to(target_device), float(metadata["relative_error"])

    # An expert-sized FP32 temporary replaces the original whole-bank expansion.
    # Per-row maxima and quantization are independent across output channels.
    quantized = torch.empty((experts * outputs, half_inputs * 2), dtype=torch.int8)
    scales = torch.empty(experts * outputs, dtype=torch.float32)
    squared_error = 0.0
    squared_weight = 0.0
    for expert in range(experts):
        start, stop = expert * outputs, (expert + 1) * outputs
        dense, codes, channel_scale = requantize_weight(packed[start:stop], fields[0][start:stop], offsets[start:stop])
        quantized[start:stop].copy_(codes)
        scales[start:stop].copy_(channel_scale)
        error = codes.float() * channel_scale.unsqueeze(1) - dense
        squared_error += float(error.square().double().sum())
        squared_weight += float(dense.square().double().sum())
        if progress is not None and ((expert + 1) % 32 == 0 or expert + 1 == experts):
            progress(expert + 1, experts)
    relative_error = (squared_error / squared_weight) ** 0.5 if squared_weight else 0.0
    quantized = tile_channel_weights(quantized.reshape(experts, outputs, half_inputs * 2))
    scales = scales.reshape(experts, outputs)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"weight": quantized, "scale": scales}
        metadata = {
            "source_sha256": source_sha256,
            "layout": "channel_int8_n16_k32_v1",
            "relative_error": str(relative_error),
            **{
                name + "_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                for name, tensor in tensors.items()
            },
        }
        temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(cache_path)
    quantized, scales = quantized.to(target_device), scales.to(target_device)
    return quantized, scales, relative_error
