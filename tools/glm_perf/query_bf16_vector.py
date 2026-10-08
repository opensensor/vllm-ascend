# SPDX-License-Identifier: Apache-2.0
"""Experimental vector conversion; CPU oracle and explicitly prepared launches."""

from pathlib import Path

import torch

DMA_HALF_ELEMENTS = 16
VECTOR_CORES = 8
VECTOR_TILE = 1024


def fp16_bits_to_bf16(bits):
    """CPU oracle, preserving the existing converter's signed canonical NaNs."""
    if bits.device.type != "cpu" or bits.dtype != torch.int16:
        raise ValueError("oracle requires CPU INT16 FP16 storage")
    values = bits.view(torch.float16).float()
    words = values.view(torch.int32).long() & 0xFFFFFFFF
    rounded = (words + 0x7FFF + ((words >> 16) & 1)) >> 16
    raw = bits.long() & 0xFFFF
    nan = (raw & 0x7C00 == 0x7C00) & (raw & 0x03FF != 0)
    return torch.where(nan, (raw & 0x8000) | 0x7FC0, rounded).to(torch.int16)


class NativeQueryVector:
    """For explicit diagnostic gates; never selected by a model automatically."""

    def __init__(self, build, namespace, *, kernel_factory=None, launch=None):
        self.device = torch.device("npu", torch.npu.current_device())
        torch.empty(1, device=self.device)
        factory = kernel_factory if kernel_factory is not None else getattr(torch.classes, namespace).Kernel
        self.kernel = factory(str(Path(build) / "glm_query_bf16_vector.bin"), "glm_query_bf16_vector_v1")
        self.launch = launch if launch is not None else getattr(torch.ops, namespace).launch
        self.configs = {}

    def prepare_counts(self, counts):
        counts = list(dict.fromkeys(counts))
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("query vector descriptors require positive integer counts")
        missing = [count for count in counts if count not in self.configs]
        if missing:
            backing = torch.tensor([[count] for count in missing], dtype=torch.int64, device=self.device)
            self.configs.update((count, backing[index]) for index, count in enumerate(missing))

    def __call__(self, value):
        if value.device != self.device or value.dtype != torch.float16 or not value.is_contiguous():
            raise ValueError("vector query conversion requires same-device contiguous FP16")
        count = value.numel()
        if count and count not in self.configs:
            raise ValueError("query vector count was not prepared outside graph capture")
        padded = (count + DMA_HALF_ELEMENTS - 1) // DMA_HALF_ELEMENTS * DMA_HALF_ELEMENTS
        backing = torch.empty(padded, device=self.device, dtype=torch.bfloat16)
        if count:
            cores = min(VECTOR_CORES, (count + VECTOR_TILE - 1) // VECTOR_TILE)
            self.launch(self.kernel, [value.view(-1), backing, self.configs[count]], cores)
        return backing[:count].view(value.shape)
