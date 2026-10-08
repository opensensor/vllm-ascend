# SPDX-License-Identifier: Apache-2.0
"""Prepared vector conversions retaining the permanent scalar bit contract."""

from pathlib import Path

import torch

MODES = (0, 1, 4, 5)
TILE = 1024
VECTOR_CORES = 8


def reference(bits, mode):
    """CPU storage oracle; signed canonical NaNs match the scalar converter."""
    if bits.device.type != "cpu" or mode not in MODES:
        raise ValueError("oracle requires CPU storage and a supported mode")
    if mode == 1:
        if bits.dtype != torch.int16:
            raise ValueError("BF16 decode requires INT16 storage")
        return bits.to(torch.int32) << 16
    if bits.dtype != torch.int32:
        raise ValueError("BF16 round requires INT32 storage")
    words = bits.long() & 0xFFFFFFFF
    rounded = (words + 0x7FFF + ((words >> 16) & 1)) >> 16
    nan = (words & 0x7F800000 == 0x7F800000) & (words & 0x007FFFFF != 0)
    rounded = torch.where(nan, (words >> 16 & 0x8000) | 0x7FC0, rounded).to(torch.int16)
    if mode == 0:
        return rounded
    wide = rounded.to(torch.int32) << 16
    if mode == 4:
        return wide
    half = wide.view(torch.float32).half().view(torch.int16)
    return torch.where(nan, (words >> 16 & 0x8000) | 0x7E00, half.long() & 0xFFFF).to(torch.int16)


class NativeBf16Vector:
    def __init__(self, build, namespace, *, kernel_factory=None, launch=None):
        self.device = torch.device("npu", torch.npu.current_device())
        factory = kernel_factory if kernel_factory is not None else getattr(torch.classes, namespace).Kernel
        self.kernel = factory(str(Path(build) / "glm_bf16_vector.bin"), "glm_bf16_vector_v1")
        self.launch = launch if launch is not None else getattr(torch.ops, namespace).launch
        self.configs = {}
        self.calls = {}

    def prepare_counts(self, counts):
        counts = list(dict.fromkeys(counts))
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("vector BF16 counts must be positive integers")
        keys = [(count, mode) for count in counts for mode in MODES if (count, mode) not in self.configs]
        if keys:
            backing = torch.tensor(keys, dtype=torch.int64, device=self.device)
            self.configs.update((key, backing[index]) for index, key in enumerate(keys))

    def convert(self, source, dtype, mode):
        expected_source = torch.bfloat16 if mode == 1 else torch.float32
        expected_output = {0: torch.bfloat16, 1: torch.float32, 4: torch.float32, 5: torch.float16}.get(mode)
        if (
            source.device != self.device
            or source.dtype != expected_source
            or dtype != expected_output
            or not source.is_contiguous()
        ):
            raise ValueError("vector BF16 requires qualified mode, precision and contiguous same-device input")
        count = source.numel()
        if count and (count, mode) not in self.configs:
            raise ValueError("vector BF16 descriptor was not prepared outside capture")
        alignment = 32 // dtype.itemsize
        padded = (count + alignment - 1) // alignment * alignment
        backing = torch.empty(padded, dtype=dtype, device=self.device)
        if count:
            self.launch(
                self.kernel,
                [source.view(-1), backing, self.configs[count, mode]],
                min(VECTOR_CORES, (count + TILE - 1) // TILE),
            )
            self.calls[mode] = self.calls.get(mode, 0) + 1
        return backing[:count].view(source.shape)
