# SPDX-License-Identifier: Apache-2.0
"""Experimental packed-weight INT8 prefill; never admitted by importing."""

import torch

from tools.qwen4exp.native_prefill import NativeGroupedPrefill

MINIMUM_INT8_ROWS = 128


def signed_nibble_lookup():
    """Immutable pairs of signed nibbles, copied as FP32 bit patterns on-chip."""
    byte = torch.arange(256, dtype=torch.int32, device="cpu")
    low, high = byte & 15, byte >> 4
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=1).half().contiguous()


class NativeHybridPrefill(NativeGroupedPrefill):
    def __init__(self, small_kernel, large_kernel, launch, device, unpack_kernel=None):
        super().__init__(small_kernel, launch)
        self.large_kernel = large_kernel
        self.unpack_kernel = unpack_kernel
        self.lookup = signed_nibble_lookup().to(device)

    def __call__(self, bank, prepared, group_ends):
        rows, experts, outputs, inputs, operands = self.validate_inputs(bank, prepared, group_ends)
        low, high, scale, _ = prepared
        if self.lookup.device != low.device:
            raise ValueError("hybrid lookup must share the operand device")
        output = torch.empty((rows, outputs), dtype=torch.float16, device=low.device)
        small_config = self.config((rows, experts, outputs, inputs, 8, 0, 1), low.device)
        large_config = self.config((rows, inputs, outputs, experts, MINIMUM_INT8_ROWS), low.device)
        # Disjoint rows, ordered on the current stream. The small partition
        # zeros peer padding; the large partition writes only >=128-row groups.
        self.launch(self.kernel, [*operands, output, small_config], 8)
        if self.unpack_kernel is not None:
            unpacked = torch.empty((rows, inputs), dtype=torch.int8, device=low.device)
            unpack_config = self.config((low.numel(),), low.device)
            self.launch(self.unpack_kernel, [low, high, self.lookup, unpacked, unpack_config], 8)
            low = unpacked
        self.launch(
            self.large_kernel,
            [
                low,
                high,
                scale,
                bank.weight,
                bank.weight_scale,
                bank.weight_offset,
                group_ends,
                self.lookup,
                output,
                large_config,
            ],
            8,
        )
        return output
