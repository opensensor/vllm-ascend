# SPDX-License-Identifier: Apache-2.0
"""Static native tables and bank geometry for channel-scale accumulation."""

from dataclasses import dataclass

import torch

from .channel_scale_math import OUTPUT_TILE, ChannelWeightBank

ROWS = 32
NZ_N = 16
LIMBS = 2
FACTOR_LAYOUT = "tile_group_channel_v1"


@dataclass(frozen=True)
class NativeWeightBank:
    """Permanent native factors; admitted and uploaded without any transposition."""

    codes: torch.Tensor
    scales: torch.Tensor
    sums: torch.Tensor
    smoothing: torch.Tensor
    group_size: int
    factor_layout: str = FACTOR_LAYOUT

    @property
    def shape(self):
        return (*self.codes.shape[:2], 2 * self.codes.shape[-1])

    def validate(self):
        if self.factor_layout != FACTOR_LAYOUT:
            raise ValueError("unsupported permanent factor layout marker")
        if self.codes.ndim != 3:
            raise ValueError("native channel-scale codes require [E,N,packedK]")
        e, n, k = self.shape
        if (
            not 1 <= e <= 288
            or not 0 < n <= 8192
            or n % OUTPUT_TILE
            or self.group_size not in (256, 512, 1024, 2048, 4096)
            or not 256 <= k <= 4096
            or k % self.group_size
        ):
            raise ValueError("unsupported permanent channel-scale bank geometry")
        expected = (e, n // OUTPUT_TILE, k // self.group_size, OUTPUT_TILE)
        if self.scales.shape != expected or self.sums.shape != expected or self.smoothing.shape != (e, k):
            raise ValueError("permanent factors require native tile/group/channel order")
        tensors = (self.codes, self.scales, self.sums, self.smoothing)
        if tuple(t.dtype for t in tensors) != (torch.int8, torch.float32, torch.int32, torch.float32):
            raise ValueError("permanent bank tensor dtypes differ")
        if any(t.device.type != "cpu" or not t.is_contiguous() for t in tensors):
            raise ValueError("permanent bank admission requires contiguous CPU storage")
        if any(not torch.isfinite(t).all() or not (t > 0).all() for t in (self.scales, self.smoothing)):
            raise ValueError("permanent factors must be finite and positive")
        return self


def logical_bank(bank):
    """CPU sensitivity/reference adapter only; never used by the serving loader."""
    bank.validate()
    e, n, k = bank.shape
    scales = bank.scales.permute(0, 1, 3, 2).reshape(e, n, k // bank.group_size).contiguous()
    sums = bank.sums.permute(0, 1, 3, 2).reshape(e, n, k // bank.group_size).contiguous()
    return ChannelWeightBank(bank.codes, scales, sums, bank.smoothing, bank.group_size).validate()


def product_offsets(rows=ROWS):
    """CPU-built CO1 NZ -> logical-channel rows, including both A8 limbs."""
    if rows not in (32, 64):
        raise ValueError("native product rows must be 32 or 64")
    tables = torch.zeros(rows // NZ_N, 4, rows, OUTPUT_TILE, dtype=torch.int64)
    for block, physical_rows in enumerate(range(NZ_N, rows + 1, NZ_N)):
        for mode, stride in ((0, 1), (2, 2)):
            for limb in range(1 if mode == 0 else LIMBS):
                for row in range(physical_rows // stride):
                    physical_row = row * stride + limb
                    for channel in range(OUTPUT_TILE):
                        physical_channel = channel // 2 + (OUTPUT_TILE // 2 if channel % 2 else 0)
                        index = (
                            physical_channel // NZ_N * physical_rows + physical_row
                        ) * NZ_N + physical_channel % NZ_N
                        tables[block, mode + limb, row, channel] = index * 4
    return tables.to(torch.int32).contiguous()


def native_weight_parameters(bank):
    bank.validate()
    # Frozen bundles have their own class identity. Use the explicit storage
    # protocol, not isinstance(), or native factors would be transposed again.
    if getattr(bank, "factor_layout", None) == FACTOR_LAYOUT:
        e, n, k = bank.shape
        if bank.scales.shape != (e, n // OUTPUT_TILE, k // bank.group_size, OUTPUT_TILE):
            raise ValueError("native factor marker and storage geometry disagree")
        return bank.codes, bank.scales, bank.sums, bank.smoothing
    experts, channels, k = bank.shape
    groups = k // bank.group_size
    # Product Gather restores logical channels before the scale epilogue.
    scales = bank.scales.reshape(experts, channels // OUTPUT_TILE, OUTPUT_TILE, groups).permute(0, 1, 3, 2).contiguous()
    sums = bank.sums.reshape(experts, channels // OUTPUT_TILE, OUTPUT_TILE, groups).permute(0, 1, 3, 2).contiguous()
    return bank.codes, scales, sums, bank.smoothing


def projection_descriptor(tokens, top_k, bank, bits, *, original_tokens):
    if not isinstance(bank, ChannelWeightBank):
        raise ValueError("projection requires explicitly prepared channel-scale bank")
    if type(tokens) is not int or tokens < 1 or type(top_k) is not int or not 1 <= top_k <= 8 or bits not in (4, 8):
        raise ValueError("unsupported channel-scale request geometry")
    bank.validate()
    experts, channels, k = bank.shape
    return (tokens, top_k, experts, channels, k, bank.group_size, bits, int(original_tokens))


def quantization_offsets():
    """Reduction maxima and broadcast indices, prepared once without scalar UB writes."""
    return torch.cat((torch.arange(64, dtype=torch.int64) * 8, torch.zeros(8, dtype=torch.int64))).to(torch.int32)
