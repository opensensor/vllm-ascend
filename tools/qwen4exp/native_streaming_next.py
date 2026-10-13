# SPDX-License-Identifier: Apache-2.0
"""Unqualified v2 projection resource; no device access on import."""

import torch

from tools.qwen4exp.native_prefill import NativeResource, _tensors_on_one_device
from tools.qwen4exp.streaming_next_memory import BLOCKS, GROUP, LANES, MAX_K, N
from tools.qwen4exp.streaming_operands import MAX_EXPERTS, MAX_ROUTES


class NativeStreamingProjectionNext(NativeResource):
    """One paired G128 schedule for gate/up and down; routed decode is separate.

    The kernel trusts stable scheduler-generated group boundaries. Python only
    checks shapes, dtypes and storage metadata; it never reads device values.
    Model, graph, native ordering and thermal gates remain mandatory.
    """

    def __init__(self, kernel, launch, columns_kernel=None):
        super().__init__(kernel, launch)
        self.columns_kernel = columns_kernel

    @property
    def tile_columns(self):
        return N

    def _validate(self, bank, prepared, group_ends):
        if len(prepared) != 4 or prepared[0].ndim != 2 or bank.weight.ndim != 3:
            raise ValueError("four packed operands and a rank-three expert bank required")
        low, high, scales, sums = prepared
        rows, packed_k = low.shape
        experts, outputs, weight_k = bank.weight.shape
        groups = packed_k * 2 // GROUP
        metadata = (bank.weight_scale, bank.weight_offset, bank.weight_sum)
        if (
            not 1 <= rows <= MAX_ROUTES
            or not 1 <= experts <= MAX_EXPERTS
            or outputs < N
            or outputs > MAX_K
            or outputs % N
            or not GROUP <= packed_k * 2 <= MAX_K
            or packed_k * 2 % GROUP
            or weight_k != packed_k
            or high.shape != low.shape
            or any(t.dtype != torch.int8 for t in (low, high, bank.weight))
            or any(t.shape != (rows, groups, LANES) or t.dtype != torch.float32 for t in (scales, sums))
            or any(t.shape != (experts, outputs, groups) or t.dtype != torch.float16 for t in metadata)
            or group_ends.shape != (experts,)
            or group_ends.dtype != torch.int64
        ):
            raise ValueError("unsupported grouped streaming geometry/dtype")
        _tensors_on_one_device((*prepared, bank.weight, *metadata, group_ends), low.device)
        return rows, experts, outputs, packed_k, metadata

    def __call__(self, bank, prepared, group_ends):
        rows, experts, outputs, packed_k, metadata = self._validate(bank, prepared, group_ends)
        low = prepared[0]
        output = torch.empty((rows, outputs), dtype=torch.float16, device=low.device)
        config = self.config((rows, experts, outputs, packed_k * 2, LANES, 0, 1), low.device)
        self.launch(self.kernel, [*prepared, bank.weight, *metadata, group_ends, output, config], BLOCKS)
        return output

    def columns(self, bank, prepared, group_ends, first_tile, tile_count):
        rows, experts, outputs, packed_k, metadata = self._validate(bank, prepared, group_ends)
        if (
            type(first_tile) is not int
            or type(tile_count) is not int
            or first_tile < 0
            or not 1 <= tile_count <= BLOCKS
            or first_tile + tile_count > outputs // N
        ):
            raise ValueError("unsupported streaming column window")
        if self.columns_kernel is None:
            raise RuntimeError("column-window kernel resource is missing")
        low = prepared[0]
        output = torch.empty((rows, tile_count * N), dtype=torch.float16, device=low.device)
        config = self.config((rows, experts, outputs, packed_k * 2, LANES, 0, 1, first_tile, tile_count), low.device)
        self.launch(self.columns_kernel, [*prepared, bank.weight, *metadata, group_ends, output, config], BLOCKS)
        return output
