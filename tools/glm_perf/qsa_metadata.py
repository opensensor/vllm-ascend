# SPDX-License-Identifier: Apache-2.0
"""Prepared fused sparse-attention metadata for GLM on Ascend 310P."""

from dataclasses import dataclass

import torch

DMA_ELEMENTS = 8
METADATA_CORES = 8
POOL_SIZE = 4


def aligned(count):
    return (count + DMA_ELEMENTS - 1) // DMA_ELEMENTS * DMA_ELEMENTS


@dataclass(frozen=True)
class Geometry:
    rows: int
    requests: int
    budget: int
    token_start: int
    ids_row_stride: int
    ids_column_stride: int
    position_stride: int
    position_bytes: int
    table_width: int
    table_row_stride: int
    table_column_stride: int
    split: int

    def __post_init__(self):
        values = self.values()
        if any(value <= 0 for index, value in enumerate(values) if index != 3) or self.token_start < 0:
            raise ValueError("QSA metadata requires positive dimensions and nonnegative token start")
        if self.position_bytes not in (4, 8) or self.split not in (1, 20):
            raise ValueError("only INT32/64 positions and 32/640-token caches are qualified")
        if self.ids_row_stride < (self.budget * POOL_SIZE - 1) * self.ids_column_stride + 1:
            raise ValueError("selected-token rows overlap")
        if self.table_row_stride < (self.table_width - 1) * self.table_column_stride + 1:
            raise ValueError("block-table rows overlap")

    def values(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__) + (aligned(self.rows),)

    @property
    def table_columns(self):
        return (self.table_width + self.split - 1) // self.split


def geometry(ids, positions, table, rows, token_start, budget, block_size):
    if ids.ndim != 2 or ids.dtype != torch.int32 or table.ndim != 2 or table.dtype != torch.int32:
        raise ValueError("QSA selected tokens and block table must be INT32 matrices")
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA positions must be an INT32/64 vector")
    if rows <= 0 or token_start < 0 or token_start + rows > ids.shape[0]:
        raise ValueError("selected-token range is outside its backing rows")
    if rows > positions.numel() or budget <= 0 or budget * POOL_SIZE > ids.shape[1]:
        raise ValueError("positions or selected-token columns do not cover the requested plan")
    if block_size not in (32, 640):
        raise ValueError("unqualified cache block size")
    return Geometry(
        rows,
        table.shape[0],
        budget,
        token_start,
        *ids.stride(),
        positions.stride(0),
        positions.element_size(),
        table.shape[1],
        *table.stride(),
        block_size // 32,
    )


def backing_span(tensor):
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return tensor.as_strided((span,), (1,))


class NativeQsaMetadata:
    def __init__(self, build, namespace, *, kernel_factory=None, launch=None):
        self.device = torch.device("npu", torch.npu.current_device())
        torch.empty(1, device=self.device)
        factory = kernel_factory if kernel_factory is not None else getattr(torch.classes, namespace).Kernel
        self.kernel = factory(str(build / "glm_qsa_metadata.bin"), "glm_qsa_metadata_v1")
        self.launch = launch if launch is not None else getattr(torch.ops, namespace).launch
        self.configs = {}
        self.calls = 0
        self.geometries = {}
        self.fallbacks = 0

    def prepare(self, cases):
        """One descriptor transfer while paused, outside every capture pool."""
        missing = list(dict.fromkeys(case for case in cases if case not in self.configs))
        if missing:
            backing = torch.tensor([case.values() for case in missing], dtype=torch.int64, device=self.device)
            self.configs.update((case, backing[index]) for index, case in enumerate(missing))

    def available(self, ids, positions, table, *, rows, token_start, budget, block_size):
        """Check host geometry only; never allocate a descriptor in capture."""
        if any(t.device != self.device for t in (ids, positions, table)):
            return False
        try:
            case = geometry(ids, positions, table, rows, token_start, budget, block_size)
        except ValueError:
            return False
        return case in self.configs

    def plan(self, ids, positions, table, *, rows, token_start, budget, block_size):
        case = geometry(ids, positions, table, rows, token_start, budget, block_size)
        if any(t.device != self.device for t in (ids, positions, table)):
            raise ValueError("metadata inputs must belong to the prepared NPU")
        if case not in self.configs:
            raise ValueError("QSA layout was not prepared outside graph capture: " + str(case))
        counts = (rows * budget, 3 * aligned(rows), table.shape[0] * case.table_columns)
        output = [torch.empty(aligned(count), dtype=torch.int32, device=self.device) for count in counts]
        self.launch(
            self.kernel,
            [backing_span(t) for t in (ids, positions, table)] + output + [self.configs[case]],
            METADATA_CORES,
        )
        self.calls += 1
        key = f"{rows}:{table.shape[0]}:{token_start}:{positions.element_size()}:{table.shape[1]}"
        self.geometries[key] = self.geometries.get(key, 0) + 1
        groups = output[0][: counts[0]].view(rows, budget)
        metadata = output[1].view(3, aligned(rows))[:, :rows]
        logical_table = output[2][: counts[2]].view(table.shape[0], case.table_columns)
        return (groups, *metadata.unbind(0)), logical_table
