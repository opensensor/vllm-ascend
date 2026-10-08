# SPDX-License-Identifier: Apache-2.0
"""Prepared selected-row copies for the gapped FP16 GLM KDA cache on 310P."""

from dataclasses import dataclass
from math import prod

import torch

MAX_SELECTED_ROWS = 4
DMA_HALF_ELEMENTS = 16
STATE_COPY_CORES = 8


@dataclass(frozen=True)
class CacheGeometry:
    shape: tuple
    stride: int

    @property
    def payload(self):
        return prod(self.shape[1:])

    @property
    def span(self):
        return (self.shape[0] - 1) * self.stride + self.payload


def cache_geometry(cache):
    if cache.dtype != torch.float16 or cache.ndim != 4 or any(size <= 0 for size in cache.shape):
        raise ValueError("state copies require a nonempty rank-four FP16 cache")
    payload = 1
    for size, stride in zip(reversed(cache.shape[1:]), reversed(cache.stride()[1:])):
        if size > 1 and stride != payload:
            raise ValueError("state payload must be contiguous inside each page")
        payload *= size
    if cache.stride(0) < payload or payload % DMA_HALF_ELEMENTS or cache.stride(0) % DMA_HALF_ELEMENTS:
        raise ValueError("state pages require nonoverlapping, DMA-aligned FP16 payloads")
    if cache.data_ptr() % (DMA_HALF_ELEMENTS * cache.element_size()):
        raise ValueError("state payload base must be DMA aligned")
    return CacheGeometry(tuple(cache.shape), cache.stride(0))


class NativeStateRows:
    MAX_SELECTED_ROWS = MAX_SELECTED_ROWS

    def __init__(self, build, namespace, *, kernel_factory=None, launch=None):
        # The CLI probe starts without a device context. Establish it before
        # the bridge calls aclrtGetDevice to load device-owned kernel code.
        self.device = torch.device("npu", torch.npu.current_device())
        # Device selection alone is lazy in torch-npu. A small allocation
        # establishes the ACL context even in a fresh standalone process.
        torch.empty(1, device=self.device)
        if kernel_factory is None:
            kernel_factory = getattr(torch.classes, namespace).Kernel
        self.gather_kernel = kernel_factory(str(build / "glm_state_rows_gather.bin"), "glm_state_rows_gather_v1")
        self.scatter_kernel = kernel_factory(str(build / "glm_state_rows_scatter.bin"), "glm_state_rows_scatter_v1")
        self.launch = launch if launch is not None else getattr(torch.ops, namespace).launch
        self.configs = {}
        self.gathers = self.scatters = 0

    def prepare(self, cache):
        """Prepare scalar launch descriptors while paused, before graph capture."""
        geometry = cache_geometry(cache)
        if cache.device != self.device:
            raise ValueError("cache must belong to the prepared copy device")
        for selected in range(1, MAX_SELECTED_ROWS + 1):
            for dtype in (torch.int32, torch.int64):
                key = (geometry, selected, dtype)
                if key not in self.configs:
                    values = (geometry.shape[0], geometry.stride, geometry.payload, selected, dtype.itemsize)
                    self.configs[key] = torch.tensor(values, dtype=torch.int64, device=self.device)
        return geometry

    def _arguments(self, cache, indices):
        geometry = cache_geometry(cache)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("state slots must be an INT32/INT64 vector")
        if cache.device != self.device or indices.device != self.device:
            raise ValueError("cache and slots must belong to the prepared copy device")
        key = (geometry, max(1, indices.numel()), indices.dtype)
        if key not in self.configs:
            raise ValueError("state copy geometry was not prepared while paused")
        # Preserve storage_offset. This is a contiguous alias of the backing
        # span, not a materialization of [all pages, heads, value, key].
        flat = cache.as_strided((geometry.span,), (1,))
        return geometry, flat, indices.contiguous(), self.configs[key]

    def gather(self, cache, indices, flags):
        geometry, flat, indices, config = self._arguments(cache, indices)
        if flags.dtype != torch.bool or flags.device != self.device or flags.numel() != indices.numel():
            raise ValueError("fresh-state flags must be a same-device BOOL vector matching slots")
        output = torch.empty((indices.numel(), *geometry.shape[1:]), dtype=torch.float16, device=self.device)
        if not indices.numel():
            return output
        self.launch(self.gather_kernel, [flat, indices, flags.contiguous(), output, config], STATE_COPY_CORES)
        self.gathers += 1
        return output

    def scatter(self, cache, indices, values):
        geometry, flat, indices, config = self._arguments(cache, indices)
        if values.dtype != torch.float16 or values.device != self.device:
            raise ValueError("carry writer requires same-device FP16 values")
        if tuple(values.shape) != (indices.numel(), *geometry.shape[1:]):
            raise ValueError("carry writer shape differs from selected cache payloads")
        if not indices.numel():
            return
        # Scheduler-owned prefill slots must be valid and distinct. As with
        # index_put_, duplicate destinations have no defined write order.
        self.launch(self.scatter_kernel, [flat, indices, values.contiguous(), config], STATE_COPY_CORES)
        self.scatters += 1
