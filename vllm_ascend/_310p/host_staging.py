# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host staging with explicit DMA lifetime ownership.

Only the copy completion event is waited on before rewriting host storage.
Device destinations must be consumed on the stream which submitted the copy.
This class does not make an independently reused device output safe.
"""

from collections.abc import Callable

import torch


class PinnedHostStaging:
    def __init__(
        self,
        capacity: tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin_memory: bool = True,
        event_factory: Callable | None = None,
    ) -> None:
        if not capacity or any(size <= 0 for size in capacity):
            raise ValueError("host staging capacity must be positive")
        # Allocate without zero-fill: pinned FP16 zero-fill has failed on this
        # host. Every submitted byte is populated before its DMA is enqueued.
        self.host = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self._event_factory = event_factory
        self._completion = None

    def copy_to(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        if source.device.type != "cpu" or source.dtype != self.host.dtype:
            raise ValueError("host staging requires CPU source in the staging dtype")
        if source.shape != destination.shape or destination.dtype != self.host.dtype:
            raise ValueError("host staging source and destination must match")
        if source.ndim != self.host.ndim or any(size > cap for size, cap in zip(source.shape, self.host.shape)):
            raise ValueError("host staging source exceeds its fixed capacity")
        if self._completion is not None and not self._completion.query():
            # A host write cannot be protected with a device stream wait.
            self._completion.synchronize()
        view = self.host[tuple(slice(0, size) for size in source.shape)]
        view.copy_(source)
        destination.copy_(view, non_blocking=True)
        if self._completion is None:
            self._completion = (self._event_factory or torch.npu.Event)()
        self._completion.record()
        return destination
