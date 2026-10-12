# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous DMA slots using separately pinned vendor buffers.

The 310P host-registration probe failed. This backend deliberately keeps one
local CPU memory copy between native CUDA-pinned and ACL-pinned buffers. Socket
payloads, device-wide barriers, and undocumented peer-pointer casts are absent.
The caller retains external allocations until completed tickets are released.
"""

import ctypes
from dataclasses import dataclass


@dataclass
class Transfer:
    sequence: int
    direction: str
    source: object
    destination: object
    size: int
    source_ready: object
    slot: object
    phase: str = "source_ready"
    released: bool = False

    @property
    def done(self):
        return self.phase == "complete"


class Slot:
    def __init__(self, cuda, ascend, capacity):
        self.resources = []
        self.transfer = None
        try:
            self.cuda_host = self.add(cuda.allocate(capacity, host=True))
            self.ascend_host = self.add(ascend.allocate(capacity, host=True))
            self.cuda_stream = self.add(cuda.stream())
            self.ascend_stream = self.add(ascend.stream())
            self.cuda_event = self.add(cuda.event())
            self.ascend_event = self.add(ascend.event())
        except BaseException:
            self.close()
            raise

    def add(self, value):
        self.resources.append(value)
        return value

    def close(self):
        for value in reversed(self.resources):
            value.close()
        self.resources.clear()


class Bus:
    def __init__(self, cuda, ascend, *, capacity, slots=2, host_copy=ctypes.memmove):
        if cuda.kind != "cuda" or ascend.kind != "ascend":
            raise ValueError("bus requires CUDA and Ascend runtimes")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (capacity, slots)):
            raise ValueError("capacity and slot count must be positive integers")
        self.cuda, self.ascend = cuda, ascend
        self.capacity, self.host_copy, self.sequence = capacity, host_copy, 0
        self.slots, self.closed, self.failed = [], False, False
        try:
            for _ in range(slots):
                self.slots.append(Slot(cuda, ascend, capacity))
        except BaseException:
            self.close()
            raise

    def submit(self, direction, source, destination, size, *, source_ready=None):
        if self.closed or self.failed:
            raise RuntimeError("bus is closed or failed")
        if direction not in ("cuda_to_ascend", "ascend_to_cuda"):
            raise ValueError("invalid bridge direction")
        expected = (self.cuda, self.ascend) if direction == "cuda_to_ascend" else (self.ascend, self.cuda)
        if (
            source.runtime is not expected[0]
            or destination.runtime is not expected[1]
            or source.host
            or destination.host
        ):
            raise ValueError("transfer endpoints must be device allocations in the specified runtimes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= min(self.capacity, source.size, destination.size)
        ):
            raise ValueError("transfer exceeds its allocation or slot capacity")
        _ = source.pointer, destination.pointer  # Fail before occupying a slot.
        for occupied in self.slots:
            active = occupied.transfer
            if active is not None and (
                overlaps(destination, size, active.source, active.size)
                or overlaps(destination, size, active.destination, active.size)
                or overlaps(source, size, active.destination, active.size)
            ):
                raise ValueError("transfer aliases an allocation owned by another ticket")
        if source_ready is not None:
            if source_ready.runtime is not expected[0]:
                raise ValueError("source event belongs to another runtime")
            source_ready.require_recorded()
        slot = next((slot for slot in self.slots if slot.transfer is None), None)
        if slot is None:
            raise BufferError("all bridge slots are owned by unreleased transfers")
        transfer = Transfer(self.sequence, direction, source, destination, size, source_ready, slot)
        self.sequence += 1
        slot.transfer = transfer
        for resource in (source, destination, source_ready):
            if resource is not None:
                resource.leases += 1
        return transfer

    def progress(self):
        if self.closed or self.failed:
            raise RuntimeError("bus is closed or failed")
        try:
            for slot in self.slots:
                if slot.transfer is not None and not slot.transfer.done:
                    self.advance(slot.transfer)
        except BaseException:
            # Retain buffers and tickets: a failed native call may already have
            # queued DMA. Never free or recycle these allocations speculatively.
            self.failed = True
            raise

    def advance(self, transfer):
        slot = transfer.slot
        forward = transfer.direction == "cuda_to_ascend"
        first, second = (self.cuda, self.ascend) if forward else (self.ascend, self.cuda)
        first_host, second_host = (slot.cuda_host, slot.ascend_host) if forward else (slot.ascend_host, slot.cuda_host)
        first_stream, second_stream = (
            (slot.cuda_stream, slot.ascend_stream) if forward else (slot.ascend_stream, slot.cuda_stream)
        )
        first_event, second_event = (
            (slot.cuda_event, slot.ascend_event) if forward else (slot.ascend_event, slot.cuda_event)
        )
        if transfer.phase == "source_ready":
            if transfer.source_ready is not None and not transfer.source_ready.ready():
                return
            first.copy(first_host.pointer, transfer.source.pointer, transfer.size, "d2h", first_stream)
            first_event.record(first_stream)
            transfer.phase = "first_dma"
        elif transfer.phase == "first_dma":
            if not first_event.ready():
                return
            self.host_copy(second_host.pointer, first_host.pointer, transfer.size)
            second.copy(transfer.destination.pointer, second_host.pointer, transfer.size, "h2d", second_stream)
            second_event.record(second_stream)
            transfer.phase = "second_dma"
        elif transfer.phase == "second_dma":
            if second_event.ready():
                transfer.phase = "complete"

    def release(self, transfer):
        if self.failed:
            raise RuntimeError("failed bus retains its DMA resources")
        if transfer.slot not in self.slots or transfer.slot.transfer is not transfer or transfer.released:
            raise ValueError("ticket does not own a live slot in this bus")
        if not transfer.done:
            raise RuntimeError("cannot release a transfer with pending DMA")
        for resource in (transfer.source, transfer.destination, transfer.source_ready):
            if resource is not None:
                resource.leases -= 1
        transfer.slot.transfer, transfer.released = None, True

    def close(self):
        if self.failed or any(slot.transfer is not None for slot in self.slots):
            raise RuntimeError("release completed tickets before closing the bus; failed buses retain resources")
        for slot in reversed(self.slots):
            slot.close()
        self.slots.clear()
        self.closed = True


def overlaps(left, left_size, right, right_size):
    return left.runtime is right.runtime and max(left.pointer, right.pointer) < min(
        left.pointer + left_size, right.pointer + right_size
    )
