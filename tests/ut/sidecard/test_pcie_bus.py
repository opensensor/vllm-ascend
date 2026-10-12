# SPDX-License-Identifier: Apache-2.0

import ctypes
import os

import pytest

from tools.sidecard.pcie.bus import Bus
from tools.sidecard.pcie.runtime import Allocation, Event, Runtime, decode_handle


class Memory:
    def __init__(self, runtime, size, host=False):
        self.runtime, self.size, self.host = runtime, size, host
        self.body = ctypes.create_string_buffer(size)
        self.leases, self.closed = 0, False

    @property
    def pointer(self):
        if self.closed:
            raise RuntimeError("allocation is closed")
        return ctypes.addressof(self.body)

    def close(self):
        assert self.leases == 0
        self.closed = True


class Marker:
    def __init__(self, runtime):
        self.runtime, self.complete, self.recorded, self.closed, self.leases = runtime, False, False, False, 0

    def record(self, stream):
        assert stream.runtime is self.runtime
        self.recorded, self.complete = True, False

    def require_recorded(self):
        if not self.recorded or self.closed:
            raise RuntimeError("event unavailable")

    def ready(self):
        self.require_recorded()
        return self.complete

    def close(self):
        assert self.leases == 0
        self.closed = True


class Lane:
    def __init__(self, runtime):
        self.runtime, self.closed = runtime, False

    def close(self):
        self.closed = True


class Backend:
    def __init__(self, kind):
        self.kind, self.events, self.copies, self.resources = kind, [], [], []
        self.fail = False

    def keep(self, resource):
        self.resources.append(resource)
        return resource

    def allocate(self, size, *, host=False):
        return self.keep(Memory(self, size, host))

    def stream(self):
        return self.keep(Lane(self))

    def event(self):
        value = self.keep(Marker(self))
        self.events.append(value)
        return value

    def copy(self, destination, source, size, direction, stream):
        if self.fail:
            raise RuntimeError("native DMA failed")
        assert stream.runtime is self and not stream.closed
        self.copies.append((size, direction))
        ctypes.memmove(destination, source, size)

    def complete(self):
        for event in self.events:
            if event.recorded:
                event.complete = True


def setup(capacity=64):
    cuda, ascend = Backend("cuda"), Backend("ascend")
    return cuda, ascend, Bus(cuda, ascend, capacity=capacity, slots=2)


@pytest.mark.parametrize("direction", ["cuda_to_ascend", "ascend_to_cuda"])
def test_async_bidirectional_slots_preserve_bytes_and_ownership(direction):
    cuda, ascend, bus = setup()
    source_runtime, destination_runtime = (cuda, ascend) if direction == "cuda_to_ascend" else (ascend, cuda)
    sources = [source_runtime.allocate(64) for _ in range(2)]
    destinations = [destination_runtime.allocate(64) for _ in range(2)]
    tickets = []
    for i, source in enumerate(sources):
        ctypes.memset(source.pointer, 17 + i, 64)
        tickets.append(bus.submit(direction, source, destinations[i], 64))
    with pytest.raises(BufferError):
        bus.submit(direction, source_runtime.allocate(64), destination_runtime.allocate(64), 64)
    with pytest.raises(RuntimeError, match="pending DMA"):
        bus.release(tickets[0])
    with pytest.raises(RuntimeError, match="release completed"):
        bus.close()
    bus.progress()
    assert all(ticket.phase == "first_dma" for ticket in tickets)
    bus.progress()
    assert destination_runtime.copies == []  # Source event still incomplete.
    source_runtime.complete()
    bus.progress()
    assert all(ticket.phase == "second_dma" for ticket in tickets)
    bus.progress()
    assert not any(ticket.done for ticket in tickets)
    destination_runtime.complete()
    bus.progress()
    for i, ticket in enumerate(tickets):
        assert ticket.done
        assert ctypes.string_at(destinations[i].pointer, 64) == bytes([17 + i]) * 64
        assert sources[i].leases == destinations[i].leases == 1
        bus.release(ticket)
        assert sources[i].leases == destinations[i].leases == 0
        with pytest.raises(ValueError):
            bus.release(ticket)
    bus.close()
    assert all(resource.closed for slot_runtime in (cuda, ascend) for resource in slot_runtime.resources[:6])


def test_imported_source_event_blocks_only_its_own_slot():
    cuda, ascend, bus = setup()
    ready = cuda.event()
    ready.record(cuda.stream())
    a = bus.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), 64, source_ready=ready)
    b = bus.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), 64)
    bus.progress()
    assert a.phase == "source_ready" and b.phase == "first_dma"
    ready.complete = True
    bus.progress()
    assert a.phase == "first_dma"
    for _ in range(3):
        cuda.complete()
        ascend.complete()
        bus.progress()
    bus.release(a)
    bus.release(b)
    assert ready.leases == 0
    bus.close()


@pytest.mark.parametrize("size", [0, -1, 65, True, 1.5])
def test_bounds_rejected_without_occupying_slots(size):
    cuda, ascend, bus = setup()
    with pytest.raises(ValueError):
        bus.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), size)
    assert all(slot.transfer is None for slot in bus.slots)
    bus.close()


def test_endpoints_source_readiness_and_foreign_ticket_fail_closed():
    cuda, ascend, bus = setup()
    with pytest.raises(ValueError):
        bus.submit("cuda_to_ascend", ascend.allocate(64), cuda.allocate(64), 64)
    with pytest.raises(ValueError):
        bus.submit("cuda_to_ascend", cuda.allocate(64, host=True), ascend.allocate(64), 64)
    with pytest.raises(RuntimeError):
        bus.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), 64, source_ready=cuda.event())
    other = Bus(cuda, ascend, capacity=64)
    ticket = other.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), 64)
    with pytest.raises(ValueError):
        bus.release(ticket)
    bus.close()


def test_native_failure_retains_owned_buffers():
    cuda, ascend, bus = setup()
    ticket = bus.submit("cuda_to_ascend", cuda.allocate(64), ascend.allocate(64), 64)
    cuda.fail = True
    with pytest.raises(RuntimeError, match="native DMA failed"):
        bus.progress()
    assert bus.failed and ticket.source.leases == ticket.destination.leases == 1
    with pytest.raises(RuntimeError):
        bus.release(ticket)
    with pytest.raises(RuntimeError):
        bus.close()
    assert not any(resource.closed for resource in cuda.resources)


def test_overlapping_inflight_writes_and_read_after_write_rejected():
    cuda, ascend, bus = setup()
    source, destination = cuda.allocate(64), ascend.allocate(64)
    bus.submit("cuda_to_ascend", source, destination, 64)
    with pytest.raises(ValueError, match="aliases"):
        bus.submit("cuda_to_ascend", cuda.allocate(64), destination, 64)
    with pytest.raises(ValueError, match="aliases"):
        bus.submit("ascend_to_cuda", destination, cuda.allocate(64), 64)
    # Sharing a read-only source between independent destinations is valid.
    bus.submit("cuda_to_ascend", source, ascend.allocate(64), 64)


def test_native_objects_guard_pending_lifetimes_and_process_owner():
    runtime = object.__new__(Runtime)
    runtime.pid, runtime.thread = os.getpid() + 1, 0
    with pytest.raises(RuntimeError, match="creating process"):
        runtime.check_owner()
    allocation = Allocation(runtime, 1, 64)
    event = Event(runtime, 1)
    allocation.leases = event.leases = 1
    for resource in (allocation, event):
        with pytest.raises(RuntimeError, match="pending bridge"):
            resource.close()


def test_ipc_handle_size_and_encoding_validation():
    assert bytes(decode_handle("ab" * 64)) == bytes([171]) * 64
    for encoded in ("ab" * 63, "zz" * 64):
        with pytest.raises(ValueError):
            decode_handle(encoded)
