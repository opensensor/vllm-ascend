# SPDX-License-Identifier: Apache-2.0
"""Startup-only ownership of an optional Ascend PCIe DMA channel.

The current driver ABI grants a reservation, not a device mapping or transfer
engine. Keep the control FD open for the entire lease. Closing it releases an
idle reservation; reset invalidates the lease, and uncertain work is quarantined
by the driver. Never invoke this interface in the per-token path.
"""

import fcntl
import os
import struct
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

DRIVER_NODE = "/dev/ascend_device_dma"
ABI_VERSION = 1
CAP_HOST_ROUTING_ISOLATED = 1 << 0
CAP_SOURCE_REGISTRATION = 1 << 1
MAX_SOURCE_BYTES = 128 << 30
LEASE_STRUCT = struct.Struct("=IIIIQQIIQQQ")
SOURCE_STRUCT = struct.Struct("=IIQQQQIIQQ")
IOC_WRITE = 1
IOC_READ = 2
IOC_SIZE_SHIFT = 16
IOC_DIR_SHIFT = 30
IOC_TYPE_SHIFT = 8
IOC_TYPE = ord("D")
ACQUIRE = (
    ((IOC_READ | IOC_WRITE) << IOC_DIR_SHIFT)
    | (LEASE_STRUCT.size << IOC_SIZE_SHIFT)
    | (IOC_TYPE << IOC_TYPE_SHIFT)
    | 0xD0
)
QUERY = ACQUIRE + 1
RELEASE = (IOC_WRITE << IOC_DIR_SHIFT) | (LEASE_STRUCT.size << IOC_SIZE_SHIFT) | (IOC_TYPE << IOC_TYPE_SHIFT) | 0xD2
SOURCE_REGISTER = ACQUIRE + 3
SOURCE_QUERY = ACQUIRE + 4
SOURCE_RELEASE = RELEASE + 3


@dataclass(frozen=True)
class LeaseInfo:
    device_id: int
    channel: int
    generation: int
    capabilities: int
    sq_depth: int
    cq_depth: int


@dataclass(frozen=True)
class SourceInfo:
    generation: int
    address: int
    length: int
    region_id: int
    page_count: int
    segment_count: int


def _source_request(generation: int, *, address: int = 0, length: int = 0, region_id: int = 0) -> bytearray:
    return bytearray(SOURCE_STRUCT.pack(ABI_VERSION, 0, generation, address, length, region_id, 0, 0, 0, 0))


def _decode_source(reply: bytearray, generation: int, address: int, length: int) -> SourceInfo:
    version, flags, gen, addr, size, region, pages, segments, *reserved = SOURCE_STRUCT.unpack(reply)
    page_size = os.sysconf("SC_PAGE_SIZE")
    expected_pages = ((address % page_size) + length + page_size - 1) // page_size
    if (
        version != ABI_VERSION
        or flags
        or any(reserved)
        or (gen, addr, size) != (generation, address, length)
        or not region
        or pages != expected_pages
        or not 0 < segments <= pages
    ):
        raise RuntimeError("unexpected device DMA source reply")
    return SourceInfo(gen, addr, size, region, pages, segments)


def _request(device_id: int) -> bytearray:
    return bytearray(LEASE_STRUCT.pack(ABI_VERSION, device_id, 0, 0, 0, 0, 0, 0, 0, 0, 0))


def _decode(reply: bytearray, device_id: int) -> LeaseInfo:
    version, device, flags, channel, generation, caps, sq, cq, *reserved = LEASE_STRUCT.unpack(reply)
    if version != ABI_VERSION or device != device_id or flags or any(reserved):
        raise RuntimeError("unexpected device DMA lease reply")
    if not generation or not sq or not cq:
        raise RuntimeError("incomplete device DMA lease reply")
    return LeaseInfo(device, channel, generation, caps, sq, cq)


class DeviceDMALease:
    """One open-file lease. Caller serializes startup/teardown operations."""

    def __init__(self, fd: int, info: LeaseInfo):
        self._fd = fd
        self.info = info
        self.source_info: SourceInfo | None = None

    @classmethod
    def acquire(cls, device_id: int, *, node: str = DRIVER_NODE) -> "DeviceDMALease":
        if isinstance(device_id, bool) or not isinstance(device_id, int) or not 0 <= device_id < 2**32:
            raise ValueError("device_id must be an unsigned 32-bit driver device ID")
        fd = os.open(node, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            reply = _request(device_id)
            fcntl.ioctl(fd, ACQUIRE, reply, True)
            return cls(fd, _decode(reply, device_id))
        except BaseException:
            os.close(fd)
            raise

    def query(self) -> LeaseInfo:
        if self._fd < 0:
            raise RuntimeError("device DMA lease is closed")
        reply = _request(self.info.device_id)
        fcntl.ioctl(self._fd, QUERY, reply, True)
        current = _decode(reply, self.info.device_id)
        if current != self.info:
            raise RuntimeError("device DMA lease changed")
        return current

    def release(self) -> None:
        """Explicitly release; preserve the FD if the driver rejects teardown."""
        if self._fd < 0:
            return
        info = self.info
        request = bytearray(
            LEASE_STRUCT.pack(
                ABI_VERSION,
                info.device_id,
                0,
                info.channel,
                info.generation,
                info.capabilities,
                info.sq_depth,
                info.cq_depth,
                0,
                0,
                0,
            )
        )
        fcntl.ioctl(self._fd, RELEASE, request, True)
        self.close()

    def register_source(self, address: int, length: int) -> SourceInfo:
        """Pin an already populated, immutable host range for device reads.

        The kernel holds long-term page references and charges RLIMIT_MEMLOCK.
        This neither submits DMA nor exposes a device-accessible mapping.
        A malformed successful reply closes the lease to release unknown pins.
        """
        if self._fd < 0:
            raise RuntimeError("device DMA lease is closed")
        if not self.info.capabilities & CAP_SOURCE_REGISTRATION:
            raise RuntimeError("driver lacks source registration")
        if self.source_info is not None:
            raise RuntimeError("a source is already registered")
        if (
            isinstance(address, bool)
            or not isinstance(address, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 < address < 2**64
            or not 0 < length <= MAX_SOURCE_BYTES
            or address + length >= 2**64
        ):
            raise ValueError("source must be a nonempty bounded host address range")
        reply = _source_request(self.info.generation, address=address, length=length)
        fcntl.ioctl(self._fd, SOURCE_REGISTER, reply, True)
        try:
            self.source_info = _decode_source(reply, self.info.generation, address, length)
        except BaseException:
            self.close()
            raise
        return self.source_info

    def query_source(self) -> SourceInfo:
        if self._fd < 0 or self.source_info is None:
            raise RuntimeError("no live registered source")
        info = self.source_info
        reply = _source_request(info.generation, region_id=info.region_id)
        fcntl.ioctl(self._fd, SOURCE_QUERY, reply, True)
        current = _decode_source(reply, info.generation, info.address, info.length)
        if current != info:
            raise RuntimeError("device DMA source changed")
        return current

    def release_source(self) -> None:
        """Unmap/unpin while retaining the idle queue reservation."""
        if self._fd < 0 or self.source_info is None:
            return
        info = self.source_info
        reply = _source_request(info.generation, region_id=info.region_id)
        fcntl.ioctl(self._fd, SOURCE_RELEASE, reply, True)
        self.source_info = None

    def close(self) -> None:
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            os.close(fd)
            self.source_info = None

    def __enter__(self) -> "DeviceDMALease":
        if self._fd < 0:
            raise RuntimeError("device DMA lease is closed")
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@contextmanager
def reserve_devices(device_ids, *, node: str = DRIVER_NODE):
    """Acquire an optional startup group, rolling back partial admission.

    Keep this context around the API/model lifetime, before ACL initialization.
    Ordinary launches pass no IDs and never open the driver. Require the fixed
    host routing capability; older reservation drivers cannot safely coexist
    with the vendor runtime. This does not enable device-initiated transfers.
    """
    ids = tuple(device_ids)
    if any(isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < 2**32 for device in ids):
        raise ValueError("device IDs must be unsigned 32-bit driver device IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate reservation device ID")
    with ExitStack() as stack:
        leases = []
        for device in ids:
            lease = stack.enter_context(DeviceDMALease.acquire(device, node=node))
            if not lease.info.capabilities & CAP_HOST_ROUTING_ISOLATED:
                raise RuntimeError(f"device {device} driver lacks isolated host DMA routing")
            leases.append(lease)
        yield tuple(leases)
