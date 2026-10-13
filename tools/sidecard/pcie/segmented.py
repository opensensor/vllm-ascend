# SPDX-License-Identifier: Apache-2.0
"""Larger logical slots using ordered transfers within the unchanged UAPI bound."""

from .peer import MAX_BYTES, KernelPeer

MAX_LOGICAL_BYTES = 2 * MAX_BYTES


def segment_sizes(size):
    if type(size) is not int or not 0 < size <= MAX_LOGICAL_BYTES:
        raise ValueError("segmented DMA exceeds its logical capacity")
    return tuple(min(MAX_BYTES, size - offset) for offset in range(0, size, MAX_BYTES))


class AllocationRegion:
    """Borrow a bounded subrange and retain its complete allocation on every lease."""

    def __init__(self, owner, offset, size):
        if (
            type(offset) is not int
            or type(size) is not int
            or offset < 0
            or not 0 < size <= MAX_BYTES
            or offset + size > owner.size
        ):
            raise ValueError("DMA region exceeds its owner allocation")
        self.owner, self.offset, self.size = owner, offset, size
        self.runtime, self.host = owner.runtime, owner.host
        self.imported = getattr(owner, "imported", False)
        self._leases = 0

    @property
    def pointer(self):
        return self.owner.pointer + self.offset

    @property
    def leases(self):
        return self._leases

    @leases.setter
    def leases(self, value):
        if type(value) is not int or value < 0:
            raise ValueError("invalid region lease count")
        self.owner.leases += value - self._leases
        self._leases = value


class ExportGroup:
    """Keep every region descriptor until all of their native leases retire."""

    def __init__(self, allocation, parts):
        self.allocation, self.parts = allocation, tuple(parts)

    @property
    def fd(self):
        return self.parts[0].fd

    def close(self):
        if any(value.fd is not None and value.allocation.leases != 1 for value in self.parts):
            raise RuntimeError("segmented export is retained by native mappings")
        for value in self.parts:
            value.close()


def export_parts(exported):
    return exported.parts if isinstance(exported, ExportGroup) else (exported,)


def export_allocation(driver, allocation):
    sizes = segment_sizes(allocation.size)
    if len(sizes) == 1:
        return driver.export(allocation)
    if allocation.leases:
        raise ValueError("segmented export requires an unleased owner allocation")
    parts = []
    try:
        for index, size in enumerate(sizes):
            parts.append(driver.export(AllocationRegion(allocation, index * MAX_BYTES, size)))
    except BaseException:
        for value in parts:
            value.close()
        raise
    return ExportGroup(allocation, parts)


def mapping_counts(receipt):
    parts = receipt.get("segments", (receipt,))
    hits = sum(value["gpu_mapping_reused"] for value in parts)
    return len(parts) - hits, hits


class SegmentedPeer:
    """Complete each bounded payload piece before the caller publishes its flag.

    Each region has its own kernel endpoint and at most two persistent GPU maps
    (input and result). An uncertain piece retains the whole logical owner.
    """

    def __init__(self, physical_chip, capacity, *, persistent_gpu_mappings=False, peer_factory=KernelPeer):
        self.capacity = capacity
        self.persistent_gpu_mappings = persistent_gpu_mappings
        self.peers = []
        try:
            for _ in segment_sizes(capacity):
                self.peers.append(peer_factory(physical_chip, persistent_gpu_mappings=persistent_gpu_mappings))
        except BaseException:
            for value in self.peers:
                value.close()
            raise

    @property
    def failed(self):
        return any(value.failed for value in self.peers)

    def copy(self, operation, gpu, npu, guard, size, *, source_ready):
        if self.failed:
            raise RuntimeError("segmented DMA is quarantined")
        if (
            type(size) is not int
            or not 0 < size <= min(self.capacity, gpu.allocation.size, npu.size)
            or npu.size > self.capacity
            or guard.size < npu.size
        ):
            raise ValueError("segmented copy exceeds its logical allocation capacity")
        parts = export_parts(gpu)
        sizes = segment_sizes(gpu.allocation.size)
        if len(parts) != len(sizes) or len(parts) > len(self.peers):
            raise ValueError("segmented GPU registration geometry differs")
        if any(part.allocation.size != expected for part, expected in zip(parts, sizes, strict=True)):
            raise ValueError("segmented GPU region capacity differs")
        if any(part.fd is None for part in parts):
            raise ValueError("segmented GPU region is closed")
        # Validate every descriptor and extent before the first native transfer.
        regions = []
        for index, count in enumerate(segment_sizes(size)):
            region = AllocationRegion(npu, index * MAX_BYTES, min(MAX_BYTES, npu.size - index * MAX_BYTES))
            regions.append((index, count, region))
        receipts = []
        for index, count, region in regions:
            receipts.append(
                self.peers[index].copy(
                    operation,
                    parts[index],
                    region,
                    guard,
                    count,
                    source_ready=source_ready,
                )
            )
        return dict(
            length=size,
            segments=receipts,
            gpu_mapping_reused=int(all(value["gpu_mapping_reused"] for value in receipts)),
        )

    def close(self):
        if self.failed:
            raise RuntimeError("segmented DMA retains its owners after uncertain completion")
        for value in self.peers:
            value.close()


def make_peer(physical_chip, capacity, *, persistent_gpu_mappings=False):
    segment_sizes(capacity)
    if capacity <= MAX_BYTES:
        return KernelPeer(physical_chip, persistent_gpu_mappings=persistent_gpu_mappings)
    return SegmentedPeer(physical_chip, capacity, persistent_gpu_mappings=persistent_gpu_mappings)
