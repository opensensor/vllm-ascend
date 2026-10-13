# SPDX-License-Identifier: Apache-2.0
"""Bounded subrange geometry, full-owner retention, and ordered split DMA."""

import os
import socket
from types import SimpleNamespace

import pytest

from tools.sidecard import completion_protocol
from tools.sidecard.completion_protocol import receive_result, send_fd
from tools.sidecard.pcie.peer import MAX_BYTES
from tools.sidecard.pcie.segmented import (
    AllocationRegion,
    ExportGroup,
    SegmentedPeer,
    export_allocation,
    mapping_counts,
    segment_sizes,
)


def owner(size=2 * MAX_BYTES, *, kind="ascend"):
    return SimpleNamespace(
        size=size,
        pointer=123456,
        leases=0,
        host=False,
        imported=False,
        runtime=SimpleNamespace(kind=kind, check_owner=lambda: None),
    )


@pytest.mark.parametrize("size", [1, MAX_BYTES, MAX_BYTES + 1, 2 * MAX_BYTES])
def test_logical_size_splits_exactly_within_native_bounds(size):
    pieces = segment_sizes(size)
    assert sum(pieces) == size and all(0 < piece <= MAX_BYTES for piece in pieces)


@pytest.mark.parametrize("size", [0, True, "16M", 1.0, 2 * MAX_BYTES + 1])
def test_invalid_logical_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="capacity"):
        segment_sizes(size)


def test_region_leases_retain_the_full_owner_independently():
    whole = owner(kind="cuda")
    left, right = AllocationRegion(whole, 0, MAX_BYTES), AllocationRegion(whole, MAX_BYTES, MAX_BYTES)
    left.leases += 1
    right.leases += 1
    assert whole.leases == 2 and left.leases == right.leases == 1
    assert right.pointer == whole.pointer + MAX_BYTES
    left.leases -= 1
    assert whole.leases == 1 and right.leases == 1
    right.leases -= 1
    assert whole.leases == 0
    with pytest.raises(ValueError, match="lease"):
        right.leases = -1


@pytest.mark.parametrize("offset,size", [(-1, 1), (True, 1), (0, 0), (0, MAX_BYTES + 1), (2 * MAX_BYTES, 1)])
def test_region_cannot_escape_owner(offset, size):
    with pytest.raises(ValueError, match="owner"):
        AllocationRegion(owner(), offset, size)


class Export:
    def __init__(self, allocation):
        self.allocation, self.fd = allocation, 1
        allocation.leases += 1

    def close(self):
        if self.fd is not None:
            assert self.allocation.leases == 1
            self.allocation.leases -= 1
            self.fd = None


@pytest.mark.parametrize("fail", [False, True])
def test_export_regions_release_all_leases_if_registration_fails(fail):
    whole = owner(kind="cuda")
    parts = []

    def export(region):
        if fail and parts:
            raise RuntimeError("second registration rejected")
        result = Export(region)
        parts.append(result)
        return result

    driver = SimpleNamespace(export=export)
    if fail:
        with pytest.raises(RuntimeError, match="second registration"):
            export_allocation(driver, whole)
        assert whole.leases == 0 and parts[0].fd is None
    else:
        result = export_allocation(driver, whole)
        assert len(result.parts) == 2 and whole.leases == 2
        parts[1].allocation.leases += 1
        with pytest.raises(RuntimeError, match="retained"):
            result.close()
        assert all(part.fd is not None for part in parts)
        parts[1].allocation.leases -= 1
        result.close()
        result.close()
        assert whole.leases == 0


def peer_fixture(*, fail_second=False):
    calls = []
    peers = []

    class Peer:
        failed = False

        def __init__(self, chip, **kwargs):
            self.index = len(peers)
            self.retained = None
            peers.append(self)

        def copy(self, operation, gpu, npu, guard, size, **kwargs):
            calls.append((self.index, operation, gpu.allocation.size, npu.pointer, npu.size, size))
            if fail_second and self.index == 1:
                self.failed = True
                npu.leases += 1
                self.retained = npu
                raise RuntimeError("uncertain native completion")
            return dict(length=size, gpu_mapping_reused=self.index)

        def close(self):
            calls.append(("closed", self.index))

    peer = SegmentedPeer(2, 2 * MAX_BYTES, peer_factory=Peer)
    gpu = ExportGroup(
        owner(kind="cuda"), [SimpleNamespace(allocation=owner(MAX_BYTES, kind="cuda"), fd=1) for _ in range(2)]
    )
    return peer, gpu, owner(), owner(), calls, peers


@pytest.mark.parametrize("size", [1, MAX_BYTES, MAX_BYTES + 1, 2 * MAX_BYTES])
def test_split_payload_completes_each_piece_before_publication(size):
    peer, gpu, npu, guard, calls, _ = peer_fixture()
    receipt = peer.copy(4, gpu, npu, guard, size, source_ready=object())
    calls.append(("publish",))
    pieces = segment_sizes(size)
    assert len(calls) == len(pieces) + 1
    for index, count in enumerate(pieces):
        assert calls[index] == (index, 4, MAX_BYTES, npu.pointer + index * MAX_BYTES, MAX_BYTES, count)
    assert sum(row["length"] for row in receipt["segments"]) == size
    assert mapping_counts(receipt) == ((1, 0) if len(pieces) == 1 else (1, 1))


def test_uncertain_second_piece_retains_full_npu_owner_and_blocks_release():
    peer, gpu, npu, guard, calls, peers = peer_fixture(fail_second=True)
    with pytest.raises(RuntimeError, match="uncertain"):
        peer.copy(4, gpu, npu, guard, 2 * MAX_BYTES, source_ready=object())
    assert peer.failed and npu.leases == 1 and peers[1].retained.owner is npu
    with pytest.raises(RuntimeError, match="retains"):
        peer.close()
    with pytest.raises(RuntimeError, match="quarantined"):
        peer.copy(4, gpu, npu, guard, 1, source_ready=object())
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["closed", "capacity", "guard", "extent"])
def test_second_piece_geometry_is_validated_before_first_native_submission(change):
    peer, gpu, npu, guard, calls, _ = peer_fixture()
    if change == "closed":
        gpu.parts[1].fd = None
    elif change == "capacity":
        gpu.parts[1].allocation.size -= 1
    elif change == "guard":
        guard.size -= 1
    else:
        npu.size += 1
    with pytest.raises(ValueError):
        peer.copy(4, gpu, npu, guard, 2 * MAX_BYTES, source_ready=object())
    assert not calls


def test_foreign_result_registration_receives_and_closes_each_region(tmp_path):
    path = tmp_path / "descriptor"
    path.write_bytes(b"fixture")
    fd = os.open(path, os.O_RDONLY)
    left, right = socket.socketpair()
    with left, right:
        for _ in range(2):
            send_fd(left, fd)
        result = receive_result(owner().runtime, right, 2 * MAX_BYTES)
    assert result.allocation.size == 2 * MAX_BYTES
    assert [part.allocation.size for part in result.parts] == [MAX_BYTES, MAX_BYTES]
    descriptors = [part.fd for part in result.parts]
    result.parts[1].allocation.leases += 1
    with pytest.raises(RuntimeError, match="retained"):
        result.close()
    for duplicate in descriptors:
        os.fstat(duplicate)
    result.parts[1].allocation.leases -= 1
    result.close()
    result.close()
    for duplicate in descriptors:
        with pytest.raises(OSError):
            os.fstat(duplicate)
    os.close(fd)


def test_incomplete_foreign_registration_releases_received_region(tmp_path, monkeypatch):
    path = tmp_path / "descriptor"
    path.write_bytes(b"fixture")
    original = os.open(path, os.O_RDONLY)
    duplicate = os.dup(original)
    received = []

    def receive(connection):
        if not received:
            received.append(duplicate)
            return duplicate
        raise ValueError("missing second descriptor")

    monkeypatch.setattr(completion_protocol, "receive_fd", receive)
    with pytest.raises(ValueError, match="missing second"):
        receive_result(owner().runtime, None, 2 * MAX_BYTES)
    with pytest.raises(OSError):
        os.fstat(duplicate)
    os.fstat(original)
    os.close(original)
