# SPDX-License-Identifier: Apache-2.0
"""Both storage sources must preserve exact FP16 bytes, duplicates and order."""

import fcntl
import itertools
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from tests.ut.qwen38_1m.test_ple_storage import checkpoint
from tools.qwen4exp import ple_sources
from tools.qwen4exp.ple_sources import FileSource, SharedRamSource
from tools.qwen4exp.ple_storage import MAX_LOOKUP_ROWS, register


@pytest.fixture
def source(tmp_path, monkeypatch):
    expected = checkpoint(tmp_path)
    storage = register(tmp_path, index_filename="index.json", shard_count=3)
    monkeypatch.setattr(ple_sources, "available_host_bytes", lambda: 10**9)
    return storage, expected


def test_shared_and_filesystem_match_all_rows_and_duplicates(source):
    storage, expected = source
    ids = [119, 0, 39, 40, 79, 80, *range(120), 3, 3]
    reference = b"".join(expected[row * 320 : (row + 1) * 320] for row in ids)
    with (
        FileSource(storage) as disk,
        SharedRamSource.load(storage, budget_bytes=storage.table_bytes, reserve_bytes=0) as ram,
    ):
        assert disk.read_rows(ids) == ram.read_rows(ids) == reference
    with pytest.raises(RuntimeError, match="closed"):
        disk.read_rows([0])
    with pytest.raises(RuntimeError, match="closed"):
        ram.read_rows([0])


def test_shared_fd_is_immutable_and_survives_owner_close(source):
    storage, expected = source
    owner = SharedRamSource.load(storage, budget_bytes=storage.table_bytes, reserve_bytes=0)
    fd = owner.duplicate_fd()
    try:
        attached = SharedRamSource(storage, fd)
        owner.close()
        with pytest.raises(OSError):
            os.pwrite(fd, b"x", 0)
        with pytest.raises(OSError):
            os.ftruncate(fd, 1)
        assert attached.read_rows([0]) == expected[:320]
        # A second process reads the same object, without serializing the table.
        result = subprocess.run(
            [sys.executable, "-c", "import os,sys;sys.stdout.buffer.write(os.pread(int(sys.argv[1]),320,0))", str(fd)],
            pass_fds=(fd,),
            capture_output=True,
            check=True,
        )
        assert result.stdout == expected[:320]
        attached.close()
    finally:
        owner.close()
        os.close(fd)


@pytest.mark.parametrize("mode", ["budget", "reserve", "available", "pressure"])
def test_memory_admission_rejects_before_publication(source, monkeypatch, mode):
    storage, _ = source
    budget, reserve = storage.table_bytes, 0
    error = ValueError
    if mode == "budget":
        budget -= 1
    elif mode == "reserve":
        reserve = -1
    elif mode == "available":
        monkeypatch.setattr(ple_sources, "available_host_bytes", lambda: storage.table_bytes)
        error = MemoryError
    else:
        values = iter([10**9, 0])
        monkeypatch.setattr(ple_sources, "available_host_bytes", lambda: next(values))
        error = MemoryError
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(error):
        SharedRamSource.load(storage, budget_bytes=budget, reserve_bytes=reserve)
    assert len(os.listdir("/proc/self/fd")) == before


def test_modified_file_is_rejected_but_shared_snapshot_is_unchanged(source):
    storage, expected = source
    with (
        FileSource(storage) as disk,
        SharedRamSource.load(storage, budget_bytes=storage.table_bytes, reserve_bytes=0) as ram,
    ):
        with open(storage.files[0].path, "r+b") as handle:
            handle.seek(storage.shards[0].offset)
            handle.write(b"changed")
        with pytest.raises(RuntimeError, match="changed"):
            disk.read_rows([0])
        assert ram.read_rows([0]) == expected[:320]


@pytest.mark.parametrize("sealed", [False, True])
def test_reject_invalid_attachment(source, sealed):
    storage, _ = source
    fd = os.memfd_create("invalid-ple", os.MFD_ALLOW_SEALING)
    try:
        os.ftruncate(fd, storage.table_bytes if not sealed else 1)
        if sealed:
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, ple_sources.IMMUTABLE_SEALS)
        with pytest.raises(ValueError):
            SharedRamSource(storage, fd)
    finally:
        os.close(fd)


def test_unbounded_row_iterator_is_bounded(source):
    storage, _ = source
    visited = []

    def rows():
        for _ in itertools.count():
            visited.append(0)
            yield 0

    with pytest.raises(ValueError):
        storage.plan(rows())
    assert len(visited) == MAX_LOOKUP_ROWS + 1


@pytest.mark.parametrize("change", [{"rows": 0}, {"row_bytes": 0}, {"file": -1}, {"offset": -1}, {"rows": 41}])
def test_invalid_direct_geometry_is_rejected(source, change):
    storage, _ = source
    with pytest.raises(ValueError):
        replace(storage, shards=(replace(storage.shards[0], **change), *storage.shards[1:]))


def test_short_read_fails(source, monkeypatch):
    storage, _ = source
    with FileSource(storage) as disk:
        monkeypatch.setattr(os, "pread", lambda *args: b"")
        with pytest.raises(RuntimeError, match="short read"):
            disk.read_rows([0])
