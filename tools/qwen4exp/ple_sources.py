# SPDX-License-Identifier: Apache-2.0
"""Cold admission and CPU reference readers for the two PLE source backends.

Shared RAM uses one sealed memfd, not one table allocation per NPU rank. Its
pages are populated but can still be swapped until the native DMA layer pins
them. Neither this module nor mlock constitutes device-initiated DMA support.
File reads use filesystem offsets and never expose raw NVMe LBAs.
"""

import fcntl
import mmap
import os
from contextlib import ExitStack
from pathlib import Path

from tools.qwen4exp.ple_storage import integer

COPY_CHUNK_BYTES = 8 * 1024 * 1024
IMMUTABLE_SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


def available_host_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host MemAvailable is unavailable")


def read_exact(fd, length, offset):
    chunks = []
    while length:
        data = os.pread(fd, length, offset)
        if not data:
            raise RuntimeError("short read from admitted PLE source")
        chunks.append(data)
        length -= len(data)
        offset += len(data)
    return b"".join(chunks)


def write_exact(fd, data, offset):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise RuntimeError("short write while building shared PLE table")
        offset += written
        view = view[written:]


class FileSource:
    """Filesystem-aware reference reader; CPU executes the page plan."""

    def __init__(self, storage):
        self.storage = storage
        self._files = []
        try:
            for record in storage.files:
                fd = os.open(record.path, os.O_RDONLY | os.O_CLOEXEC)
                self._files.append(fd)
                record.verify(fd)
        except BaseException:
            self.close()
            raise

    def verify(self):
        if len(self._files) != len(self.storage.files):
            raise RuntimeError("PLE source is closed")
        for record, fd in zip(self.storage.files, self._files):
            record.verify(fd)

    def read_rows(self, row_ids):
        self.verify()
        plan = self.storage.plan(row_ids)
        pages = [read_exact(self._files[item.file], item.length, item.offset) for item in plan.reads]
        row_bytes = self.storage.shards[0].row_bytes
        output = bytearray(plan.useful_bytes)
        for item in plan.slices:
            start = item.destination_row * row_bytes + item.destination_offset
            output[start : start + item.length] = pages[item.read][
                item.source_offset : item.source_offset + item.length
            ]
        self.verify()
        return bytes(output)

    def close(self):
        while self._files:
            os.close(self._files.pop())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SharedRamSource:
    """Immutable shared table with explicit cold-path memory admission.

    Pass duplicate_fd() through SCM_RIGHTS or process inheritance to attach
    ranks to the same physical pages. Descriptors are process-local integers,
    never serializable handles. No transport may outlive its own mapping/FD.
    """

    def __init__(self, storage, fd):
        self.storage = storage
        self._fd = None
        self._mapping = None
        if os.fstat(fd).st_size != storage.table_bytes:
            raise ValueError("shared PLE table size does not match admission")
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & IMMUTABLE_SEALS != IMMUTABLE_SEALS:
            raise ValueError("shared PLE source must be immutable and size-sealed")
        owned = os.dup(fd)
        try:
            self._mapping = mmap.mmap(owned, storage.table_bytes, access=mmap.ACCESS_READ)
        except BaseException:
            os.close(owned)
            raise
        self._fd = owned

    @classmethod
    def load(cls, storage, *, budget_bytes, reserve_bytes):
        budget_bytes, reserve_bytes = integer(budget_bytes), integer(reserve_bytes)
        if budget_bytes < storage.table_bytes or reserve_bytes < 0:
            raise ValueError("shared PLE table exceeds budget or reserve is invalid")
        # Include bounded temporary read buffers; no allocation scales by ranks.
        temporary = 2 * min(COPY_CHUNK_BYTES, storage.table_bytes)
        if storage.table_bytes + temporary + reserve_bytes > available_host_bytes():
            raise MemoryError("insufficient available RAM for shared PLE table and reserve")
        with ExitStack() as stack:
            source = stack.enter_context(FileSource(storage))
            fd = os.memfd_create("qfn-ple-shared", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            stack.callback(os.close, fd)
            os.ftruncate(fd, storage.table_bytes)
            destination = 0
            for shard in storage.shards:
                remaining, offset = shard.rows * shard.row_bytes, shard.offset
                while remaining:
                    count = min(COPY_CHUNK_BYTES, remaining)
                    if reserve_bytes + 2 * count > available_host_bytes():
                        raise MemoryError("host reserve exhausted during shared PLE admission")
                    data = read_exact(source._files[shard.file], count, offset)
                    write_exact(fd, data, destination)
                    destination += count
                    offset += count
                    remaining -= count
            source.verify()
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, IMMUTABLE_SEALS)
            return cls(storage, fd)

    def duplicate_fd(self):
        if self._fd is None:
            raise RuntimeError("shared PLE source is closed")
        return os.dup(self._fd)

    def read_rows(self, row_ids):
        if self._mapping is None:
            raise RuntimeError("shared PLE source is closed")
        ids = self.storage.validate_rows(row_ids)
        row_bytes = self.storage.shards[0].row_bytes
        return b"".join(self._mapping[row * row_bytes : (row + 1) * row_bytes] for row in ids)

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
