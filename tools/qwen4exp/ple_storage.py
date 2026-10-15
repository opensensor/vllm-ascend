# SPDX-License-Identifier: Apache-2.0
"""Read-only PLE file registration and bounded row/page descriptors.

This is the admission layer for an experimental device-driven lookup path.
Descriptors contain FILE offsets, never physical LBAs. File metadata checks do
not pin filesystem extents: a raw NVMe backend must obtain its own filesystem
lease and DMA ownership before consuming any physical mapping. No I/O queue,
device mapping, or serving behavior is changed by this module.
"""

import hashlib
import json
import operator
import os
import struct
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_LOOKUP_ROWS = 4 * 3 * 16
SHARD_SUFFIX = "ngram_embedding.shard_{}.weight"


def integer(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a row or geometry value")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("integer row or geometry value required") from error


@dataclass(frozen=True)
class RegisteredFile:
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    header_sha256: str

    def verify(self, fd=None):
        stat = Path(self.path).stat() if fd is None else os.fstat(fd)
        actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if actual != (self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns):
            raise RuntimeError("registered PLE file identity or contents changed")


@dataclass(frozen=True)
class Shard:
    file: int
    offset: int
    rows: int
    row_bytes: int


@dataclass(frozen=True)
class PageRead:
    file: int
    offset: int
    length: int


@dataclass(frozen=True)
class RowSlice:
    """Scatter one piece of a file read into a requested output row."""

    read: int
    source_offset: int
    destination_row: int
    destination_offset: int
    length: int


@dataclass(frozen=True)
class LookupPlan:
    reads: tuple[PageRead, ...]
    slices: tuple[RowSlice, ...]
    useful_bytes: int

    @property
    def fetched_bytes(self):
        return sum(read.length for read in self.reads)


@dataclass(frozen=True)
class PLEStorage:
    files: tuple[RegisteredFile, ...]
    shards: tuple[Shard, ...]

    def __post_init__(self):
        if not self.files or not self.shards:
            raise ValueError("nonempty PLE files and shards required")
        rows, row_bytes = integer(self.shards[0].rows), integer(self.shards[0].row_bytes)
        if rows < 1 or row_bytes < 1:
            raise ValueError("positive PLE row geometry required")
        for shard in self.shards:
            file, offset = integer(shard.file), integer(shard.offset)
            if (
                not 0 <= file < len(self.files)
                or offset < 0
                or integer(shard.rows) != rows
                or integer(shard.row_bytes) != row_bytes
                or offset + rows * row_bytes > self.files[file].size
            ):
                raise ValueError("invalid PLE shard geometry or file range")

    @property
    def rows(self):
        return sum(shard.rows for shard in self.shards)

    @property
    def table_bytes(self):
        return sum(shard.rows * shard.row_bytes for shard in self.shards)

    def manifest(self):
        return dict(
            schema="qfn-ple-files-v1", **asdict(self), table_bytes=self.table_bytes, physical_extents_pinned=False
        )

    def plan(self, row_ids, *, page_bytes=4096):
        """Bounded reference planner; preserves duplicates and request order."""
        page_bytes = integer(page_bytes)
        if page_bytes < 512 or page_bytes & (page_bytes - 1):
            raise ValueError("power-of-two storage page of at least 512 bytes required")
        ids = self.validate_rows(row_ids)
        reads, slices, positions = [], [], {}
        shard_rows = self.shards[0].rows
        for destination, row in enumerate(ids):
            shard_index, local_row = divmod(row, shard_rows)
            shard = self.shards[shard_index]
            start = shard.offset + local_row * shard.row_bytes
            end = start + shard.row_bytes
            cursor = start
            while cursor < end:
                page = cursor // page_bytes * page_bytes
                key = (shard.file, page)
                if key not in positions:
                    length = min(page_bytes, self.files[shard.file].size - page)
                    positions[key] = len(reads)
                    reads.append(PageRead(shard.file, page, length))
                read = positions[key]
                length = min(end - cursor, reads[read].offset + reads[read].length - cursor)
                slices.append(RowSlice(read, cursor - page, destination, cursor - start, length))
                cursor += length
        return LookupPlan(tuple(reads), tuple(slices), len(ids) * self.shards[0].row_bytes)

    def validate_rows(self, row_ids):
        ids = tuple(integer(row) for row in islice(iter(row_ids), MAX_LOOKUP_ROWS + 1))
        if not 1 <= len(ids) <= MAX_LOOKUP_ROWS or any(row < 0 or row >= self.rows for row in ids):
            raise ValueError("lookup exceeds the admitted row count or table bounds")
        return ids


def register(root, *, index_filename, shard_count=128, head_dim=160):
    """Inspect safetensors headers only; no weight loading or extent mapping."""
    root = Path(root).resolve(strict=True)
    shard_count, head_dim = integer(shard_count), integer(head_dim)
    if shard_count < 1 or head_dim < 1 or Path(index_filename).name != index_filename:
        raise ValueError("positive geometry and a plain index filename required")
    index = (root / index_filename).resolve(strict=True)
    if not index.is_relative_to(root):
        raise ValueError("PLE index escapes the checkpoint root")
    weights = json.loads(index.read_text())["weight_map"]
    files, headers, shards = [], {}, []
    for number in range(shard_count):
        suffix = SHARD_SUFFIX.format(number)
        names = [key for key in weights if key.endswith(suffix)]
        if len(names) != 1:
            raise ValueError("exactly one tensor per PLE shard is required")
        name = names[0]
        relative = Path(weights[name])
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("PLE shard escapes the checkpoint root")
        if path not in headers:
            before = path.stat()
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated safetensors header")
                size = struct.unpack("<Q", prefix)[0]
                if not 1 <= size <= min(MAX_HEADER_BYTES, before.st_size - 8):
                    raise ValueError("safetensors header length exceeds bounds")
                raw = stream.read(size)
            record = RegisteredFile(
                str(path),
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                hashlib.sha256(prefix + raw).hexdigest(),
            )
            record.verify()
            headers[path] = (len(files), 8 + size, json.loads(raw))
            files.append(record)
        file, base, header = headers[path]
        entry = header[name]
        rows, columns = map(integer, entry["shape"])
        start, stop = map(integer, entry["data_offsets"])
        if (
            entry["dtype"] != "F16"
            or rows < 1
            or columns != head_dim
            or start < 0
            or stop - start != rows * head_dim * 2
            or base + stop > files[file].size
            or (shards and rows != shards[0].rows)
        ):
            raise ValueError("PLE shard layout differs from the FP16 equal-shard contract")
        shards.append(Shard(file, base + start, rows, head_dim * 2))
    for file in range(len(files)):
        spans = sorted(
            (shard.offset, shard.offset + shard.rows * shard.row_bytes) for shard in shards if shard.file == file
        )
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("PLE tensor spans overlap")
    return PLEStorage(tuple(files), tuple(shards))
