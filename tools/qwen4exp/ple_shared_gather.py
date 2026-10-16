# SPDX-License-Identifier: Apache-2.0
"""Read-only host gather from the already admitted native PLE RAM table.

This adds a virtual mapping of the existing table, not another physical table.
It preserves the model's hash, pinned staging, projection and decode transport.
Ownership is explicit: retain this object while an installed method can use it.
"""

import mmap
import os
from pathlib import Path

import numpy as np

ROW_COLUMNS = 160
ROW_BYTES = ROW_COLUMNS * 2


class SharedPLEGather:
    def __init__(self, method, metadata):
        import torch

        if (
            method.embedding_dim != ROW_COLUMNS
            or method.dtype != torch.float16
            or method.num_embeddings != metadata["geometry"]["table_rows"]
            or Path(method._checkpoint_dir).resolve() != Path(metadata["model"]).resolve()
            or metadata["bytes"] != method.num_embeddings * ROW_BYTES
        ):
            raise ValueError("shared table differs from the loaded PLE method")
        self.original = method.gather_rows
        self.rows = method.num_embeddings
        self.fd = os.open(metadata["fd_path"], os.O_RDONLY | os.O_CLOEXEC)
        self.mapping = None
        self.table = None
        try:
            stat = os.fstat(self.fd)
            if (stat.st_dev, stat.st_ino, stat.st_size) != (metadata["device"], metadata["inode"], metadata["bytes"]):
                raise ValueError("shared PLE table identity changed")
            self.mapping = mmap.mmap(self.fd, metadata["bytes"], access=mmap.ACCESS_READ)
            self.table = np.ndarray((self.rows, ROW_COLUMNS), dtype=np.float16, buffer=self.mapping)
        except Exception:
            if self.mapping is not None:
                self.mapping.close()
            os.close(self.fd)
            raise

    def gather_rows(self, ids):
        import torch

        if ids.device.type != "cpu":
            return self.original(ids)
        indices = ids.reshape(-1).long().numpy()
        if not indices.size:
            return torch.empty((0, ROW_COLUMNS), dtype=torch.float16)
        if indices.min() < 0 or indices.max() >= self.rows:
            raise IndexError("PLE row id outside the shared table")
        # Advanced indexing returns owned writable storage, preserving duplicate
        # IDs and order; callers never receive aliases of the read-only table.
        return torch.from_numpy(self.table[indices])


class SharedPLEBinding:
    """Reversible instance binding with bounded CPU-only shadow verification."""

    def __init__(self, method, gather, verify_calls=4):
        if type(verify_calls) is not int or verify_calls < 1:
            raise ValueError("positive shadow verification count required")
        if method.gather_rows != gather.original:
            raise RuntimeError("PLE gather ownership changed since qualification")
        self.method, self.gather = method, gather
        self.original = method.gather_rows
        self.previous = vars(method).get("gather_rows")
        self.owned = "gather_rows" in vars(method)
        self.remaining = verify_calls
        self.checked = self.calls = 0
        self.failed = self.installed = False

    def install(self):
        if self.installed or self.method.gather_rows != self.original:
            raise RuntimeError("PLE gather ownership changed before binding")
        self.method.gather_rows = self
        self.installed = True

    def restore(self):
        if not self.installed or self.method.gather_rows is not self:
            raise RuntimeError("PLE gather ownership changed before restoration")
        if self.owned:
            self.method.gather_rows = self.previous
        else:
            delattr(self.method, "gather_rows")
        self.installed = False

    def __call__(self, ids):
        import torch

        if self.failed or ids.device.type != "cpu":
            return self.original(ids)
        actual = self.gather.gather_rows(ids)
        self.calls += 1
        if self.remaining:
            expected = self.original(ids)
            self.failed = (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
            )
            if not self.failed:
                self.checked += 1
                self.remaining -= 1
            return expected
        return actual

    def status(self):
        return dict(
            installed=self.installed,
            failed=self.failed,
            checked=self.checked,
            remaining=self.remaining,
            calls=self.calls,
        )
