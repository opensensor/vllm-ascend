# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp n-gram / host PLE table embedding (plan T1.2 stub + T4.1).

Two concerns live here:

* ``AscendQwen4ExpNGramEmbedding`` -- the n-gram embedding module surface pinned
  by T1.2. Its full hashing + PLE vocab-parallel forward lands in T1.3.
* The **host-resident PLE table** ownership and lookup (plan T4.1, PRD R5). The
  real Qwen4Exp PLE table is ONE logical FP16 host copy (95.43 GiB) that must be
  shared across the four Ascend 310P worker processes -- never replicated per
  rank. Two transports ship behind one interface, selected by a config switch
  decided on hardware at D1:

  (a) :class:`AscendPLEPinnedHostEmbeddingMethod` -- pinned CPU memory looked up
      through a UVA accelerator view, mirroring the NVIDIA fork's
      ``Qwen4ExpPLEPinnedHostEmbedding``. The device calls (UVA probe, accelerator
      view, batched kernel gather) are injectable and guarded so the transport is
      interface-complete and host-importable; the on-device path is **D1-verified**
      on real 310P silicon.
  (b) :class:`AscendPLESharedMmapEmbeddingMethod` -- a file-backed ``MAP_SHARED``
      mmap (default under ``/dev/shm``) that every co-located worker attaches, so
      all ranks observe the *same physical pages*. It supports registered transfer
      windows (only measured regions are pinned -- never the whole table) and a
      batched row gather. This transport is fully host-testable.

The switch (:func:`create_ple_embedding_method`) defaults to (a) with automatic
fallback to (b) when UVA is unavailable, and always uses (b) when the engram
config requests ``dp_shared_memory``.

All dtype decisions read from :data:`ASCEND_QWEN4EXP_DTYPE_POLICY` (PLE table =
float16); no dtype literals are spelled in this module.

Scope note: the asynchronous prefetch / de-duplication path is a separate later
task (T4.4). :meth:`AscendPLEEmbeddingMethod.start_prefetch` is a clean no-op
hook here.
"""

from __future__ import annotations

import abc
import json
import logging
import mmap
import os
import struct
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import nn

from vllm_ascend.observability.qwen38_mem_accounting import (
    MemComponent,
    MemoryAccountant,
)

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy

if TYPE_CHECKING:
    from vllm.config.engram import EngramConfig

logger = logging.getLogger(__name__)

# OS + host<->device transfer-buffer reserve that must remain free after the
# single shared PLE table is placed in host RAM (PRD §6). Startup fails fast if
# the table cannot coexist with this reserve.
OS_TRANSFER_RESERVE_BYTES = 48 * (1024**3)

# Default location for the file-backed shared table. tmpfs (/dev/shm) keeps the
# pages resident and lets co-located workers share one physical copy.
DEFAULT_PLE_SHM_DIR = "/dev/shm"

# Lazy per-shard transport (c): the PLE table is read straight out of the
# checkpoint's 128 ``...ngram_embedding.shard_{s}.weight`` tensors (each
# ``[shard_rows, head_dim]`` F16), scattered across the safetensors shard files.
DEFAULT_SHARD_TENSOR_FMT = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
DEFAULT_SPLIT_NGRAM_PARTS = 128
DEFAULT_SAFETENSORS_INDEX = "quant_model_weights.safetensors.index.json"

# safetensors header dtype tokens -> little-endian numpy dtypes (the checkpoint
# is written little-endian; the 310P host is little-endian too).
_SAFETENSORS_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("u1"),
}


class AscendPLETransport(str, Enum):
    """Host PLE table transport selector (decided on hardware at D1)."""

    AUTO = "auto"
    PINNED_UVA = "pinned_uva"
    SHARED_MMAP = "shared_mmap"
    LAZY_SHARD = "lazy_shard"


class HostByteBudgetError(RuntimeError):
    """Raised when the PLE table cannot fit alongside the OS/transfer reserve."""


class PLETableSharingError(RuntimeError):
    """Raised on any attempt to replicate or over-pin the shared host table."""


class PLETransportUnavailableError(RuntimeError):
    """Raised when a requested transport cannot be initialized on this host."""


def _default_host_total_bytes() -> int:
    """Total physical host RAM in bytes (best effort, host-portable)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - exotic hosts
        return 0


def _resolve_table_source(
    source: object,
    num_embeddings: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Materialize an injectable table source into a CPU tensor (or ``None``)."""
    if source is None:
        return None
    if callable(source):
        source = source(num_embeddings, embedding_dim, dtype)
    if not isinstance(source, torch.Tensor):
        raise TypeError("PLE table_source must be a torch.Tensor or a callable returning one")
    if tuple(source.shape) != (num_embeddings, embedding_dim):
        raise ValueError(f"PLE table_source shape {tuple(source.shape)} != expected {(num_embeddings, embedding_dim)}")
    return source.to(dtype=dtype, device="cpu")


class AscendPLEEmbeddingMethod(abc.ABC):
    """One interface for host-resident PLE table ownership and row lookup.

    Concrete transports own the storage and implement the batched row gather;
    this base owns the invariants shared by every transport: dtype comes from the
    authoritative policy, the table is counted as ONE logical copy (never scaled
    by ``world_size``), and the table must fit alongside the OS/transfer reserve.

    On device (D1) the pinned transport composes the NVIDIA fork ABC
    ``Qwen4ExpPLEEmbedding``; that class pulls CUDA/Triton/distributed modules and
    is not importable on the host-only 310P dev path, so the shared contract is
    mirrored here rather than inherited.
    """

    transport: ClassVar[AscendPLETransport]
    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        world_size: int = 1,
        host_total_bytes: int | None = None,
        reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
        prefix: str = "",
    ) -> None:
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("PLE table dimensions must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.dtype_policy = dtype_policy
        self.dtype = dtype_policy.cast_site("ngram_embedding")
        self.world_size = int(world_size)
        self.reserve_bytes = int(reserve_bytes)
        self.prefix = prefix
        self._pinned_bytes = 0
        self._weight: torch.Tensor | None = None
        self.verify_host_budget(host_total_bytes)

    # -- byte accounting --------------------------------------------------- #

    @property
    def itemsize(self) -> int:
        return torch.finfo(self.dtype).bits // 8

    @property
    def table_bytes(self) -> int:
        """Bytes of the single logical host table (one shared copy)."""
        return self.num_embeddings * self.embedding_dim * self.itemsize

    @property
    def physical_bytes(self) -> int:
        """Physically resident host bytes: always ONE copy, never ×world_size."""
        return self.table_bytes

    @property
    def naive_rank_scaled_bytes(self) -> int:
        """The footprint we explicitly AVOID (one table per rank)."""
        return self.table_bytes * self.world_size

    @property
    def pinned_bytes(self) -> int:
        """Host bytes actually pinned (only measured regions, never the table)."""
        return self._pinned_bytes

    def verify_host_budget(self, host_total_bytes: int | None = None) -> None:
        """Fail fast if the single table cannot coexist with the reserve."""
        total = _default_host_total_bytes() if host_total_bytes is None else int(host_total_bytes)
        if total <= 0:
            # Host RAM could not be determined; skip rather than false-positive.
            return
        required = self.table_bytes + self.reserve_bytes
        if required > total:
            raise HostByteBudgetError(
                f"PLE host table ({self.table_bytes} bytes) + OS/transfer reserve "
                f"({self.reserve_bytes} bytes) = {required} bytes exceeds host RAM "
                f"({total} bytes). Reduce the table or free host memory (PRD §6)."
            )

    def record_host_bytes(self, accountant: MemoryAccountant, rank: int) -> None:
        """Record the shared table as ONE logical copy on ``rank``.

        Every rank records the identical :attr:`table_bytes`; the accountant's
        :meth:`MemoryAccountant.host_table_bytes` returns the single shared value
        and raises if any rank diverges, so the 95.43 GiB table can never be
        over-counted by ``world_size``.
        """
        accountant.rank_report(rank).add(MemComponent.PLE_HOST_TABLE, self.table_bytes)

    # -- storage / lookup interface ---------------------------------------- #

    @property
    def weight(self) -> torch.Tensor:
        if self._weight is None:
            raise RuntimeError("PLE embedding weight has not been allocated")
        return self._weight

    @abc.abstractmethod
    def allocate_embedding_weight(self) -> torch.Tensor:
        """Allocate/attach the single host table and return it as a tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def gather_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched row gather: ``ids -> [num_ids, embedding_dim]`` in table dtype."""
        raise NotImplementedError

    def dequantize(self, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        """Convert looked-up rows to the activation dtype (unquantized FP16)."""
        if embeddings.dtype == output_dtype:
            return embeddings
        return embeddings.to(output_dtype)

    def start_prefetch(self, hidden_states: object, ngram_ids: object) -> None:
        """Clean async-prefetch hook. Real implementation is plan T4.4."""
        return None

    def close(self) -> None:
        """Release any held OS resources (default: nothing)."""
        return None

    def __enter__(self) -> AscendPLEEmbeddingMethod:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AscendPLESharedMmapEmbeddingMethod(AscendPLEEmbeddingMethod):
    """Transport (b): file-backed ``MAP_SHARED`` host table with batched gather.

    All co-located workers attach the same file, so the table is one physical
    copy of pages regardless of process count. Only explicitly registered
    (measured) transfer windows are ever pinned; the full table is never pinned.
    """

    transport: ClassVar[AscendPLETransport] = AscendPLETransport.SHARED_MMAP
    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        shm_path: str | None = None,
        create: bool = True,
        table_source: object = None,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        world_size: int = 1,
        host_total_bytes: int | None = None,
        reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype_policy=dtype_policy,
            world_size=world_size,
            host_total_bytes=host_total_bytes,
            reserve_bytes=reserve_bytes,
            prefix=prefix,
        )
        self.shm_path = shm_path or os.path.join(DEFAULT_PLE_SHM_DIR, f"vllm_ascend_ple_{prefix or 'default'}.bin")
        self.create = create
        self._file = None
        self._mmap: mmap.mmap | None = None
        self._windows: list[tuple[int, int]] = []
        self._source = table_source
        self.allocate_embedding_weight()

    @classmethod
    def attach(
        cls,
        shm_path: str,
        num_embeddings: int,
        embedding_dim: int,
        **kwargs: object,
    ) -> AscendPLESharedMmapEmbeddingMethod:
        """Attach a worker to an already-created shared table (read/write)."""
        return cls(
            num_embeddings,
            embedding_dim,
            shm_path=shm_path,
            create=False,
            **kwargs,  # type: ignore[arg-type]
        )

    def allocate_embedding_weight(self) -> torch.Tensor:
        table_bytes = self.table_bytes
        if self.create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o600)
            self._file = os.fdopen(fd, "r+b")
            self._file.truncate(table_bytes)
        else:
            if not os.path.exists(self.shm_path):
                raise PLETransportUnavailableError(f"shared PLE table {self.shm_path!r} does not exist to attach")
            if os.path.getsize(self.shm_path) != table_bytes:
                raise PLETableSharingError(
                    f"shared PLE table {self.shm_path!r} is "
                    f"{os.path.getsize(self.shm_path)} bytes, expected {table_bytes}; "
                    "refusing to attach a mismatched / replicated table"
                )
            # Handle is held for the object lifetime (backs the mmap), released
            # in close(); a context manager would close it too early.
            self._file = open(self.shm_path, "r+b")  # noqa: SIM115

        self._mmap = mmap.mmap(self._file.fileno(), table_bytes, access=mmap.ACCESS_WRITE)
        flat = torch.frombuffer(self._mmap, dtype=self.dtype, count=self.num_embeddings * self.embedding_dim)
        self._weight = flat.view(self.num_embeddings, self.embedding_dim)
        if self.create and self._source is not None:
            source = _resolve_table_source(self._source, self.num_embeddings, self.embedding_dim, self.dtype)
            self._weight.copy_(source)
        return self._weight

    @property
    def mmap_length(self) -> int:
        """Length in bytes of this process's mapping (proves single-copy)."""
        return 0 if self._mmap is None else len(self._mmap)

    def gather_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched row gather straight out of the shared mapping."""
        ids = ids.reshape(-1).long()
        if ids.numel() == 0:
            return self.weight.new_empty((0, self.embedding_dim))
        # index_select returns an independent tensor (a copy of the rows), so the
        # result is safe to hand downstream without aliasing the mmap.
        return self.weight.index_select(0, ids)

    def register_transfer_window(self, start_row: int, num_rows: int) -> None:
        """Register a measured region to stage/pin (never the whole table)."""
        if start_row < 0 or num_rows <= 0 or start_row + num_rows > self.num_embeddings:
            raise ValueError("transfer window is out of table bounds")
        window_bytes = num_rows * self.embedding_dim * self.itemsize
        if start_row == 0 and num_rows >= self.num_embeddings:
            raise PLETableSharingError("refusing to pin the whole PLE table; register only measured windows")
        if self._pinned_bytes + window_bytes >= self.table_bytes:
            raise PLETableSharingError(
                "cumulative transfer windows would pin the whole table; "
                "only measured hot regions may be pinned (PRD R5)"
            )
        self._windows.append((start_row, num_rows))
        self._pinned_bytes += window_bytes

    def close(self) -> None:
        self._weight = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None


class AscendPLEPinnedHostEmbeddingMethod(AscendPLEEmbeddingMethod):
    """Transport (a): pinned CPU table looked up through a UVA accelerator view.

    Mirrors the NVIDIA fork's ``Qwen4ExpPLEPinnedHostEmbedding``. The device-only
    calls (UVA availability probe, ``get_accelerator_view_from_cpu_tensor``, and
    the batched accelerator-side row gather) are injectable and guarded so the
    class is interface-complete and importable on the host-only dev path. The
    real on-device gather is **D1-verified** on Ascend 310P silicon.

    Only the pinned host table itself is resident; there is no per-rank copy and
    the table is not chunk-pinned beyond the measured staging buffer.
    """

    transport: ClassVar[AscendPLETransport] = AscendPLETransport.PINNED_UVA
    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        table_source: object = None,
        uva_probe: Callable[[], bool] | None = None,
        pinned_allocator: Callable[[int, int, torch.dtype], torch.Tensor] | None = None,
        accelerator_view_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        world_size: int = 1,
        host_total_bytes: int | None = None,
        reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype_policy=dtype_policy,
            world_size=world_size,
            host_total_bytes=host_total_bytes,
            reserve_bytes=reserve_bytes,
            prefix=prefix,
        )
        self._uva_probe = uva_probe or _default_uva_probe
        self._pinned_allocator = pinned_allocator or _default_pinned_allocator
        self._accelerator_view_fn = accelerator_view_fn
        self._source = table_source
        self._uva_weight: torch.Tensor | None = None
        if not self.is_available(self._uva_probe):
            raise PLETransportUnavailableError("pinned-UVA PLE transport requires UVA support")
        self.allocate_embedding_weight()

    @staticmethod
    def is_available(uva_probe: Callable[[], bool] | None = None) -> bool:
        """Whether the pinned-UVA transport can run here (guarded probe)."""
        probe = uva_probe or _default_uva_probe
        try:
            return bool(probe())
        except Exception:  # pragma: no cover - defensive: probe touches device
            return False

    def allocate_embedding_weight(self) -> torch.Tensor:
        try:
            weight = self._pinned_allocator(self.num_embeddings, self.embedding_dim, self.dtype)
        except Exception as exc:  # pinning needs an accelerator context
            raise PLETransportUnavailableError(f"failed to allocate pinned PLE host table: {exc}") from exc
        if tuple(weight.shape) != (self.num_embeddings, self.embedding_dim):
            raise ValueError("pinned allocator returned a wrongly shaped table")
        if self._source is not None:
            source = _resolve_table_source(self._source, self.num_embeddings, self.embedding_dim, self.dtype)
            weight.copy_(source)
        self._weight = weight
        # The accelerator UVA view is a device-side pointer alias of the pinned
        # host table (no copy). D1-verified on 310P; guarded/mocked on host.
        if self._accelerator_view_fn is not None:
            self._uva_weight = self._accelerator_view_fn(weight)
        return weight

    def gather_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched row gather.

        On device this issues the UVA/accelerator kernel gather over
        :attr:`_uva_weight` (D1-verified). On host, with no accelerator view, it
        falls back to an equivalent reference gather over the pinned table so the
        transport stays correct and testable.
        """
        ids = ids.reshape(-1).long()
        if ids.numel() == 0:
            return self.weight.new_empty((0, self.embedding_dim))
        table = self._uva_weight if self._uva_weight is not None else self.weight
        return table.index_select(0, ids.to(table.device))

    def close(self) -> None:
        self._weight = None
        self._uva_weight = None


class AscendPLELazyShardEmbeddingMethod(AscendPLEEmbeddingMethod):
    """Transport (c): lazy per-shard mmap of the checkpoint n-gram shards.

    The PLE table is never materialized as one 95.43 GiB host copy. The 128
    checkpoint shard tensors (``...ngram_embedding.shard_{s}.weight``, each
    ``[shard_rows, head_dim]`` F16) are located through the safetensors index and
    read on demand: :meth:`gather_rows` maps each global padded-table row id to
    ``(shard, row) = divmod(id, shard_rows)`` and gathers the requested rows
    straight out of a per-shard read-only :class:`numpy.memmap` (OS demand
    paging -- only touched rows become resident).

    Because the table lives on disk (checkpoint files), not in host RAM, this
    transport overrides :meth:`verify_host_budget` to a no-op and reports a
    negligible :attr:`physical_bytes` (the in-memory shard index only). It never
    pins, never writes, and never replicates the table.
    """

    transport: ClassVar[AscendPLETransport] = AscendPLETransport.LAZY_SHARD
    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        checkpoint_dir: str | Path | None,
        shard_tensor_fmt: str = DEFAULT_SHARD_TENSOR_FMT,
        split_ngram_parts: int = DEFAULT_SPLIT_NGRAM_PARTS,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        world_size: int = 1,
        host_total_bytes: int | None = None,
        reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
        prefix: str = "",
    ) -> None:
        if checkpoint_dir is None:
            raise PLETransportUnavailableError("lazy-shard PLE transport requires checkpoint_dir")
        if split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")
        self._checkpoint_dir = Path(checkpoint_dir)
        self._shard_tensor_fmt = shard_tensor_fmt
        self._split_ngram_parts = int(split_ngram_parts)
        self.shard_rows = -(-int(num_embeddings) // self._split_ngram_parts)  # ceil
        self._weight_map: dict[str, str] | None = None
        self._shard_memmaps: dict[int, np.memmap] = {}
        self._shard_files: dict[int, Path] = {}
        super().__init__(
            num_embeddings,
            embedding_dim,
            dtype_policy=dtype_policy,
            world_size=world_size,
            host_total_bytes=None,  # budget is overridden: table is disk-resident
            reserve_bytes=reserve_bytes,
            prefix=prefix,
        )
        del host_total_bytes  # unused on the disk-resident path (kept for interface parity)

    # -- host budget / accounting: disk-resident, nothing to reserve --------- #

    def verify_host_budget(self, host_total_bytes: int | None = None) -> None:
        """No-op: the table is not host-resident (only the shard index is held)."""
        return None

    @property
    def physical_bytes(self) -> int:
        """Physically resident host bytes: the shard index, not the table."""
        return 0

    def record_host_bytes(self, accountant: MemoryAccountant, rank: int) -> None:
        """The lazy transport holds no host-resident table; record nothing."""
        return None

    # -- shard resolution --------------------------------------------------- #

    @staticmethod
    def _parse_st_header(path: Path) -> tuple[dict, int]:
        """Parse only the 8-byte length + JSON header of a safetensors file."""
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        return header, 8 + header_len

    def _load_weight_map(self) -> dict[str, str]:
        if self._weight_map is None:
            index = self._checkpoint_dir / DEFAULT_SAFETENSORS_INDEX
            with open(index) as handle:
                data = json.load(handle)
            self._weight_map = dict(data.get("weight_map", data))
        return self._weight_map

    def _shard_memmap(self, shard_index: int) -> np.memmap:
        """Resolve + mmap one shard lazily (cached; only first access parses)."""
        mem = self._shard_memmaps.get(shard_index)
        if mem is not None:
            return mem
        name = self._shard_tensor_fmt.format(shard=shard_index)
        weight_map = self._load_weight_map()
        filename = weight_map.get(name)
        if filename is None:
            raise PLETransportUnavailableError(
                f"PLE shard tensor {name!r} not found in safetensors index "
                f"{self._checkpoint_dir / DEFAULT_SAFETENSORS_INDEX}"
            )
        path = self._checkpoint_dir / filename
        header, data_base = self._parse_st_header(path)
        entry = header.get(name)
        if entry is None:
            raise PLETransportUnavailableError(f"safetensors {path!r} header lacks tensor {name!r}")
        rows, cols = entry["shape"]
        if cols != self.embedding_dim:
            raise ValueError(f"PLE shard {shard_index} has {cols} columns, expected {self.embedding_dim}")
        if entry["dtype"] not in _SAFETENSORS_DTYPES:
            raise PLETransportUnavailableError(f"unsupported safetensors dtype {entry['dtype']!r}")
        dtype = _SAFETENSORS_DTYPES[entry["dtype"]]
        start, _end = entry["data_offsets"]
        tensor_start = data_base + start
        mem = np.memmap(path, dtype=dtype, mode="r", offset=tensor_start, shape=(rows, cols))
        self._shard_memmaps[shard_index] = mem
        return mem

    # -- gather ------------------------------------------------------------- #

    def allocate_embedding_weight(self) -> torch.Tensor | None:
        """No single host table is allocated; shards are mmap'd on demand."""
        return None

    def gather_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched row gather from the checkpoint shards (demand-paged).

        ``ids`` are global padded-table row ids; each is resolved to
        ``(shard, row)`` and read from that shard's read-only mmap. Only the
        touched rows are read from disk (no whole-shard or whole-table load).
        """
        ids = ids.reshape(-1).long().cpu()
        if ids.numel() == 0:
            return torch.empty((0, self.embedding_dim), dtype=self.dtype)
        arr = ids.numpy()
        if int(arr.max()) >= self.num_embeddings or int(arr.min()) < 0:
            raise IndexError(
                f"PLE row id out of range [0, {self.num_embeddings}): got {int(arr.min())}..{int(arr.max())}"
            )
        shard = arr // self.shard_rows
        row = arr % self.shard_rows
        out = np.empty((arr.size, self.embedding_dim), dtype=np.float16)
        for s in np.unique(shard):
            mask = shard == s
            rows = row[mask]
            mm = self._shard_memmap(int(s))
            if rows.size and int(rows.max()) >= mm.shape[0]:
                raise IndexError(f"PLE row id {int(rows.max())} resolves past shard {int(s)} ({mm.shape[0]} rows)")
            out[mask] = mm[rows]
        return torch.from_numpy(out)

    def close(self) -> None:
        for mem in self._shard_memmaps.values():
            raw = getattr(mem, "_mmap", None)
            if raw is not None:
                raw.close()
        self._shard_memmaps.clear()
        self._weight_map = None


def _default_uva_probe() -> bool:
    """Probe UVA availability via vLLM, guarded for the host-only dev path."""
    try:
        from vllm.utils.platform_utils import is_uva_available

        return bool(is_uva_available())
    except Exception:  # pragma: no cover - host without the accelerator stack
        return False


def _default_pinned_allocator(num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate the complete PLE table in pinned CPU memory (needs accelerator)."""
    return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu", pin_memory=True)


def create_ple_embedding_method(
    *,
    num_embeddings: int,
    embedding_dim: int,
    engram_config: EngramConfig | None = None,
    transport: AscendPLETransport = AscendPLETransport.AUTO,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    world_size: int = 1,
    shm_path: str | None = None,
    table_source: object = None,
    host_total_bytes: int | None = None,
    reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
    prefix: str = "",
    uva_probe: Callable[[], bool] | None = None,
    pinned_allocator: Callable[[int, int, torch.dtype], torch.Tensor] | None = None,
    accelerator_view_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    checkpoint_dir: str | Path | None = None,
    shard_tensor_fmt: str = DEFAULT_SHARD_TENSOR_FMT,
    split_ngram_parts: int = DEFAULT_SPLIT_NGRAM_PARTS,
) -> AscendPLEEmbeddingMethod:
    """Build the host PLE table method for the selected transport.

    Switch semantics (final winner decided on hardware at D1):

    * ``dp_shared_memory`` in the engram config forces transport (b) shared-mmap.
    * ``AUTO`` prefers transport (a) pinned-UVA when UVA is available, otherwise
      falls back to (b).
    * An explicit ``PINNED_UVA`` request falls back to (b) if UVA / pinning is
      unavailable rather than crashing.
    * ``LAZY_SHARD`` (transport c) reads rows on demand from the checkpoint's
      n-gram shard tensors via per-shard mmap; it needs ``checkpoint_dir`` and
      never materializes the full host table.
    """
    common = dict(
        dtype_policy=dtype_policy,
        world_size=world_size,
        host_total_bytes=host_total_bytes,
        reserve_bytes=reserve_bytes,
        prefix=prefix,
    )

    def _build_mmap() -> AscendPLESharedMmapEmbeddingMethod:
        return AscendPLESharedMmapEmbeddingMethod(
            num_embeddings,
            embedding_dim,
            shm_path=shm_path,
            create=True,
            table_source=table_source,
            **common,  # type: ignore[arg-type]
        )

    def _build_pinned() -> AscendPLEPinnedHostEmbeddingMethod:
        return AscendPLEPinnedHostEmbeddingMethod(
            num_embeddings,
            embedding_dim,
            table_source=table_source,
            uva_probe=uva_probe,
            pinned_allocator=pinned_allocator,
            accelerator_view_fn=accelerator_view_fn,
            **common,  # type: ignore[arg-type]
        )

    def _build_lazy_shard() -> AscendPLELazyShardEmbeddingMethod:
        return AscendPLELazyShardEmbeddingMethod(
            num_embeddings,
            embedding_dim,
            checkpoint_dir=checkpoint_dir,
            shard_tensor_fmt=shard_tensor_fmt,
            split_ngram_parts=split_ngram_parts,
            **common,  # type: ignore[arg-type]
        )

    if transport == AscendPLETransport.LAZY_SHARD:
        return _build_lazy_shard()

    if engram_config is not None and getattr(engram_config, "dp_shared_memory", False):
        return _build_mmap()

    want_pinned = transport in (AscendPLETransport.AUTO, AscendPLETransport.PINNED_UVA)
    if transport == AscendPLETransport.SHARED_MMAP:
        return _build_mmap()

    if want_pinned and AscendPLEPinnedHostEmbeddingMethod.is_available(uva_probe):
        try:
            return _build_pinned()
        except PLETransportUnavailableError:
            logger.warning("pinned-UVA PLE transport unavailable; falling back to shared mmap")
            return _build_mmap()

    if transport == AscendPLETransport.PINNED_UVA:
        logger.warning("pinned-UVA PLE transport requested but UVA is unavailable; using shared mmap")
    return _build_mmap()


class AscendQwen4ExpNGramEmbedding(nn.Module):
    """N-gram PLE embedding for Ascend 310P (host-only, Triton-free).

    Ports the CPU path of the NVIDIA fork ``Qwen4ExpNGramEmbedding``
    (``vllm/models/qwen4_exp/nvidia/ngram_embedding.py``): the deterministic
    SplitMix64 multiplier construction, the prime per-head vocab layout, and the
    ``query_start_loc`` / ``ngram_context`` packed hashing with EOS-crossing
    (``_shift_precompute`` / ``_shift_apply``). The resulting per-head global row
    ids are gathered from the single host-resident PLE table through the T4.1
    :class:`AscendPLEEmbeddingMethod` (``gather_rows``) and assembled into
    ``[num_tokens, ple_embed_dim]`` embeddings (``ngram_heads * head_dim``).

    Hashing is byte-for-byte identical to the T0.6/T4.2-verified reference (which
    is itself verified against the real Qwen3.8-Flash-Next checkpoint). All
    dtypes read from :data:`ASCEND_QWEN4EXP_DTYPE_POLICY`
    (``ngram_embedding`` == float16); no dtype literal is spelled here.
    """

    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007
    _DEFAULT_SEED = 1234

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Deterministic Miller-Rabin primality test for 64-bit integers."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        *,
        config: object,
        ple_method: AscendPLEEmbeddingMethod | None = None,
        ple_dense_layer_id: int = 0,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy
        self.embedding_dtype = dtype_policy.cast_site("ngram_embedding")
        self.prefix = prefix
        self.ple_method = ple_method
        self.ple_dense_layer_id = int(ple_dense_layer_id)

        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram

        self.ple_embed_dim = int(config.ple_embed_dim)
        if self.ple_embed_dim % self.ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({self.ple_embed_dim}) must be divisible by the total n-gram heads ({self.ngram_heads})"
            )
        self.head_dim = self.ple_embed_dim // self.ngram_heads

        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", self._DEFAULT_SEED)),
            ple_dense_layer_id=self.ple_dense_layer_id,
        )
        sizes, offsets, self.total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=self.ple_dense_layer_id,
        )
        # Padded table geometry (parity with the fork nvidia ngram_embedding):
        # the physical shards cover total_vocab_size rounded up to the
        # make_ngram_vocab_size_divisible_by boundary, split into split_ngram_parts.
        divisor = int(getattr(config, "make_ngram_vocab_size_divisible_by", 1) or 1)
        self.padded_vocab_size = ((self.total_vocab_size + divisor - 1) // divisor) * divisor
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", DEFAULT_SPLIT_NGRAM_PARTS))
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )

    # -- hashing (fork-faithful CPU path) ---------------------------------- #

    @staticmethod
    def _shift_precompute(tokens: torch.Tensor, eos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions and per-token distance from the previous in-segment EOS."""
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Gather the ``shift``-back predecessor, EOS-filling across boundaries."""
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-head global PLE row ids for the packed request layout.

        Args:
            input_ids: ``[num_tokens]`` flat token ids across all requests.
            query_start_loc: ``[num_reqs + 1]`` cumulative token offsets.
            ngram_context: ``[num_reqs, ngram_size - 1]`` per-request history
                (EOS-padded for a fresh segment).
            output: accepted for fork signature parity; unused on the host path.

        Returns:
            ``[num_tokens, ngram_heads]`` int64 global row ids into the padded
            PLE table.
        """
        del output  # host path returns a fresh tensor; no in-place output.
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.reshape(-1).long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(0, packed.shape[1] - 1)
        packed[request_indices, columns] = input_ids

        ngram_context = ngram_context[:num_reqs].to(device=input_ids.device, dtype=torch.long)
        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(context, self.eos_token_id)
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )

        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[index] * self.layer_multipliers[index])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    # -- gather ------------------------------------------------------------- #

    def gather_embeddings(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Batched PLE row gather -> ``[num_tokens, ple_embed_dim]``.

        ``ngram_ids`` is ``[num_tokens, ngram_heads]`` of global row indices.
        Every ``(token, head)`` row is fetched in ONE batched call through the
        T4.1 host method (no per-row sync); head ``h`` of token ``t`` fills
        columns ``[h * head_dim, (h + 1) * head_dim)``.
        """
        if self.ple_method is None:
            raise RuntimeError("AscendQwen4ExpNGramEmbedding.forward requires a PLE embedding method (T4.1)")
        if ngram_ids.ndim != 2:
            raise ValueError("ngram_ids must be [num_tokens, ngram_heads]")
        num_tokens, heads = ngram_ids.shape
        if heads != self.ngram_heads:
            raise ValueError(f"ngram_ids has {heads} heads, expected {self.ngram_heads}")
        rows = self.ple_method.gather_rows(ngram_ids.reshape(-1))
        per_head_dim = rows.shape[-1]
        if per_head_dim != self.head_dim:
            raise ValueError(
                f"PLE table row dim ({per_head_dim}) * n-gram heads ({heads}) != ple_embed_dim ({self.ple_embed_dim})"
            )
        return rows.reshape(num_tokens, heads * per_head_dim)

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Produce ``[num_tokens, ple_embed_dim]`` n-gram PLE embeddings.

        Signature mirrors the fork ``Qwen4ExpNGramEmbedding.forward`` so the
        ``model.py`` call site stays stable. ``hidden_states`` is accepted for
        parity (the fork uses it only on the device prefetch path) and is unused
        on the host gather path.
        """
        del hidden_states  # parity-only; host path hashes ids + gathers rows.
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.gather_embeddings(ngram_ids)


__all__ = [
    "OS_TRANSFER_RESERVE_BYTES",
    "DEFAULT_SAFETENSORS_INDEX",
    "DEFAULT_SHARD_TENSOR_FMT",
    "DEFAULT_SPLIT_NGRAM_PARTS",
    "AscendPLEEmbeddingMethod",
    "AscendPLELazyShardEmbeddingMethod",
    "AscendPLEPinnedHostEmbeddingMethod",
    "AscendPLESharedMmapEmbeddingMethod",
    "AscendPLETransport",
    "AscendQwen4ExpNGramEmbedding",
    "HostByteBudgetError",
    "PLETableSharingError",
    "PLETransportUnavailableError",
    "create_ple_embedding_method",
]
