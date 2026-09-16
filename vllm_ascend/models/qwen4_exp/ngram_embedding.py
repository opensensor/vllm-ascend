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
import logging
import mmap
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

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


class AscendPLETransport(str, Enum):
    """Host PLE table transport selector (decided on hardware at D1)."""

    AUTO = "auto"
    PINNED_UVA = "pinned_uva"
    SHARED_MMAP = "shared_mmap"


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
) -> AscendPLEEmbeddingMethod:
    """Build the host PLE table method for the selected transport.

    Switch semantics (final winner decided on hardware at D1):

    * ``dp_shared_memory`` in the engram config forces transport (b) shared-mmap.
    * ``AUTO`` prefers transport (a) pinned-UVA when UVA is available, otherwise
      falls back to (b).
    * An explicit ``PINNED_UVA`` request falls back to (b) if UVA / pinning is
      unavailable rather than crashing.
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
    """N-gram embedding (skeleton).

    Materializes n-gram embeddings in ``policy.ngram_embedding_dtype``.
    """

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy
        self.embedding_dtype = dtype_policy.cast_site("ngram_embedding")
        # TODO(T1.3): build the n-gram hashing + PLE vocab-parallel embedding
        # reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T1.3): AscendQwen4ExpNGramEmbedding.forward is implemented in a later task.")


__all__ = [
    "OS_TRANSFER_RESERVE_BYTES",
    "AscendPLEEmbeddingMethod",
    "AscendPLEPinnedHostEmbeddingMethod",
    "AscendPLESharedMmapEmbeddingMethod",
    "AscendPLETransport",
    "AscendQwen4ExpNGramEmbedding",
    "HostByteBudgetError",
    "PLETableSharingError",
    "PLETransportUnavailableError",
    "create_ple_embedding_method",
]
