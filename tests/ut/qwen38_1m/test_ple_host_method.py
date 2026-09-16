# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the host-resident Qwen4Exp PLE table method (plan T4.1, R5).

The PLE table is one logical FP16 host copy (95.43 GiB on the real model) shared
across the four Ascend 310P worker processes. These tests exercise the fully
host-testable shared-mmap transport (b): single physical copy across processes,
exact byte accounting, fail-fast host budget, and no full-table pin. Transport
(a) (pinned-UVA) is interface-complete with its device calls mocked (D1-verified
on hardware).

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \\
        tests/ut/qwen38_1m/test_ple_host_method.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    OS_TRANSFER_RESERVE_BYTES,
    AscendPLEEmbeddingMethod,
    AscendPLEPinnedHostEmbeddingMethod,
    AscendPLESharedMmapEmbeddingMethod,
    AscendPLETransport,
    HostByteBudgetError,
    PLETableSharingError,
    PLETransportUnavailableError,
    create_ple_embedding_method,
)
from vllm_ascend.observability.qwen38_mem_accounting import (
    MemComponent,
    MemoryAccountant,
)

_GIB = 1024**3
# Small synthetic table (rows, dim) that is host-cheap but still multi-row.
_ROWS = 4096
_DIM = 64
_PLE_DTYPE = ASCEND_QWEN4EXP_DTYPE_POLICY.cast_site("ngram_embedding")


def _known_table(num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """A deterministic table where row r encodes r for exact verification."""
    base = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
    return (base * 1000.0 + cols).to(dtype)


def _shm_path() -> str:
    return f"/dev/shm/test_ple_{uuid.uuid4().hex}.bin"


def _make_mmap_method(path: str, *, create: bool, table_source=None, **kwargs):
    return AscendPLESharedMmapEmbeddingMethod(
        _ROWS,
        _DIM,
        shm_path=path,
        create=create,
        table_source=table_source,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Byte accounting                                                             #
# --------------------------------------------------------------------------- #


def test_host_byte_accounting_is_exact():
    path = _shm_path()
    method = _make_mmap_method(path, create=True)
    try:
        expected = _ROWS * _DIM * torch.finfo(_PLE_DTYPE).bits // 8
        assert method.dtype is _PLE_DTYPE
        assert method.table_bytes == expected
        assert method.physical_bytes == expected
        # The mmap really is exactly one table on disk.
        assert method.mmap_length == expected
        assert os.path.getsize(path) == expected
    finally:
        method.close()
        os.unlink(path)


def test_accounting_records_single_shared_copy_across_ranks():
    path = _shm_path()
    world_size = 4
    method = _make_mmap_method(path, create=True, world_size=world_size)
    try:
        accountant = MemoryAccountant(world_size=world_size)
        for rank in range(world_size):
            method.record_host_bytes(accountant, rank)
        # host_table_bytes() returns ONE logical copy, never world_size copies.
        assert accountant.host_table_bytes() == method.table_bytes
        assert method.naive_rank_scaled_bytes == method.table_bytes * world_size
        # The per-rank recorded value is identical (the shared table).
        for rank in range(world_size):
            report = accountant.ranks[rank]
            assert report.components[MemComponent.PLE_HOST_TABLE] == method.table_bytes
    finally:
        method.close()
        os.unlink(path)


def test_rank_scaled_copies_are_rejected():
    path = _shm_path()
    method = _make_mmap_method(path, create=True)
    try:
        accountant = MemoryAccountant(world_size=2)
        # A correct rank records one shared copy...
        method.record_host_bytes(accountant, 0)
        # ...but a buggy rank that stored a per-rank (x2) copy must be caught by
        # the accountant's single-copy invariant.
        accountant.rank_report(1).add(MemComponent.PLE_HOST_TABLE, method.table_bytes * 2)
        with pytest.raises(ValueError):
            accountant.host_table_bytes()
    finally:
        method.close()
        os.unlink(path)


# --------------------------------------------------------------------------- #
# Host budget fail-fast                                                        #
# --------------------------------------------------------------------------- #


def test_host_budget_fail_fast_against_reserve():
    path = _shm_path()
    table_bytes = _ROWS * _DIM * torch.finfo(_PLE_DTYPE).bits // 8
    # Not enough headroom: table + 48 GiB reserve exceeds host RAM.
    too_small = table_bytes + OS_TRANSFER_RESERVE_BYTES - 1
    with pytest.raises(HostByteBudgetError):
        _make_mmap_method(path, create=True, host_total_bytes=too_small)
    assert not os.path.exists(path)
    # Exactly enough headroom passes.
    just_enough = table_bytes + OS_TRANSFER_RESERVE_BYTES
    method = _make_mmap_method(path, create=True, host_total_bytes=just_enough)
    try:
        assert method.reserve_bytes == OS_TRANSFER_RESERVE_BYTES
    finally:
        method.close()
        os.unlink(path)


# --------------------------------------------------------------------------- #
# Row correctness                                                             #
# --------------------------------------------------------------------------- #


def test_shared_mmap_row_reads_are_correct():
    path = _shm_path()
    table = _known_table(_ROWS, _DIM, _PLE_DTYPE)
    method = _make_mmap_method(path, create=True, table_source=table)
    try:
        ids = torch.tensor([0, 1, 5, 100, _ROWS - 1, 42], dtype=torch.long)
        rows = method.gather_rows(ids)
        assert rows.shape == (ids.numel(), _DIM)
        assert rows.dtype is _PLE_DTYPE
        torch.testing.assert_close(rows, table[ids])
        # Empty gather returns a correctly shaped empty tensor.
        empty = method.gather_rows(torch.empty(0, dtype=torch.long))
        assert empty.shape == (0, _DIM)
    finally:
        method.close()
        os.unlink(path)


# --------------------------------------------------------------------------- #
# No full-table pin                                                            #
# --------------------------------------------------------------------------- #


def test_no_full_table_pin_by_default():
    path = _shm_path()
    method = _make_mmap_method(path, create=True)
    try:
        # Nothing pinned right after allocation.
        assert method.pinned_bytes == 0
        # Registering a bounded measured window pins only that region.
        method.register_transfer_window(0, 128)
        assert method.pinned_bytes == 128 * _DIM * (torch.finfo(_PLE_DTYPE).bits // 8)
        assert method.pinned_bytes < method.table_bytes
        # Trying to pin the whole table is refused.
        with pytest.raises(PLETableSharingError):
            method.register_transfer_window(0, _ROWS)
    finally:
        method.close()
        os.unlink(path)


# --------------------------------------------------------------------------- #
# 4-process shared-page identity check                                         #
# --------------------------------------------------------------------------- #


def _shm_worker(path, num_embeddings, embedding_dim, rank, barrier, queue):
    # Re-import inside the child (fork inherits, but be explicit).
    from vllm_ascend.models.qwen4_exp.ngram_embedding import (
        AscendPLESharedMmapEmbeddingMethod as Method,
    )

    method = Method.attach(path, num_embeddings, embedding_dim)
    try:
        # Each process writes a unique sentinel into its own row, column 0.
        method.weight[rank, 0] = float(rank + 1)
        barrier.wait()
        # After the barrier every process must observe ALL sentinels if and
        # only if they share one physical copy of the pages.
        seen = [float(method.weight[r, 0]) for r in range(4)]
        queue.put((rank, seen, method.mmap_length))
    finally:
        method.close()


def test_four_processes_share_one_physical_copy():
    path = _shm_path()
    table = torch.zeros(_ROWS, _DIM, dtype=_PLE_DTYPE)
    method = _make_mmap_method(path, create=True, table_source=table)
    table_bytes = method.table_bytes
    method.close()  # children re-attach; parent no longer needs the mapping
    try:
        ctx = mp.get_context("fork")
        barrier = ctx.Barrier(4)
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_shm_worker,
                args=(path, _ROWS, _DIM, rank, barrier, queue),
            )
            for rank in range(4)
        ]
        for proc in procs:
            proc.start()
        results = [queue.get(timeout=30) for _ in range(4)]
        for proc in procs:
            proc.join(timeout=30)
            assert proc.exitcode == 0

        expected_sentinels = [1.0, 2.0, 3.0, 4.0]
        for rank, seen, mmap_length in results:
            # Every process saw every other process's write -> shared pages.
            assert seen == expected_sentinels, f"rank {rank} saw {seen}"
            # Each mapping is exactly one table, never world_size copies.
            assert mmap_length == table_bytes
    finally:
        if os.path.exists(path):
            os.unlink(path)


# --------------------------------------------------------------------------- #
# Transport switch + fallback                                                  #
# --------------------------------------------------------------------------- #


def test_auto_transport_prefers_pinned_then_falls_back():
    path = _shm_path()
    try:
        # UVA available -> pinned-UVA transport (a).
        method = create_ple_embedding_method(
            num_embeddings=_ROWS,
            embedding_dim=_DIM,
            transport=AscendPLETransport.AUTO,
            shm_path=path,
            uva_probe=lambda: True,
            pinned_allocator=lambda n, d, dt: torch.zeros(n, d, dtype=dt),
        )
        assert isinstance(method, AscendPLEPinnedHostEmbeddingMethod)
        assert method.transport is AscendPLETransport.PINNED_UVA
        method.close()

        # UVA unavailable -> auto fallback to shared-mmap transport (b).
        method = create_ple_embedding_method(
            num_embeddings=_ROWS,
            embedding_dim=_DIM,
            transport=AscendPLETransport.AUTO,
            shm_path=path,
            uva_probe=lambda: False,
        )
        assert isinstance(method, AscendPLESharedMmapEmbeddingMethod)
        assert method.transport is AscendPLETransport.SHARED_MMAP
        method.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_explicit_pinned_request_falls_back_when_unavailable():
    path = _shm_path()
    try:
        method = create_ple_embedding_method(
            num_embeddings=_ROWS,
            embedding_dim=_DIM,
            transport=AscendPLETransport.PINNED_UVA,
            shm_path=path,
            uva_probe=lambda: False,
        )
        # Requested (a) but hardware cannot; must degrade to (b), not crash.
        assert isinstance(method, AscendPLESharedMmapEmbeddingMethod)
        method.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_dp_shared_memory_config_selects_mmap():
    path = _shm_path()

    class _FakeEngram:
        cpu_offload = True
        dp_shared_memory = True
        embedding_across_dp = False

    try:
        method = create_ple_embedding_method(
            num_embeddings=_ROWS,
            embedding_dim=_DIM,
            engram_config=_FakeEngram(),
            shm_path=path,
            uva_probe=lambda: True,  # even with UVA, dp_shared_memory wins
        )
        assert isinstance(method, AscendPLESharedMmapEmbeddingMethod)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# --------------------------------------------------------------------------- #
# Pinned-UVA transport interface completeness (device calls mocked)           #
# --------------------------------------------------------------------------- #


def test_pinned_transport_is_interface_complete():
    assert issubclass(AscendPLEPinnedHostEmbeddingMethod, AscendPLEEmbeddingMethod)
    assert AscendPLEPinnedHostEmbeddingMethod.transport is AscendPLETransport.PINNED_UVA
    assert AscendPLEPinnedHostEmbeddingMethod.supports_prefetch is True

    # With device calls mocked, the pinned transport constructs and gathers.
    table = _known_table(_ROWS, _DIM, _PLE_DTYPE)

    def _fake_alloc(n, d, dt):
        weight = torch.empty(n, d, dtype=dt)
        weight.copy_(table)
        return weight

    method = AscendPLEPinnedHostEmbeddingMethod(
        _ROWS,
        _DIM,
        uva_probe=lambda: True,
        pinned_allocator=_fake_alloc,
    )
    try:
        ids = torch.tensor([1, 2, 3, _ROWS - 1], dtype=torch.long)
        rows = method.gather_rows(ids)
        torch.testing.assert_close(rows, table[ids])
        # Prefetch hook exists and is a safe no-op here (real impl is T4.4).
        assert method.start_prefetch(None, ids) is None
    finally:
        method.close()


def test_pinned_transport_unavailable_raises_when_alloc_fails():
    def _boom(n, d, dt):
        raise RuntimeError("cannot pin without accelerator")

    with pytest.raises(PLETransportUnavailableError):
        AscendPLEPinnedHostEmbeddingMethod(
            _ROWS,
            _DIM,
            uva_probe=lambda: True,
            pinned_allocator=_boom,
        )


# --------------------------------------------------------------------------- #
# D1 micro-benchmark script                                                    #
# --------------------------------------------------------------------------- #


def test_transport_bench_script_runs():
    script = Path(__file__).parents[3] / "tools" / "qwen38_1m" / "ple_transport_bench.py"
    assert script.exists(), f"missing bench script {script}"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rows",
            "2048",
            "--dim",
            "64",
            "--num-ids",
            "512",
            "--iters",
            "3",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "shared_mmap" in result.stdout
