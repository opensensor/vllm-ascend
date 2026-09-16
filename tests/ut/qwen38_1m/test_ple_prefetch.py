# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for async PLE row prefetch, dedup and metrics (plan T4.4, R5).

Decode requests several n-gram rows per token with a high cross-step repeat rate.
The prefetcher must (a) batch + dedup those rows, (b) overlap the H2D transfer
with compute on a dedicated side stream and issue exactly ONE batched wait per
step (never a wait per row), and (c) emit the TOBS metric schema (host bytes,
page faults, transfer bytes/step, hit rate, lookup latency).

There is no NPU here, so the side stream is a :class:`SimulatedStream` whose
timeline makes the overlap and the no-per-row-sync property directly assertable.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \\
        tests/ut/qwen38_1m/test_ple_prefetch.py
"""

from __future__ import annotations

import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLESharedMmapEmbeddingMethod,
)
from vllm_ascend.models.qwen4_exp.ple_prefetch import (
    PLE_PREFETCH_METRIC_FIELDS,
    AscendRowPrefetcher,
    PLEPrefetchMetrics,
    PLEPrefetchStepMetrics,
    SimulatedStream,
)

_ROWS = 2048
_DIM = 32
_PLE_DTYPE = ASCEND_QWEN4EXP_DTYPE_POLICY.cast_site("ngram_embedding")


def _known_table(num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Row r encodes r across all columns (r + col) for exact verification."""
    base = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
    return (base + cols).to(dtype)


def _make_method(tmp_path, name: str) -> AscendPLESharedMmapEmbeddingMethod:
    """A fully host-testable shared-mmap PLE table with a known content."""
    shm_path = str(tmp_path / f"ple_prefetch_{name}.bin")
    return AscendPLESharedMmapEmbeddingMethod(
        _ROWS,
        _DIM,
        shm_path=shm_path,
        create=True,
        table_source=_known_table,
        # Large host budget so the tiny synthetic table always fits.
        host_total_bytes=1 << 40,
    )


def _repeated_trace(num_steps: int, tokens: int, heads: int, hot: int, seed: int = 0):
    """A realistic decode trace: each step's rows are drawn from a small hot set.

    High repeat rate both within a step (dedup) and across steps (cache hits).
    """
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(num_steps):
        ids = torch.randint(0, hot, (tokens, heads), generator=g, dtype=torch.long)
        steps.append(ids)
    return steps


# --------------------------------------------------------------------------- #
# Dedup ratio
# --------------------------------------------------------------------------- #


def test_dedup_ratio_on_repeated_trace(tmp_path):
    """Dedup ratio matches (requested - unique) / requested exactly."""
    method = _make_method(tmp_path, "dedup")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream(), cache_capacity_rows=0)
        # 8 tokens x 4 heads = 32 requested rows, all drawn from 5 hot ids.
        ids = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3], [4, 4, 4, 4], [0, 0, 1, 1]] + [[2, 3, 4, 0]] * 4,
            dtype=torch.long,
        )
        requested = ids.numel()
        expected_unique = int(torch.unique(ids).numel())

        handle = prefetcher.prefetch_step(ids)
        prefetcher.wait(handle)

        step = prefetcher.metrics
        assert step.requested_rows == requested
        assert step.unique_rows == expected_unique
        expected_dedup = (requested - expected_unique) / requested
        assert abs(step.dedup_ratio - expected_dedup) < 1e-9
        assert step.dedup_ratio > 0.5  # trace really is highly repeated
    finally:
        method.close()


def test_dedup_gives_correct_rows(tmp_path):
    """Dedup + reassembly returns the exact per-(token,head) table rows."""
    method = _make_method(tmp_path, "correct")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream())
        ids = torch.tensor([[10, 20, 20], [10, 5, 5]], dtype=torch.long)
        out = prefetcher.decode_step(ids)
        assert out.shape == (2, 3, _DIM)
        expected = method.gather_rows(ids.reshape(-1)).reshape(2, 3, _DIM)
        assert torch.equal(out.to(expected.dtype), expected)
    finally:
        method.close()


# --------------------------------------------------------------------------- #
# No per-row sync / overlap
# --------------------------------------------------------------------------- #


def test_single_batched_wait_per_step_no_per_row_sync(tmp_path):
    """Each decode step joins the side stream exactly once, regardless of rows."""
    method = _make_method(tmp_path, "nosync")
    try:
        stream = SimulatedStream()
        prefetcher = AscendRowPrefetcher(method, stream=stream, cache_capacity_rows=0)
        # 64 requested rows in one step -> must NOT produce 64 waits.
        ids = torch.randint(0, 100, (16, 4), dtype=torch.long)
        prefetcher.decode_step(ids)

        assert stream.wait_count == 1
        assert stream.waits_in_timeline() == 1
        # One batched transfer submit, not one per row.
        submits = [t for k, t in stream.timeline if k == "submit"]
        assert len(submits) == 1
    finally:
        method.close()


def test_wait_count_equals_step_count(tmp_path):
    """Over many steps the join count equals the step count (one per step)."""
    method = _make_method(tmp_path, "steps")
    try:
        stream = SimulatedStream()
        prefetcher = AscendRowPrefetcher(method, stream=stream)
        trace = _repeated_trace(num_steps=12, tokens=8, heads=4, hot=24)
        for ids in trace:
            prefetcher.decode_step(ids)
        assert stream.wait_count == len(trace)
        assert stream.waits_in_timeline() == len(trace)
        # Never a wait per row: rows requested hugely exceeds waits.
        assert prefetcher.metrics.requested_rows > 10 * stream.wait_count
    finally:
        method.close()


def test_compute_overlaps_transfer_before_join(tmp_path):
    """The transfer is in flight (pending) while compute runs, then one join."""
    method = _make_method(tmp_path, "overlap")
    try:
        stream = SimulatedStream()
        prefetcher = AscendRowPrefetcher(method, stream=stream, cache_capacity_rows=0)
        ids = torch.randint(0, 50, (8, 4), dtype=torch.long)

        observed = {}

        def compute():
            # At this point the transfer is submitted but NOT yet joined.
            observed["pending_at_compute"] = stream.pending_count
            observed["waits_at_compute"] = stream.wait_count

        prefetcher.decode_step(ids, compute_fn=compute)

        assert observed["pending_at_compute"] == 1  # transfer in flight
        assert observed["waits_at_compute"] == 0  # not joined during compute
        assert stream.wait_count == 1  # exactly one join afterwards
    finally:
        method.close()


# --------------------------------------------------------------------------- #
# Cache hits / page faults across steps
# --------------------------------------------------------------------------- #


def test_cache_hits_eliminate_repeat_transfers(tmp_path):
    """A repeated row across steps is transferred once (page fault) then hits."""
    method = _make_method(tmp_path, "cache")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream())
        ids = torch.tensor([[7, 8, 9, 7]], dtype=torch.long)  # unique {7,8,9}

        prefetcher.decode_step(ids)  # step 1: all miss
        m1 = list(prefetcher.metrics.transfer_bytes_per_step)
        assert prefetcher.metrics.page_faults == 3

        prefetcher.decode_step(ids)  # step 2: all resident -> no transfer
        m2 = list(prefetcher.metrics.transfer_bytes_per_step)
        assert m1[-1] > 0
        assert m2[-1] == 0
        # Second step is a full cache hit on the deduped set.
        assert prefetcher.metrics.transfer_bytes_per_step[-1] == 0
    finally:
        method.close()


def test_hit_rate_computed_over_deduped_rows(tmp_path):
    """hit_rate = cache_hits / unique across the run and stays in [0, 1]."""
    method = _make_method(tmp_path, "hitrate")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream())
        trace = _repeated_trace(num_steps=20, tokens=8, heads=4, hot=16, seed=3)
        for ids in trace:
            prefetcher.decode_step(ids)
        m = prefetcher.metrics
        assert 0.0 <= m.hit_rate <= 1.0
        assert m.cache_hits + m.page_faults == m.unique_rows
        # With only 16 hot rows and 20 steps, cross-step reuse must occur.
        assert m.hit_rate > 0.0
        # Total page faults cannot exceed the hot-set size (each row fetched once).
        assert m.page_faults <= 16
    finally:
        method.close()


# --------------------------------------------------------------------------- #
# Metrics schema (TOBS)
# --------------------------------------------------------------------------- #


def test_metrics_emit_all_required_fields(tmp_path):
    """Every required TOBS field is present on both step and aggregate dicts."""
    method = _make_method(tmp_path, "schema")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream())
        ids = torch.randint(0, 64, (8, 4), dtype=torch.long)
        prefetcher.decode_step(ids)

        required = {"host_bytes", "page_faults", "transfer_bytes", "hit_rate", "lookup_latency"}
        agg = prefetcher.metrics.to_dict()
        assert required <= set(agg)
        # Full canonical schema present.
        assert set(PLE_PREFETCH_METRIC_FIELDS) <= set(agg)

        # Step-level metrics carry the same schema.
        step = PLEPrefetchStepMetrics(
            requested_rows=32,
            unique_rows=8,
            cache_hits=2,
            page_faults=6,
            transfer_bytes=6 * _DIM * (torch.finfo(_PLE_DTYPE).bits // 8),
            host_bytes=128,
            lookup_latency=1.5,
        )
        sd = step.to_dict()
        assert required <= set(sd)
        assert set(PLE_PREFETCH_METRIC_FIELDS) <= set(sd)
        assert abs(sd["dedup_ratio"] - (32 - 8) / 32) < 1e-9
        assert abs(sd["hit_rate"] - 2 / 8) < 1e-9
    finally:
        method.close()


def test_metrics_values_are_consistent(tmp_path):
    """Aggregate host bytes, transfer bytes and latency are non-degenerate."""
    method = _make_method(tmp_path, "values")
    try:
        prefetcher = AscendRowPrefetcher(method, stream=SimulatedStream())
        trace = _repeated_trace(num_steps=6, tokens=8, heads=4, hot=40, seed=7)
        for ids in trace:
            prefetcher.decode_step(ids)
        m = prefetcher.metrics
        row_bytes = _DIM * (torch.finfo(_PLE_DTYPE).bits // 8)

        assert m.steps == len(trace)
        assert m.host_bytes == prefetcher.resident_rows * row_bytes
        assert m.transfer_bytes == m.page_faults * row_bytes
        assert m.lookup_latency > 0.0  # some transfers happened
        assert len(m.transfer_bytes_per_step) == len(trace)
        assert m.mean_transfer_bytes_per_step >= 0.0
        assert isinstance(prefetcher.metrics.human_summary(), str)
    finally:
        method.close()


def test_empty_step_is_safe(tmp_path):
    """An empty request submits no transfer and still records a step."""
    method = _make_method(tmp_path, "empty")
    try:
        stream = SimulatedStream()
        prefetcher = AscendRowPrefetcher(method, stream=stream)
        out = prefetcher.decode_step(torch.zeros((0, 4), dtype=torch.long))
        assert out.shape == (0, 4, _DIM)
        assert stream.wait_count == 1  # a single (empty) join
        assert prefetcher.metrics.transfer_bytes == 0
        assert prefetcher.metrics.page_faults == 0
    finally:
        method.close()


def test_default_stream_is_simulated(tmp_path):
    """Prefetcher provisions a SimulatedStream when none is injected."""
    method = _make_method(tmp_path, "default")
    try:
        prefetcher = AscendRowPrefetcher(method)
        assert isinstance(prefetcher.stream, SimulatedStream)
        prefetcher.decode_step(torch.randint(0, 32, (4, 4), dtype=torch.long))
        assert prefetcher.stream.wait_count == 1
    finally:
        method.close()


def test_metrics_container_defaults():
    """A bare aggregate has zeroed, in-range derived fields."""
    m = PLEPrefetchMetrics()
    assert m.dedup_ratio == 0.0
    assert m.hit_rate == 0.0
    assert m.mean_transfer_bytes_per_step == 0.0
    assert set(PLE_PREFETCH_METRIC_FIELDS) <= set(m.to_dict())
