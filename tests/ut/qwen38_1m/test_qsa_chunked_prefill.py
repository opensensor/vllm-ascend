# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for Qwen4Exp QSA chunked prefill on Ascend 310P (plan T6.3).

The QSA raw-key ring and compressed-history side caches are advanced one
scheduler chunk at a time during prefill through the Triton-free MRV2 metadata
builder (``ops.qsa_cache.build_qsa_indexer_metadata``). The chunk-boundary
invariant this task guarantees: the ring / compressed cache state advances
*identically* whether a sequence is processed whole or split into chunks -- for
any ``chunk_size >= ring_capacity``, including a ragged last chunk and the
``chunk == block_size`` edge case.

Because both paths are lossless gather/scatter of the *same* keys (no float
math), the declared tolerance is exact bit-for-bit equality
(:data:`QSA_CHUNK_PREFILL_ATOL` ``== 0``): a boundary/off-by-one bug shifts a
cache row by O(1) and fails immediately.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import here):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_qsa_chunked_prefill.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.models.qwen4_exp.chunk_config import (
    QSA_DEFAULT_PREFILL_CHUNK_SIZE,
    QSAChunkPrefillPolicy,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_cache import (
    build_qsa_prefill_chunk_metadata,
    plan_qsa_prefill_chunks,
    run_qsa_prefill,
)

# --- Declared tolerance (PRD Sec 8.1: named constant before any assert) -----
# Chunked and whole prefill are lossless scatters of identical key rows into the
# same physical slots; the cache states must match bit-for-bit. Any nonzero
# difference is a slot-mapping / chunk-boundary bug, so the bound is exact.
QSA_CHUNK_PREFILL_ATOL = 0.0

# Pinned QSA indexer geometry (mirrors kv_cache.py / qsa_indexer_reference).
_COMPRESS_RATIO = 4
_RING_SIZE = 4  # qsa_ring_capacity(compress_ratio=4, spec=0)
_STORAGE_BLOCK_SIZE = 128
_HEAD_DIM = 128


def _make_keys(seq_len: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(seq_len, _HEAD_DIM, generator=gen, dtype=torch.float32)


def _run(keys: torch.Tensor, chunk_size: int, *, resume_from: int = 0):
    return run_qsa_prefill(
        keys,
        chunk_size=chunk_size,
        compress_ratio=_COMPRESS_RATIO,
        ring_size=_RING_SIZE,
        storage_block_size=_STORAGE_BLOCK_SIZE,
        resume_from=resume_from,
    )


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------
def test_plan_covers_sequence_with_ragged_last_chunk():
    chunks = plan_qsa_prefill_chunks(9000, QSA_DEFAULT_PREFILL_CHUNK_SIZE)
    assert chunks == [(0, 4096), (4096, 4096), (8192, 808)]
    # Contiguous, non-overlapping, exact cover.
    covered = 0
    for start, length in chunks:
        assert start == covered
        covered += length
    assert covered == 9000


def test_plan_single_chunk_when_chunk_ge_seqlen():
    assert plan_qsa_prefill_chunks(300, 4096) == [(0, 300)]


def test_plan_rejects_bad_args():
    with pytest.raises(ValueError):
        plan_qsa_prefill_chunks(10, 0)
    with pytest.raises(ValueError):
        plan_qsa_prefill_chunks(10, 4, resume_from=11)


# ---------------------------------------------------------------------------
# Full vs chunked cache-state equality (the acceptance criterion)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seq_len, chunk_size, label",
    [
        (9000, QSA_DEFAULT_PREFILL_CHUNK_SIZE, "default-4096-ragged-last"),
        (8192, QSA_DEFAULT_PREFILL_CHUNK_SIZE, "default-4096-exact-multiple"),
        (517, _STORAGE_BLOCK_SIZE, "chunk==block_size-ragged"),
        (512, _STORAGE_BLOCK_SIZE, "chunk==block_size-exact"),
        (300, 64, "small-chunks-ragged"),
        (256, 64, "small-chunks-exact"),
        (100, _RING_SIZE, "chunk==ring_size"),
        (37, 8, "ragged-tail-shorter-than-ring"),
    ],
)
def test_full_vs_chunked_cache_state_identical(seq_len, chunk_size, label):
    keys = _make_keys(seq_len, seed=hash(label) & 0xFFFF)

    ring_whole, compressed_whole = _run(keys, seq_len)  # single whole pass
    ring_chunked, compressed_chunked = _run(keys, chunk_size)

    # Declared tolerance is exact equality (QSA_CHUNK_PREFILL_ATOL == 0).
    assert torch.equal(ring_chunked, ring_whole), f"ring mismatch: {label}"
    assert torch.equal(compressed_chunked, compressed_whole), f"compressed mismatch: {label}"
    assert (ring_chunked - ring_whole).abs().max().item() <= QSA_CHUNK_PREFILL_ATOL
    assert (compressed_chunked - compressed_whole).abs().max().item() <= QSA_CHUNK_PREFILL_ATOL


def test_ring_holds_trailing_suffix_and_compressed_holds_group_keys():
    # Direct semantic check against the whole-sequence definition.
    seq_len = 300
    keys = _make_keys(seq_len, seed=7)
    ring, compressed = _run(keys, 64)

    # Ring: trailing RING_SIZE positions land at pos % RING_SIZE.
    for pos in range(seq_len - _RING_SIZE, seq_len):
        assert torch.equal(ring[pos % _RING_SIZE], keys[pos])
    # Compressed: each completed group's closing token lands at pos // ratio.
    for pos in range(seq_len):
        if (pos + 1) % _COMPRESS_RATIO == 0:
            assert torch.equal(compressed[pos // _COMPRESS_RATIO], keys[pos])


def test_chunk_metadata_logical_positions_are_absolute():
    # A chunk starting at chunk_start must resolve to absolute logical positions.
    block_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    meta = build_qsa_prefill_chunk_metadata(
        block_table,
        chunk_start=260,
        chunk_len=40,
        compress_ratio=_COMPRESS_RATIO,
        ring_size=_RING_SIZE,
        storage_block_size=_STORAGE_BLOCK_SIZE,
    )
    expected = torch.arange(260, 300)
    assert torch.equal(meta.logical_positions, expected)


# ---------------------------------------------------------------------------
# Preemption-aware recompute policy
# ---------------------------------------------------------------------------
def test_policy_default_is_4096_and_fail_closed():
    policy = QSAChunkPrefillPolicy()
    assert policy.chunk_size == QSA_DEFAULT_PREFILL_CHUNK_SIZE == 4096
    assert policy.recompute_from_scratch is True
    # Fail-closed: a preempted request recomputes from position 0.
    assert policy.resume_offset(committed_tokens=5000) == 0


def test_policy_retained_cache_resumes_from_committed():
    policy = QSAChunkPrefillPolicy(chunk_size=2048, recompute_from_scratch=False)
    assert policy.resume_offset(committed_tokens=5000) == 5000


def test_policy_validate_for_ring_rejects_undersized_chunk():
    QSAChunkPrefillPolicy(chunk_size=4096).validate_for_ring(_RING_SIZE)  # ok
    with pytest.raises(ValueError):
        QSAChunkPrefillPolicy(chunk_size=2).validate_for_ring(_RING_SIZE)


def test_preempt_recompute_matches_uninterrupted_run():
    # Fail-closed recompute (resume_from=0) reproduces the whole cache exactly,
    # which is the guarantee the preemption policy relies on.
    seq_len = 777
    keys = _make_keys(seq_len, seed=11)
    policy = QSAChunkPrefillPolicy(chunk_size=256)

    ring_ref, compressed_ref = _run(keys, seq_len)
    resume_from = policy.resume_offset(committed_tokens=333)  # -> 0 (fail-closed)
    ring_resumed, compressed_resumed = _run(keys, policy.chunk_size, resume_from=resume_from)

    assert torch.equal(ring_resumed, ring_ref)
    assert torch.equal(compressed_resumed, compressed_ref)
