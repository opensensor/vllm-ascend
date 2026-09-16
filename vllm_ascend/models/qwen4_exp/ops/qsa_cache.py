# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA side-cache slot mappings + metadata (torch fallback), Triton-free.

Ports the pure-torch formulas from the vLLM CUDA fork's
``models/qwen4_exp/common/qsa_cache.py`` -- specifically
``circular_qsa_slot_mapping`` (the raw-key ring, one physical row per token),
``compressed_qsa_slot_mapping`` (the compressed history, one row per completed
group of ``compress_ratio`` tokens) and the ``_build_qsa_metadata_torch``
fallback (logical positions, visible-block counts, slot mappings) -- to plain
PyTorch so the Ascend 310P indexer can run without Triton or a paged CUDA
attention backend.

A "slot" is a flat physical row index into the cache storage viewed as
``[num_blocks * block_size, head_dim]``: ``slot = physical_block * block_size +
offset``. Invalid / masked entries carry :data:`PAD_SLOT_ID`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Mirrors ``vllm.v1.attention.backends.utils.PAD_SLOT_ID`` (kept local so this
# module imports in isolation, without pulling the full vLLM attention stack).
PAD_SLOT_ID = -1


def qsa_token_to_req(
    query_start_loc: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Map each packed token to its request index.

    ``query_start_loc`` is the ``[num_reqs + 1]`` prefix-sum of per-request
    query lengths (with a terminal offset). Token ``t`` belongs to the last
    request whose start is ``<= t``.
    """
    if num_tokens == 0:
        return query_start_loc.new_empty((0,), dtype=torch.long)
    starts = query_start_loc.to(torch.long)
    tokens = torch.arange(num_tokens, device=query_start_loc.device)
    # Request whose [start, end) window contains the token.
    req = torch.searchsorted(starts[1:], tokens, right=True)
    num_reqs = starts.shape[0] - 1
    return req.clamp_max(num_reqs - 1)


def qsa_logical_positions(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Absolute (context-relative) logical position of each packed token.

    Port of ``_logical_positions``: ``pos = seq_len[req] - query_len[req] +
    (token - query_start[req])``.
    """
    if num_tokens == 0:
        return seq_lens.new_empty((0,), dtype=torch.long)
    starts = query_start_loc.to(torch.long)
    arange = torch.arange(num_tokens, device=query_start_loc.device)
    req = token_to_req[:num_tokens].to(torch.long)
    query_lens = torch.diff(starts)
    within_query = arange - starts.index_select(0, req)
    return seq_lens.to(torch.long).index_select(0, req) - query_lens.index_select(0, req) + within_query


def qsa_visible_blocks(
    logical_positions: torch.Tensor,
    row_seq_lens: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Number of fully-formed causal compressed blocks visible to each token.

    Port of the ``visible_blocks`` computation: ``min((pos + 1) // ratio,
    seq_len // ratio)`` clamped to ``>= 0``.
    """
    visible = torch.minimum(
        (logical_positions + 1) // compress_ratio,
        row_seq_lens.to(logical_positions.dtype) // compress_ratio,
    )
    return visible.clamp_min(0).to(torch.int32)


def _logical_to_physical_qsa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Paged logical position -> flat physical slot. Port of the fork helper."""
    if block_size <= 0:
        raise ValueError("QSA cache block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if request_indices.shape != logical_positions.shape:
        request_indices = torch.broadcast_to(request_indices, logical_positions.shape)

    requests = request_indices.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    if not all(block_table.shape):
        return torch.full_like(positions, PAD_SLOT_ID)

    valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
    logical_blocks = torch.div(positions.clamp_min(0), block_size, rounding_mode="floor")
    valid &= logical_blocks < block_table.shape[1]
    safe_requests = requests.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_blocks = logical_blocks.clamp(0, max(block_table.shape[1] - 1, 0))
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, torch.full_like(slots, PAD_SLOT_ID))


def circular_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressor_state_size: int,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw-key ring slot for each token (one physical row per token).

    Port of ``circular_qsa_slot_mapping``: each request owns one fixed physical
    block used as a circular buffer of ``compressor_state_size`` rows; token at
    logical ``pos`` lands at ``pos % compressor_state_size``. When
    ``query_start_loc`` is given, only the trailing ``compressor_state_size``
    tokens of each request are retained (the ring keeps just the open group's
    suffix); earlier tokens map to :data:`PAD_SLOT_ID`.
    """
    if compressor_state_size <= 0:
        raise ValueError("QSA circular buffer size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")

    requests = token_to_req.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    if not all(block_table.shape):
        slots = torch.full_like(positions, PAD_SLOT_ID)
    else:
        valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
        safe_requests = requests.clamp(0, block_table.shape[0] - 1)
        physical_blocks = block_table[safe_requests, 0].long()
        valid &= physical_blocks >= 0
        slots = physical_blocks * compressor_state_size + positions.remainder(compressor_state_size)
        slots = torch.where(valid, slots, torch.full_like(slots, PAD_SLOT_ID))

    if query_start_loc is not None:
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
            raise ValueError("QSA query starts must contain a terminal offset")
        starts = query_start_loc.to(device=block_table.device, dtype=torch.long)
        num_requests = starts.shape[0] - 1
        safe_requests = requests.clamp(0, num_requests - 1)
        request_ends = starts.index_select(0, safe_requests + 1)
        rows = torch.arange(slots.numel(), device=slots.device)
        keep = (requests >= 0) & (requests < num_requests) & (rows + compressor_state_size >= request_ends)
        slots = torch.where(keep, slots, torch.full_like(slots, PAD_SLOT_ID))

    return slots.to(torch.long)


def compressed_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Compressed-history slot for each token (one row per completed group).

    Port of ``compressed_qsa_slot_mapping``: only the boundary token that
    *closes* a group -- ``(pos + 1) % compress_ratio == 0`` -- writes a row, at
    compressed position ``pos // compress_ratio``. All other tokens map to
    :data:`PAD_SLOT_ID`.
    """
    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA block size and compression ratio must be positive")
    compressed_positions = torch.div(logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor")
    slots = _logical_to_physical_qsa_slots(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & ((logical_positions + 1).remainder(compress_ratio) == 0)
    return torch.where(valid, slots, torch.full_like(slots, PAD_SLOT_ID)).to(torch.long)


def qsa_scatter_rows(
    cache_flat: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Scatter ``rows`` into a flat ``[num_slots, head_dim]`` cache in place.

    Rows whose slot is :data:`PAD_SLOT_ID` (or otherwise negative) are skipped,
    matching the paged store kernels' masked writes.
    """
    if cache_flat.ndim != 2:
        raise ValueError("QSA cache must be a flat [num_slots, head_dim] view")
    if rows.shape[0] != slot_mapping.shape[0]:
        raise ValueError("slot mapping and rows must share the token dimension")
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].to(torch.long)
    cache_flat.index_copy_(0, slots, rows[valid].to(cache_flat.dtype))


def qsa_gather_rows(
    cache_flat: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Gather rows from a flat cache by slot; negative slots return zeros."""
    if cache_flat.ndim != 2:
        raise ValueError("QSA cache must be a flat [num_slots, head_dim] view")
    valid = (slots >= 0).unsqueeze(-1)
    safe = slots.clamp_min(0).to(torch.long)
    gathered = cache_flat.index_select(0, safe)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


@dataclass(frozen=True)
class QSAIndexerMetadata:
    """Per-forward QSA indexer metadata (torch fallback).

    Mirrors the fields ``_build_qsa_metadata_torch`` produces that the indexer
    consumes: token->request map, logical positions, visible-block counts, and
    the raw-ring / compressed slot mappings.
    """

    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    visible_blocks: torch.Tensor
    ring_slot_mapping: torch.Tensor
    compressed_slot_mapping: torch.Tensor
    num_actual_tokens: int


def build_qsa_indexer_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    *,
    compress_ratio: int,
    ring_size: int,
    storage_block_size: int,
) -> QSAIndexerMetadata:
    """Assemble QSA indexer metadata from batch descriptors (torch fallback).

    This is the Triton-free port of ``_build_qsa_metadata_torch`` restricted to
    the fields the weight-free indexer needs. ``query_start_loc`` is the
    ``[num_reqs + 1]`` query prefix-sum; ``seq_lens`` the per-request context
    length; ``block_table`` the ``[num_reqs, max_blocks]`` paged block table.
    """
    token_to_req = qsa_token_to_req(query_start_loc, num_tokens)
    logical_positions = qsa_logical_positions(query_start_loc, seq_lens, token_to_req, num_tokens)
    row_seq_lens = seq_lens.index_select(0, token_to_req.to(torch.long))
    visible_blocks = qsa_visible_blocks(logical_positions, row_seq_lens, compress_ratio)
    ring_slot_mapping = circular_qsa_slot_mapping(
        block_table,
        token_to_req,
        logical_positions,
        ring_size,
        query_start_loc=query_start_loc,
    )
    compressed_slot_mapping = compressed_qsa_slot_mapping(
        block_table,
        token_to_req,
        logical_positions,
        storage_block_size,
        compress_ratio,
    )
    return QSAIndexerMetadata(
        token_to_req=token_to_req,
        logical_positions=logical_positions,
        visible_blocks=visible_blocks,
        ring_slot_mapping=ring_slot_mapping,
        compressed_slot_mapping=compressed_slot_mapping,
        num_actual_tokens=num_tokens,
    )


# =========================================================================
# Chunked prefill (plan T6.3)
# =========================================================================
# The QSA side caches are advanced one scheduler chunk at a time during prefill.
# The 310P MRV2 metadata builder invokes :func:`build_qsa_indexer_metadata` (the
# Triton-free ``_build_qsa_metadata_torch`` fallback) once per chunk with that
# chunk's ``query_start_loc`` / ``seq_lens``. The helpers below plan the chunk
# split, build the per-chunk metadata, and advance the ring / compressed caches
# so the final cache state is *bit-identical* whether a sequence is processed
# whole or split into chunks (see :func:`run_qsa_prefill`).

# Convenience re-export of the chunk-size knob (authoritative default lives in
# ``vllm_ascend.models.qwen4_exp.chunk_config``).
QSA_DEFAULT_PREFILL_CHUNK_SIZE = 4096


def plan_qsa_prefill_chunks(
    seq_len: int,
    chunk_size: int,
    *,
    resume_from: int = 0,
) -> list[tuple[int, int]]:
    """Split ``[resume_from, seq_len)`` into ``(chunk_start, chunk_len)`` tiles.

    Chunks are ``chunk_size`` tokens each with a (possibly) ragged final chunk.
    ``resume_from`` is the preemption-aware recompute offset (0 recomputes the
    whole side cache on resume, the fail-closed default).
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= resume_from <= seq_len:
        raise ValueError("resume_from must lie in [0, seq_len]")
    chunks: list[tuple[int, int]] = []
    start = resume_from
    while start < seq_len:
        length = min(chunk_size, seq_len - start)
        chunks.append((start, length))
        start += length
    return chunks


def build_qsa_prefill_chunk_metadata(
    block_table: torch.Tensor,
    chunk_start: int,
    chunk_len: int,
    *,
    compress_ratio: int,
    ring_size: int,
    storage_block_size: int,
    req_index: int = 0,
) -> QSAIndexerMetadata:
    """Build QSA metadata for one prefill chunk of a single request.

    Presents the chunk to :func:`build_qsa_indexer_metadata` exactly as the MRV2
    metadata builder would: a single-request ``query_start_loc = [0, chunk_len]``
    and ``seq_lens = [chunk_start + chunk_len]`` so the logical positions resolve
    to the *absolute* ``chunk_start + arange(chunk_len)``. Slot mappings are thus
    keyed by absolute position and address the same physical rows a whole-sequence
    build would.
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    if chunk_start < 0:
        raise ValueError("chunk_start must be non-negative")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    device = block_table.device
    req_block_table = block_table[req_index : req_index + 1]
    query_start_loc = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([chunk_start + chunk_len], dtype=torch.int32, device=device)
    return build_qsa_indexer_metadata(
        query_start_loc,
        seq_lens,
        req_block_table,
        chunk_len,
        compress_ratio=compress_ratio,
        ring_size=ring_size,
        storage_block_size=storage_block_size,
    )


def advance_qsa_prefill_caches(
    ring_cache_flat: torch.Tensor,
    compressed_cache_flat: torch.Tensor,
    metadata: QSAIndexerMetadata,
    chunk_keys: torch.Tensor,
) -> None:
    """Scatter one chunk's keys into the ring + compressed caches in place.

    The raw-key ring keeps the open group's suffix (one physical row per retained
    token); the compressed history keeps one row per completed compression group.
    Both writes are masked scatters, so tokens the metadata marks
    :data:`PAD_SLOT_ID` are skipped -- matching the paged store kernels.
    """
    qsa_scatter_rows(ring_cache_flat, metadata.ring_slot_mapping, chunk_keys)
    qsa_scatter_rows(compressed_cache_flat, metadata.compressed_slot_mapping, chunk_keys)


def _cdiv_int(a: int, b: int) -> int:
    """Ceil division for positive ints (local, keeps this module import-light)."""
    return -(-a // b)


def _qsa_prefill_block_table(seq_len: int, storage_block_size: int, device: torch.device) -> torch.Tensor:
    """Identity single-request block table covering ``seq_len`` compressed rows.

    Physical block ``i`` maps to logical block ``i`` (``block_table = arange``),
    so a compressed group at logical position ``p`` lands at flat slot ``p``
    regardless of chunking. Column 0 doubles as the ring's fixed physical block.
    """
    num_compressed_rows = max(seq_len, 1)  # <= one compressed row per token
    num_cols = _cdiv_int(num_compressed_rows, storage_block_size)
    return torch.arange(num_cols, dtype=torch.int32, device=device).unsqueeze(0)


def run_qsa_prefill(
    keys: torch.Tensor,
    *,
    chunk_size: int,
    compress_ratio: int,
    ring_size: int,
    storage_block_size: int,
    resume_from: int = 0,
    block_table: torch.Tensor | None = None,
    ring_cache: torch.Tensor | None = None,
    compressed_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the QSA ring + compressed caches over one request's prefill.

    Splits ``keys`` (``[seq_len, head_dim]``) into ``chunk_size`` chunks, builds
    per-chunk metadata and scatters each chunk into the side caches. The returned
    ``(ring_cache, compressed_cache)`` are *bit-identical* to a single whole
    sequence pass (``chunk_size >= seq_len``), which is the chunk-boundary
    invariant T6.3 guarantees. ``resume_from`` supports the preemption-aware
    recompute policy (recompute from 0 by default).
    """
    if keys.ndim != 2:
        raise ValueError("keys must be [seq_len, head_dim]")
    seq_len, head_dim = keys.shape
    device = keys.device
    if block_table is None:
        block_table = _qsa_prefill_block_table(seq_len, storage_block_size, device)
    if ring_cache is None:
        ring_cache = torch.zeros((ring_size, head_dim), dtype=keys.dtype, device=device)
    if compressed_cache is None:
        num_blocks = int(block_table.shape[1])
        compressed_cache = torch.zeros((num_blocks * storage_block_size, head_dim), dtype=keys.dtype, device=device)
    for chunk_start, chunk_len in plan_qsa_prefill_chunks(seq_len, chunk_size, resume_from=resume_from):
        metadata = build_qsa_prefill_chunk_metadata(
            block_table,
            chunk_start,
            chunk_len,
            compress_ratio=compress_ratio,
            ring_size=ring_size,
            storage_block_size=storage_block_size,
        )
        chunk_keys = keys[chunk_start : chunk_start + chunk_len]
        advance_qsa_prefill_caches(ring_cache, compressed_cache, metadata, chunk_keys)
    return ring_cache, compressed_cache


__all__ = [
    "PAD_SLOT_ID",
    "QSA_DEFAULT_PREFILL_CHUNK_SIZE",
    "QSAIndexerMetadata",
    "advance_qsa_prefill_caches",
    "build_qsa_indexer_metadata",
    "build_qsa_prefill_chunk_metadata",
    "plan_qsa_prefill_chunks",
    "run_qsa_prefill",
    "circular_qsa_slot_mapping",
    "compressed_qsa_slot_mapping",
    "qsa_gather_rows",
    "qsa_logical_positions",
    "qsa_scatter_rows",
    "qsa_token_to_req",
    "qsa_visible_blocks",
]
