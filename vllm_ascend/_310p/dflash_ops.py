# SPDX-License-Identifier: Apache-2.0
"""Triton-free draft primitives; isolated parity does not prove model support."""

import torch


def _copy_prefix(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Write exact logical indices without a sliced destination copy.

    The 310P probe observed valid prefix values but overwritten sentinel tails
    after sliced copies in the expansion sequence. Indexed writes avoid that
    path and preserved tails in the matched device probe. Full-model and graph
    validation remain separate requirements.
    """
    indices = torch.arange(source.numel(), dtype=torch.int64, device=destination.device)
    destination.index_copy_(0, indices, source.reshape(-1).to(destination.dtype))


def copy_and_expand_dflash_inputs(
    *,
    next_token_ids_ptr,
    target_positions_ptr,
    context_slot_mapping_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    out_token_indices_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    batch_size,
    HAS_NUM_REJECTED=False,
    SAMPLE_FROM_ANCHOR=False,
    DCP_SIZE=1,
    DCP_RANK=0,
    CP_INTERLEAVE_SIZE=1,
):
    """Expand anchor + masked queries, preserving separate RoPE/KV positions.

    Matches the non-DCP kernel used by DFlash and DSpark MRv1. DSpark
    samples every query, including the anchor, so K proposals need K query
    rows; DFlash samples after the anchor and needs K+1 query rows.
    Metadata remains on device; no tensor values are read by Python. Every
    scheduled request must contain at least one context row, even if rejected.
    """
    if DCP_SIZE != 1 or DCP_RANK != 0 or CP_INTERLEAVE_SIZE != 1:
        raise ValueError("310P draft expansion supports only non-DCP inputs")
    expected_queries = num_speculative_tokens + (0 if SAMPLE_FROM_ANCHOR else 1)
    if num_speculative_tokens < 1 or num_query_per_req != expected_queries:
        raise ValueError("Draft query count must match the anchor sampling layout")
    if block_size <= 0 or block_table_stride != block_table_ptr.stride(0):
        raise ValueError("Invalid KV block size or block-table stride")

    # MRv1 forwards [3, tokens] MRoPE views with a capacity-sized row stride.
    # The Triton pointer path reads the first row only; DFlash's own position
    # buffers are one-dimensional. Preserve that existing draft convention.
    target_positions = target_positions_ptr[0] if target_positions_ptr.ndim == 2 else target_positions_ptr
    if target_positions.ndim != 1:
        raise ValueError("Expected target positions [tokens] or [axes, tokens]")
    _copy_prefix(out_context_positions_ptr, target_positions[:total_input_tokens])
    _copy_prefix(out_context_slot_mapping_ptr, context_slot_mapping_ptr[:total_input_tokens])
    if batch_size == 0:
        return

    starts = query_start_loc_ptr[:batch_size].to(torch.int64)
    ends = query_start_loc_ptr[1 : batch_size + 1].to(torch.int64)
    lengths = seq_lens_ptr[:batch_size]
    if HAS_NUM_REJECTED:
        rejected = num_rejected_tokens_ptr[:batch_size]
        ends = ends - rejected
        lengths = lengths - rejected
    valid = ends > starts
    last_indices = torch.where(valid, ends - 1, starts)
    last_positions = target_positions[last_indices]
    # A fully rejected step restarts at its own first position, never the
    # preceding request's last position.
    first_positions = last_positions + valid.to(last_positions.dtype)
    offsets = torch.arange(num_query_per_req, device=lengths.device, dtype=torch.int32)
    query_positions = first_positions[:, None] + offsets
    kv_positions = lengths[:, None] + offsets
    block_columns = torch.div(kv_positions, block_size, rounding_mode="floor").to(torch.int64)
    blocks = block_table_ptr[:batch_size].gather(1, block_columns).to(torch.int64)
    slots = blocks * block_size + kv_positions.remainder(block_size)

    total_queries = batch_size * num_query_per_req
    input_ids = torch.where(
        offsets[None, :] == 0,
        next_token_ids_ptr[:batch_size].reshape(batch_size, 1),
        parallel_drafting_token_id,
    )
    _copy_prefix(out_input_ids_ptr, input_ids)
    _copy_prefix(out_query_positions_ptr, query_positions)
    _copy_prefix(out_query_slot_mapping_ptr, slots)
    sample_indices = torch.arange(total_queries, device=lengths.device, dtype=torch.int32)
    sample_start = 0 if SAMPLE_FROM_ANCHOR else 1
    _copy_prefix(
        out_token_indices_ptr,
        sample_indices.reshape(batch_size, num_query_per_req)[:, sample_start:].flatten(),
    )


def greedy_select_path_310p(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Walk candidate edges on device, with the smallest index winning ties."""
    batch, steps, top_k = candidate_ids.shape
    if top_k < 1 or scores.shape != (batch, steps, top_k, top_k):
        raise ValueError("Expected scores [batch, steps, top_k, top_k]")
    output = candidate_ids.new_empty((batch, steps))
    previous = torch.zeros((batch, 1, 1), dtype=torch.int64, device=candidate_ids.device)
    for step in range(steps):
        row = scores[:, step].gather(1, previous.expand(-1, 1, top_k)).squeeze(1).float()
        previous = row.argmax(dim=-1).reshape(batch, 1, 1)
        output[:, step].copy_(candidate_ids[:, step].gather(1, previous.squeeze(2)).squeeze(1))
    return output
