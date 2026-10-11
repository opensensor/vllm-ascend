# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled, batched 310P QSA attention over selected paged KV tokens."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .qsa_gather_nz_310 import (
    prepare_qsa_gather_metadata,
    qsa_gather_key_transposed_nz_310,
    qsa_gather_value_nz_310,
)
from .qsa_indexer import QSAGroupSelection
from .qsa_sparse_attention_310 import _validate_contract

_NZ_INNER = 16
_PREFILL_QUERY_TILE = 64
# On 310P, direct NZ gather loses to ND gather/PV at 16 groups but wins at 256.
_NZ_VALUE_GATHER_MIN_GROUPS = 256
_GROUPED_MATMUL_SINGLE_OUTPUT = 3


class QSAPrefillGatherStreams:
    """Lazily create the persistent streams used by QSA prefill gathers."""

    def __init__(self) -> None:
        self._streams: tuple[object, object] | None = None

    def get(self) -> tuple[object, object]:
        if self._streams is None:
            # Do not create NPU resources during CPU model construction. The
            # first sparse prefill owns stream creation on the worker device.
            self._streams = (torch.npu.Stream(), torch.npu.Stream())
        return self._streams


def _request_slices(query_lens: Sequence[int], num_tokens: int, num_requests: int) -> tuple[tuple[int, int], ...]:
    """Validate host query lengths and return packed-token request slices."""
    lengths = tuple(int(length) for length in query_lens)
    if len(lengths) != num_requests:
        raise ValueError("query_lens must contain one length per block-table row")
    if any(length < 0 for length in lengths):
        raise ValueError("query_lens must be nonnegative")
    if sum(lengths) != num_tokens:
        raise ValueError("query_lens must sum to the packed query token count")
    start = 0
    slices = []
    for length in lengths:
        end = start + length
        slices.append((start, end))
        start = end
    return tuple(slices)


def qsa_batched_prefill_310(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float,
    compress_ratio: int = 4,
    visible_blocks: int | None = None,
    query_tile: int = _PREFILL_QUERY_TILE,
    decode_group_list: torch.Tensor | None = None,
    query_lens: Sequence[int] | None = None,
    gather_streams: tuple[object, object] | None = None,
) -> torch.Tensor:
    """Use matrix operations for QSA prefill or small decode.

    Query tiling bounds the temporary selected K/V tensors even for a 128K
    cache. Selection and page translation remain device-side; no host sync is
    introduced in the prefill hot path. Small decode batches also benefit from
    the direct NZ gather and matrix operations at a wide 2,048-token budget.
    """
    # Keep torch_npu isolated from CPU-only imports of the model package.
    import torch_npu

    _validate_contract(query, key_cache, value_cache, selection, block_table, query_start_loc, compress_ratio)
    if query.device.type != "npu":
        raise RuntimeError("batched 310P QSA prefill requires an Ascend NPU")
    if query_tile < 1:
        raise ValueError("query_tile must be positive")

    num_tokens, num_query_heads, head_dim = query.shape
    if query_lens is None:
        if block_table.shape[0] != 1:
            raise ValueError("multi-request QSA attention requires host query_lens")
        request_slices = ((0, num_tokens),)
    else:
        request_slices = _request_slices(query_lens, num_tokens, block_table.shape[0])
    block_size = key_cache.shape[2]
    num_blocks = key_cache.shape[0]
    num_kv_heads = key_cache.shape[1] // (head_dim // _NZ_INNER)
    heads_per_kv_head = num_query_heads // num_kv_heads
    selected_width = selection.group_indices.shape[1]
    use_nz_gather = selected_width >= _NZ_VALUE_GATHER_MIN_GROUPS
    if len(request_slices) > 1 and not use_nz_gather:
        raise ValueError("multi-request QSA attention requires the NZ gather path")
    if decode_group_list is not None:
        if not use_nz_gather:
            raise ValueError("grouped decode requires the NZ gather path")
        if decode_group_list.device != query.device or decode_group_list.dtype != torch.int64:
            raise ValueError("decode_group_list must be an int64 tensor on the query device")
        if decode_group_list.numel() < num_tokens * num_kv_heads:
            raise ValueError("decode_group_list does not cover all query groups")
    selected_tokens = selected_width * compress_ratio + compress_ratio
    padded_tokens = ((selected_tokens + _NZ_INNER - 1) // _NZ_INNER) * _NZ_INNER
    tail_width = padded_tokens - selected_width * compress_ratio if use_nz_gather else compress_ratio
    parallel_gather = use_nz_gather and decode_group_list is None and gather_streams is not None
    main_stream = None
    key_gather_stream = None
    value_gather_stream = None
    if parallel_gather:
        from vllm_ascend.utils import current_stream

        main_stream = current_stream()
        key_gather_stream, value_gather_stream = gather_streams
    if visible_blocks is not None:
        if visible_blocks < 1 or visible_blocks > block_table.shape[1]:
            raise ValueError("visible_blocks must fit within the block table")
        # Reorder only pages belonging to this request into logical order.
        # Early prefill chunks can then avoid converting the entire 128K cache.
        if not use_nz_gather:
            page_ids = block_table[0, :visible_blocks]
            key_cache = torch.index_select(key_cache, 0, page_ids)
            value_cache = torch.index_select(value_cache, 0, page_ids)
        num_blocks = visible_blocks

    def token_major(cache: torch.Tensor) -> torch.Tensor:
        return (
            torch_npu.npu_format_cast(cache, 0)
            .view(num_blocks, num_kv_heads, head_dim // _NZ_INNER, block_size, _NZ_INNER)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(num_blocks * block_size, num_kv_heads, head_dim)
        )

    key_rows = None if use_nz_gather else token_major(key_cache)
    value_rows = None if use_nz_gather else token_major(value_cache)
    # The model uses 1/sqrt(256) = 1/16. For an exact power-of-two scale,
    # moving the multiply onto Q avoids scaling the much larger score tensor.
    prescale_query = scale > 0 and math.frexp(scale)[0] == 0.5
    score_query = query * scale if prescale_query else query
    group_offsets = torch.arange(compress_ratio, dtype=torch.int32, device=query.device) if not use_nz_gather else None
    # Include the NZ-only padding lanes in the mask itself. Padding the
    # [T, selected_tokens] mask after concatenation dispatches PadV3 to AICPU
    # once per query tile, even though these extra lanes are always invalid.
    tail_offsets = torch.arange(tail_width, dtype=torch.int32, device=query.device)
    group_ranks = torch.arange(selected_width, dtype=torch.int32, device=query.device)
    selected_keys_buffer = None
    selected_values_buffer = None
    if use_nz_gather:
        # Metadata is produced on the main stream before any K/V readiness
        # event. Contiguous row slices remain views and each gather's public
        # adapter can reuse these operands without a conversion or allocation.
        selection, block_table = prepare_qsa_gather_metadata(selection, block_table)
        max_tile_tokens = min(query_tile, num_tokens)
        selected_keys_buffer = torch_npu.empty_with_format(
            size=(max_tile_tokens, num_kv_heads, head_dim, padded_tokens),
            dtype=key_cache.dtype,
            device=key_cache.device,
            acl_format=29,
        )
        selected_values_buffer = torch_npu.empty_with_format(
            size=(max_tile_tokens, num_kv_heads, padded_tokens, head_dim),
            dtype=value_cache.dtype,
            device=value_cache.device,
            acl_format=29,
        )
        if parallel_gather:
            assert key_gather_stream is not None and value_gather_stream is not None
            selected_keys_buffer.record_stream(key_gather_stream)
            selected_values_buffer.record_stream(value_gather_stream)
            key_cache.record_stream(key_gather_stream)
            value_cache.record_stream(value_gather_stream)
    outputs = []
    for start in range(0, num_tokens, query_tile):
        end = min(start + query_tile, num_tokens)
        tile_tokens = end - start
        group_valid = group_ranks.unsqueeze(0) < selection.group_counts[start:end].unsqueeze(1)
        group_token_valid = (
            group_valid.unsqueeze(-1)
            .expand(-1, -1, compress_ratio)
            .reshape(tile_tokens, selected_width * compress_ratio)
        )
        tail_valid = tail_offsets.unsqueeze(0) < selection.tail_counts[start:end].unsqueeze(1)
        valid = torch.cat((group_token_valid, tail_valid), dim=1)
        key_gather_done = None
        value_gather_done = None
        if use_nz_gather:
            assert selected_keys_buffer is not None and selected_values_buffer is not None
            selected_keys_nz = (
                selected_keys_buffer
                if tile_tokens == selected_keys_buffer.shape[0]
                else selected_keys_buffer[:tile_tokens]
            )
            selected_values_nz = (
                selected_values_buffer
                if tile_tokens == selected_values_buffer.shape[0]
                else selected_values_buffer[:tile_tokens]
            )
            gather_parts = []
            output_start = 0
            for request, (request_start, request_end) in enumerate(request_slices):
                part_start = max(start, request_start)
                part_end = min(end, request_end)
                if part_start >= part_end:
                    continue
                part_selection = QSAGroupSelection(
                    group_indices=selection.group_indices[part_start:part_end],
                    group_counts=selection.group_counts[part_start:part_end],
                    tail_starts=selection.tail_starts[part_start:part_end],
                    tail_counts=selection.tail_counts[part_start:part_end],
                )
                request_table = block_table[request : request + 1]
                part_tokens = part_end - part_start
                output_end = output_start + part_tokens
                key_output = (
                    selected_keys_nz
                    if output_start == 0 and output_end == tile_tokens
                    else selected_keys_nz[output_start:output_end]
                )
                value_output = (
                    selected_values_nz
                    if output_start == 0 and output_end == tile_tokens
                    else selected_values_nz[output_start:output_end]
                )
                gather_parts.append((part_selection, request_table, key_output, value_output))
                output_start = output_end
            if output_start != tile_tokens:
                raise ValueError("query_lens do not cover the current query tile")
            if parallel_gather:
                from vllm_ascend.utils import npu_stream_switch

                assert main_stream is not None
                assert key_gather_stream is not None and value_gather_stream is not None
                # This event also protects the reused buffers: after the first
                # tile it is recorded behind the preceding PV consumer.
                buffers_ready = main_stream.record_event()
                with npu_stream_switch(key_gather_stream):
                    key_gather_stream.wait_event(buffers_ready)
                    for part_selection, request_table, key_output, _ in gather_parts:
                        for tensor in (
                            part_selection.group_indices,
                            part_selection.group_counts,
                            part_selection.tail_starts,
                            part_selection.tail_counts,
                            request_table,
                        ):
                            tensor.record_stream(key_gather_stream)
                        qsa_gather_key_transposed_nz_310(
                            key_cache,
                            part_selection,
                            request_table,
                            head_dim=head_dim,
                            output=key_output,
                        )
                    key_gather_done = key_gather_stream.record_event()
                with npu_stream_switch(value_gather_stream):
                    value_gather_stream.wait_event(buffers_ready)
                    for part_selection, request_table, _, value_output in gather_parts:
                        for tensor in (
                            part_selection.group_indices,
                            part_selection.group_counts,
                            part_selection.tail_starts,
                            part_selection.tail_counts,
                            request_table,
                        ):
                            tensor.record_stream(value_gather_stream)
                        qsa_gather_value_nz_310(
                            value_cache,
                            part_selection,
                            request_table,
                            head_dim=head_dim,
                            output=value_output,
                        )
                    value_gather_done = value_gather_stream.record_event()
            else:
                for part_selection, request_table, key_output, value_output in gather_parts:
                    qsa_gather_key_transposed_nz_310(
                        key_cache,
                        part_selection,
                        request_table,
                        head_dim=head_dim,
                        output=key_output,
                    )
                    qsa_gather_value_nz_310(
                        value_cache,
                        part_selection,
                        request_table,
                        head_dim=head_dim,
                        output=value_output,
                    )
        else:
            assert key_rows is not None and value_rows is not None and group_offsets is not None
            groups = selection.group_indices[start:end].to(torch.int32)
            group_tokens = (groups.unsqueeze(-1) * compress_ratio + group_offsets).reshape(
                tile_tokens, selected_width * compress_ratio
            )
            tail_tokens = selection.tail_starts[start:end].to(torch.int32).unsqueeze(1) + tail_offsets
            token_ids = torch.cat((group_tokens, tail_tokens), dim=1)
            token_ids = torch.where(valid, token_ids, 0)
            if visible_blocks is None:
                logical_blocks = (token_ids // block_size).reshape(-1)
                physical_blocks = torch.index_select(block_table[0], 0, logical_blocks).view_as(token_ids)
                slot_ids = (physical_blocks * block_size + token_ids % block_size).reshape(-1)
            else:
                slot_ids = token_ids.reshape(-1)
            selected_keys = torch.index_select(key_rows, 0, slot_ids).view(
                tile_tokens, selected_tokens, num_kv_heads, head_dim
            )
            selected_values = torch.index_select(value_rows, 0, slot_ids).view(
                tile_tokens, selected_tokens, num_kv_heads, head_dim
            )
        tile_query = score_query[start:end].contiguous().view(tile_tokens, num_kv_heads, heads_per_kv_head, head_dim)
        if key_gather_done is not None:
            assert main_stream is not None
            main_stream.wait_event(key_gather_done)
        if decode_group_list is not None:
            group_count = tile_tokens * num_kv_heads
            tile_group_list = decode_group_list[:group_count]
            logits = torch_npu.npu_grouped_matmul(
                [tile_query.reshape(group_count * heads_per_kv_head, head_dim)],
                [selected_keys_nz.reshape(group_count, head_dim, padded_tokens)],
                group_list=tile_group_list,
                split_item=_GROUPED_MATMUL_SINGLE_OUTPUT,
                group_type=0,
            )[0].view(tile_tokens, num_kv_heads, heads_per_kv_head, padded_tokens)
        else:
            logits = torch.matmul(tile_query, selected_keys_nz if use_nz_gather else selected_keys.permute(0, 2, 3, 1))
        logits = (logits.float() if prescale_query else logits.float() * scale).masked_fill(
            ~valid[:, None, None, :], -torch.inf
        )
        probabilities = torch.softmax(logits, dim=-1).to(query.dtype)
        if decode_group_list is not None:
            tile_output = torch_npu.npu_grouped_matmul(
                [probabilities.reshape(group_count * heads_per_kv_head, padded_tokens)],
                [selected_values_nz.reshape(group_count, padded_tokens, head_dim)],
                group_list=tile_group_list,
                split_item=_GROUPED_MATMUL_SINGLE_OUTPUT,
                group_type=0,
            )[0].view(tile_tokens, num_kv_heads, heads_per_kv_head, head_dim)
        elif use_nz_gather:
            if value_gather_done is not None:
                assert main_stream is not None
                main_stream.wait_event(value_gather_done)
            tile_output = torch.matmul(probabilities, selected_values_nz)
        else:
            tile_output = torch.matmul(probabilities, selected_values.permute(0, 2, 1, 3))
        outputs.append(tile_output.reshape(tile_tokens, num_query_heads, head_dim))
    return torch.cat(outputs, dim=0)
