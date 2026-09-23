# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated Ascend 310P QSA sparse-attention entry point.

The native operator consumes the compact learned-indexer result directly.  A
selected index denotes a four-token compression group; the kernel expands that
group while reading the existing 128-token paged K/V cache in its native NZ
layout.  No dense token-index expansion, K/V gather, or second main cache is
materialized.

The torch reference in this module is intentionally host-safe.  It defines the
operator's address mapping and numerical contract for unit tests; serving calls
the native ``_C_ascend.npu_qsa_sparse_attention_310`` implementation only.
"""

from __future__ import annotations

import torch

from .qsa_indexer import QSAGroupSelection

_NZ_INNER = 16
_COMPRESS_RATIO = 4


def _validate_contract(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    compress_ratio: int,
) -> None:
    if query.ndim != 3:
        raise ValueError("query must be [T, Nq, D]")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("key/value cache must be identical [blocks, HD/16, block, 16] NZ tensors")
    if key_cache.shape[-1] != _NZ_INNER:
        raise ValueError("NZ cache inner dimension must be 16")
    if query.shape[-1] % _NZ_INNER:
        raise ValueError("query head dimension must be divisible by 16")
    if selection.group_indices.ndim != 2 or selection.group_indices.shape[0] != query.shape[0]:
        raise ValueError("group_indices must be [T, K]")
    for name, tensor in (
        ("group_counts", selection.group_counts),
        ("tail_starts", selection.tail_starts),
        ("tail_counts", selection.tail_counts),
    ):
        if tensor.shape != (query.shape[0],):
            raise ValueError(f"{name} must be [T]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be [B, max_blocks]")
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] != block_table.shape[0] + 1:
        raise ValueError("query_start_loc must be [B+1]")
    if compress_ratio != _COMPRESS_RATIO:
        raise ValueError("the dedicated 310P kernel currently requires compress_ratio=4")
    head_dim_blocks = query.shape[-1] // _NZ_INNER
    if key_cache.shape[1] % head_dim_blocks:
        raise ValueError("NZ cache channel blocks are incompatible with query head dimension")
    num_kv_heads = key_cache.shape[1] // head_dim_blocks
    if query.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")


def qsa_sparse_attention_310(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float | None = None,
    compress_ratio: int = _COMPRESS_RATIO,
) -> torch.Tensor:
    """Run the native 310P learned-QSA sparse attention operator."""
    _validate_contract(
        query,
        key_cache,
        value_cache,
        selection,
        block_table,
        query_start_loc,
        compress_ratio,
    )
    if query.device.type != "npu":
        raise RuntimeError("qsa_sparse_attention_310 is an Ascend NPU-only operator")
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    op = None if op_namespace is None else getattr(op_namespace, "npu_qsa_sparse_attention_310", None)
    if op is None:
        raise RuntimeError("vLLM Ascend was built without the dedicated 310P QSA sparse-attention operator")
    attention_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return op(
        query.contiguous(),
        key_cache,
        value_cache,
        selection.group_indices.to(dtype=torch.int32).contiguous(),
        selection.group_counts.to(dtype=torch.int32).contiguous(),
        selection.tail_starts.to(dtype=torch.int32).contiguous(),
        selection.tail_counts.to(dtype=torch.int32).contiguous(),
        block_table.to(dtype=torch.int32).contiguous(),
        query_start_loc.to(dtype=torch.int32).contiguous(),
        attention_scale,
        compress_ratio,
    )


def _nz_row(
    cache: torch.Tensor,
    physical_block: int,
    token_offset: int,
    kv_head: int,
    head_dim: int,
) -> torch.Tensor:
    """Read one logical ``[D]`` row from the 310P NZ cache (reference only)."""
    dim_blocks = head_dim // _NZ_INNER
    first = kv_head * dim_blocks
    return cache[physical_block, first : first + dim_blocks, token_offset, :].reshape(head_dim)


def qsa_sparse_attention_310_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    scale: float | None = None,
    compress_ratio: int = _COMPRESS_RATIO,
) -> torch.Tensor:
    """Host reference for NZ address mapping, GQA mapping, and QSA semantics."""
    _validate_contract(
        query,
        key_cache,
        value_cache,
        selection,
        block_table,
        query_start_loc,
        compress_ratio,
    )
    attention_scale = query.shape[-1] ** -0.5 if scale is None else scale
    num_tokens, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1] // (head_dim // _NZ_INNER)
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    cache_block_size = key_cache.shape[2]
    output = torch.empty_like(query)

    boundaries = query_start_loc.to(device="cpu", dtype=torch.int64).tolist()
    for row in range(num_tokens):
        request = next(request for request in range(len(boundaries) - 1) if row < boundaries[request + 1])
        count = int(selection.group_counts[row])
        groups = selection.group_indices[row, :count].to(device="cpu", dtype=torch.int64).tolist()
        token_ids = [group * compress_ratio + offset for group in groups for offset in range(compress_ratio)]
        tail_start = int(selection.tail_starts[row])
        token_ids.extend(tail_start + offset for offset in range(int(selection.tail_counts[row])))

        for query_head in range(num_query_heads):
            kv_head = query_head // query_heads_per_kv_head
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for token in token_ids:
                logical_block, token_offset = divmod(token, cache_block_size)
                physical_block = int(block_table[request, logical_block])
                keys.append(_nz_row(key_cache, physical_block, token_offset, kv_head, head_dim))
                values.append(_nz_row(value_cache, physical_block, token_offset, kv_head, head_dim))
            if not keys:
                output[row, query_head].zero_()
                continue
            key_rows = torch.stack(keys).to(torch.float32)
            value_rows = torch.stack(values).to(torch.float32)
            logits = torch.matmul(key_rows, query[row, query_head].to(torch.float32)) * attention_scale
            output[row, query_head] = torch.matmul(torch.softmax(logits, dim=0), value_rows).to(output.dtype)
    return output


__all__ = ["qsa_sparse_attention_310", "qsa_sparse_attention_310_reference"]
