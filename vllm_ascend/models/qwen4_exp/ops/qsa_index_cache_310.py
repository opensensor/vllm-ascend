# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent compressed QSA index-cache operations for Ascend 310P."""

from __future__ import annotations

import torch

QSA_310_DEFAULT_BLOCK_SIZE = 128
QSA_310_COMPRESS_RATIO = 4
QSA_310_SCRATCH_ROWS = QSA_310_COMPRESS_RATIO - 1
QSA_310_SUPPORTED_BLOCK_SIZES = (64, 128)


def qsa_index_cache_shape(
    num_blocks: int,
    head_dim: int,
    *,
    block_size: int = QSA_310_DEFAULT_BLOCK_SIZE,
) -> tuple[int, int, int]:
    """Return the physical 310P index-cache shape.

    The leading rows hold complete four-token means. The last three rows hold
    the open group's raw keys, allowing an incomplete group to cross scheduler
    invocations without request-indexed host state.
    """
    if num_blocks < 0 or head_dim <= 0:
        raise ValueError("num_blocks must be non-negative and head_dim must be positive")
    if block_size not in QSA_310_SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {QSA_310_SUPPORTED_BLOCK_SIZES}")
    groups_per_block = block_size // QSA_310_COMPRESS_RATIO
    return num_blocks, groups_per_block + QSA_310_SCRATCH_ROWS, head_dim


def _validate_update_contract(
    compressed_key_cache: torch.Tensor,
    index_keys: torch.Tensor,
    query_start_loc: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    block_size: int,
    rotary_dim: int,
) -> None:
    if compressed_key_cache.ndim != 3:
        raise ValueError("compressed_key_cache must be [blocks,group_rows+3,D]")
    expected_rows = qsa_index_cache_shape(0, compressed_key_cache.shape[2], block_size=block_size)[1]
    if compressed_key_cache.shape[1] != expected_rows:
        raise ValueError(f"compressed_key_cache must have {expected_rows} rows per block")
    if index_keys.ndim != 2:
        raise ValueError("index_keys must be [T,D]")
    if index_keys.shape[1] != compressed_key_cache.shape[2]:
        raise ValueError("index_keys and cache must have the same head dimension")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be one-dimensional")
    if slot_mapping.shape != (index_keys.shape[0],):
        raise ValueError("slot_mapping must be [T]")
    if key_norm_weight.shape != (index_keys.shape[1],):
        raise ValueError("key_norm_weight must be [D]")
    if rope_cos.shape != (index_keys.shape[0], rotary_dim) or rope_sin.shape != rope_cos.shape:
        raise ValueError("rope_cos and rope_sin must be [T, rotary_dim]")
    if rotary_dim <= 0 or rotary_dim > index_keys.shape[1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and no larger than D")


def qsa_index_cache_update_310(
    compressed_key_cache: torch.Tensor,
    index_keys: torch.Tensor,
    query_start_loc: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    block_size: int = QSA_310_DEFAULT_BLOCK_SIZE,
    rotary_dim: int = 64,
    norm_eps: float = 1e-6,
) -> None:
    """Pool raw keys and persist normalized/rotated complete QSA groups."""
    _validate_update_contract(
        compressed_key_cache,
        index_keys,
        query_start_loc,
        slot_mapping,
        key_norm_weight,
        rope_cos,
        rope_sin,
        block_size,
        rotary_dim,
    )
    if compressed_key_cache.device.type != "npu":
        raise RuntimeError("qsa_index_cache_update_310 is an Ascend NPU-only path")
    op_namespace = getattr(torch.ops, "_C_ascend", None)
    op = None if op_namespace is None else getattr(op_namespace, "qsa_index_cache_update_310", None)
    if op is None:
        raise RuntimeError("vLLM Ascend was built without the dedicated 310P QSA cache-update operator")
    op(
        compressed_key_cache,
        index_keys.contiguous(),
        query_start_loc.to(dtype=torch.int32).contiguous(),
        slot_mapping.to(dtype=torch.int32).contiguous(),
        key_norm_weight.contiguous(),
        rope_cos.contiguous(),
        rope_sin.contiguous(),
        block_size,
        QSA_310_COMPRESS_RATIO,
        rotary_dim,
        norm_eps,
    )


def qsa_index_cache_update_310_reference(
    compressed_key_cache: torch.Tensor,
    index_keys: torch.Tensor,
    query_start_loc: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    block_size: int = QSA_310_DEFAULT_BLOCK_SIZE,
    rotary_dim: int = 64,
    norm_eps: float = 1e-6,
) -> None:
    """Host reference for page ownership and cross-call scratch semantics."""
    _validate_update_contract(
        compressed_key_cache,
        index_keys,
        query_start_loc,
        slot_mapping,
        key_norm_weight,
        rope_cos,
        rope_sin,
        block_size,
        rotary_dim,
    )
    if compressed_key_cache.device.type != "cpu":
        raise ValueError("the reference cache updater expects CPU tensors")
    for request in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        for row in range(start, end):
            slot = int(slot_mapping[row])
            if slot < 0:
                continue
            physical_block, token_offset = divmod(slot, block_size)
            group_row, group_offset = divmod(token_offset, QSA_310_COMPRESS_RATIO)
            groups_per_block = block_size // QSA_310_COMPRESS_RATIO
            if group_offset < QSA_310_SCRATCH_ROWS:
                compressed_key_cache[
                    physical_block,
                    groups_per_block + group_offset,
                ].copy_(index_keys[row])
                continue
            raw_group = torch.vstack(
                (
                    compressed_key_cache[
                        physical_block,
                        groups_per_block : groups_per_block + QSA_310_SCRATCH_ROWS,
                    ],
                    index_keys[row].unsqueeze(0),
                )
            )
            pooled = raw_group.float().mean(dim=0)
            pooled = pooled * torch.rsqrt(pooled.square().mean() + norm_eps)
            pooled = pooled * (1.0 + key_norm_weight.float())
            half = rotary_dim // 2
            first = pooled[:half].clone()
            second = pooled[half:rotary_dim].clone()
            cos = rope_cos[row].float()
            sin = rope_sin[row].float()
            pooled[:half] = first * cos[:half] - second * sin[:half]
            pooled[half:rotary_dim] = second * cos[half:] + first * sin[half:]
            compressed_key_cache[physical_block, group_row].copy_(pooled.to(compressed_key_cache.dtype))
