# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental 310P QSA gather directly into Cube-friendly NZ storage."""

from __future__ import annotations

import torch

from .qsa_indexer import QSAGroupSelection

_NZ_INNER = 16
_COMPRESS_RATIO = 4


def _qsa_gather_nz_310(
    cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    *,
    head_dim: int,
    transpose_output: bool,
) -> torch.Tensor:
    """Gather selected QSA K/V without materializing an ND tensor.

    ``S`` includes a finite four-token tail and 16-token padding. The value
    layout is ``[T,H,S,D]``; the score-operand layout is ``[T,H,D,S]``.
    """
    import torch_npu

    if cache.device.type != "npu" or cache.dtype != torch.float16:
        raise ValueError("QSA NZ gather requires an NPU float16 cache")
    if cache.ndim != 4 or cache.shape[-1] != _NZ_INNER:
        raise ValueError("value cache must be [blocks, channels, block, 16]")
    if head_dim < _NZ_INNER or head_dim % _NZ_INNER:
        raise ValueError("head_dim must be a positive multiple of 16")
    if cache.shape[1] % (head_dim // _NZ_INNER):
        raise ValueError("cache channels must be divisible by head dimension blocks")
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("QSA NZ gather currently supports a single request")
    if selection.group_indices.ndim != 2:
        raise ValueError("selection group indices must be [T,K]")
    token_count, group_width = selection.group_indices.shape
    if token_count == 0 or group_width == 0:
        raise ValueError("selection group indices must be nonempty")
    if any(
        tensor.shape != (token_count,)
        for tensor in (selection.group_counts, selection.tail_starts, selection.tail_counts)
    ):
        raise ValueError("selection count/tail vectors must match the token count")
    if cache.shape[2] == 0 or cache.shape[2] % _COMPRESS_RATIO:
        raise ValueError("cache block size must be divisible by compression ratio")
    num_kv_heads = cache.shape[1] // (head_dim // _NZ_INNER)
    selected_tokens = group_width * _COMPRESS_RATIO + _COMPRESS_RATIO
    padded_tokens = ((selected_tokens + _NZ_INNER - 1) // _NZ_INNER) * _NZ_INNER
    output_shape = (
        (token_count, num_kv_heads, head_dim, padded_tokens)
        if transpose_output
        else (token_count, num_kv_heads, padded_tokens, head_dim)
    )
    output = torch_npu.empty_with_format(
        size=output_shape,
        dtype=cache.dtype,
        device=cache.device,
        acl_format=29,
    )
    namespace = getattr(torch.ops, "_C_ascend", None)
    op = None if namespace is None else getattr(namespace, "qsa_gather_value_nz_310", None)
    if op is None:
        raise RuntimeError("vLLM Ascend was built without the 310P QSA NZ value gather operator")
    op(
        cache,
        selection.group_indices.to(dtype=torch.int32).contiguous(),
        selection.group_counts.to(dtype=torch.int32).contiguous(),
        selection.tail_starts.to(dtype=torch.int32).contiguous(),
        selection.tail_counts.to(dtype=torch.int32).contiguous(),
        block_table.to(dtype=torch.int32).contiguous(),
        output,
        num_kv_heads,
        head_dim,
        transpose_output,
    )
    return output


def qsa_gather_value_nz_310(
    value_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Gather selected QSA values in ``[T,H,S,D]`` NZ layout."""
    return _qsa_gather_nz_310(value_cache, selection, block_table, head_dim=head_dim, transpose_output=False)


def qsa_gather_key_transposed_nz_310(
    key_cache: torch.Tensor,
    selection: QSAGroupSelection,
    block_table: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Gather selected QSA keys directly as a ``[T,H,D,S]`` NZ score operand."""
    return _qsa_gather_nz_310(key_cache, selection, block_table, head_dim=head_dim, transpose_output=True)


__all__ = ["qsa_gather_key_transposed_nz_310", "qsa_gather_value_nz_310"]
