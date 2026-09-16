# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P Qwen4Exp weight-free QSA indexer (plan T6.1).

Torch / NPU port of the fork's ``nvidia/indexer_qsa.py`` +
``nvidia/ops/qsa_indexer.py`` / ``qsa_pre_indexer.py``. The indexer is *weight
free*: it compresses the raw index keys into per-group mean-pooled blocks, scores
each visible block against the query heads (per-head ReLU, summed), selects the
top blocks under the token budget, and expands the selection back to token ids
plus the causal tail of the open group.

On the 310P host path there is no Triton and no paged CUDA attention backend, so
the raw-key ring write (one row per token) and the compressed history (one row
per completed group of ``compress_ratio`` tokens) go through the torch slot
mappings in :mod:`vllm_ascend.models.qwen4_exp.ops.qsa_cache` (the port of the
fork's ``_build_qsa_metadata_torch`` fallback). The selection math lives in
:mod:`vllm_ascend.models.qwen4_exp.ops.qsa_indexer`.

Dtypes are read from :data:`ASCEND_QWEN4EXP_DTYPE_POLICY`: the indexer q / k and
side caches ride ``qsa_indexer_dtype`` (float16), while the block scores
accumulate in ``attention_accumulation_dtype`` (float32). No dtype literals are
spelled in this module.

The scoring / top-k / expand output is exposed cleanly for the QSA sparse
attention kernel (T6.2); this module does NOT implement sparse attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .ops.qsa_cache import (
    build_qsa_indexer_metadata,
    qsa_gather_rows,
    qsa_scatter_rows,
)
from .ops.qsa_indexer import compress_keys, qsa_indexer_select

# Default indexer geometry (plan T6.1): 4 q-heads, 1 k-head, dim 128,
# compression ratio 4, selection budget 2,048 tokens.
_DEFAULT_INDEXER_N_HEADS = 4
_DEFAULT_INDEXER_KV_HEADS = 1
_DEFAULT_INDEXER_HEAD_DIM = 128
_DEFAULT_INDEXER_COMPRESS_RATIO = 4
_DEFAULT_INDEXER_BUDGET = 2048


@dataclass(frozen=True)
class QSAIndexerOutput:
    """Selection output for one indexer forward.

    Attributes:
        token_indices: ``[T, output_width]`` int64, ``-1``-padded request-relative
            token ids selected for each query token.
        valid_counts: ``[T]`` int64 count of valid (non ``-1``) entries per row --
            the sparse-attention tile-loop bound (T6.2).
        ring_cache: ``[ring_size, D]`` raw-key circular buffer state (the open
            group's retained suffix), for incremental decode continuation.
        compressed_cache: ``[N, D]`` compressed (mean-pooled) key history read
            back from the paged compressed cache.
    """

    token_indices: torch.Tensor
    valid_counts: torch.Tensor
    ring_cache: torch.Tensor
    compressed_cache: torch.Tensor

    @property
    def packed(self) -> torch.Tensor:
        """NVIDIA-style packed buffer ``[T, output_width + 1]`` (int32).

        Leading ``output_width`` columns are the ``-1``-padded token indices; the
        trailing column carries each row's valid-entry count (the attention
        kernel's loop bound, never a token index).
        """
        num_tokens, output_width = self.token_indices.shape
        buffer = self.token_indices.new_empty((num_tokens, output_width + 1), dtype=torch.int32)
        buffer[:, :output_width] = self.token_indices.to(torch.int32)
        buffer[:, output_width] = self.valid_counts.to(torch.int32)
        return buffer


class AscendQwen4ExpQSAIndexer(nn.Module):
    """Weight-free QSA sparse-index producer for the Ascend 310P path.

    Compresses raw index keys, scores visible blocks against the query heads,
    and deterministically selects the top blocks under the token budget. The
    raw-ring and compressed side caches are maintained through torch slot
    mappings (Triton-free); the top-k is bitwise-stable (stable descending sort,
    ties broken by ascending block index).
    """

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.dtype_policy = dtype_policy
        # Storage dtype for q / k and the side caches; scores accumulate in fp32.
        self.indexer_dtype = dtype_policy.cast_site("qsa_indexer")
        self.accumulation_dtype = dtype_policy.cast_site("attention_accumulation")

        self.index_n_heads = int(getattr(config, "indexer_n_heads", _DEFAULT_INDEXER_N_HEADS))
        self.index_kv_heads = int(getattr(config, "indexer_kv_heads", _DEFAULT_INDEXER_KV_HEADS))
        self.index_head_dim = int(getattr(config, "indexer_head_dim", _DEFAULT_INDEXER_HEAD_DIM))
        self.token_topk = int(getattr(config, "indexer_budget", _DEFAULT_INDEXER_BUDGET))
        self.compress_ratio = int(getattr(config, "indexer_compress_ratio", _DEFAULT_INDEXER_COMPRESS_RATIO))
        if self.index_kv_heads != 1:
            raise NotImplementedError("QSA indexer supports a single kv head (MQA)")
        if self.token_topk % self.compress_ratio != 0:
            raise ValueError("indexer_budget must be divisible by the compression ratio")

    @property
    def block_topk(self) -> int:
        """Number of compressed blocks selected per query."""
        return self.token_topk // self.compress_ratio

    @property
    def output_width(self) -> int:
        """Selection (index) columns per row."""
        return self.token_topk + self.compress_ratio - 1

    @property
    def packed_output_width(self) -> int:
        """Packed selection-buffer width (indices + trailing valid-count column)."""
        return self.output_width + 1

    @property
    def ring_size(self) -> int:
        """Raw-key circular buffer capacity (whole groups covering the open group).

        With no speculative tokens the ring retains exactly one open group of
        ``compress_ratio`` raw keys; the fork rounds the span up to whole groups.
        """
        num_speculative = int(getattr(self.config, "num_speculative_tokens", 0))
        span = self.compress_ratio + max(num_speculative, 0)
        groups = -(-span // self.compress_ratio)  # ceil division
        return self.compress_ratio * groups

    def forward(
        self,
        indexer_query: torch.Tensor,
        indexer_keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> QSAIndexerOutput:
        """Select token indices for one sequence's query tokens.

        Args:
            indexer_query: ``[T, H, D]`` indexer queries (already projected /
                normed / RoPE'd upstream).
            indexer_keys: ``[S, D]`` (or ``[S, 1, D]``) raw index keys for the
                whole causal context (single kv head).
            positions: ``[T]`` logical positions of the query tokens.

        Returns:
            :class:`QSAIndexerOutput` with the ``-1``-padded token indices and
            per-row valid counts.
        """
        if indexer_keys.ndim == 3:
            if indexer_keys.shape[1] != 1:
                raise ValueError("QSA indexer keys must have a single kv head")
            indexer_keys = indexer_keys[:, 0, :]
        if indexer_keys.ndim != 2:
            raise ValueError("indexer_keys must be [S, D] or [S, 1, D]")
        if indexer_query.ndim != 3:
            raise ValueError("indexer_query must be [T, H, D]")
        if positions.ndim != 1 or positions.shape[0] != indexer_query.shape[0]:
            raise ValueError("positions must be [T] aligned with the queries")

        head_dim = indexer_keys.shape[1]
        num_keys = indexer_keys.shape[0]
        ratio = self.compress_ratio
        num_blocks = num_keys // ratio
        device = indexer_keys.device

        query = indexer_query.to(self.indexer_dtype)
        keys = indexer_keys.to(self.indexer_dtype)

        # --- Side-cache metadata for the KEY tokens (single request context) ---
        # Keys occupy logical positions 0..S-1 of one request.
        storage_block_size = max(num_blocks, 1)
        key_query_start_loc = torch.tensor([0, num_keys], dtype=torch.int32, device=device)
        key_seq_lens = torch.tensor([num_keys], dtype=torch.int32, device=device)
        # One physical block each for the ring and the compressed history.
        key_block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        metadata = build_qsa_indexer_metadata(
            key_query_start_loc,
            key_seq_lens,
            key_block_table,
            num_keys,
            compress_ratio=ratio,
            ring_size=self.ring_size,
            storage_block_size=storage_block_size,
        )

        # --- Raw-key ring write (one row per token, retained suffix only) ------
        ring_cache = torch.zeros((self.ring_size, head_dim), dtype=self.indexer_dtype, device=device)
        qsa_scatter_rows(ring_cache, metadata.ring_slot_mapping, keys)

        # --- Compressed history write (one row per completed group) -----------
        compressed_direct = compress_keys(keys, ratio, out_dtype=self.indexer_dtype)
        compressed_cache = torch.zeros((storage_block_size, head_dim), dtype=self.indexer_dtype, device=device)
        if num_blocks > 0:
            token_group = (torch.arange(num_keys, device=device) // ratio).clamp_max(num_blocks - 1)
            rows_per_key = compressed_direct.index_select(0, token_group)
        else:
            rows_per_key = keys.new_zeros((num_keys, head_dim))
        qsa_scatter_rows(compressed_cache, metadata.compressed_slot_mapping, rows_per_key)

        # Read the compressed history back through the paged cache (slots 0..N-1).
        compressed_slots = torch.arange(num_blocks, device=device, dtype=torch.long)
        compressed_keys = qsa_gather_rows(compressed_cache, compressed_slots)

        # --- Score / deterministic top-k / expand -----------------------------
        token_indices, valid_counts = qsa_indexer_select(
            query,
            compressed_keys,
            positions,
            compress_ratio=ratio,
            token_topk=self.token_topk,
            accum_dtype=self.accumulation_dtype,
        )

        return QSAIndexerOutput(
            token_indices=token_indices,
            valid_counts=valid_counts,
            ring_cache=ring_cache,
            compressed_cache=compressed_keys,
        )


__all__ = ["AscendQwen4ExpQSAIndexer", "QSAIndexerOutput"]
