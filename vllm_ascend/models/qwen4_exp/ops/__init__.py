# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch (Triton-free) ops for the Ascend 310P Qwen4Exp QSA indexer (plan T6.1).

These ops port the *formulas* from the vLLM CUDA fork's
``models/qwen4_exp/nvidia/ops/qsa_indexer.py`` / ``qsa_pre_indexer.py`` and the
side-cache slot mappings / metadata torch fallback from
``models/qwen4_exp/common/qsa_cache.py`` (``_build_qsa_metadata_torch``) to plain
PyTorch. No Triton, no CUDA, no NPU kernel is imported on the 310P host path.

Modules:
    qsa_cache   -- QSA side-cache slot mappings, metadata (torch fallback),
                   and paged scatter/gather helpers (raw ring + compressed).
    qsa_indexer -- weight-free indexer scoring, deterministic top-k block
                   selection, and causal-tail expansion.
"""

from .qsa_cache import (
    PAD_SLOT_ID,
    QSAIndexerMetadata,
    build_qsa_indexer_metadata,
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
    qsa_gather_rows,
    qsa_logical_positions,
    qsa_scatter_rows,
    qsa_token_to_req,
    qsa_visible_blocks,
)
from .qsa_indexer import (
    compress_keys,
    expand_block_selection,
    indexer_block_scores,
    qsa_indexer_select,
    select_topk_blocks,
)

__all__ = [
    "PAD_SLOT_ID",
    "QSAIndexerMetadata",
    "build_qsa_indexer_metadata",
    "circular_qsa_slot_mapping",
    "compress_keys",
    "compressed_qsa_slot_mapping",
    "expand_block_selection",
    "indexer_block_scores",
    "qsa_gather_rows",
    "qsa_indexer_select",
    "qsa_logical_positions",
    "qsa_scatter_rows",
    "qsa_token_to_req",
    "qsa_visible_blocks",
    "select_topk_blocks",
]
