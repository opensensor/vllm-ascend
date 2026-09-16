# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate B of the Qwen4Exp 1M-cache decision: QSA-aware Decode Context Parallel.

A host-measurement PROTOTYPE (plan T8.2) that sequence-shards the main QSA K/V
cache across the four Ascend 310P ranks, keeps the indexer history replicated,
exchanges only the indexer-selected rows, and reduces the per-rank partial
attention deterministically. It pre-decides nothing; D4 decides on hardware.

No NPU / Triton import here -- validated purely on host with a 4-process ``gloo``
simulation against the T6.2/T6.4 single-rank eager reference.
"""

from __future__ import annotations

from .attention import (
    QSAPartialState,
    compute_rank_partial,
    finalize_output,
    merge_partials,
    partial_from_owned_rows,
    qsa_dcp_sparse_attention,
)
from .decoder import run_qsa_dcp_decoder_attention
from .sharding import QSAShardPlan
from .transfer import TransferLedger, row_bytes

__all__ = [
    "QSAPartialState",
    "QSAShardPlan",
    "TransferLedger",
    "compute_rank_partial",
    "finalize_output",
    "merge_partials",
    "partial_from_owned_rows",
    "qsa_dcp_sparse_attention",
    "row_bytes",
    "run_qsa_dcp_decoder_attention",
]
