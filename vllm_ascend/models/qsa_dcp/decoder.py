# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA decoder-layer path with DCP-sharded main attention (plan T8.2, Candidate B).

Mirrors the T6.4 single-rank composition
(:func:`vllm_ascend.models.qwen4_exp.qsa.run_qsa_decoder_attention`) exactly --
same projections, same replicated indexer selection, same Q/K GemmaRMSNorm +
partial RoPE and output gate, same out-projection -- but replaces the dense
gather-based sparse attention with the sequence-sharded DCP attention in
:mod:`.attention`. Because everything outside the attention core is bit-identical
and the DCP core is a mathematically exact online-softmax reduction over the same
selected rows, this path reproduces the single-rank decoder output within the
QSA attention tolerance while moving only the selected rows across ranks.

Imports (never edits) the T6.1 indexer, the T6.2/T6.4 attention module, and the
T6.4 projection dataclass, honouring the T8.2 scope boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.models.qwen4_exp.qsa import (
    AscendQwen4ExpQSAAttention,
    QSADecoderProjections,
)

from .attention import qsa_dcp_sparse_attention
from .sharding import QSAShardPlan
from .transfer import TransferLedger


def run_qsa_dcp_decoder_attention(
    block_input: torch.Tensor,
    positions: torch.Tensor,
    *,
    projections: QSADecoderProjections,
    indexer: nn.Module,
    attention: AscendQwen4ExpQSAAttention,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    store_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    num_ranks: int,
    plan: QSAShardPlan | None = None,
) -> tuple[torch.Tensor, TransferLedger]:
    """T6.4 decoder attention with the main K/V cache sharded across ``num_ranks``.

    Args mirror :func:`run_qsa_decoder_attention`; ``num_ranks`` / ``plan`` add the
    DCP sharding. Returns ``(out, ledger)`` -- the ``[T, hidden]`` block output in
    ``store_dtype`` and the selected-row transfer ledger.
    """
    seq_len = block_input.shape[0]
    accum = attention.accumulation_dtype

    def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(compute_dtype), weight.to(compute_dtype))

    query = _linear(block_input, projections.q_proj).view(seq_len, num_query_heads, head_dim)
    key = _linear(block_input, projections.k_proj).view(seq_len, num_kv_heads, head_dim)
    value = _linear(block_input, projections.v_proj).view(seq_len, num_kv_heads, head_dim)
    gate = _linear(block_input, projections.gate_proj).view(seq_len, num_query_heads, head_dim)
    index_q = _linear(block_input, projections.index_q_proj).view(seq_len, index_n_heads, index_head_dim)
    index_k = _linear(block_input, projections.index_k_proj)

    selection = indexer(index_q.to(store_dtype), index_k.to(store_dtype), positions)

    # Q/K GemmaRMSNorm + partial RoPE in the model's order (T6.2), reusing the
    # attention module so the DCP path shares the single-rank projection code.
    q = query.to(store_dtype)
    k = key.to(store_dtype)
    v = value.to(store_dtype)
    gate_cast = gate.to(store_dtype)
    context_len = k.shape[0]
    key_positions = torch.arange(context_len, device=k.device)
    q, k = attention.project_qk(q, k, positions, key_positions, accum_dtype=accum)

    attn_out, ledger = qsa_dcp_sparse_attention(
        q,
        k,
        v,
        gate_cast,
        selection.token_indices,
        selection.valid_counts,
        num_kv_heads,
        num_ranks=num_ranks,
        plan=plan,
        scale=attention.scaling,
        accum_dtype=accum,
        apply_output_gate=True,
    )
    out = attn_out.to(store_dtype).reshape(seq_len, num_query_heads * head_dim)
    return _linear(out, projections.out_proj).to(store_dtype), ledger


__all__ = ["run_qsa_dcp_decoder_attention"]
