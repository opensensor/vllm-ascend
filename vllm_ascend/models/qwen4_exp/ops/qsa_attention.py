# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse attention ops (torch, Triton-free) for the Ascend 310P (plan T6.2).

Ports the *formulas* (never the Triton code) from the vLLM CUDA fork's
``models/qwen4_exp/nvidia/ops/qsa.py`` (``_qsa_sparse_paged_gqa_splitk_kernel``)
to plain PyTorch: softmax attention over the *selected* token set with GQA head
grouping, a ``head_dim ** -0.5`` softmax scale, selection-count semantics (the
packed buffer's trailing count column bounds the tile loop), and an
``out * sigmoid(gate)`` output gate.

The selected rows are pulled from the main K/V caches through the T6.1 gather
(:func:`vllm_ascend.models.qwen4_exp.ops.qsa_cache.qsa_gather_rows`); this module
does NOT reimplement cache mapping or the indexer selection. Dtypes are supplied
by the caller from the authoritative dtype policy (no dtype literals here except
the fp32 accumulation *default*, which callers override from the policy).

Numerics are torch-eager sized for correctness (a fused NPU kernel replaces the
inner tile loop later); the gather order is the packed selection order and the
softmax is a stable reduction, so the op is run-to-run bitwise deterministic.

C8 quant seam (Candidate A / task T8.1): :func:`qsa_write_kv_to_cache` accepts an
optional ``quant_hook`` applied to the K/V rows before the paged scatter. The
hook is left unset here (BF16/float16 write path); T8.1 fills it in.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .qsa_cache import qsa_gather_rows, qsa_scatter_rows

# Type of the reserved C8 write-path quant hook (Candidate A, task T8.1).
# ``(key_rows, value_rows) -> (key_rows_q, value_rows_q)`` applied before scatter.
QSAKVQuantHook = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _flatten_cache(cache: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """View a ``[S, Hkv, D]`` (or ``[S, Hkv*D]``) cache as flat ``[S, Hkv*D]``.

    Returns the flat view plus ``(num_kv_heads, head_dim)`` when recoverable.
    """
    if cache.ndim == 3:
        num_slots, num_kv_heads, head_dim = cache.shape
        return cache.reshape(num_slots, num_kv_heads * head_dim), num_kv_heads, head_dim
    if cache.ndim == 2:
        return cache, -1, -1
    raise ValueError("QSA K/V cache must be [S, Hkv, D] or [S, Hkv*D]")


def gather_selected_rows(
    cache: torch.Tensor,
    token_indices_row: torch.Tensor,
    valid_count: int,
) -> torch.Tensor:
    """Gather one query's selected context rows via the T6.1 paged gather.

    Honors the selection-count semantics: only the first ``valid_count`` packed
    entries are considered, and ``-1`` padding is dropped (gather order = packed
    order, so the result is deterministic).

    Args:
        cache: ``[S, Hkv, D]`` or flat ``[S, Hkv*D]`` context cache.
        token_indices_row: ``[W]`` packed selection ids for one query token.
        valid_count: number of valid leading entries (the tile-loop bound).

    Returns:
        Gathered rows shaped ``[n, Hkv, D]`` (or ``[n, Hkv*D]`` for a flat cache).
    """
    flat, num_kv_heads, head_dim = _flatten_cache(cache)
    count = max(int(valid_count), 0)
    idx = token_indices_row[:count]
    idx = idx[idx >= 0].to(torch.long)
    rows = qsa_gather_rows(flat, idx)
    if num_kv_heads > 0:
        return rows.reshape(rows.shape[0], num_kv_heads, head_dim)
    return rows


def qsa_sparse_gqa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    packed_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    num_kv_heads: int,
    *,
    scale: float | None = None,
    accum_dtype: torch.dtype = torch.float32,
    apply_output_gate: bool = True,
) -> torch.Tensor:
    """Gather-based sparse GQA attention with an output gate (torch-eager).

    Mirrors the fork's ``_qsa_sparse_paged_gqa_splitk_kernel`` math and matches
    the T0.6 reference :func:`sparse_gqa_attention`: per query token, gather the
    selected context rows, run GQA softmax attention (scale ``head_dim**-0.5``)
    over exactly the selected set, then multiply by ``sigmoid(gate)``.

    Args:
        query: ``[T, Hq, D]`` (already Q-normed / RoPE'd upstream).
        key_cache, value_cache: ``[S, Hkv, D]`` full-context caches.
        gate: ``[T, Hq, D]`` pre-sigmoid output gate.
        packed_indices: ``[T, W]`` selected token ids (``-1`` padded).
        valid_counts: ``[T]`` valid-entry count per row (the loop bound).
        num_kv_heads: KV head count for GQA grouping.
        scale: softmax scale; defaults to ``head_dim ** -0.5``.
        accum_dtype: score / softmax / weighted-sum accumulation dtype.
        apply_output_gate: apply ``* sigmoid(gate)`` when ``True``.

    Returns:
        ``[T, Hq, D]`` gated attention output in ``accum_dtype``.
    """
    if query.ndim != 3:
        raise ValueError("query must be [T, Hq, D]")
    seq_len, num_q_heads, head_dim = query.shape
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by kv heads")
    group_size = num_q_heads // num_kv_heads
    softmax_scale = head_dim**-0.5 if scale is None else float(scale)

    q = query.to(accum_dtype)
    gate_acc = gate.to(accum_dtype)
    out = torch.zeros((seq_len, num_q_heads, head_dim), dtype=accum_dtype, device=query.device)

    for t in range(seq_len):
        count = int(valid_counts[t].item())
        k_sel = gather_selected_rows(key_cache, packed_indices[t], count).to(accum_dtype)
        if k_sel.shape[0] == 0:
            continue
        v_sel = gather_selected_rows(value_cache, packed_indices[t], count).to(accum_dtype)
        # [Hkv, group, D] query grouped onto the shared kv head.
        q_grp = q[t].reshape(num_kv_heads, group_size, head_dim)
        # scores[kv, g, n] = (q_grp . k_sel) * scale
        scores = torch.einsum("kgd,nkd->kgn", q_grp, k_sel) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("kgn,nkd->kgd", probs, v_sel)
        out[t] = ctx.reshape(num_q_heads, head_dim)

    if apply_output_gate:
        out = out * torch.sigmoid(gate_acc)
    return out


def qsa_write_kv_to_cache(
    key_cache_flat: torch.Tensor,
    value_cache_flat: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_rows: torch.Tensor,
    value_rows: torch.Tensor,
    *,
    quant_hook: QSAKVQuantHook | None = None,
) -> None:
    """Scatter this step's K/V rows into the flat main caches (paged write).

    Uses the T6.1 masked scatter (:func:`qsa_scatter_rows`); PAD slots are
    skipped. The BF16/float16 write path stores the rows as-is.

    C8 quant seam (Candidate A / task T8.1): when ``quant_hook`` is provided it is
    applied to ``(key_rows, value_rows)`` *before* the scatter, so an 8-bit
    compressed layout can be dropped in without touching this call site. It is
    left unset here -- do NOT implement C8 in this task.
    """
    if quant_hook is not None:
        key_rows, value_rows = quant_hook(key_rows, value_rows)
    qsa_scatter_rows(key_cache_flat, slot_mapping, key_rows)
    qsa_scatter_rows(value_cache_flat, slot_mapping, value_rows)


__all__ = [
    "QSAKVQuantHook",
    "gather_selected_rows",
    "qsa_sparse_gqa_attention",
    "qsa_write_kv_to_cache",
]
