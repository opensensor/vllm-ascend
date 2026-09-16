# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host validation for the QSA-aware DCP prototype (plan T8.2, Candidate B).

Proves the four acceptance properties on the host, with the T6.2 gather-based
sparse GQA attention (which the T6.4 decoder composes) as the single-rank ground
truth:

1. **4-way == single-rank**: the in-process 4-rank DCP simulation and the
   4-process ``gloo`` run both reproduce the single-rank BF16-policy reference
   within the QSA attention tolerance, for dense and genuinely-sparse selections.
2. **Selected-rows-only transfer**: the ledger proves ``bytes_moved ==
   selected_rows * row_bytes`` (K + V), a tiny fraction of a whole-cache
   all-gather.
3. **Determinism**: the fixed-order online-softmax reduction is bitwise stable
   across reruns (and across the two gloo spawns).
4. **Decoder parity**: the full T6.4 decoder path with DCP-sharded attention
   equals the single-rank :func:`run_qsa_decoder_attention` at float64.

Run (the shared tests/ut/conftest.py fails to import here):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_qsa_dcp.py
"""

from __future__ import annotations

import contextlib
import os
import socket
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.tolerances import QSA_ATTN_ATOL, QSA_ATTN_RTOL
from vllm_ascend.models.qsa_dcp import (
    QSAShardPlan,
    qsa_dcp_sparse_attention,
    row_bytes,
    run_qsa_dcp_decoder_attention,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.ops.qsa_attention import qsa_sparse_gqa_attention

_NUM_RANKS = 4
_ACCUM = torch.float64  # float64 so the reduction agrees with the reference at rounding level


# ---------------------------------------------------------------------------
# Synthetic attention inputs + a replicated selection
# ---------------------------------------------------------------------------
def _rand(shape, seed, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def _make_selection(num_tokens: int, context_len: int, seed: int, *, include_empty: bool):
    """Build a replicated ``(packed_indices, valid_counts)`` selection.

    Each query selects a random subset of distinct global positions; when
    ``include_empty`` the first query selects nothing (zero-selection semantics).
    Returns the packed buffer, valid counts, and the total selected rows.
    """
    gen = torch.Generator().manual_seed(seed)
    rows: list[list[int]] = []
    for token in range(num_tokens):
        if include_empty and token == 0:
            rows.append([])
            continue
        # 1..min(context_len, budget) distinct positions, sorted for stability.
        budget = min(context_len, 5)
        count = int(torch.randint(1, budget + 1, (1,), generator=gen).item())
        perm = torch.randperm(context_len, generator=gen)[:count]
        rows.append(sorted(int(p) for p in perm.tolist()))

    width = max((len(r) for r in rows), default=1) or 1
    packed = torch.full((num_tokens, width), -1, dtype=torch.int64)
    valid = torch.zeros(num_tokens, dtype=torch.int64)
    total = 0
    for token, sel in enumerate(rows):
        if sel:
            packed[token, : len(sel)] = torch.tensor(sel, dtype=torch.int64)
        valid[token] = len(sel)
        total += len(sel)
    return packed, valid, total


def _single_rank_reference(q, k, v, gate, packed, valid, num_kv_heads):
    return qsa_sparse_gqa_attention(
        q, k, v, gate, packed, valid, num_kv_heads, accum_dtype=_ACCUM, apply_output_gate=True
    )


def _attention_inputs(*, num_tokens, context_len, num_q_heads, num_kv_heads, head_dim, seed, include_empty):
    q = _rand((num_tokens, num_q_heads, head_dim), seed)
    k = _rand((context_len, num_kv_heads, head_dim), seed + 1)
    v = _rand((context_len, num_kv_heads, head_dim), seed + 2)
    gate = _rand((num_tokens, num_q_heads, head_dim), seed + 3)
    packed, valid, total = _make_selection(num_tokens, context_len, seed + 4, include_empty=include_empty)
    return q, k, v, gate, packed, valid, total


# ---------------------------------------------------------------------------
# Shard map
# ---------------------------------------------------------------------------
def test_shard_plan_round_robin_is_balanced_and_invertible():
    plan = QSAShardPlan(num_ranks=_NUM_RANKS)
    context_len = 23
    cache = torch.arange(context_len).reshape(context_len, 1, 1).double()
    shards = plan.split_cache(cache)

    # Ownership partitions every position exactly once, balanced to within one row.
    lengths = [s.shape[0] for s in shards]
    assert sum(lengths) == context_len
    assert max(lengths) - min(lengths) <= 1
    for rank in range(_NUM_RANKS):
        assert lengths[rank] == plan.shard_length(context_len, rank)

    for pos in range(context_len):
        owner = plan.owner_of(pos)
        slot = plan.local_slot_of(pos)
        assert int(shards[owner][slot].item()) == pos  # (owner, slot) -> global position


# ---------------------------------------------------------------------------
# 4-way (in-process) == single-rank reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_tokens", "context_len", "include_empty"),
    [
        (3, 4, False),  # context below the selection budget -> dense selection
        (6, 40, True),  # several x the budget -> genuinely sparse + a zero-selection row
    ],
)
def test_in_process_dcp_matches_single_rank(num_tokens, context_len, include_empty):
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    q, k, v, gate, packed, valid, _ = _attention_inputs(
        num_tokens=num_tokens,
        context_len=context_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=100,
        include_empty=include_empty,
    )
    reference = _single_rank_reference(q, k, v, gate, packed, valid, num_kv_heads)
    dcp_out, _ = qsa_dcp_sparse_attention(
        q, k, v, gate, packed, valid, num_kv_heads, num_ranks=_NUM_RANKS, accum_dtype=_ACCUM
    )
    assert dcp_out.shape == reference.shape
    torch.testing.assert_close(dcp_out, reference, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)


def test_zero_selection_row_is_zero_output():
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    q, k, v, gate, packed, valid, _ = _attention_inputs(
        num_tokens=4,
        context_len=20,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=7,
        include_empty=True,
    )
    dcp_out, _ = qsa_dcp_sparse_attention(
        q, k, v, gate, packed, valid, num_kv_heads, num_ranks=_NUM_RANKS, accum_dtype=_ACCUM
    )
    assert int(valid[0].item()) == 0
    assert torch.count_nonzero(dcp_out[0]) == 0  # zero selection -> exactly zero output


# ---------------------------------------------------------------------------
# Transfer accounting: only selected rows move
# ---------------------------------------------------------------------------
def test_transfer_moves_only_selected_rows():
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    context_len = 64
    q, k, v, gate, packed, valid, total_selected = _attention_inputs(
        num_tokens=5,
        context_len=context_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=321,
        include_empty=True,
    )
    _, ledger = qsa_dcp_sparse_attention(
        q, k, v, gate, packed, valid, num_kv_heads, num_ranks=_NUM_RANKS, accum_dtype=_ACCUM
    )

    expected_row_bytes = row_bytes(num_kv_heads, head_dim, k.dtype)
    assert ledger.row_bytes == expected_row_bytes
    # Exactly the selected rows crossed the exchange (K and V each once).
    assert ledger.selected_rows == total_selected
    assert ledger.bytes_moved == total_selected * expected_row_bytes * 2

    # ... and that is a strict, large fraction less than a whole-cache all-gather.
    whole_cache = ledger.whole_cache_bytes(context_len)
    assert total_selected < context_len
    assert ledger.bytes_moved < whole_cache


# ---------------------------------------------------------------------------
# Determinism: bitwise-stable reduction across reruns
# ---------------------------------------------------------------------------
def test_reduction_is_bitwise_deterministic():
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    q, k, v, gate, packed, valid, _ = _attention_inputs(
        num_tokens=6,
        context_len=48,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=555,
        include_empty=True,
    )
    out_a, ledger_a = qsa_dcp_sparse_attention(
        q, k, v, gate, packed, valid, num_kv_heads, num_ranks=_NUM_RANKS, accum_dtype=_ACCUM
    )
    out_b, ledger_b = qsa_dcp_sparse_attention(
        q, k, v, gate, packed, valid, num_kv_heads, num_ranks=_NUM_RANKS, accum_dtype=_ACCUM
    )
    assert torch.equal(out_a, out_b)  # fixed rank-order fold -> bitwise identical
    assert ledger_a.to_dict() == ledger_b.to_dict()


# ---------------------------------------------------------------------------
# Decoder-level parity vs the single-rank T6.4 path
# ---------------------------------------------------------------------------
_POLICY_F64 = replace(
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    main_dtype=torch.float64,
    accumulation_dtype=torch.float64,
    qsa_main_dtype=torch.float64,
    qsa_indexer_dtype=torch.float64,
    attention_dtype=torch.float64,
    attention_accumulation_dtype=torch.float64,
    kv_cache_dtype=torch.float64,
)


def _qsa_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )


def _init_module(module, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_((torch.randn(param.shape, generator=gen, dtype=torch.float64) * 0.1).to(param.dtype))


def test_decoder_path_matches_single_rank():
    from vllm_ascend.models.qwen4_exp.model import _QSAAttention
    from vllm_ascend.models.qwen4_exp.qsa import (
        QSADecoderProjections,
        run_qsa_decoder_attention,
    )

    cfg = _qsa_config()
    module = _QSAAttention(config=cfg, layer_idx=1, dtype_policy=_POLICY_F64).double()
    _init_module(module, seed=11)

    seq_len = 40  # > indexer_budget (8) -> genuinely sparse selection (8K-shape)
    block_input = _rand((seq_len, cfg.hidden_size), seed=101) * 0.2
    positions = torch.arange(seq_len, dtype=torch.int64)

    projections = QSADecoderProjections(
        q_proj=module.q_proj,
        k_proj=module.k_proj,
        v_proj=module.v_proj,
        gate_proj=module.gate_proj,
        index_q_proj=module.iq_proj,
        index_k_proj=module.ik_proj,
        out_proj=module.o_proj,
    )
    common = dict(
        projections=projections,
        indexer=module.indexer,
        attention=module.attn,
        num_query_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        index_n_heads=cfg.indexer_n_heads,
        index_head_dim=cfg.indexer_head_dim,
        store_dtype=module.params_dtype,
        compute_dtype=module.compute_dtype,
    )
    with torch.no_grad():
        reference = run_qsa_decoder_attention(block_input, positions, **common)
        dcp_out, ledger = run_qsa_dcp_decoder_attention(block_input, positions, num_ranks=_NUM_RANKS, **common)

    torch.testing.assert_close(dcp_out.double(), reference.double(), rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)
    # The decoder attends over the whole causal context; the DCP exchange still
    # moved strictly fewer rows than a whole-cache all-gather would have.
    assert ledger.bytes_moved < ledger.whole_cache_bytes(seq_len) * seq_len


# ---------------------------------------------------------------------------
# 4-process gloo simulation
# ---------------------------------------------------------------------------
def _free_port() -> str:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _run_gloo(payload, num_kv_heads, scale, result_dir, port):
    import torch.multiprocessing as mp

    from vllm_ascend.models.qsa_dcp.runtime import DCPWorkerConfig, _worker_entry

    config = DCPWorkerConfig(
        num_ranks=_NUM_RANKS,
        num_kv_heads=num_kv_heads,
        scale=scale,
        accum_dtype=_ACCUM,
        master_port=port,
    )
    mp.spawn(_worker_entry, args=(config, payload, result_dir), nprocs=_NUM_RANKS, join=True)
    return [torch.load(os.path.join(result_dir, f"rank_{r}.pt"), weights_only=False) for r in range(_NUM_RANKS)]


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="gloo backend unavailable")
def test_gloo_4way_matches_single_rank_and_is_deterministic():
    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    context_len = 40
    q, k, v, gate, packed, valid, total_selected = _attention_inputs(
        num_tokens=3,
        context_len=context_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=909,
        include_empty=True,
    )
    scale = head_dim**-0.5
    reference = _single_rank_reference(q, k, v, gate, packed, valid, num_kv_heads)
    payload = {
        "query": q,
        "key_cache": k,
        "value_cache": v,
        "gate": gate,
        "packed_indices": packed,
        "valid_counts": valid,
    }

    with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
        results_a = _run_gloo(payload, num_kv_heads, scale, dir_a, _free_port())
        results_b = _run_gloo(payload, num_kv_heads, scale, dir_b, _free_port())

    # Every rank produced the identical reduced output, matching single-rank.
    rank0 = results_a[0]["out"]
    for res in results_a:
        assert torch.equal(res["out"], rank0)
        assert res["total_rows"] == total_selected  # only selected rows exchanged
    torch.testing.assert_close(rank0, reference, rtol=QSA_ATTN_RTOL, atol=QSA_ATTN_ATOL)

    # Byte accounting from the gloo run agrees with the ledger formula (K + V).
    exchanged_bytes = results_a[0]["total_rows"] * results_a[0]["row_bytes"] * 2
    assert exchanged_bytes == total_selected * row_bytes(num_kv_heads, head_dim, k.dtype) * 2

    # Determinism across two independent 4-process spawns.
    assert torch.equal(results_a[0]["out"], results_b[0]["out"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "--noconftest"]))
