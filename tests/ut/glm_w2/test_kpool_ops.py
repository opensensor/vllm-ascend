# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.models.glm5next.kpool_ops import (
    compress_kpool,
    dense_kpool_token_indices,
    expand_kpool_groups,
    hadamard128,
    score_and_select_kpool_tokens,
    score_kpool,
    select_kpool_groups,
)


def test_hadamard_preserves_float32_dot_products():
    torch.manual_seed(11)
    a = torch.randn(3, 128)
    b = torch.randn(4, 128)
    rotated = hadamard128(a) @ hadamard128(b).T
    torch.testing.assert_close(rotated, a @ b.T, atol=2e-5, rtol=2e-5)


def test_compression_uses_gate_and_ape_per_dimension():
    keys = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    keys[0, :, 0] = torch.tensor([1, 2, 4, 8], dtype=torch.bfloat16)
    gate = torch.zeros(1, 4, 128)
    ape = torch.zeros(4, 128)
    ape[3, 0] = 12
    compressed = compress_kpool(keys, gate, ape)
    unrotated = torch.zeros(1, 128, dtype=torch.bfloat16)
    unrotated[0, 0] = 8
    expected = hadamard128(unrotated)
    # Only dimension zero is nonzero before the rotation.
    torch.testing.assert_close(compressed[0, 0].float(), expected[0, 0].float(), atol=0.1, rtol=0)
    assert compressed.shape == (1, 128)


def test_score_uses_relu_before_signed_head_weighting():
    queries = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    queries[0, 0, 0] = 1
    queries[0, 1, 0] = -1
    keys = torch.zeros(2, 128, dtype=torch.bfloat16)
    keys[0, 0] = 2
    keys[1, 0] = -3
    rotated = hadamard128(keys).to(torch.bfloat16)
    weights = torch.tensor([[2, -1]], dtype=torch.bfloat16)
    scores = score_kpool(queries, weights, rotated)
    torch.testing.assert_close(scores, torch.tensor([[4.0, -3.0]]), atol=0.1, rtol=0)


def test_head_weighted_query_changes_kpool_ranking():
    """Keep the per-head ReLU when optimizing the pooled-key score."""
    queries = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    queries[0, 0, 0] = 1
    queries[0, 1, 1] = 1
    keys = torch.zeros(2, 128, dtype=torch.bfloat16)
    keys[0, 0], keys[0, 1] = 1, -2
    keys[1, 0], keys[1, 1] = 0.25, 0.25
    rotated_keys = hadamard128(keys).to(torch.bfloat16)
    weights = torch.ones(1, 2, dtype=torch.bfloat16)

    reference = score_kpool(queries, weights, rotated_keys)
    folded_query = hadamard128((queries.float() * weights[:, :, None].float()).sum(1)).to(torch.bfloat16)
    folded = folded_query.float() @ rotated_keys.float().T

    assert reference[0, 0] > reference[0, 1]
    assert folded[0, 0] < folded[0, 1]


def test_topk_keeps_completed_pools_and_current_tail():
    logits = torch.tensor([[1.0, 4.0, 2.0], [1.0, 4.0, 2.0], [1.0, 4.0, 2.0]])
    positions = torch.tensor([2, 4, 11], dtype=torch.int32)
    groups, counts, starts, tails = select_kpool_groups(logits, positions, 8, 4)
    torch.testing.assert_close(counts, torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(starts, torch.tensor([0, 4, 12], dtype=torch.int32))
    torch.testing.assert_close(tails, torch.tensor([3, 1, 0], dtype=torch.int32))
    torch.testing.assert_close(groups, torch.tensor([[-1, -1], [0, -1], [1, 2]], dtype=torch.int32))
    tokens = expand_kpool_groups(groups, starts, tails, 4)
    torch.testing.assert_close(
        tokens,
        torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2],
                [0, 1, 2, 3, -1, -1, -1, -1, 4, -1, -1],
                [4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_dense_kpool_indices_match_zero_score_selection_through_budget_boundary():
    positions = torch.tensor([0, 1, 2, 3, 4, 7, 8, 2047, 2050], dtype=torch.int32)
    logits = torch.zeros(positions.numel(), 512)
    selected, _, tail_starts, tail_counts = select_kpool_groups(logits, positions, 2048, 4)
    expected = expand_kpool_groups(selected, tail_starts, tail_counts, 4)

    actual = dense_kpool_token_indices(positions, 2048, 4)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (positions.numel(), 2051)


def test_future_pool_cannot_change_earlier_selection():
    torch.manual_seed(19)
    logits = torch.randn(6, 8)
    positions = torch.tensor([0, 2, 3, 6, 10, 15], dtype=torch.int32)
    original = select_kpool_groups(logits, positions, 8, 4)[0]
    changed = logits.clone()
    changed[:, 4:] += 10000
    after = select_kpool_groups(changed, positions, 8, 4)[0]
    torch.testing.assert_close(after, original)


def test_chunked_kpool_scoring_matches_full_topk():
    torch.manual_seed(31)
    queries = torch.randn(12, 3, 128, dtype=torch.bfloat16)
    weights = torch.randn(12, 3)
    keys = torch.randn(11, 128, dtype=torch.float16)
    positions = torch.arange(12, 24, dtype=torch.int32)
    selected, _, starts, counts = select_kpool_groups(score_kpool(queries, weights, keys), positions, 8, 4)
    expected = expand_kpool_groups(selected, starts, counts, 4)

    with (
        patch("vllm_ascend.models.glm5next.kpool_ops.MAX_KPOOL_SCORE_ELEMENTS", 2 * 3 * 11),
        patch("vllm_ascend.models.glm5next.kpool_ops.score_kpool", wraps=score_kpool) as score,
    ):
        actual = score_and_select_kpool_tokens(queries, weights, keys, positions, 8, 4)

    torch.testing.assert_close(actual, expected)
    assert score.call_count == 6
    assert all(call.args[0].shape[0] <= 2 for call in score.call_args_list)


def test_indexer_completes_a_pool_across_prefill_chunks():
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    torch.manual_seed(29)
    keys = torch.randn(9, 128, dtype=torch.bfloat16)
    gates = torch.randn(9, 128)
    ape = torch.randn(4, 128)
    queries = torch.randn(9, 2, 128, dtype=torch.bfloat16)
    weights = torch.ones(9, 2, dtype=torch.bfloat16)
    key_cache = torch.zeros(4, 4, 1, 128, dtype=torch.float16)
    state_cache = torch.zeros(4, 4, 256)
    index_layer = SimpleNamespace(kv_cache=key_cache, prefix="index")
    state_layer = SimpleNamespace(kv_cache=state_cache, prefix="state")
    topk_buffer = torch.full((16, 8), -1, dtype=torch.int32)
    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.k_cache = index_layer
    indexer.tail_cache = state_layer
    indexer.topk_tokens = 4
    indexer.head_dim = 128
    indexer.topk_indices_buffer = topk_buffer
    indexer.skip_k_cache_insert = False

    compression_batch_sizes = []
    for start, end in ((0, 6), (6, 7), (7, 9)):
        positions = torch.arange(start, end, dtype=torch.int32)
        slots = torch.full((end - start,), -1, dtype=torch.int32)
        completed = (positions + 1) % 4 == 0
        slots[completed] = positions[completed] // 4
        metadata = {
            "index": SimpleNamespace(
                num_actual_tokens=end - start,
                slot_mapping=slots,
                seq_lens_cpu=torch.tensor([end // 4]),
                raw_seq_lens=torch.tensor([end]),
                block_table=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
                cum_query_lens=torch.tensor([end - start], dtype=torch.int32),
                cum_query_lens_cpu=torch.tensor([0, end - start], dtype=torch.int32),
            ),
            "state": SimpleNamespace(slot_mapping=positions),
        }
        with (
            patch(
                "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=metadata),
            ),
            patch(
                "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.compress_kpool",
                wraps=compress_kpool,
            ) as compress,
        ):
            indexer.forward_oot(
                keys[start:end],
                queries[start:end],
                keys[start:end],
                weights[start:end],
                gate_score=gates[start:end],
                compress_ape=ape,
                index_kpool=4,
                positions=positions,
            )
            compression_batch_sizes.append(compress.call_args.args[0].shape[0])

    # Capture-safe compression keeps a fixed row count for each invocation.
    assert compression_batch_sizes == [6, 1, 2]
    expected = compress_kpool(keys[4:8].unsqueeze(0), gates[4:8].unsqueeze(0), ape)
    torch.testing.assert_close(key_cache[0, 1, 0], expected[0].to(key_cache.dtype))
    # One complete pool is selected; token 8 remains the mandatory tail.
    selected = topk_buffer[1]
    assert selected[0] in (0, 4)
    assert selected[4] == 8


def test_indexer_keeps_two_requests_in_separate_cache_pages():
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    keys = torch.ones(8, 128, dtype=torch.bfloat16)
    keys[4:] *= 3
    gates = torch.zeros(8, 128)
    ape = torch.zeros(4, 128)
    key_cache = torch.zeros(2, 4, 1, 128, dtype=torch.float16)
    state_cache = torch.zeros(2, 4, 256)
    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.k_cache = SimpleNamespace(kv_cache=key_cache, prefix="index")
    indexer.tail_cache = SimpleNamespace(kv_cache=state_cache, prefix="state")
    indexer.topk_tokens = 8
    indexer.head_dim = 128
    indexer.topk_indices_buffer = torch.full((8, 12), 99, dtype=torch.int32)
    indexer.skip_k_cache_insert = False
    metadata = {
        "index": SimpleNamespace(
            num_actual_tokens=8,
            slot_mapping=torch.tensor([-1, -1, -1, 0, -1, -1, -1, 4]),
            seq_lens_cpu=torch.tensor([1, 1]),
            raw_seq_lens=torch.tensor([4, 4]),
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            cum_query_lens=torch.tensor([4, 8], dtype=torch.int32),
            cum_query_lens_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        ),
        "state": SimpleNamespace(slot_mapping=torch.arange(8)),
    }
    positions = torch.arange(4, dtype=torch.int32).repeat(2)
    with (
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=metadata),
        ),
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.dense_kpool_token_indices",
            wraps=dense_kpool_token_indices,
        ) as dense,
    ):
        selected = indexer.forward_oot(
            keys,
            torch.ones(8, 1, 128, dtype=torch.bfloat16),
            keys,
            torch.ones(8, 1),
            gate_score=gates,
            compress_ape=ape,
            index_kpool=4,
            positions=positions,
        )
    torch.testing.assert_close(key_cache[0, 0, 0], compress_kpool(keys[:4][None], gates[:4][None], ape)[0].half())
    torch.testing.assert_close(key_cache[1, 0, 0], compress_kpool(keys[4:][None], gates[4:][None], ape)[0].half())
    expected_tokens = torch.arange(4, dtype=torch.int32)
    torch.testing.assert_close(selected[3, :4], expected_tokens)
    torch.testing.assert_close(selected[7, :4], expected_tokens)
    assert selected[0, 8] == selected[4, 8] == 0
    assert torch.all(selected[:, 11] == -1)
    assert dense.call_count == 1  # both requests share one batched write


def _reference_pool_write(keys, gates, ape, positions, pool_size, index_meta, state_meta, state, cache):
    """Original masked writer, retained as a CPU oracle for graph-safe writes."""
    num_tokens = keys.shape[0]
    slots = state_meta.slot_mapping[:num_tokens].long()
    safe_slots = slots.clamp_min(0)
    blocks = torch.div(safe_slots, pool_size, rounding_mode="floor")
    old_state = state[blocks]
    offsets = torch.arange(pool_size - 1, -1, -1)
    local = torch.arange(num_tokens)[:, None] - offsets[None, :]
    safe_local = local.clamp_min(0)
    request_ids = torch.searchsorted(index_meta.cum_query_lens, torch.arange(num_tokens), right=True)
    same_pool = (
        (local >= 0)
        & (positions[safe_local] == positions[:, None] - offsets[None, :])
        & (request_ids[safe_local] == request_ids[:, None])
    )
    pool_keys = torch.where(same_pool[:, :, None], keys[safe_local].float(), old_state[:, :, :128])
    pool_gates = torch.where(same_pool[:, :, None], gates[safe_local].float(), old_state[:, :, 128:])
    final_positions = index_meta.raw_seq_lens[request_ids].long() - 1
    final_starts = torch.div(final_positions, pool_size, rounding_mode="floor") * pool_size
    valid_state = (slots >= 0) & (positions >= final_starts)
    state[blocks[valid_state], safe_slots[valid_state] % pool_size] = torch.cat(
        (keys[valid_state].float(), gates[valid_state].float()), dim=-1
    )
    completed = ((positions + 1) % pool_size == 0) & (index_meta.slot_mapping[:num_tokens] >= 0)
    compressed = compress_kpool(pool_keys[completed], pool_gates[completed], ape)
    pool_slots = index_meta.slot_mapping[:num_tokens][completed].long()
    for slot, value in zip(pool_slots.tolist(), compressed):
        cache[slot // cache.shape[1], slot % cache.shape[1], 0] = value.to(cache.dtype)


@pytest.mark.parametrize("key_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("positions", "state_slots", "key_slots", "query_ends", "raw_lengths"),
    [
        ([0, 1, 2], [0, 1, 2], [-1, -1, -1], [3], [3]),
        ([3, 4, 5, 6], [3, 4, 5, 6], [0, -1, -1, -1], [4], [7]),
        ([7, 8], [7, 8], [1, -1], [2], [9]),
        ([0, 1, 2, 3, 0, 1, 2, 3], list(range(8)), [-1, -1, -1, 0, -1, -1, -1, 4], [4, 8], [4, 4]),
        ([3, 0], [-1, 0], [-1, -1], [1, 2], [4, 1]),
        ([0, 1, 2, 3], [0, 1, 2, 3], [-1, -1, -1, 0], [4], [4]),
        ([], [], [], [0], [0]),
    ],
)
def test_graph_safe_pool_writer_matches_masked_reference(
    positions, state_slots, key_slots, query_ends, raw_lengths, key_dtype
):
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    torch.manual_seed(81)
    count = len(positions)
    keys = torch.randn(count, 128, dtype=torch.bfloat16)
    gates = torch.randn(count, 128)
    ape = torch.randn(4, 128)
    state_before = torch.randn(3, 4, 256)
    key_before = torch.randn(3, 4, 1, 128, dtype=key_dtype)
    state_actual, state_expected = state_before.clone(), state_before.clone()
    key_actual, key_expected = key_before.clone(), key_before.clone()
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor(key_slots, dtype=torch.int32),
        raw_seq_lens=torch.tensor(raw_lengths, dtype=torch.int32),
        cum_query_lens=torch.tensor(query_ends, dtype=torch.int32),
    )
    state_metadata = SimpleNamespace(slot_mapping=torch.tensor(state_slots, dtype=torch.int32))
    positions_tensor = torch.tensor(positions, dtype=torch.int32)
    _reference_pool_write(keys, gates, ape, positions_tensor, 4, metadata, state_metadata, state_expected, key_expected)
    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.head_dim = 128
    indexer.k_cache = SimpleNamespace(kv_cache=key_actual)
    indexer.tail_cache = SimpleNamespace(kv_cache=state_actual)
    indexer._write_pools(keys, gates, ape, positions_tensor, 4, metadata, state_metadata)

    torch.testing.assert_close(state_actual, state_expected, rtol=0, atol=0)
    torch.testing.assert_close(key_actual, key_expected, rtol=0, atol=0)


def test_graph_safe_pool_writer_preserves_strided_storage_and_reused_page():
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    torch.manual_seed(92)
    backing = torch.randn(2, 8, 1, 128, dtype=torch.float16)
    actual_backing, expected_backing = backing.clone(), backing.clone()
    actual_keys, expected_keys = actual_backing[:, ::2], expected_backing[:, ::2]
    actual_state = torch.randn(2, 4, 256)
    expected_state = actual_state.clone()
    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.head_dim = 128
    indexer.k_cache = SimpleNamespace(kv_cache=actual_keys)
    indexer.tail_cache = SimpleNamespace(kv_cache=actual_state)
    ape = torch.randn(4, 128)

    for key_value in (1.0, -2.0):
        keys = torch.full((4, 128), key_value, dtype=torch.bfloat16)
        gates = torch.randn(4, 128)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = SimpleNamespace(
            slot_mapping=torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
            raw_seq_lens=torch.tensor([4], dtype=torch.int32),
            cum_query_lens=torch.tensor([4], dtype=torch.int32),
        )
        state_metadata = SimpleNamespace(slot_mapping=torch.arange(4, dtype=torch.int32))
        _reference_pool_write(keys, gates, ape, positions, 4, metadata, state_metadata, expected_state, expected_keys)
        indexer._write_pools(keys, gates, ape, positions, 4, metadata, state_metadata)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(actual_backing, expected_backing, rtol=0, atol=0)


def test_graph_capture_keeps_kpool_selection_eager_with_current_metadata():
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.topk_indices_buffer = torch.empty(4, 12, dtype=torch.int32)
    indexer.topk_tokens = 8
    indexer.k_cache = SimpleNamespace(prefix="index.cache")
    indexer.tail_cache = SimpleNamespace(prefix="index.tail")
    indexer.skip_k_cache_insert = False
    indexer._write_pools = Mock()
    indexer._select_tokens = Mock(return_value=indexer.topk_indices_buffer)
    captured_metadata = SimpleNamespace(num_actual_tokens=1)
    replay_metadata = SimpleNamespace(num_actual_tokens=2, seq_lens_cpu=None)
    context = SimpleNamespace(
        attn_metadata={
            "index.cache": captured_metadata,
            "index.tail": SimpleNamespace(),
        }
    )

    class Capture:
        _capturing = True

        def add_eager(self, fn):
            context.attn_metadata["index.cache"] = replay_metadata
            fn()

    query = torch.randn(2, 2, 4)
    keys = torch.randn(2, 4)
    weights = torch.randn(2, 2)
    positions = torch.arange(2, dtype=torch.int32)
    with (
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.BreakableCUDAGraphCapture.current",
            return_value=Capture(),
        ),
    ):
        actual = indexer.forward_oot(
            torch.empty(2, 1),
            query,
            keys,
            weights,
            gate_score=torch.randn(2, 4),
            compress_ape=torch.randn(4, 4),
            index_kpool=4,
            positions=positions,
        )

    assert actual is indexer.topk_indices_buffer
    indexer._select_tokens.assert_called_once()
    assert indexer._select_tokens.call_args.args[-2] == 4
    assert indexer._select_tokens.call_args.args[-1] is replay_metadata


@pytest.mark.parametrize(("pool_length", "expect_eager"), [(1, False), (2, True)])
def test_graph_capture_skips_eager_kpool_selection_below_budget(pool_length, expect_eager):
    from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool

    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.topk_indices_buffer = torch.full((1, 12), 99, dtype=torch.int32)
    indexer.topk_tokens = 8
    indexer.k_cache = SimpleNamespace(kv_cache=torch.zeros(1, 4, 1, 4), prefix="index.cache")
    indexer.tail_cache = SimpleNamespace(prefix="index.tail")
    indexer.skip_k_cache_insert = False
    indexer._write_pools = Mock()
    indexer._select_tokens = Mock(return_value=indexer.topk_indices_buffer)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        seq_lens_cpu=torch.tensor([pool_length]),
        block_table=torch.tensor([[0]]),
    )
    context = SimpleNamespace(attn_metadata={"index.cache": metadata, "index.tail": SimpleNamespace()})

    class Capture:
        _capturing = True

        def add_eager(self, fn):
            fn()

    with (
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
            return_value=context,
        ),
        patch(
            "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.BreakableCUDAGraphCapture.current",
            return_value=Capture(),
        ),
    ):
        output = indexer.forward_oot(
            torch.empty(1, 1),
            torch.empty(1, 1, 4),
            torch.empty(1, 4),
            torch.empty(1, 1),
            gate_score=torch.empty(1, 4),
            compress_ape=torch.empty(4, 4),
            index_kpool=4,
            positions=torch.tensor([0], dtype=torch.int32),
        )

    assert output is indexer.topk_indices_buffer
    assert torch.all(output[0] == -1)
    if expect_eager:
        indexer._select_tokens.assert_called_once()
    else:
        indexer._select_tokens.assert_not_called()


def test_glm_indexer_scales_head_weights_in_310p_supported_float32():
    from vllm_ascend.models.glm5next.attention import Indexer

    indexer = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_head = 1
    indexer.head_dim = 128
    indexer.rope_dim = 0
    indexer.softmax_scale = 128**-0.5
    indexer.index_kpool = 4
    indexer.index_kpool_compress_ape = torch.zeros(4, 128)
    indexer._wk_weight_f32 = torch.zeros(129, 4)
    indexer._wk_weight_f32[-1] = 1
    indexer._gate_weight_f32 = torch.zeros(128, 4)
    indexer.wq_b = lambda qr: (torch.ones(2, 128, dtype=torch.bfloat16), None)
    indexer.k_norm = lambda k: k
    captured = {}

    def capture_indexer_op(hidden_states, q, k, weights, **kwargs):
        captured["weights"] = weights
        return torch.empty(0)

    indexer.indexer_op = capture_indexer_op
    indexer.forward(
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.arange(2),
        None,
    )
    assert captured["weights"].dtype == torch.float32
