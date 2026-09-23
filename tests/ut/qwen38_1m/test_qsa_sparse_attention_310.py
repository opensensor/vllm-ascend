# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import (
    qsa_sparse_attention_310,
    qsa_sparse_attention_310_reference,
)


def _pack_nz(rows: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pack dense [B,S,H,D] rows into [P,H*D/16,block,16] NZ pages."""
    batch, seq_len, num_heads, head_dim = rows.shape
    num_physical_blocks = int(block_table.max()) + 1
    cache = rows.new_zeros((num_physical_blocks, num_heads * head_dim // 16, block_size, 16))
    for request in range(batch):
        for token in range(seq_len):
            logical_block, token_offset = divmod(token, block_size)
            physical_block = int(block_table[request, logical_block])
            cache[physical_block, :, token_offset, :] = rows[request, token].reshape(-1, 16)
    return cache


def test_reference_reads_permuted_nz_pages_and_maps_gqa_heads() -> None:
    torch.manual_seed(17)
    batch = 2
    seq_len = 12
    block_size = 4
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 16
    query = torch.randn(2, num_query_heads, head_dim, dtype=torch.float16)
    dense_key = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16)
    dense_value = torch.randn_like(dense_key)
    block_table = torch.tensor([[4, 1, 5], [2, 0, 3]], dtype=torch.int32)
    key_cache = _pack_nz(dense_key, block_table, block_size)
    value_cache = _pack_nz(dense_value, block_table, block_size)
    selection = QSAGroupSelection(
        group_indices=torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        group_counts=torch.tensor([2, 2], dtype=torch.int64),
        tail_starts=torch.tensor([8, 8], dtype=torch.int64),
        tail_counts=torch.tensor([2, 3], dtype=torch.int64),
    )
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)

    actual = qsa_sparse_attention_310_reference(
        query,
        key_cache,
        value_cache,
        selection,
        block_table,
        query_start_loc,
        scale=0.25,
    )

    expected = torch.empty_like(actual)
    selected = ([4, 5, 6, 7, 0, 1, 2, 3, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for request in range(batch):
        for query_head in range(num_query_heads):
            kv_head = query_head // (num_query_heads // num_kv_heads)
            keys = dense_key[request, selected[request], kv_head].float()
            values = dense_value[request, selected[request], kv_head].float()
            logits = keys @ query[request, query_head].float() * 0.25
            expected[request, query_head] = (torch.softmax(logits, dim=0) @ values).half()

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_native_entry_point_rejects_cpu_instead_of_silently_falling_back() -> None:
    query = torch.zeros((1, 2, 16), dtype=torch.float16)
    cache = torch.zeros((1, 2, 4, 16), dtype=torch.float16)
    selection = QSAGroupSelection(
        group_indices=torch.zeros((1, 1), dtype=torch.int64),
        group_counts=torch.ones(1, dtype=torch.int64),
        tail_starts=torch.zeros(1, dtype=torch.int64),
        tail_counts=torch.zeros(1, dtype=torch.int64),
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="NPU-only"):
        qsa_sparse_attention_310(query, cache, cache, selection, block_table, query_start_loc)


def test_contract_rejects_expanded_token_indices() -> None:
    query = torch.zeros((1, 2, 16), dtype=torch.float16)
    cache = torch.zeros((1, 2, 4, 16), dtype=torch.float16)
    selection = QSAGroupSelection(
        group_indices=torch.zeros((1, 1), dtype=torch.int64),
        group_counts=torch.ones(1, dtype=torch.int64),
        tail_starts=torch.zeros(1, dtype=torch.int64),
        tail_counts=torch.zeros(1, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="compress_ratio=4"):
        qsa_sparse_attention_310_reference(
            query,
            cache,
            cache,
            selection,
            torch.zeros((1, 1), dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            compress_ratio=1,
        )
