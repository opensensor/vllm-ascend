# SPDX-License-Identifier: Apache-2.0
"""Ascend 310P regression for full-width QSA attention score reduction."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from vllm_ascend.models.qwen4_exp.ops.qsa_index_cache_310 import (
    qsa_index_cache_shape,
    qsa_index_cache_update_310,
    qsa_index_cache_update_310_reference,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    QSAGroupSelection,
    _repair_native_group_indices,
    _stable_topk_indices,
    qsa_indexer_score_310_reference,
    qsa_indexer_select_groups_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import (
    qsa_sparse_attention_310,
    qsa_sparse_attention_310_reference,
)
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize(
    "num_tokens,num_groups,key_scale",
    [
        (2, 512, 0.1),
        (2, 512, 4.0),
        (8, 16, 0.1),
        (8, 16, 4.0),
        (8, 257, 0.1),
        (64, 256, 0.1),
        (64, 512, 0.1),
        (64, 512, 4.0),
    ],
)
@pytest.mark.parametrize("use_visible_blocks", [False, True])
def test_batched_prefill_matches_native_with_paged_cache_and_tail(
    num_tokens: int, num_groups: int, key_scale: float, use_visible_blocks: bool
) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(313)
    device = "npu:0"
    block_size = 128
    num_blocks = max(4, (num_groups * 4 + 4 + block_size - 1) // block_size)
    cache_blocks = num_blocks + 17
    query = torch.randn((num_tokens, 24, 256), dtype=torch.float16, device=device)
    key_cache = torch_npu.npu_format_cast(
        (torch.randn((cache_blocks, 32, block_size, 16), dtype=torch.float16) * key_scale).to(device), 29
    )
    value_cache = torch_npu.npu_format_cast(
        (torch.randn((cache_blocks, 32, block_size, 16), dtype=torch.float16) * 0.1).to(device), 29
    )
    group_ids = torch.randperm(num_groups, dtype=torch.int32, device=device).expand(num_tokens, -1).contiguous()
    group_counts = torch.full((num_tokens,), num_groups, dtype=torch.int32, device=device)
    group_counts[1::2] -= 1
    selection = QSAGroupSelection(
        group_indices=group_ids,
        group_counts=group_counts,
        tail_starts=torch.full((num_tokens,), num_groups * 4, dtype=torch.int32, device=device),
        tail_counts=torch.arange(num_tokens, dtype=torch.int32, device=device) % 5,
    )
    block_table = torch.randperm(cache_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    expected = qsa_sparse_attention_310(query, key_cache, value_cache, selection, block_table, query_start_loc)
    actual = qsa_batched_prefill_310(
        query,
        key_cache,
        value_cache,
        selection,
        block_table,
        query_start_loc,
        scale=256**-0.5,
        visible_blocks=num_blocks if use_visible_blocks else None,
    )
    # Compare on CPU: this test need not invoke torch_npu's unsupported
    # float64 IsClose tolerance path after the kernels have completed.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=2e-3)
    if num_tokens == 2 and num_groups == 512:
        decode_group_list = torch.arange(1, 2 * 2 + 1, dtype=torch.int64, device=device) * 12

        def grouped_decode() -> torch.Tensor:
            return qsa_batched_prefill_310(
                query,
                key_cache,
                value_cache,
                selection,
                block_table,
                query_start_loc,
                scale=256**-0.5,
                visible_blocks=num_blocks if use_visible_blocks else None,
                decode_group_list=decode_group_list,
            )

        grouped = grouped_decode()
        torch.testing.assert_close(grouped.cpu(), expected.cpu(), rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("scale", [256**-0.5, 0.07])
def test_batched_prefill_score_scale_matches_reference(scale: float) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(314)
    device = "npu:0"
    query = torch.randn((1, 24, 256), dtype=torch.float16)
    key_cache = torch.randn((2, 32, 128, 16), dtype=torch.float16) * 0.1
    value_cache = torch.randn((2, 32, 128, 16), dtype=torch.float16) * 0.1
    selection = QSAGroupSelection(
        group_indices=torch.arange(16, dtype=torch.int32).unsqueeze(0),
        group_counts=torch.tensor([16], dtype=torch.int32),
        tail_starts=torch.tensor([64], dtype=torch.int32),
        tail_counts=torch.tensor([2], dtype=torch.int32),
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    expected = qsa_sparse_attention_310_reference(
        query, key_cache, value_cache, selection, block_table, query_start_loc, scale=scale
    )
    selection_npu = QSAGroupSelection(
        *(getattr(selection, field).to(device) for field in selection.__dataclass_fields__)
    )
    actual = qsa_batched_prefill_310(
        query.to(device),
        torch_npu.npu_format_cast(key_cache.to(device), 29),
        torch_npu.npu_format_cast(value_cache.to(device), 29),
        selection_npu,
        block_table.to(device),
        query_start_loc.to(device),
        scale=scale,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("width", [512, 1712, 32768])
def test_native_stable_topk_keeps_lowest_index_on_ties(width: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    scores = torch.full((2, width), -torch.finfo(torch.float32).max, dtype=torch.float32)
    scores[:, :1024] = 1.0
    scores[:, width - 1] = 2.0
    expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :512]
    actual = _stable_topk_indices(scores.to("npu:0"), 512).cpu()
    assert torch.equal(actual, expected)


def test_native_reduces_all_128_score_dimensions() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    device = "npu:0"
    query = torch.ones((1, 2, 128), dtype=torch.float16)
    key_cache = torch.zeros((1, 8, 128, 16), dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    # Only dimensions 64..127 of the second key contribute to its score.
    # Reducing a single 64-element FP32 vector repeat incorrectly returns 0.
    key_cache[:, 4:8, 1] = 1
    value_cache[:, :, 1] = 1
    selection = QSAGroupSelection(
        group_indices=torch.zeros((1, 1), dtype=torch.int32),
        group_counts=torch.zeros(1, dtype=torch.int32),
        tail_starts=torch.zeros(1, dtype=torch.int32),
        tail_counts=torch.full((1,), 2, dtype=torch.int32),
    )
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    expected = qsa_sparse_attention_310_reference(
        query, key_cache, value_cache, selection, block_table, query_start_loc
    )
    selection_npu = QSAGroupSelection(
        *(getattr(selection, field).to(device) for field in selection.__dataclass_fields__)
    )
    actual = qsa_sparse_attention_310(
        query.to(device),
        torch_npu.npu_format_cast(key_cache.to(device), 29),
        torch_npu.npu_format_cast(value_cache.to(device), 29),
        selection_npu,
        block_table.to(device),
        query_start_loc.to(device),
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_native_online_softmax_with_changing_maximum() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    device = "npu:0"
    query = torch.ones((2, 2, 256), dtype=torch.float16)
    key_cache = torch.zeros((2, 16, 64, 16), dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    for token in range(80):
        # Most tokens leave the maximum unchanged; tokens 32 and 64 raise it.
        # Swap physical pages to exercise the cache address calculation.
        physical_block = 1 - token // 64
        token_offset = token % 64
        key_cache[physical_block, :, token_offset] = 1.0 if token == 64 else (0.5 if token == 32 else -0.5)
        value_cache[physical_block, :, token_offset] = (token % 7) / 7
    selection = QSAGroupSelection(
        group_indices=torch.arange(20, dtype=torch.int32).expand(2, -1).contiguous(),
        group_counts=torch.full((2,), 20, dtype=torch.int32),
        tail_starts=torch.zeros(2, dtype=torch.int32),
        tail_counts=torch.zeros(2, dtype=torch.int32),
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    expected = qsa_sparse_attention_310_reference(
        query, key_cache, value_cache, selection, block_table, query_start_loc
    )
    selection_npu = QSAGroupSelection(
        *(getattr(selection, field).to(device) for field in selection.__dataclass_fields__)
    )
    actual = qsa_sparse_attention_310(
        query.to(device),
        torch_npu.npu_format_cast(key_cache.to(device), 29),
        torch_npu.npu_format_cast(value_cache.to(device), 29),
        selection_npu,
        block_table.to(device),
        query_start_loc.to(device),
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_native_reuses_kv_across_twelve_query_heads(num_tokens: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(310)
    device = "npu:0"
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    query = torch.randn((num_tokens, num_query_heads, head_dim), dtype=torch.float16)
    key_cache = torch.randn((2, num_kv_heads * head_dim // 16, 64, 16), dtype=torch.float16) * 0.1
    value_cache = torch.randn_like(key_cache) * 0.1
    selection = QSAGroupSelection(
        group_indices=torch.tensor([[0, 1, 2, 3, 16, 17, 18, 19], [4, 5, 6, 7, 20, 21, 22, 23]], dtype=torch.int32)[
            :num_tokens
        ],
        group_counts=torch.full((num_tokens,), 8, dtype=torch.int32),
        tail_starts=torch.tensor([96, 100], dtype=torch.int32)[:num_tokens],
        tail_counts=torch.full((num_tokens,), 4, dtype=torch.int32),
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    expected = qsa_sparse_attention_310_reference(
        query, key_cache, value_cache, selection, block_table, query_start_loc
    )
    selection_npu = QSAGroupSelection(
        *(getattr(selection, field).to(device) for field in selection.__dataclass_fields__)
    )
    actual = qsa_sparse_attention_310(
        query.to(device),
        torch_npu.npu_format_cast(key_cache.to(device), 29),
        torch_npu.npu_format_cast(value_cache.to(device), 29),
        selection_npu,
        block_table.to(device),
        query_start_loc.to(device),
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("key_scale", [0.1, 1.0, 4.0])
def test_native_long_selection_matches_vectorized_reference(key_scale: float) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(311)
    device = "npu:0"
    head_dim = 256
    num_kv_heads = 2
    num_query_heads = 24
    block_size = 512
    num_groups = 512
    num_tokens = num_groups * 4
    query = torch.randn((1, num_query_heads, head_dim), dtype=torch.float16)
    cache_shape = (num_tokens // block_size, num_kv_heads * head_dim // 16, block_size, 16)
    key_cache = torch.randn(cache_shape, dtype=torch.float16) * key_scale
    value_cache = torch.randn(cache_shape, dtype=torch.float16) * 0.1
    block_table = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32)
    token_ids = torch.arange(num_tokens)
    physical_blocks = block_table[0, token_ids // block_size].long()
    token_offsets = token_ids % block_size
    key_rows = key_cache[physical_blocks, :, token_offsets, :].reshape(num_tokens, num_kv_heads, head_dim)
    value_rows = value_cache[physical_blocks, :, token_offsets, :].reshape(num_tokens, num_kv_heads, head_dim)
    kv_head_indices = torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)
    keys_for_query = key_rows[:, kv_head_indices, :].float()
    values_for_query = value_rows[:, kv_head_indices, :].float()
    logits = (keys_for_query * query[0].float().unsqueeze(0)).sum(dim=-1) * head_dim**-0.5
    weights = torch.softmax(logits, dim=0)
    expected = (weights.unsqueeze(-1) * values_for_query).sum(dim=0).half()

    selection = QSAGroupSelection(
        group_indices=torch.arange(num_groups, dtype=torch.int32, device=device).unsqueeze(0),
        group_counts=torch.tensor([num_groups], dtype=torch.int32, device=device),
        tail_starts=torch.zeros(1, dtype=torch.int32, device=device),
        tail_counts=torch.zeros(1, dtype=torch.int32, device=device),
    )
    actual = qsa_sparse_attention_310(
        query.to(device),
        torch_npu.npu_format_cast(key_cache.to(device), 29),
        torch_npu.npu_format_cast(value_cache.to(device), 29),
        selection,
        block_table.to(device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
    ).cpu()[0]
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


def test_native_index_score_reduces_all_128_dimensions() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    query = torch.ones((1, 4, 128), dtype=torch.float16)
    cache = torch.zeros((1, 35, 128), dtype=torch.float16)
    cache[0, 0, 64:] = 1
    cache[0, 1, :64] = 1
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    positions = torch.tensor([7], dtype=torch.int32)
    expected = qsa_indexer_score_310_reference(query, cache, block_table, query_start_loc, positions)
    actual = torch.ops._C_ascend.npu_qsa_indexer_score_310(
        query.npu(),
        cache.npu(),
        block_table.npu(),
        query_start_loc.npu(),
        positions.npu(),
        4,
    ).cpu()
    torch.testing.assert_close(actual[:, :2], expected[:, :2], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("num_tokens", "num_pages", "first_position"),
    [
        (1024, 64, 2048),
        (512, 512, 8192),
        (384, 2048, 130000),
    ],
)
def test_matmul_prefill_selects_same_groups_as_native_index_score(
    num_tokens: int, num_pages: int, first_position: int
) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(42)
    device = "npu:0"
    query = torch.randn((num_tokens, 4, 128), dtype=torch.float16, device=device)
    cache = torch.randn((num_pages + 32, 19, 128), dtype=torch.float16, device=device)
    block_table = torch.randperm(cache.shape[0], device=device)[:num_pages].to(torch.int32).unsqueeze(0)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    positions = torch.arange(first_position, first_position + num_tokens, dtype=torch.int32, device=device)

    native_scores = torch.ops._C_ascend.npu_qsa_indexer_score_310(
        query, cache, block_table, query_start_loc, positions, 4
    )
    native_groups = _stable_topk_indices(native_scores, 512)
    selected = qsa_indexer_select_groups_310(
        query,
        cache,
        block_table,
        query_start_loc,
        positions,
        compress_ratio=4,
        token_topk=2048,
    )
    # FP32 GEMM and the native kernel can round nearly tied scores in a
    # different order, but sparse attention must receive the same group set.
    torch.testing.assert_close(
        selected.group_indices.sort(dim=1).values,
        native_groups.sort(dim=1).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(selected.group_counts, torch.full_like(selected.group_counts, 512))


def test_native_index_cache_norm_reduces_all_128_dimensions() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    cache = torch.zeros(qsa_index_cache_shape(1, 128), dtype=torch.float16)
    keys = torch.zeros((4, 128), dtype=torch.float16)
    keys[:, 64:] = 1
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    slot_mapping = torch.arange(4, dtype=torch.int32)
    norm_weight = torch.zeros(128, dtype=torch.float16)
    rope_cos = torch.ones((4, 64), dtype=torch.float16)
    rope_sin = torch.zeros_like(rope_cos)
    expected = cache.clone()
    qsa_index_cache_update_310_reference(expected, keys, query_start_loc, slot_mapping, norm_weight, rope_cos, rope_sin)
    actual = cache.npu()
    qsa_index_cache_update_310(
        actual,
        keys.npu(),
        query_start_loc.npu(),
        slot_mapping.npu(),
        norm_weight.npu(),
        rope_cos.npu(),
        rope_sin.npu(),
    )
    torch.testing.assert_close(actual.cpu()[0, 0], expected[0, 0], rtol=2e-3, atol=2e-3)


def test_native_index_cache_and_score_across_2048_token_chunk() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(312)
    block_size = 128
    first_chunk = 2048
    num_tokens = 2188
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(qsa_index_cache_shape(num_blocks, 128), dtype=torch.float16)
    keys = torch.randn((num_tokens, 128), dtype=torch.float16) * 0.1
    slots = torch.arange(num_tokens, dtype=torch.int32)
    norm_weight = torch.zeros(128, dtype=torch.float16)
    rope_cos = torch.ones((num_tokens, 64), dtype=torch.float16)
    rope_sin = torch.zeros_like(rope_cos)
    expected = cache.clone()
    actual = cache.to("npu:0")

    for start, end in ((0, first_chunk), (first_chunk, num_tokens)):
        boundaries = torch.tensor([0, end - start], dtype=torch.int32)
        qsa_index_cache_update_310_reference(
            expected,
            keys[start:end],
            boundaries,
            slots[start:end],
            norm_weight,
            rope_cos[start:end],
            rope_sin[start:end],
            block_size=block_size,
        )
        qsa_index_cache_update_310(
            actual,
            keys[start:end].to("npu:0"),
            boundaries.to("npu:0"),
            slots[start:end].to("npu:0"),
            norm_weight.to("npu:0"),
            rope_cos[start:end].to("npu:0"),
            rope_sin[start:end].to("npu:0"),
            block_size=block_size,
        )

    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)

    query = torch.randn((3, 4, 128), dtype=torch.float16)
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    boundaries = torch.tensor([0, 3], dtype=torch.int32)
    positions = torch.tensor([first_chunk - 1, first_chunk, num_tokens - 1], dtype=torch.int32)
    expected_scores = qsa_indexer_score_310_reference(query, expected, block_table, boundaries, positions)
    actual_scores = torch.ops._C_ascend.npu_qsa_indexer_score_310(
        query.to("npu:0"),
        actual,
        block_table.to("npu:0"),
        boundaries.to("npu:0"),
        positions.to("npu:0"),
        4,
    ).cpu()
    for row, position in enumerate(positions.tolist()):
        visible_groups = (position + 1) // 4
        torch.testing.assert_close(
            actual_scores[row, :visible_groups],
            expected_scores[row, :visible_groups],
            rtol=2e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("num_tokens", [2048, 2049])
def test_native_sparse_attention_reads_reshape_and_cache_output(block_size: int, num_tokens: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(313)
    device = "npu:0"
    num_kv_heads = 2
    num_query_heads = 24
    head_dim = 256
    num_blocks = (num_tokens + block_size - 1) // block_size
    key = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=torch.float16) * 0.1
    value = torch.randn_like(key) * 0.1
    query = torch.randn((1, num_query_heads, head_dim), dtype=torch.float16)
    cache_shape = (num_blocks, num_kv_heads * head_dim // 16, block_size, 16)
    key_cache = torch_npu.empty_with_format(size=cache_shape, dtype=torch.float16, device=device, acl_format=29)
    value_cache = torch_npu.empty_with_format(size=cache_shape, dtype=torch.float16, device=device, acl_format=29)
    torch_npu._npu_reshape_and_cache(
        key=key.to(device),
        value=value.to(device),
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=torch.arange(num_tokens, dtype=torch.int32, device=device),
    )

    kv_head_indices = torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)
    logits = (key[:, kv_head_indices, :].float() * query[0].float().unsqueeze(0)).sum(dim=-1)
    weights = torch.softmax(logits * head_dim**-0.5, dim=0)
    expected = (weights.unsqueeze(-1) * value[:, kv_head_indices, :].float()).sum(dim=0).half()
    full_groups = num_tokens // 4
    selection = QSAGroupSelection(
        group_indices=torch.arange(full_groups, dtype=torch.int32, device=device).unsqueeze(0),
        group_counts=torch.tensor([full_groups], dtype=torch.int32, device=device),
        tail_starts=torch.tensor([full_groups * 4], dtype=torch.int32, device=device),
        tail_counts=torch.tensor([num_tokens % 4], dtype=torch.int32, device=device),
    )
    actual = qsa_sparse_attention_310(
        query.to(device),
        key_cache,
        value_cache,
        selection,
        torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
    ).cpu()[0]
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("num_tokens", [2048, 2049])
@pytest.mark.parametrize("extra_block_table_entries", [0, 1000])
def test_native_index_selection_and_sparse_attention_at_budget_boundary(
    num_tokens: int, extra_block_table_entries: int
) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(314)
    device = "npu:0"
    block_size = 128
    num_query_heads = 24
    num_kv_heads = 2
    head_dim = 256
    num_blocks = (num_tokens + block_size - 1) // block_size
    key = torch.randn((num_tokens, num_kv_heads, head_dim), dtype=torch.float16) * 0.1
    value = torch.randn_like(key) * 0.1
    query = torch.randn((1, num_query_heads, head_dim), dtype=torch.float16)
    cache_shape = (num_blocks, num_kv_heads * head_dim // 16, block_size, 16)
    key_cache = torch_npu.empty_with_format(size=cache_shape, dtype=torch.float16, device=device, acl_format=29)
    value_cache = torch_npu.empty_with_format(size=cache_shape, dtype=torch.float16, device=device, acl_format=29)
    torch_npu._npu_reshape_and_cache(
        key=key.to(device),
        value=value.to(device),
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=torch.arange(num_tokens, dtype=torch.int32, device=device),
    )
    block_table = torch.zeros((1, num_blocks + extra_block_table_entries), dtype=torch.int32, device=device)
    block_table[0, :num_blocks] = torch.arange(num_blocks, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    index_cache = torch.randn(qsa_index_cache_shape(num_blocks, block_size), dtype=torch.float16, device=device)
    index_query = torch.randn((1, 4, 128), dtype=torch.float16, device=device)
    selection = qsa_indexer_select_groups_310(
        index_query,
        index_cache,
        block_table,
        query_start_loc,
        torch.tensor([num_tokens - 1], dtype=torch.int32, device=device),
        compress_ratio=4,
        token_topk=2048,
        max_visible_groups=num_tokens // 4,
    )
    full_groups = num_tokens // 4
    if extra_block_table_entries:
        unbounded_selection = qsa_indexer_select_groups_310(
            index_query,
            index_cache,
            block_table,
            query_start_loc,
            torch.tensor([num_tokens - 1], dtype=torch.int32, device=device),
            compress_ratio=4,
            token_topk=2048,
        )
        assert torch.equal(selection.group_indices[:, :full_groups], unbounded_selection.group_indices[:, :full_groups])
    assert selection.group_counts.cpu().tolist() == [full_groups]
    assert selection.tail_counts.cpu().tolist() == [num_tokens % 4]
    assert sorted(selection.group_indices[0, :full_groups].cpu().tolist()) == list(range(full_groups))

    if num_tokens == 2049 and extra_block_table_entries == 0:
        # A live 2,049-token decode returned INT32_MIN as one selected group,
        # causing the native sparse kernel to read an invalid KV page and NaN.
        malformed = selection.group_indices.clone()
        malformed[0, -1] = torch.iinfo(torch.int32).min
        repaired = _repair_native_group_indices(malformed, selection.group_counts)
        assert repaired.cpu().tolist() == [list(range(512))]
        selection = QSAGroupSelection(repaired, selection.group_counts, selection.tail_starts, selection.tail_counts)

    kv_head_indices = torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)
    logits = (key[:, kv_head_indices, :].float() * query[0].float().unsqueeze(0)).sum(dim=-1)
    weights = torch.softmax(logits * head_dim**-0.5, dim=0)
    expected = (weights.unsqueeze(-1) * value[:, kv_head_indices, :].float()).sum(dim=0).half()
    actual = qsa_sparse_attention_310(
        query.to(device), key_cache, value_cache, selection, block_table, query_start_loc
    ).cpu()[0]
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
