# SPDX-License-Identifier: Apache-2.0
"""Ascend 310P regression for full-width QSA attention score reduction."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_index_cache_310 import (
    qsa_index_cache_shape,
    qsa_index_cache_update_310,
    qsa_index_cache_update_310_reference,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection, qsa_indexer_score_310_reference
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import (
    qsa_sparse_attention_310,
    qsa_sparse_attention_310_reference,
)
from vllm_ascend.utils import enable_custom_op


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
