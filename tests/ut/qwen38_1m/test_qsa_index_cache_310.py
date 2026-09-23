# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops.qsa_index_cache_310 import (
    qsa_index_cache_shape,
    qsa_index_cache_update_310,
    qsa_index_cache_update_310_reference,
)


def _norm_rope_args(rows: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.zeros(dim, dtype=torch.float16)
    cos = torch.ones((rows, dim), dtype=torch.float16)
    sin = torch.zeros((rows, dim), dtype=torch.float16)
    return weight, cos, sin


def _normalized(row: torch.Tensor) -> torch.Tensor:
    value = row.float()
    return (value * torch.rsqrt(value.square().mean() + 1e-6)).half()


def test_index_cache_shape_reserves_page_local_open_group() -> None:
    assert qsa_index_cache_shape(7, 128) == (7, 35, 128)
    assert qsa_index_cache_shape(7, 128, block_size=64) == (7, 19, 128)


def test_reference_pools_complete_group_into_physical_page() -> None:
    cache = torch.zeros(qsa_index_cache_shape(3, 4), dtype=torch.float16)
    keys = torch.arange(16, dtype=torch.float16).view(4, 4)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    slot_mapping = torch.tensor([256, 257, 258, 259], dtype=torch.int32)

    qsa_index_cache_update_310_reference(
        cache, keys, query_start_loc, slot_mapping, *_norm_rope_args(4, 4), rotary_dim=4
    )

    torch.testing.assert_close(cache[2, 0], _normalized(keys.float().mean(dim=0)))
    torch.testing.assert_close(cache[2, 32:], keys[:3])
    assert torch.count_nonzero(cache[:2]) == 0


def test_reference_carries_open_group_across_invocations() -> None:
    cache = torch.zeros(qsa_index_cache_shape(1, 2), dtype=torch.float16)
    first_keys = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float16)
    qsa_index_cache_update_310_reference(
        cache,
        first_keys,
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([4, 5, 6], dtype=torch.int32),
        *_norm_rope_args(3, 2),
        rotary_dim=2,
    )
    qsa_index_cache_update_310_reference(
        cache,
        torch.tensor([[7, 8]], dtype=torch.float16),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        *_norm_rope_args(1, 2),
        rotary_dim=2,
    )

    expected = _normalized(torch.tensor([4, 5], dtype=torch.float16))
    torch.testing.assert_close(cache[0, 1], expected)


def test_reference_handles_multiple_requests_and_ignored_slots() -> None:
    cache = torch.zeros(qsa_index_cache_shape(2, 2), dtype=torch.float16)
    keys = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [9, 10]], dtype=torch.float16)
    qsa_index_cache_update_310_reference(
        cache,
        keys,
        torch.tensor([0, 4, 5], dtype=torch.int32),
        torch.tensor([128, 129, 130, 131, -1], dtype=torch.int32),
        *_norm_rope_args(5, 2),
        rotary_dim=2,
    )

    torch.testing.assert_close(cache[1, 0], _normalized(torch.tensor([2.5, 3.5], dtype=torch.float16)))
    assert torch.count_nonzero(cache[0]) == 0


def test_native_entrypoint_fails_closed_off_npu() -> None:
    cache = torch.zeros(qsa_index_cache_shape(1, 4), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="NPU-only"):
        qsa_index_cache_update_310(
            cache,
            torch.zeros((1, 4), dtype=torch.float16),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            *_norm_rope_args(1, 4),
            rotary_dim=4,
        )
