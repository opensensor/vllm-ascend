# SPDX-License-Identifier: Apache-2.0
"""Ascend 310P regression for direct paged QSA value gather into NZ."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_gather_nz_310 import (
    qsa_gather_key_transposed_nz_310,
    qsa_gather_value_nz_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.utils import enable_custom_op

FULL_CONTEXT_TABLE_BLOCKS = 2048
LOCAL_TABLE_ALIGNMENT = 8


@pytest.mark.parametrize(
    "num_groups,table_width",
    [(5, None), (5, LOCAL_TABLE_ALIGNMENT), (512, None), (512, FULL_CONTEXT_TABLE_BLOCKS)],
)
@pytest.mark.parametrize("transpose_output", [False, True])
def test_gather_preserves_selected_values_with_paged_cache_and_tail(
    num_groups: int, table_width: int | None, transpose_output: bool
) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires an Ascend 310P NPU")

    enable_custom_op()
    torch.manual_seed(841)
    block_size = 64
    num_kv_heads = 2
    head_dim = 256
    num_tokens = 2
    num_blocks = (num_groups * 4 + 4 + block_size - 1) // block_size
    cache_blocks = num_blocks + 3
    cache = torch.randn(cache_blocks, num_kv_heads * head_dim // 16, block_size, 16, dtype=torch.float16)
    if table_width is None:
        pages = torch.randperm(cache_blocks, dtype=torch.int32)[:num_blocks].unsqueeze(0)
    else:
        pages = torch.randint(cache_blocks, (1, table_width), dtype=torch.int32)
        pages[0, :num_blocks] = torch.randperm(cache_blocks, dtype=torch.int32)[:num_blocks]
    groups = torch.randperm(num_groups, dtype=torch.int32).expand(num_tokens, -1).contiguous()
    selection_cpu = QSAGroupSelection(
        group_indices=groups,
        group_counts=torch.tensor([num_groups, num_groups - 1], dtype=torch.int32),
        tail_starts=torch.full((num_tokens,), num_groups * 4, dtype=torch.int32),
        tail_counts=torch.tensor([3, 1], dtype=torch.int32),
    )
    selection_npu = QSAGroupSelection(
        *(getattr(selection_cpu, field).to("npu:0") for field in selection_cpu.__dataclass_fields__)
    )
    gather = qsa_gather_key_transposed_nz_310 if transpose_output else qsa_gather_value_nz_310
    actual_nz = gather(
        torch_npu.npu_format_cast(cache.to("npu:0"), 29),
        selection_npu,
        pages.to("npu:0"),
        head_dim=head_dim,
    )
    actual = torch_npu.npu_format_cast(actual_nz, 0).cpu()
    if transpose_output:
        actual = actual.transpose(-1, -2)
    assert torch_npu.get_npu_format(actual_nz) == 29
    assert actual.shape == (num_tokens, num_kv_heads, ((num_groups * 4 + 4 + 15) // 16) * 16, head_dim)

    for row in range(num_tokens):
        selected = selection_cpu.group_indices[row, : selection_cpu.group_counts[row]]
        token_ids = torch.cat(
            (
                (selected.unsqueeze(-1) * 4 + torch.arange(4)).reshape(-1),
                selection_cpu.tail_starts[row] + torch.arange(selection_cpu.tail_counts[row]),
            )
        )
        physical_pages = pages[0, token_ids // block_size].long()
        expected = cache[physical_pages, :, token_ids % block_size, :].reshape(-1, num_kv_heads, head_dim)
        expected = expected.permute(1, 0, 2)
        if row == 1:
            # The second row has one invalid group slot before its tail.
            actual_valid = torch.cat(
                (actual[row, :, : (num_groups - 1) * 4], actual[row, :, num_groups * 4 : num_groups * 4 + 1]), dim=1
            )
        else:
            actual_valid = torch.cat(
                (actual[row, :, : num_groups * 4], actual[row, :, num_groups * 4 : num_groups * 4 + 3]), dim=1
            )
        torch.testing.assert_close(actual_valid, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
