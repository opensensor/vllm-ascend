# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.models.qwen4_exp.model import _QSAAttention


def _selector(*, budget: int, sums: torch.Tensor) -> _QSAAttention:
    module = object.__new__(_QSAAttention)
    nn.Module.__init__(module)
    module.indexer_budget = budget
    module.index_head_dim = sums.shape[1]
    module.compute_dtype = torch.float32
    module.params_dtype = sums.dtype
    module.ik_proj = nn.Parameter(torch.eye(sums.shape[1], dtype=sums.dtype))
    module.register_buffer("_qsa_page_key_sums", sums, persistent=False)
    module.register_buffer(
        "_qsa_page_key_counts",
        torch.ones((sums.shape[0], 1), dtype=torch.float32),
        persistent=False,
    )
    return module


def test_qsa_decode_page_selection_pins_recent_pages():
    sums = torch.zeros((24, 2), dtype=torch.float16)
    sums[11] = torch.tensor([4.0, 0.0])
    module = _selector(budget=8, sums=sums)
    metadata = SimpleNamespace(
        seq_lens_list=[21],
        block_tables=torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
        seq_lens=torch.tensor([21], dtype=torch.int32),
    )
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float16)

    selected = module._select_decode_pages(query, metadata, block_size=4)

    assert selected is not None
    block_tables, context_lens = selected
    # Two-page budget is entirely reserved for recency.
    torch.testing.assert_close(block_tables, torch.tensor([[14, 15]], dtype=torch.int32))
    torch.testing.assert_close(context_lens, torch.tensor([5], dtype=torch.int32))


def test_qsa_decode_page_selection_uses_remaining_budget_for_content():
    sums = torch.zeros((24, 2), dtype=torch.float16)
    sums[11] = torch.tensor([4.0, 0.0])
    sums[12] = torch.tensor([2.0, 0.0])
    module = _selector(budget=40, sums=sums)
    metadata = SimpleNamespace(
        seq_lens_list=[45],
        block_tables=torch.tensor([list(range(10, 22))], dtype=torch.int32),
        seq_lens=torch.tensor([45], dtype=torch.int32),
    )
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float16)

    selected = module._select_decode_pages(query, metadata, block_size=4)

    assert selected is not None
    block_tables, context_lens = selected
    torch.testing.assert_close(
        block_tables,
        torch.tensor([[11, 12, 14, 15, 16, 17, 18, 19, 20, 21]], dtype=torch.int32),
    )
    torch.testing.assert_close(context_lens, torch.tensor([37], dtype=torch.int32))


def test_qsa_decode_page_selection_leaves_short_context_dense():
    module = _selector(budget=8, sums=torch.zeros((4, 2), dtype=torch.float16))
    metadata = SimpleNamespace(
        seq_lens_list=[8],
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
    )

    assert module._select_decode_pages(torch.zeros((1, 1, 2)), metadata, block_size=4) is None


def test_qsa_page_key_update_projects_only_final_row_per_touched_page():
    module = _selector(budget=8, sums=torch.zeros((5, 2), dtype=torch.float32))
    metadata = SimpleNamespace(
        num_actual_tokens=5,
        slot_mapping=torch.tensor([0, 1, 3, 4, 5], dtype=torch.int32),
    )
    block_input = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 5.0]],
    )

    module._update_page_key_cache(block_input, metadata, block_size=4)

    torch.testing.assert_close(module._qsa_page_key_sums[0], block_input[2])
    torch.testing.assert_close(module._qsa_page_key_sums[1], block_input[4])
    torch.testing.assert_close(module._qsa_page_key_counts[:2], torch.ones((2, 1)))
