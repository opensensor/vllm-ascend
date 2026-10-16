# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Draft slots must address the selected cache geometry, including on 310P."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROPOSER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"


def load_update_method():
    """Run the full production method without importing the NPU runtime."""
    tree = ast.parse(PROPOSER_PATH.read_text(encoding="utf-8"))
    method = next(
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "attn_update_stack_num_spec_norm"
    )
    namespace = {
        "torch": torch,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "AscendAttentionState": SimpleNamespace(SpecDecoding="spec", ChunkedPrefill="prefill"),
        "PADDING_SLOT_ID": -1,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(PROPOSER_PATH), "exec"),
        namespace,
    )
    return namespace[method.name]


def exercise_slot_mapping(method, device, block_size, has_gdn, full_graph, mrope, draft_index, constant_positions):
    """Also used by the private six-NPU gate; never writes a model KV cache."""
    capacity_blocks = 26884
    max_model_len = 4 * block_size
    batch_size = 3
    padded_size = 5
    # High physical pages reproduce the overflow into neighbouring allocations.
    table_values = [[17602, 26883, 14999, 21003], [26883, 17602, 21003, 14999], [21003, 14999, 17602, 26883]]
    initial_positions = [block_size - 1, 2 * block_size - 1, max_model_len - 1]
    positions = torch.tensor(initial_positions, device=device)
    if mrope:
        positions = positions.repeat(3, 1)
    original_lengths = torch.tensor(initial_positions, dtype=torch.int32, device=device) + 1
    table = torch.tensor(table_values, dtype=torch.int32, device=device)
    common = SimpleNamespace(
        block_table_tensor=table,
        seq_lens=original_lengths.clone(),
        seq_lens_cpu=None,
        _seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        positions=torch.zeros(padded_size, dtype=torch.int64, device=device),
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
    )

    def adjust(tensor, size):
        result = tensor.new_zeros((size, *tensor.shape[1:]))
        result[: len(tensor)].copy_(tensor)
        return result

    proposer = SimpleNamespace(
        block_size=block_size,
        kernel_block_size=128,
        has_gdn=has_gdn,
        method="mtp",
        runner=SimpleNamespace(),
        shallow_copy_metadata=copy,
        _adjust_tensor=adjust,
        arange=torch.arange(padded_size + 1, dtype=torch.int32, device=device),
        token_arange_np=np.arange(padded_size + 1, dtype=np.int32),
        constant_draft_positions=constant_positions,
        uses_mrope=mrope,
        max_model_len=max_model_len,
        sliding_window=None,
        slot_mapping_group=[torch.full((padded_size,), 777, dtype=torch.int32, device=device) for _ in range(3)],
        seq_lens_group=[torch.zeros(padded_size, dtype=torch.int32, device=device) for _ in range(3)],
        query_start_loc_group=[torch.zeros(padded_size + 1, dtype=torch.int32, device=device) for _ in range(3)],
        use_compress=False,
        _common_attn_metadata_for_draft_group=lambda metadata, *_: metadata,
    )
    builder = SimpleNamespace(build_for_drafting=lambda metadata, *_: metadata)
    group = SimpleNamespace(get_metadata_builder=lambda: builder)
    updated, built = method(
        proposer,
        draft_index,
        common,
        batch_size,
        padded_size,
        positions,
        "full" if full_graph else "none",
        attn_group=group,
    )
    next_positions = [p + (not constant_positions) for p in initial_positions]
    expected = [
        table_values[i][p // block_size] * block_size + p % block_size if p < max_model_len else -1
        for i, p in enumerate(next_positions)
    ] + [-1] * (padded_size - batch_size)
    actual = updated.slot_mapping.cpu().tolist()
    assert actual == expected
    assert all(slot == -1 or 0 <= slot < capacity_blocks * block_size for slot in actual)
    assert updated.slot_mapping.data_ptr() == proposer.slot_mapping_group[draft_index].data_ptr()
    assert built is updated
    assert torch.equal(common.seq_lens, original_lengths)
    assert torch.equal(common.block_table_tensor, table)
    assert updated.positions[:batch_size].cpu().tolist() == [p if p < max_model_len else 0 for p in next_positions]
    return actual


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("has_gdn", [False, True])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("draft_index", [1, 2])
@pytest.mark.parametrize("constant_positions", [False, True])
def test_draft_slots_use_selected_cache_block_size(
    block_size, has_gdn, full_graph, mrope, draft_index, constant_positions
):
    exercise_slot_mapping(
        load_update_method(), "cpu", block_size, has_gdn, full_graph, mrope, draft_index, constant_positions
    )
