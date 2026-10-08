# SPDX-License-Identifier: Apache-2.0
"""Run the MLA metadata builder on CPU without importing NPU modules."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def make_builder(requests, tokens, graph_tokens, *, disable_padded_drafter=False):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    source = ast.parse(path.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAMetadataBuilder")
    names = {
        "build_decode_metadata",
        "get_block_table_size",
        "pad_actual_seq_len_q_mtp_enable_pad",
        "pad_actual_seq_len_q_mtp_disable_pad",
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    subset = ast.ClassDef(name="Builder", bases=[], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, subset], type_ignores=[]))
    scope = {
        "torch": torch,
        "np": np,
        "PAD_SLOT_ID": -1,
        "BUILD_METADATA_STEP_PREFILL": 0,
        "BUILD_METADATA_STEP_DECODE": 1,
    }
    exec(compile(module, str(path), "exec"), scope)
    builder = scope["Builder"]()
    builder.num_actual_tokens = builder.num_decode_tokens = tokens
    builder.num_decodes = requests
    builder.graph_pad_size = graph_tokens
    builder.use_mla_rope = False
    builder.device = torch.device("cpu")
    builder.seq_lens = torch.full((requests,), 25, dtype=torch.int32)
    builder.speculative_config = SimpleNamespace(disable_padded_drafter_batch=disable_padded_drafter)
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    builder.decode_metadata_cls = SimpleNamespace
    builder.nope_zero_rope_cache = {}
    common = SimpleNamespace(
        num_reqs=requests,
        positions=torch.arange(24, 24 + tokens),
        query_start_loc_cpu=torch.arange(requests + 1) * (tokens // requests),
        block_table_tensor=torch.arange(requests * 4, dtype=torch.int32).reshape(requests, 4),
        slot_mapping=torch.arange(24, 24 + tokens, dtype=torch.int32),
        decode_token_per_req=tokens // requests,
        actual_seq_lengths_q=list(range(tokens // requests, graph_tokens + 1, tokens // requests)),
    )
    builder.block_table = common.block_table_tensor
    builder.slot_mapping = common.slot_mapping
    return builder, common


@pytest.mark.parametrize("requests,tokens", [(1, 2), (4, 8), (4, 4)])
def test_capture_metadata_keeps_runner_storage_when_no_padding_needed(requests, tokens):
    builder, common = make_builder(requests, tokens, tokens)
    captured = builder.build_decode_metadata(0, common)
    assert captured.input_positions.data_ptr() == common.positions.data_ptr()
    assert captured.block_table.data_ptr() == common.block_table_tensor.data_ptr()
    assert builder.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    # Simulate scheduler updates after capture. Captured tensors must see new
    # positions, moved request pages and new write slots without being rebuilt.
    common.positions.add_(100)
    common.block_table_tensor.add_(40)
    common.slot_mapping.add_(100)
    assert torch.equal(captured.input_positions, common.positions)
    assert torch.equal(captured.block_table, common.block_table_tensor)
    assert torch.equal(builder.slot_mapping, common.slot_mapping)


def test_actual_padding_still_fills_unused_tokens_and_requests():
    builder, common = make_builder(1, 2, 8)
    metadata = builder.build_decode_metadata(0, common)
    assert metadata.input_positions.tolist() == [24, 25, 0, 0, 0, 0, 0, 0]
    assert builder.slot_mapping.tolist() == [24, 25, -1, -1, -1, -1, -1, -1]
    assert metadata.block_table.shape == (4, 4)
    assert torch.equal(metadata.block_table[0], common.block_table_tensor[0])
    assert metadata.actual_seq_lengths_q == [2, 4, 6, 8]


def test_unpadded_drafter_retains_its_request_padding_contract():
    builder, common = make_builder(1, 2, 2, disable_padded_drafter=True)
    metadata = builder.build_decode_metadata(0, common)
    assert metadata.actual_seq_lengths_q == [2, 2]
    assert metadata.seq_lens_list == [25, 0]
    assert metadata.block_table.shape == (2, 4)
