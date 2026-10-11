# SPDX-License-Identifier: Apache-2.0
"""Use different first-pass and later-pass layouts in padded MTP captures."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
tree = ast.parse(SOURCE.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer")
method = next(node for node in cls.body if getattr(node, "name", None) == "_prepare_draft_graph_capture_step")
helper_spec = importlib.util.spec_from_file_location(
    "qwen_mtp_capture_helpers", SOURCE.parents[1] / "_310p/qwen4exp_mtp.py"
)
helper_module = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper_module)
namespace = {
    "torch": torch,
    "PADDING_SLOT_ID": -1,
    "stage_qwen_later_draft_block_table": helper_module.stage_qwen_later_draft_block_table,
}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
prepare = namespace["_prepare_draft_graph_capture_step"]


def fixture(rows, architecture="Qwen4ExpMTP"):
    requests = rows // 3
    proposer = SimpleNamespace(
        method="mtp",
        runner=SimpleNamespace(max_num_reqs=6),
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=[3, 18])),
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        shallow_copy_metadata=copy.copy,
        arange=torch.arange(65, dtype=torch.int32),
        token_arange_np=np.arange(65, dtype=np.int32),
        seq_lens_group=[torch.full((64,), 7, dtype=torch.int32) for _ in range(2)],
        slot_mapping_group=[torch.full((64,), 99, dtype=torch.int32) for _ in range(2)],
        _adjust_tensor=lambda tensor, count: torch.cat(
            (tensor, torch.zeros(count - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype))
        ),
    )
    common = SimpleNamespace(
        num_input_tokens=rows,
        num_reqs=requests,
        num_actual_tokens=rows,
        max_query_len=3,
        decode_token_per_req=3,
        query_start_loc=torch.arange(0, rows + 1, 3, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(0, rows + 1, 3, dtype=torch.int32),
        seq_lens=torch.full((requests,), 3, dtype=torch.int32),
        slot_mapping=torch.full((rows,), 99, dtype=torch.int32),
        block_table_tensor=torch.ones(requests, 4, dtype=torch.int32),
    )
    return proposer, common


@pytest.mark.parametrize("rows", [3, 18])
def test_later_capture_uses_full_single_query_bucket_and_safe_slots(rows):
    proposer, common = fixture(rows)
    original_slots = common.slot_mapping.clone()
    pointer = proposer.slot_mapping_group[1].data_ptr()
    prepared = prepare(proposer, common, 1)
    assert prepared is not common
    assert prepared.num_reqs == prepared.num_actual_tokens == prepared.num_input_tokens == rows
    assert prepared.query_start_loc.tolist() == list(range(rows + 1))
    assert prepared.query_start_loc_cpu.tolist() == list(range(rows + 1))
    assert prepared.seq_lens.tolist() == prepared.seq_lens_cpu.tolist() == [1] * rows
    assert prepared._seq_lens_cpu.data_ptr() != prepared.seq_lens_cpu.data_ptr()
    assert prepared.max_query_len == prepared.decode_token_per_req == 1
    assert prepared.actual_seq_lengths_q == list(range(1, rows + 1))
    assert prepared.block_table_tensor.shape == (rows, 4)
    assert prepared.slot_mapping.data_ptr() == pointer
    assert prepared.slot_mapping.tolist() == [-1] * 64
    assert torch.equal(common.slot_mapping, original_slots)
    assert common.num_reqs == rows // 3
    assert proposer.slot_mapping_group[0].tolist() == [99] * 64


@pytest.mark.parametrize(
    "step,architecture,method_name", [(0, "Qwen4ExpMTP", "mtp"), (1, "OtherMTP", "mtp"), (1, "Qwen4ExpMTP", "eagle3")]
)
def test_first_pass_and_other_model_capture_layouts_are_unchanged(step, architecture, method_name):
    proposer, common = fixture(18, architecture)
    proposer.method = method_name
    assert prepare(proposer, common, step) is common


def test_dummy_run_prepares_geometry_before_binding_persistent_operands():
    dummy = next(node for node in cls.body if getattr(node, "name", None) == "dummy_run")
    prepare_call = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_prepare_draft_graph_capture_step"
    )
    copies = [
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "slot_mapping" for target in node.targets)
    ]
    assert any(node.lineno > prepare_call.lineno for node in copies)


@pytest.mark.parametrize("rows", [3, 18])
def test_production_capture_loop_binds_later_geometry_without_aliased_copies(rows, monkeypatch):
    proposer, common = fixture(rows)
    proposer.num_speculative_tokens = 2
    proposer.use_compress = False
    proposer.dcp_size = 1
    proposer.query_start_loc_group = [torch.zeros(64, dtype=torch.int32) for _ in range(2)]
    proposer._prepare_draft_graph_capture_step = lambda metadata, step: prepare(proposer, metadata, step)
    proposer._build_multi_group_graph_capture_metadata = lambda metadata, step: {"qsa": metadata}
    original_copy = torch.Tensor.copy_

    def reject_alias_copy(destination, source, *args, **kwargs):
        if destination.data_ptr() == source.data_ptr():
            raise AssertionError("NPU capture cannot copy overlapping staging views")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", reject_alias_copy)
    dummy = next(node for node in cls.body if getattr(node, "name", None) == "dummy_run")
    loop = next(
        node
        for node in ast.walk(dummy)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "draft_index"
    )
    maps = []
    scope = {
        "self": proposer,
        "common_attn_metadata": common,
        "dcp_manager": None,
        "multi_steps_attn_metadata": maps,
        "PADDING_SLOT_ID": -1,
    }
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SOURCE), "exec"), scope)
    first = maps[0]["qsa"]
    later = maps[1]["qsa"]
    assert first.query_start_loc.tolist() == list(range(0, rows + 1, 3))
    assert later.query_start_loc.tolist() == list(range(rows + 1))
    assert later.seq_lens.shape == (rows,)
    assert later.seq_lens.tolist() == [1] * rows
    assert later.slot_mapping.tolist() == [-1] * 64
