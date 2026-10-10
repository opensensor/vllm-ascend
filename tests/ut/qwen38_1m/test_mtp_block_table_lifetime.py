# SPDX-License-Identifier: Apache-2.0
"""Captured later drafts must own and refresh their paged-cache mappings."""

import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/qwen4exp_mtp.py"
spec = importlib.util.spec_from_file_location("qwen_mtp_table_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stage = module.stage_qwen_later_draft_block_table


def proposer(architecture="Qwen4ExpMTP"):
    return SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=[3, 18])),
        runner=SimpleNamespace(max_num_reqs=6),
    )


@pytest.mark.parametrize("rows", [3, 18])
def test_capture_survives_temporary_metadata_and_refreshes_for_real_requests(rows):
    owner = proposer()
    source = torch.arange(rows * 8, dtype=torch.int32).reshape(rows, 8).clone()
    metadata = SimpleNamespace(num_reqs=rows, block_table_tensor=source)
    captured = stage(owner, metadata, 1)
    expected = source.clone()
    assert captured.block_table_tensor.data_ptr() != source.data_ptr()
    pointer = captured.block_table_tensor.data_ptr()
    source.fill_(999)
    del source, metadata
    gc.collect()
    assert torch.equal(captured.block_table_tensor, expected)
    # Replay uses the capture's fixed address with fresh pages; the inactive
    # rows of a partly occupied bucket must no longer reference old requests.
    real_table = torch.tensor([[61, 23, 9, 8, 7, 6, 5, 4], [71, 33, 19, 18, 17, 16, 15, 14]], dtype=torch.int32)
    real_metadata = SimpleNamespace(num_reqs=rows, block_table_tensor=real_table)
    refreshed = stage(owner, real_metadata, 1)
    assert refreshed.block_table_tensor.data_ptr() == pointer
    assert torch.equal(captured.block_table_tensor[:2], real_table)
    assert torch.count_nonzero(captured.block_table_tensor[2:]) == 0
    assert real_metadata.block_table_tensor is real_table
    # A subsequent small capture does not resize or move the large bucket.
    small = stage(owner, SimpleNamespace(num_reqs=3, block_table_tensor=real_table), 1)
    assert small.block_table_tensor.data_ptr() == pointer
    assert owner._qwen_draft_block_table_buffers[1].shape == (18, 8)


def test_later_steps_do_not_overwrite_each_others_tables():
    owner = proposer()
    first = stage(owner, SimpleNamespace(num_reqs=3, block_table_tensor=torch.ones(3, 8, dtype=torch.int32)), 1)
    second = stage(owner, SimpleNamespace(num_reqs=3, block_table_tensor=torch.full((3, 8), 2, dtype=torch.int32)), 2)
    assert first.block_table_tensor.data_ptr() != second.block_table_tensor.data_ptr()
    assert first.block_table_tensor.tolist() == [[1] * 8] * 3
    assert second.block_table_tensor.tolist() == [[2] * 8] * 3
    assert stage(owner, first, 1).block_table_tensor.data_ptr() == first.block_table_tensor.data_ptr()


@pytest.mark.parametrize("rows", [4, 24])
@pytest.mark.parametrize("step", [1, 2])
def test_three_draft_profile_refreshes_both_later_steps(rows, step):
    owner = proposer()
    owner.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 24]
    captured = stage(
        owner, SimpleNamespace(num_reqs=rows, block_table_tensor=torch.ones(rows, 8, dtype=torch.int32)), step
    )
    pointer = captured.block_table_tensor.data_ptr()
    source = torch.full((2, 8), 31 + step, dtype=torch.int32)
    refreshed = stage(owner, SimpleNamespace(num_reqs=rows, block_table_tensor=source), step)
    assert refreshed.block_table_tensor.data_ptr() == pointer
    assert torch.equal(captured.block_table_tensor[:2], source)
    assert not torch.count_nonzero(captured.block_table_tensor[2:])
    assert owner._qwen_draft_block_table_buffers[step].shape == (24, 8)


@pytest.mark.parametrize("step,architecture", [(0, "Qwen4ExpMTP"), (1, "OtherMTP")])
def test_first_draft_and_other_models_keep_existing_tables(step, architecture):
    owner = proposer(architecture)
    metadata = SimpleNamespace(num_reqs=3, block_table_tensor=torch.ones(3, 8, dtype=torch.int32))
    assert stage(owner, metadata, step) is metadata
    assert not hasattr(owner, "_qwen_draft_block_table_buffers")


@pytest.mark.parametrize("rows,dtype", [(19, torch.int32), (0, torch.int32), (3, torch.int64)])
def test_invalid_table_contract_is_rejected(rows, dtype):
    with pytest.raises(ValueError):
        stage(proposer(), SimpleNamespace(num_reqs=rows, block_table_tensor=torch.ones(3, 8, dtype=dtype)), 1)


def test_existing_captured_storage_cannot_change_width():
    owner = proposer()
    stage(owner, SimpleNamespace(num_reqs=3, block_table_tensor=torch.ones(3, 8, dtype=torch.int32)), 1)
    with pytest.raises(ValueError, match="cannot change"):
        stage(owner, SimpleNamespace(num_reqs=3, block_table_tensor=torch.ones(3, 9, dtype=torch.int32)), 1)


@pytest.mark.parametrize("owned_tables", [False, True])
def test_runtime_skips_intermediate_clone_only_with_owned_later_tables(owned_tables, monkeypatch):
    import ast

    path = SOURCE.parents[1] / "spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    klass = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    propose = next(node for node in klass.body if getattr(node, "name", None) == "_propose")
    branch = next(
        node
        for node in ast.walk(propose)
        if isinstance(node, ast.If)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "block_table_tensor_clone"
            for assignment in ast.walk(node)
            if isinstance(assignment, ast.Assign)
            for target in assignment.targets
        )
    )
    owner = proposer()
    owner.dcp_size = 1
    owner.use_cuda_graph = True
    source = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    metadata = SimpleNamespace(num_reqs=3, block_table_tensor=source)
    if owned_tables:
        stage(owner, metadata, 1)
    clones = []
    original_clone = torch.Tensor.clone

    def counted_clone(tensor, *args, **kwargs):
        clones.append(tensor.numel())
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counted_clone)
    namespace = {"self": owner, "common_attn_metadata": metadata}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), namespace)
    assert clones == ([] if owned_tables else [source.numel()])
    assert (metadata.block_table_tensor.data_ptr() == source.data_ptr()) == owned_tables
    captured = stage(owner, metadata, 1)
    assert torch.equal(captured.block_table_tensor, source)
    source.fill_(999)
    assert captured.block_table_tensor.tolist() == [list(range(row * 8, row * 8 + 8)) for row in range(3)]
