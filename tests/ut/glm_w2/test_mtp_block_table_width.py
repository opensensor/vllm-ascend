# SPDX-License-Identifier: Apache-2.0
"""Exercise actual draft table cropping without importing an NPU runtime."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def proposer(context, physical_size, kernel_size, *, indexer=False):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/multi_kv_cache_group_proposer.py"
    source = ast.parse(path.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
    names = {"_draft_block_table_width", "_common_attn_metadata_for_draft_group"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    subset = ast.ClassDef(name="Proposer", bases=[], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"copy": copy}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, subset], type_ignores=[])), str(path), "exec"), scope
    )
    blocks = (context + physical_size - 1) // physical_size
    split = physical_size // kernel_size
    builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(max_num_blocks_per_req=lambda *args: blocks))
    if indexer:
        builder.logical_block_size = physical_size
        builder.kernel_blocks_per_logical_block = split
    group = SimpleNamespace(kv_cache_group_id=0, get_metadata_builder=lambda: builder)
    instance = scope["Proposer"]()
    instance.max_model_len = context
    instance.vllm_config = None
    instance.kv_cache_gid = 0
    instance._uses_multi_group_kv_cache = True
    instance.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(blocks_per_phys_block=split)])
    )
    return instance, group, blocks, split


@pytest.mark.parametrize("context", [32768, 131072, 196608, 311040])
@pytest.mark.parametrize("physical_size,kernel_size", [(640, 32), (384, 32), (128, 128), (4, 4)])
@pytest.mark.parametrize("indexer", [False, True])
def test_draft_preserves_every_kernel_page(context, physical_size, kernel_size, indexer):
    instance, group, blocks, split = proposer(context, physical_size, kernel_size, indexer=indexer)
    width = blocks * split
    table = torch.arange(width, dtype=torch.int32).unsqueeze(0)
    common = SimpleNamespace(num_reqs=1, block_table_tensor=table, slot_mapping=torch.tensor([3]))
    actual = instance._common_attn_metadata_for_draft_group(common, group, 1)
    assert actual is not common
    assert actual.block_table_tensor.shape == (1, width)
    assert actual.block_table_tensor.data_ptr() == table.data_ptr()
    assert actual.slot_mapping is common.slot_mapping
    physical = actual.block_table_tensor[:, ::split] // split
    assert physical.shape[1] * physical_size >= context
    assert physical[0, (context - 1) // physical_size] == (context - 1) // physical_size


def test_311k_old_crop_explains_16k_boundary_and_fix_covers_full_context():
    instance, group, blocks, split = proposer(311040, 640, 32)
    table = torch.arange(blocks * split).unsqueeze(0)
    old_physical_table = table[:, :blocks][:, ::split] // split
    assert old_physical_table.shape[1] * 640 == 16000
    new_physical_table = table[:, : instance._draft_block_table_width(group)][:, ::split] // split
    for position in [15999, 16000, 16639, 16640, 311039]:
        assert new_physical_table[0, position // 640] == position // 640


def prefill_functions():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/attention/mla_v1_310.py"
    tree = ast.parse(path.read_text())
    names = {"_qsa_cache_block_table", "_forward_prefill_paged_latent"}
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch, "_QSA_KERNEL_BLOCK_SIZE": 32}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_truncated_draft_table_fails_before_qsa_dispatch():
    functions = prefill_functions()
    # Regression geometry: 486 scheduler pages were incorrectly used to crop
    # a table of 32-token kernel pages. QSA receives only ceil(486/20)=25 pages.
    cache = torch.empty(30, 1, 640, 16, dtype=torch.float16)
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            input_positions=torch.arange(16000, 16640),
            actual_seq_lengths_q=[640],
            max_seq_lens=16640,
            block_table=torch.arange(486, dtype=torch.int32).unsqueeze(0),
        )
    )
    owner = SimpleNamespace(W_UK_T=torch.eye(16, dtype=torch.float16).unsqueeze(0))
    # No QSA op is even present on this owner: rejection precedes lookup/launch.
    with pytest.raises(ValueError, match="16000 addressable tokens < 16640"):
        functions["_forward_prefill_paged_latent"](
            owner, torch.ones(640, 1, 16, dtype=torch.float16), (cache, cache), metadata
        )
