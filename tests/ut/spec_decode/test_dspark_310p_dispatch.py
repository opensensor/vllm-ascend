# SPDX-License-Identifier: Apache-2.0
"""Exercise MRv1 DSpark input dispatch without importing the NPU runtime."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("on_310p", [False, True])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("draft_tokens", [1, 3, 7])
def test_dspark_inputs_dispatch_and_per_group_metadata(on_310p, sample_from_anchor, draft_tokens):
    root = Path(__file__).resolve().parents[3]
    ops_path = root / "vllm_ascend/_310p/dflash_ops.py"
    spec = importlib.util.spec_from_file_location("dspark_test_ops", ops_path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    path = root / "vllm_ascend/spec_decode/dspark_proposer.py"
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AscendDSparkProposer")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_inputs_first_pass"
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    launches = []

    def grid(context, query):
        assert not on_310p, "310P must not query Triton launch properties"
        return 1

    class Kernel:
        def __getitem__(self, launch_grid):
            launches.append(launch_grid)
            return ops.copy_and_expand_dflash_inputs

    namespace = {
        "torch": torch,
        "is_310p": lambda: on_310p,
        "copy_and_expand_dflash_inputs": ops.copy_and_expand_dflash_inputs,
        "copy_and_expand_dflash_and_dspark_inputs_kernel": Kernel(),
        "_compute_num_programs": grid,
        "AscendAttentionState": SimpleNamespace(ChunkedPrefill="chunked"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    batch, context, hidden = 2, 5, 4
    queries = draft_tokens + (0 if sample_from_anchor else 1)
    total_queries = batch * queries
    capacity = total_queries + 8
    starts = torch.tensor([0, 2, 5], dtype=torch.int32)
    lengths = torch.tensor([14, 19], dtype=torch.int32)
    rejected = torch.tensor([1, 0], dtype=torch.int32)
    positions = torch.tensor([5, 6, 10, 11, 12], dtype=torch.int32)
    group_ids = [0, 2]
    tables = {gid: torch.arange(32).reshape(2, 16).int() + 10 + gid * 100 for gid in group_ids}
    slots = {gid: torch.arange(context, dtype=torch.int32) + gid * 100 for gid in group_ids}

    def buffer():
        return torch.full((capacity,), -777, dtype=torch.int32)

    proposer = SimpleNamespace(
        _dspark_seed_buffer=torch.full((4,), -777, dtype=torch.int64),
        _dflash_hidden_states=torch.zeros((context + 2, hidden)),
        _context_positions_buffer=buffer(),
        _per_group_block_tables=tables,
        _per_group_slot_mappings=slots,
        _per_group_kernel_block_sizes={0: 4, 2: 8},
        _per_group_context_slot_mapping_buffers={gid: buffer() for gid in group_ids},
        _per_group_query_slot_mapping_buffers={gid: buffer() for gid in group_ids},
        _layer_group_idx=[2, 0, 2],
        draft_attn_groups=[SimpleNamespace(kv_cache_group_id=gid) for gid in group_ids],
        num_query_per_req=queries,
        num_speculative_tokens=draft_tokens,
        sample_from_anchor=sample_from_anchor,
        parallel_drafting_token_id=999,
        positions=buffer(),
        input_ids=buffer(),
        arange_dflash=torch.arange(32, dtype=torch.int32),
        token_arange_np=np.arange(32),
        device=torch.device("cpu"),
        model=SimpleNamespace(),
    )
    cpu_lengths = lengths - rejected
    cad = SimpleNamespace(
        num_reqs=batch,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=lengths,
        _seq_lens_cpu=cpu_lengths,
        seq_lens_cpu=cpu_lengths,
        max_seq_len=19,
    )
    count, indices, result, long_seq = namespace["set_inputs_first_pass"](
        proposer,
        torch.arange(context),
        torch.tensor([41, 42]),
        positions,
        torch.ones(context, hidden),
        None,
        cad,
        rejected,
    )
    assert launches == ([] if on_310p else [(1,), (1,)])
    assert count == total_queries and result is cad and long_seq is None
    expected_indices = [r * queries + q for r in range(batch) for q in range(0 if sample_from_anchor else 1, queries)]
    assert indices.tolist() == expected_indices
    assert proposer._dspark_seed_buffer.tolist() == [41, 42, 0, 0]
    assert cad.query_start_loc.tolist() == [0, queries, total_queries]
    assert cad.query_start_loc_cpu.tolist() == [0, queries, total_queries]
    assert cad.seq_lens.tolist() == [13 + queries, 19 + queries]
    assert cad._seq_lens_cpu.tolist() == cad.seq_lens.tolist() == cad.seq_lens_cpu.tolist()
    assert cpu_lengths.tolist() == [13, 19]  # input host mirror not mutated
    assert cad.causal is False and cad.attn_state == "chunked"
    assert proposer.positions[:total_queries].tolist() == list(range(6, 6 + queries)) + list(range(13, 13 + queries))
    assert proposer.positions[total_queries:].tolist() == [-777] * 8
    for gid in group_ids:
        block = proposer._per_group_kernel_block_sizes[gid]
        expected_slots = [
            int(tables[gid][r, pos // block]) * block + pos % block
            for r, start in enumerate([13, 19])
            for pos in range(start, start + queries)
        ]
        assert proposer._per_group_query_slot_mapping_buffers[gid].tolist() == expected_slots + [-777] * 8
        assert proposer._per_group_context_slot_mapping_buffers[gid].tolist() == slots[gid].tolist() + [-777] * (
            capacity - context
        )
    for gid, actual in zip(proposer._layer_group_idx, proposer._context_slot_mapping_buffers):
        assert actual is proposer._per_group_context_slot_mapping_buffers[gid]
