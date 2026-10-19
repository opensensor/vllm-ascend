# SPDX-License-Identifier: Apache-2.0
"""CPU parity for the 310P primitives without importing the NPU runtime."""

import ast
import importlib.util
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/dflash_ops.py"
_SPEC = importlib.util.spec_from_file_location("dflash_310p_ops", _PATH)
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


@pytest.mark.parametrize("draft_tokens", [1, 3, 7])
@pytest.mark.parametrize("rejections", [None, [0, 2, 3], [2, 4, 3]])
@pytest.mark.parametrize("position_layout", ["flat", "mrope"])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_expand_preserves_positions_slots_and_buffer_tails(
    draft_tokens, rejections, position_layout, sample_from_anchor
):
    # Repeated/discontinuous MRoPE positions must not control KV addresses.
    positions = torch.tensor([100, 100, 20, 21, 21, 45, 80, 80, 81], dtype=torch.int32)
    target_positions = positions
    if position_layout == "mrope":
        # Real runner shape: three axes sliced from a larger capacity buffer.
        # Different spatial axes catch accidentally flattening all three rows.
        backing = torch.full((3, 16), -999, dtype=positions.dtype)
        target_positions = backing[:, : len(positions)]
        target_positions.copy_(torch.stack((positions, positions + 1000, positions + 2000)))
        assert not target_positions.is_contiguous()
        assert target_positions.stride() == (16, 1)
    starts = torch.tensor([0, 2, 6, 9], dtype=torch.int32)
    seq_lens = torch.tensor([15, 24, 31], dtype=torch.int32)
    context_slots = torch.tensor([9, 10, 55, 56, 57, 58, 80, 81, 82], dtype=torch.int64)
    # Non-contiguous row storage catches ignoring the actual block-table stride.
    table = torch.arange(3 * 32, dtype=torch.int32).reshape(3, 32)[:, :16]
    batch, block = 3, 4
    queries_per_req = draft_tokens + (0 if sample_from_anchor else 1)
    query_count = batch * queries_per_req
    outputs = {
        "out_input_ids_ptr": torch.full((query_count + 5,), -777, dtype=torch.int64),
        "out_context_positions_ptr": torch.full((14,), -777, dtype=torch.int32),
        "out_query_positions_ptr": torch.full((query_count + 5,), -777, dtype=torch.int32),
        "out_context_slot_mapping_ptr": torch.full((14,), -777, dtype=torch.int32),
        "out_query_slot_mapping_ptr": torch.full((query_count + 5,), -777, dtype=torch.int32),
        "out_token_indices_ptr": torch.full((batch * draft_tokens + 5,), -777, dtype=torch.int32),
    }
    bonus = torch.tensor([11, 22, 33])
    ops.copy_and_expand_dflash_inputs(
        next_token_ids_ptr=bonus,
        target_positions_ptr=target_positions,
        context_slot_mapping_ptr=context_slots,
        block_table_ptr=table,
        block_table_stride=table.stride(0),
        query_start_loc_ptr=starts,
        seq_lens_ptr=seq_lens,
        num_rejected_tokens_ptr=0 if rejections is None else torch.tensor(rejections),
        parallel_drafting_token_id=999,
        block_size=block,
        num_query_per_req=queries_per_req,
        num_speculative_tokens=draft_tokens,
        total_input_tokens=9,
        batch_size=batch,
        HAS_NUM_REJECTED=rejections is not None,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        **outputs,
    )
    expected_ids, expected_positions, expected_slots, expected_indices = [], [], [], []
    for req in range(batch):
        rejected = 0 if rejections is None else rejections[req]
        begin, end = int(starts[req]), int(starts[req + 1]) - rejected
        first_position = int(positions[end - 1]) + 1 if end > begin else int(positions[begin])
        for step in range(queries_per_req):
            linear_position = int(seq_lens[req]) - rejected + step
            expected_ids.append(int(bonus[req]) if step == 0 else 999)
            expected_positions.append(first_position + step)
            expected_slots.append(int(table[req, linear_position // block]) * block + linear_position % block)
            if sample_from_anchor or step:
                expected_indices.append(req * queries_per_req + step)
    expected = {
        "out_input_ids_ptr": expected_ids,
        "out_context_positions_ptr": positions.tolist(),
        "out_query_positions_ptr": expected_positions,
        "out_context_slot_mapping_ptr": context_slots.tolist(),
        "out_query_slot_mapping_ptr": expected_slots,
        "out_token_indices_ptr": expected_indices,
    }
    for name, values in expected.items():
        assert outputs[name].tolist() == values + [-777] * (len(outputs[name]) - len(values))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"DCP_SIZE": 2}, "non-DCP"),
        ({"DCP_RANK": 1}, "non-DCP"),
        ({"CP_INTERLEAVE_SIZE": 2}, "non-DCP"),
        ({"num_query_per_req": 4}, "anchor sampling"),
        ({"num_speculative_tokens": 0}, "anchor sampling"),
    ],
)
def test_expand_rejects_unsupported_layout_before_buffer_access(overrides, message):
    # None buffers make any accidental read/write fail before the intended
    # validation error. Invalid layout must never partially mutate KV metadata.
    kwargs = {
        name: None
        for name, param in inspect.signature(ops.copy_and_expand_dflash_inputs).parameters.items()
        if param.default is inspect.Parameter.empty
    }
    kwargs.update(num_query_per_req=3, num_speculative_tokens=3, SAMPLE_FROM_ANCHOR=True)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        ops.copy_and_expand_dflash_inputs(**kwargs)


@pytest.mark.parametrize("batch,steps,top_k", [(1, 1, 1), (3, 7, 16), (17, 15, 8), (2, 0, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_selector_matches_sequential_walk_and_ties(batch, steps, top_k, dtype):
    generator = torch.Generator().manual_seed(91)
    ids = torch.randint(100, 2000, (batch, steps, top_k), generator=generator)
    scores = torch.randint(-2, 3, (batch, steps, top_k, top_k), generator=generator).to(dtype)
    expected = torch.empty((batch, steps), dtype=ids.dtype)
    for req in range(batch):
        previous = 0
        for step in range(steps):
            row = scores[req, step, previous].tolist()
            previous = row.index(max(row))
            expected[req, step] = ids[req, step, previous]
    torch.testing.assert_close(ops.greedy_select_path_310p(ids, scores), expected, rtol=0, atol=0)


def test_selector_rejects_mismatched_edge_shape():
    with pytest.raises(ValueError, match="Expected scores"):
        ops.greedy_select_path_310p(torch.zeros(1, 3, 4, dtype=torch.int64), torch.zeros(1, 3, 4))


@pytest.mark.parametrize("count", [1, 6, 9, 12, 21])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16, torch.float32])
def test_prefix_write_preserves_padded_tail(count, dtype):
    # CPU contract regression for the device-observed sliced-copy corruption.
    # Non-aligned lengths leave live sentinel elements in the same padded span.
    destination = torch.full((count + 13,), -777, dtype=dtype)
    source = (torch.arange(count, dtype=torch.float32) + 0.5).reshape(1, count)
    expected = destination.clone()
    expected[:count] = source.flatten().to(dtype)
    ops._copy_prefix(destination, source)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [None, [], [7], [7, 8]])
def test_rtx_benchmark_requires_actual_token_ids(monkeypatch, tokens):
    path = _PATH.parents[2] / "artifacts/qwen38_dflash2_20261019/bench.py"
    spec = importlib.util.spec_from_file_location("dflash_rtx_bench", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    response = {"tokens_predicted": 2, "content": "example", "timings": {}}
    if tokens is not None:
        response["tokens"] = tokens

    def fake_urlopen(request, timeout):
        assert json.loads(request.data)["return_tokens"] is True
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(bench, "urlopen", fake_urlopen)
    if tokens == [7, 8]:
        assert bench.complete("http://unused", "prompt", 2)["tokens"] == tokens
    else:
        with pytest.raises(RuntimeError, match="token ID"):
            bench.complete("http://unused", "prompt", 2)


@pytest.mark.parametrize("has_triton,on_310p", [(True, True), (False, False), (True, False)])
def test_prepare_padded_dispatch_preserves_rejection_counts(has_triton, on_310p):
    # Compile the real method without importing the unavailable NPU dependency
    # graph. This exercises both dispatch and tensor math, not a copied guard.
    path = _PATH.parents[1] / "spec_decode/llm_base_proposer.py"
    module = ast.parse(path.read_text())
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs_padded"
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            launches.append(grid)

            def run(cumulative, valid, starts, indices, rejected, num_reqs, **kwargs):
                assert num_reqs == 3
                assert kwargs["BLOCK_SIZE"] == 4
                rejected.copy_(torch.tensor([0, 2, 8], dtype=torch.int32))
                indices.copy_(torch.tensor([3, 5, 7], dtype=torch.int32))

            return run

    class Metadata(SimpleNamespace):
        def __getattr__(self, name):
            return None

    namespace = {
        "torch": torch,
        "HAS_TRITON": has_triton,
        "ascend_utils": SimpleNamespace(is_310p=lambda: on_310p),
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        "get_vectorcore_num": lambda: 8,
        "_PREPARE_INPUTS_BLOCK_SIZE": 4,
        "prepare_inputs_padded_kernel": Kernel(),
        "AscendCommonAttentionMetadata": SimpleNamespace,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    starts = torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    cad = Metadata(query_start_loc=starts, query_start_loc_cpu=starts, num_reqs=3)
    proposer = SimpleNamespace(arange=torch.arange(32), runner=Metadata())
    result = namespace["prepare_inputs_padded"](
        proposer,
        cad,
        SimpleNamespace(cu_num_draft_tokens=torch.tensor([0, 3, 10], dtype=torch.int32)),
        torch.tensor([1, 2, 0], dtype=torch.int32),
    )
    assert launches == ([(1,)] if has_triton and not on_310p else [])
    assert result[0].num_actual_tokens == 16
    assert result[1].tolist() == list(range(16))
    assert result[2].tolist() == [3, 5, 7]
    assert result[3].tolist() == [0, 2, 8]
