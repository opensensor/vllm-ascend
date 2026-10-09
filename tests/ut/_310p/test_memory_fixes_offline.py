# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute transfer/ordering logic on CPU without importing hardware packages.

AST extraction isolates real production functions from unrelated NPU imports;
these checks do not qualify NPU streams, kernels or graph capture.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend._310p.graph_update_ordering import GraphUpdateOrdering
from vllm_ascend._310p.host_staging import PinnedHostStaging

ROOT = Path(__file__).resolve().parents[3]


def _extract(path, names, scope=None, baseline=False):
    source = (ROOT / path).read_text()
    tree = ast.parse(source)
    if baseline:
        # Restore only the old layout/cast operations for numerical comparison.
        source = source.replace("k_kernel.to(torch.float32)", "k.transpose(1, 2).contiguous().to(torch.float32)")
        source = source.replace(
            ".to(dtype=torch.float32, memory_format=torch.contiguous_format)", ".contiguous().to(torch.float32)"
        )
        tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names)
    namespace = dict(torch=torch, np=np, Any=object, **(scope or {}))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


class _Event:
    def __init__(self):
        self.complete = False
        self.records = []
        self.waits = 0

    def query(self):
        return self.complete

    def synchronize(self):
        self.waits += 1
        self.complete = True

    def record(self, stream=None):
        self.records.append(stream)
        self.complete = False


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_host_dma_reuse_waits_before_host_overwrite(dtype):
    event = _Event()
    stage = PinnedHostStaging((4, 8), dtype, pin_memory=False, event_factory=lambda: event)
    source = torch.arange(16).reshape(2, 8).to(dtype)
    first = torch.empty_like(source)
    stage.copy_to(source, first)
    pointer = stage.host.data_ptr()
    second = torch.empty_like(source)
    stage.copy_to(source + 3, second)
    assert event.waits == 1
    assert stage.host.data_ptr() == pointer
    torch.testing.assert_close(first, source)
    torch.testing.assert_close(second, source + 3)
    event.complete = True
    stage.copy_to(source, second)
    assert event.waits == 1


@pytest.mark.parametrize("shape,dtype", [((5, 8), torch.int32), ((4, 9), torch.int32), ((2, 8), torch.float16)])
def test_host_stage_rejects_capacity_or_dtype_mismatch(shape, dtype):
    stage = PinnedHostStaging((4, 8), torch.int32, pin_memory=False, event_factory=_Event)
    with pytest.raises(ValueError):
        stage.copy_to(torch.empty(shape, dtype=dtype), torch.empty(shape, dtype=dtype))


def test_graph_host_mutation_waits_for_previous_replay_and_device_updates():
    guard = GraphUpdateOrdering(_Event)
    main, update = Mock(), Mock()
    guard.before_update(main, update)
    assert guard.previous_replay.waits == 0
    update.wait_stream.assert_called_once_with(main)
    ready = guard.ready(main, update)
    main.wait_event.assert_called_once_with(guard.update_done)
    assert guard.update_done.records == [update]
    assert ready.records == [main]
    guard.replay_submitted(main)
    guard.before_update(main, update)
    assert guard.previous_replay.waits == 1
    guard.previous_replay.complete = True
    guard.before_update(main, update)
    assert guard.previous_replay.waits == 1


@pytest.mark.parametrize("kind", ["disjoint", "overlap", "same", "strided"])
def test_mamba_copy_retains_overlap_semantics_without_disjoint_scratch(monkeypatch, kind):
    copy = _extract("vllm_ascend/patch/worker/patch_mamba_utils.py", ["_copy_mamba_state"])["_copy_mamba_state"]
    storage = torch.arange(24, dtype=torch.float32)
    src = storage[:8]
    dst = {"disjoint": storage[12:20], "overlap": storage[3:11], "same": src, "strided": storage[1:17:2]}[kind]
    expected = storage.clone()
    expected_dst = {
        "disjoint": expected[12:20],
        "overlap": expected[3:11],
        "same": expected[:8],
        "strided": expected[1:17:2],
    }[kind]
    expected_dst.copy_(src.clone())
    calls = []
    original = torch.Tensor.clone
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda value, *a, **kw: (calls.append(value), original(value, *a, **kw))[1]
    )
    copy(src, dst)
    torch.testing.assert_close(storage, expected)
    assert len(calls) == (1 if kind in ("overlap", "strided") else 0)


def test_gdn_nonuniform_state_rows_and_padding():
    flatten = _extract(
        "vllm_ascend/_310p/ops/fla/gdn_310.py",
        ["_flatten_state_indices"],
        {"_EXTRA_CTX": SimpleNamespace(capturing=False)},
    )["_flatten_state_indices"]
    idx = torch.tensor([[8, 9, -1], [20, -1, -1], [-1, -1, -1]])
    result = flatten(idx, torch.tensor([0, 2, 3, 3]), 3)
    torch.testing.assert_close(result, torch.tensor([8, 9, 20], dtype=torch.int32))
    assert flatten(idx, torch.zeros(4, dtype=torch.int32), 0).numel() == 0
    uniform = flatten(idx[:2, :2], torch.tensor([0, 2, 4]), 4, True)
    torch.testing.assert_close(uniform, idx[:2, :2].flatten().int())


def test_gdn_host_bounds_never_read_device_boundaries(monkeypatch):
    plans = []
    planner = lambda bounds, chunk: plans.append(bounds.tolist()) or (bounds.tolist(), chunk)
    cached = _extract(
        "vllm_ascend/_310p/ops/fla/gdn_310.py",
        ["_cached_chunk_plan"],
        {"CHUNK_SIZE": 64, "build_varlen_chunk_plan": planner},
    )["_cached_chunk_plan"]
    cu = torch.tensor([0, 1, 66, 66])
    metadata = SimpleNamespace(_gdn_host_chunk_source=cu, _gdn_host_chunk_bounds=(0, 1, 66, 66))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda _: pytest.fail("unexpected boundary readback"))
    first = cached(metadata, cu)
    for _ in range(35):
        assert cached(metadata, cu) is first
    assert plans == [[0, 1, 66, 66]]


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("heads", [(2, 2), (4, 12)])
def test_wy_layout_fusion_preserves_fp32_math(grouped, heads):
    names = [
        "_expand_qk_to_v_heads",
        "_upper_incl_diag_mask",
        "_strictly_lower_decay",
        "_inv_small_unit_lower",
        "_inv_unit_lower_recursive",
        "_inv_unit_lower_triangular",
        "_ut_transform",
        "_compute_kernel_inputs_from_torch_wy",
    ]
    scope = {
        "_WY_GROUPED_GRAM": grouped,
        "_UT_USE_BLOCKED_INVERSE": True,
        "_UT_INVERSE_BLOCK": 8,
        "_DECAY_MASK_CACHE": {},
    }
    current = _extract("vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py", names, scope)
    baseline = _extract("vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py", names, scope, baseline=True)
    torch.manual_seed(33)
    k_heads, v_heads = heads
    q, k = [torch.randn(1, 128, k_heads, 128).half() * 0.02 for _ in range(2)]
    v = torch.randn(1, 128, v_heads, 128).half()
    g = -torch.rand(1, 128, v_heads).float() * 0.1
    beta = torch.rand(1, 128, v_heads).half() * 0.3
    args = (q, k, v, g, beta, 64)
    expected = baseline["_compute_kernel_inputs_from_torch_wy"](*args)
    actual = current["_compute_kernel_inputs_from_torch_wy"](*args)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left, right)
        assert left.is_contiguous()
    assert actual[-1].dtype == torch.float32


@dataclass
class _SamplerOutput:
    sampled_token_ids: torch.Tensor


def _snapshot_runner(parent):
    tree = ast.parse((ROOT / "vllm_ascend/_310p/model_runner_310p.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
    methods = [
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_bookkeeping_sync", "_update_states_after_model_execute")
    ]
    cls = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    ns = {"Parent": parent, "replace": replace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "<runner snapshot>", "exec"), ns)
    result = ns["Runner"]()
    result.input_batch = SimpleNamespace(req_ids=["a", "b"])
    result._qwen4exp_mtp_ple = True
    result.use_async_scheduling = False
    result.need_accepted_tokens = True
    return result


@pytest.mark.parametrize("reorder", [False, True])
def test_accepted_snapshot_precedes_discard_filter_and_expires(reorder):
    seen = []

    class Parent:
        def _bookkeeping_sync(self, scheduler, sampler, *args, **kwargs):
            # A discarded request has empty parsed output, but its raw state
            # acceptance remains 2 and must not be reset by that filter.
            return [[], [4]]

        def _update_states_after_model_execute(self, ids, scheduler):
            seen.append(getattr(self.input_batch, "_mamba_accepted_counts_snapshot", None))

    runner = _snapshot_runner(Parent)
    sampled = torch.tensor([[7, 8, -1], [4, -1, -1]])
    assert runner._bookkeeping_sync(None, _SamplerOutput(sampled)) == [[], [4]]
    if reorder:
        runner.input_batch.req_ids.reverse()
    runner._update_states_after_model_execute(sampled, None)
    if reorder:
        assert seen == [None]
    else:
        torch.testing.assert_close(seen[0], torch.tensor([2, 1]))
    assert runner._mamba_sample_snapshot is None
    assert not hasattr(runner.input_batch, "_mamba_accepted_counts_snapshot")
    runner._update_states_after_model_execute(sampled, None)
    assert seen[-1] is None


def test_snapshot_clears_after_failed_bookkeeping_or_state_update():
    class Parent:
        def _bookkeeping_sync(self, *args, **kwargs):
            raise RuntimeError("bookkeeping failed")

        def _update_states_after_model_execute(self, *args, **kwargs):
            raise RuntimeError("state update failed")

    runner = _snapshot_runner(Parent)
    sampled = torch.tensor([[7, -1], [4, -1]])
    with pytest.raises(RuntimeError):
        runner._bookkeeping_sync(None, _SamplerOutput(sampled))
    assert runner._mamba_sample_snapshot is None
    runner._mamba_sample_snapshot = (sampled, ("a", "b"), torch.tensor([1, 1]))
    with pytest.raises(RuntimeError):
        runner._update_states_after_model_execute(sampled, None)
    assert not hasattr(runner.input_batch, "_mamba_accepted_counts_snapshot")


@pytest.mark.parametrize("fail", [False, True])
def test_runner_scopes_ready_event_to_target_forward_and_orders_failed_replay(monkeypatch, fail):
    tree = ast.parse((ROOT / "vllm_ascend/_310p/model_runner_310p.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_model_forward")
    ctx = SimpleNamespace(cudagraph_runtime_mode="FULL", capturing=False)
    main, update = Mock(), Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: main), raising=False)
    guard = GraphUpdateOrdering(_Event)
    namespace = {
        "torch": torch,
        "partial": partial,
        "get_forward_context": lambda: ctx,
        "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<target forward>", "exec"), namespace)

    def model(**kwargs):
        assert ctx._ascend_replay_ready_event is guard.replay_ready
        ctx._ascend_replay_ready_event.synchronize()
        if fail:
            raise RuntimeError("replay failed")
        return "hidden states"

    runner = SimpleNamespace(
        uses_mrope=False,
        _qwen4exp_mtp_ple=True,
        speculative_config=object(),
        enable_enpu=False,
        update_stream=update,
        _qwen_graph_ordering=guard,
        vllm_config=SimpleNamespace(additional_config={"qwen_graph_update_ordering": "completion_events"}),
        model=model,
        _stage_qwen4exp_ple_inputs=lambda n: (None, None),
        input_ids=SimpleNamespace(cpu=torch.zeros(3, dtype=torch.int32)),
        _ple_query_start_loc_cpu=None,
        _ple_context_cpu=None,
        _update_full_graph_params_if_needed=Mock(),
    )
    forward = namespace["_model_forward"]
    if fail:
        with pytest.raises(RuntimeError):
            forward(runner, 3)
    else:
        assert forward(runner, 3) == "hidden states"
    assert guard.previous_replay.records == [main]
    assert not hasattr(ctx, "_ascend_replay_ready_event")
    assert guard.replay_ready.waits == 1
    assert main.synchronize.call_count == 0


def test_compact_table_stage_preserves_scheduler_table_and_has_no_device_temporary(monkeypatch):
    tree = ast.parse((ROOT / "vllm_ascend/_310p/model_runner_310p.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_copy_compact_mamba_table"
    )
    make_stage = lambda shape, dtype: PinnedHostStaging(shape, dtype, pin_memory=False, event_factory=_Event)
    ns = {"torch": torch, "PinnedHostStaging": make_stage}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<table staging>", "exec"), ns)
    monkeypatch.setattr(torch, "as_tensor", lambda *a, **kw: pytest.fail("device temporary forbidden"))
    runner = SimpleNamespace()
    table = torch.zeros((4, 7), dtype=torch.int32)
    mapped = np.array([[0, 2, 3, 0, 0, 0, 0], [0, 8, 9, 10, 0, 0, 0]], dtype=np.int32)
    original = mapped.copy()
    ns["_copy_compact_mamba_table"](runner, 1, mapped, table)
    assert np.array_equal(mapped, original)
    assert np.array_equal(table[:2].numpy(), mapped)
    assert not table[2:].any()
    stage = runner._compact_mamba_host_stages[1]
    ns["_copy_compact_mamba_table"](runner, 1, mapped[:, ::-1].copy(), table)
    assert runner._compact_mamba_host_stages[1] is stage
    assert stage._completion.waits == 1


def test_gdn_metadata_bounds_snapshot_does_not_follow_mutable_scheduler_storage():
    tree = ast.parse((ROOT / "vllm_ascend/ops/gdn_attn_builder.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendGDNAttentionMetadataBuilder"
    )
    build = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    attach = next(
        node for node in build.body if isinstance(node, ast.If) and "_gdn_host_chunk_bounds" in ast.unparse(node)
    )
    cu_cpu = torch.tensor([0, 1, 66, 66], dtype=torch.int32)
    device_bounds = object()
    metadata = SimpleNamespace(non_spec_query_start_loc=device_bounds)
    scope = {"torch": torch, "num_prefills": 2, "non_spec_query_start_loc_cpu": cu_cpu, "attn_metadata": metadata}
    exec(compile(ast.Module(body=[attach], type_ignores=[]), "<GDN builder attachment>", "exec"), scope)
    cu_cpu.add_(100)
    assert metadata._gdn_host_chunk_source is device_bounds
    assert metadata._gdn_host_chunk_bounds == (0, 1, 66, 66)


def test_wy_fp16_layout_copy_count_regression():
    names = [
        "_expand_qk_to_v_heads",
        "_upper_incl_diag_mask",
        "_strictly_lower_decay",
        "_inv_small_unit_lower",
        "_inv_unit_lower_recursive",
        "_inv_unit_lower_triangular",
        "_ut_transform",
        "_compute_kernel_inputs_from_torch_wy",
    ]
    scope = {"_WY_GROUPED_GRAM": True, "_UT_USE_BLOCKED_INVERSE": True, "_UT_INVERSE_BLOCK": 8, "_DECAY_MASK_CACHE": {}}
    current = _extract("vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py", names, scope)
    baseline = _extract("vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py", names, scope, baseline=True)

    class Copies(TorchDispatchMode):
        def __init__(self):
            self.half_clones = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.clone.default and args[0].dtype == torch.float16:
                self.half_clones += 1
            return func(*args, **(kwargs or {}))

    q = torch.randn(1, 64, 4, 128).half() * 0.02
    v = torch.randn(1, 64, 12, 128).half()
    g = -torch.rand(1, 64, 12)
    beta = torch.rand(1, 64, 12).half() * 0.2
    counts = []
    for ns in (baseline, current):
        with Copies() as copies:
            ns["_compute_kernel_inputs_from_torch_wy"](q, q, v, g, beta, 64)
        counts.append(copies.half_clones)
    assert counts == [5, 2]


@pytest.mark.parametrize(
    "running,src_block,accepted,expected_accepted,expected_copies",
    [(128, 0, 3, 1, 0), (126, 0, 3, 1, 1), (128, 1, 3, 3, 1), (20, 0, 2, 2, 0)],
)
def test_mamba_align_reuses_raw_host_acceptance_and_preserves_reset_rule(
    running, src_block, accepted, expected_accepted, expected_copies
):
    state = torch.arange(48).reshape(12, 4).float()
    copies = []

    def state_copy(tensor, block_ids, index, bias):
        copies.append((index, int(bias)))
        return SimpleNamespace(start_addr=tensor[block_ids[index]].data_ptr(), num_elements=4)

    def view(tensor, address, count):
        offset = (address - tensor.data_ptr()) // tensor.element_size()
        return tensor.reshape(-1)[offset : offset + count]

    scope = {
        "mamba_utils": SimpleNamespace(_get_mamba_spec_for_layer=lambda *args: SimpleNamespace(mamba_type="gdn")),
        "get_mamba_postprocess_block_ids": lambda *args: [10, 11],
        "_tensor_view_from_data_ptr": view,
        "_copy_mamba_state": lambda src, dst: dst.copy_(src.clone()),
    }
    fallback = _extract(
        "vllm_ascend/patch/worker/patch_mamba_utils.py", ["_postprocess_mamba_align_gpu_cpu_fallback"], scope
    )["_postprocess_mamba_align_gpu_cpu_fallback"]
    ctx = SimpleNamespace(
        mamba_state_idx_buf=SimpleNamespace(np=np.array([src_block])),
        num_scheduled_tokens_buf=SimpleNamespace(np=np.array([3])),
        num_computed_tokens_buf=SimpleNamespace(np=np.array([running - 1])),
        num_draft_tokens_buf=SimpleNamespace(np=np.array([2])),
        block_size=128,
        mamba_group_ids=[0],
    )
    counts = torch.zeros(1, dtype=torch.int32)
    fallback(
        bufs=SimpleNamespace(postprocess_align=ctx),
        num_reqs=1,
        num_accepted_tokens_gpu=object(),
        num_accepted_tokens_cpu_tensor=counts,
        input_batch=SimpleNamespace(_mamba_accepted_counts_snapshot=torch.tensor([accepted])),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]),
        forward_context={"layer": SimpleNamespace(kv_cache=[state])},
        mamba_state_copy_funcs={"gdn": [state_copy]},
    )
    assert counts.tolist() == [expected_accepted]
    assert len(copies) == expected_copies


@pytest.mark.parametrize("ready,draft", [(False, False), (True, False), (True, True)])
def test_breakable_graph_uses_scoped_completion_and_preserves_draft_exception(monkeypatch, ready, draft):
    source = ROOT / "vllm_ascend/compilation/breakable_aclgraph.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BreakableACLGraphWrapper")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_replay")

    class Parent:
        def _replay(self, entry, args, kwargs):
            entry.replayed = True

    wrapper = ast.ClassDef(
        name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    stream = Mock()
    event = Mock()
    ctx = SimpleNamespace(cudagraph_runtime_mode="FULL")
    if ready:
        ctx._ascend_replay_ready_event = event
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: stream), raising=False)
    scope = {
        "torch": torch,
        "Parent": Parent,
        "get_forward_context": lambda: ctx,
        "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=draft),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "<graph wrapper>", "exec"),
        scope,
    )
    instance = scope["Wrapper"]()
    instance.use_eagle = True
    instance.enable_enpu = False
    entry = SimpleNamespace(output="output", replayed=False)
    assert instance._replay(entry, (), {}) == "output"
    assert entry.replayed
    assert stream.synchronize.call_count == (1 if not ready and not draft else 0)
    assert event.synchronize.call_count == (1 if ready and not draft else 0)
