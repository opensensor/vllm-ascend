# SPDX-License-Identifier: Apache-2.0
"""Offline ordering, native state bytes and collective dependency contracts."""

import ast
import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tools.qwen4exp import native_prefill
from tools.qwen4exp.native_state_layout import NativeStateLayout
from tools.qwen4exp.tp_prefill_pipeline import pipelined_prefill
from vllm_ascend._310p.prefix_mamba_state import (
    PrefixMambaStateTier,
    apply_prefix_mamba_updates,
    remap_prefix_mamba_rows,
)
from vllm_ascend._310p.transfer_audit import TransferLedger, copy_direction

ROOT = Path(__file__).resolve().parents[3]


def tier():
    state = torch.zeros(3, 2, dtype=torch.float32)
    return PrefixMambaStateTier([(state,)], 3, device_archive_slots=2), state


def test_ledger_is_bounded_and_never_reads_devices(monkeypatch):
    for name in ("item", "cpu", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, lambda *a: pytest.fail("device read"))
    ledger = TransferLedger(3)
    for i in range(8):
        ledger.record("d2d", "swap", nbytes=32)
    ledger.record("barrier", "phase", elapsed_ns=7)
    snapshot = ledger.snapshot()
    assert snapshot["counts"]["d2d_bytes"] == 256
    assert snapshot["counts"]["barrier_host_ns"] == 7
    assert [event["sequence"] for event in snapshot["events"]] == [7, 8, 9]
    assert snapshot["dropped_events"] == 6
    assert snapshot["measured_bus_bytes"] is False
    snapshot["counts"]["d2d_bytes"] = 0
    assert ledger.snapshot()["counts"]["d2d_bytes"] == 256


@pytest.mark.parametrize(
    "source,target,expected",
    [("cpu", "cpu", "host_copy"), ("cpu", "npu", "h2d"), ("npu", "cpu", "d2h"), ("npu", "npu", "d2d")],
)
def test_copy_direction_uses_only_device_metadata(source, target, expected):
    assert copy_direction(SimpleNamespace(type=source), SimpleNamespace(type=target)) == expected


def test_all_checkpoint_copy_bytes_include_swap_and_cow():
    t, state = tier()
    for block in (101, 102, 103):
        t.remap_table(np.array([[block]], dtype=np.int32), 1)
        state[t.slot_for(block)].fill_(block)
    before = t.transfer_ledger.snapshot()["counts"]["host_copy_bytes"]
    t.remap_table(np.array([[101]], dtype=np.int32), 1)
    counts = t.transfer_ledger.snapshot()["counts"]
    assert counts["host_copy_bytes"] - before == 3 * 8
    assert counts["reason:archive_swap_stage:calls"] == 1
    t.copy(101, 102)
    assert t.transfer_ledger.snapshot()["counts"]["reason:copy_on_write:calls"] == 1


@pytest.mark.parametrize("batched", [False, True])
def test_update_phase_preserves_cow_chain_and_has_one_drain(monkeypatch, batched):
    tiers = {group: tier()[0] for group in (1, 2, 3)}
    drains = []
    for group, t in tiers.items():
        t.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
        t.layer_states[0][0][t.slot_for(101)].fill_(group * 7)
        monkeypatch.setattr(t, "_synchronize_device_state", lambda g=group: drains.append(g))
    apply_prefix_mamba_updates(
        tiers, {g: [102] for g in tiers}, {g: [(101, 102), (102, 103)] for g in tiers}, batched=batched
    )
    assert len(drains) == (1 if batched else 9)
    # Sequential copies must read the just-written target, not an old snapshot.
    remap_prefix_mamba_rows(
        tiers, {g: (np.array([[103]], dtype=np.int32), [1], [(0,)]) for g in tiers}, batched=batched
    )
    for group, t in tiers.items():
        assert t.layer_states[0][0][t.slot_for(103)].tolist() == [group * 7, group * 7]


def test_admission_phase_has_one_drain_and_never_reuses_previous_phase(monkeypatch):
    tiers = {g: tier()[0] for g in (1, 2, 3)}
    drains = []
    for group, t in tiers.items():
        t.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
        monkeypatch.setattr(t, "_synchronize_device_state", lambda g=group: drains.append(g))
    updates = {g: [(101, 102)] for g in tiers}
    apply_prefix_mamba_updates(tiers, {g: [] for g in tiers}, updates, batched=True)
    plans = {g: (np.array([[103, 104]], dtype=np.int32), [2], [(0, 1)]) for g in tiers}
    remap_prefix_mamba_rows(tiers, plans, batched=True)
    assert len(drains) == 2
    remap_prefix_mamba_rows(tiers, plans, batched=True)
    assert len(drains) == 2
    apply_prefix_mamba_updates(tiers, {g: [] for g in tiers}, {g: [] for g in tiers}, batched=True)
    assert len(drains) == 2


def test_phase_drain_failure_preserves_all_groups(monkeypatch):
    tiers = {g: tier()[0] for g in (1, 2)}
    for t in tiers.values():
        t.remap_table(np.array([[101]], dtype=np.int32), 1)
        t.layer_states[0][0][t.slot_for(101)].fill_(3)
    monkeypatch.setattr(tiers[1], "_synchronize_device_state", Mock(side_effect=RuntimeError("writer")))
    with pytest.raises(RuntimeError, match="writer"):
        apply_prefix_mamba_updates(tiers, {g: [101] for g in tiers}, {g: [] for g in tiers}, batched=True)
    assert all(t.layer_states[0][0][t.slot_for(101)].tolist() == [3, 3] for t in tiers.values())


@pytest.fixture(scope="module")
def state_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("state-layout")
    wrapper = directory / "cpu.cpp"
    wrapper.write_text(
        '#include "kernel_operator.h"\n'
        f'#include "{ROOT / "tools/qwen4exp/native_state_layout.cpp"}"\n'
        'extern "C" void gather(void* a,void* b,void* c,void* d,void* e) {\n'
        "for(int i=0;i<8;++i){AscendC::blockIndex=i;qwen_state_gather_v1(a,b,c,d,e);}}\n"
        'extern "C" void scatter(void* a,void* b,void* c,void* d,void* e) {\n'
        "for(int i=0;i<8;++i){AscendC::blockIndex=i;qwen_state_scatter_v1(a,b,c,d,e);}}\n"
    )
    path = directory / "cpu.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{ROOT / 'tests/ut/qwen38_1m/prefill_cpu_stubs'}",
            str(wrapper),
            "-o",
            str(path),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(path))
    for name in ("gather", "scatter"):
        getattr(library, name).argtypes = [ctypes.c_void_p] * 5
        getattr(library, name).restype = None
    return library


@pytest.mark.parametrize("heads,sequences", [(1, 1), (12, 3), (48, 4)])
def test_native_state_body_exact_gather_mask_transpose_scatter(monkeypatch, state_library, heads, sequences):
    monkeypatch.setattr(native_prefill, "_on_npu", lambda device: True)
    monkeypatch.setattr(native_prefill, "_capturing", lambda device: False)

    def launch(kernel, tensors, blocks):
        assert blocks == 8
        getattr(state_library, kernel)(*[t.data_ptr() for t in tensors])

    native = NativeStateLayout("gather", "scatter", launch)
    cache = torch.randn(7, heads, 128, 128)
    slots = torch.tensor([6, 1, 3, 4][:sequences], dtype=torch.int32)
    valid = torch.tensor([False, True, True, False][:sequences])
    cache[slots[~valid].long()] = float("nan")
    expected = cache[slots.long()].clone()
    expected[~valid] = 0
    packed = native.gather(cache, slots, valid)
    assert torch.equal(packed, expected.transpose(-1, -2).contiguous())
    initial = cache.clone()
    final = torch.randn_like(packed)
    native.scatter(cache, slots, valid, final)
    initial[slots.long()] = final.transpose(-1, -2)
    torch.testing.assert_close(cache, initial, rtol=0, atol=0, equal_nan=True)


def test_native_layout_chunk_seam_avoids_both_state_materializations():
    path = ROOT / "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "chunk_gated_delta_rule_310"
    )
    observed = []

    def fwd_h(*args, **kwargs):
        observed.append(kwargs["initial_state"])
        return None, None, kwargs["initial_state"] + 3

    scope = {
        "torch": torch,
        "Callable": __import__("collections.abc").abc.Callable,
        "VarlenChunkPlan": object,
        "CHUNK_SIZE": 64,
        "_normalize_chunk_inputs": lambda q, k, v, g, beta, cu: (q, k, v, g, beta, False),
        "_require_ascend_chunk_ops": lambda *args: None,
        "_maybe_l2norm": lambda value, enabled: value,
        "_pad_bthd_to_chunk": lambda q, k, v, g, beta, size: (q, k, v, g, beta, [(0, 0, 64)], None),
        "_compute_kernel_inputs_from_torch_wy": lambda q, k, v, g, beta, size: (q, k, None, v, g),
        "_unpad_chunk_output": lambda output, *args: output,
    }
    fake = SimpleNamespace(
        chunk_gated_delta_rule_fwd_h=fwd_h, chunk_fwd_o_vllm=lambda *a, **k: torch.zeros(1, 2, 64, 128)
    )
    scope["torch"] = SimpleNamespace(
        **{name: getattr(torch, name) for name in ("Tensor", "float32", "zeros")}, ops=SimpleNamespace(_C_ascend=fake)
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    function = scope["chunk_gated_delta_rule_310"]
    q = torch.zeros(1, 64, 1, 8)
    v = torch.zeros(1, 64, 2, 128)
    g = torch.zeros(1, 64, 2)
    state = torch.randn(1, 2, 8, 128)
    out, final = function(q, q, v, g, g, initial_state=state, output_final_state=True, state_is_kernel_layout=True)
    assert observed[-1] is state
    assert torch.equal(final, state + 3)
    old = state.transpose(-1, -2).contiguous()
    out, final = function(q, q, v, g, g, initial_state=old, output_final_state=True)
    assert torch.equal(final, (state + 3).transpose(-1, -2))


@pytest.mark.parametrize("tokens", [129, 1024, 2561, 4100])
def test_tp_pipeline_combines_shared_before_reduce_and_bounds_dependencies(tokens):
    torch.manual_seed(1)
    inputs = torch.randn(tokens, 4).half()
    order = []
    pending = set()
    serial = 0
    module = SimpleNamespace(
        shared_expert_replicated=False,
        expert_tp_size=4,
        grouped_routing=True,
        gate=torch.eye(4).half(),
        top_k=1,
        renormalize=True,
        routed_scaling_factor=1,
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        has_shared_expert=True,
        _forward_grouped_chunk=lambda x, w, ids: x.float() * 2,
        _forward_shared=lambda x: x.float() * 3,
    )

    def submit(local):
        nonlocal serial
        serial += 1
        pending.add(serial)
        order.append(("submit", serial))
        assert len(pending) <= 2
        return local * 4, serial

    def wait(event):
        pending.remove(event)
        order.append(("wait", event))

    actual = pipelined_prefill(
        module, inputs, lambda logits, *a, **kw: (logits, logits), submit, wait, chunk_tokens=1024
    )
    assert torch.equal(actual, (inputs.float() * 20).half())
    assert not pending
    assert len([x for x in order if x[0] == "submit"]) == (tokens + 1023) // 1024
    assert order[-1] == ("wait", serial)


@pytest.mark.parametrize("columns,groups", [(64, 1), (64, 20), (80, 5), (160, 20)])
def test_cached_native_metadata_matches_strided_loads(columns, groups):
    torch.manual_seed(3)
    banks = torch.randn(3, columns // 16, groups, 16).half()
    cached = banks.flatten(1)
    for group in range(groups):
        old = torch.stack([bank[:, group, :].flatten() for bank in banks]).float()
        new = torch.stack(
            [
                torch.cat(
                    [cache[(nb * groups + group) * 16 : (nb * groups + group + 1) * 16] for nb in range(columns // 16)]
                )
                for cache in cached
            ]
        ).float()
        assert torch.equal(old, new)
    # Every row tile formerly loaded these bytes; the candidate loads them once.
    assert cached.numel() * cached.element_size() == 3 * columns * groups * 2


@pytest.mark.parametrize("native", [False, True])
def test_custom_qwen_serving_method_consumes_state_and_wy_hooks(monkeypatch, native):
    import sys
    from types import ModuleType

    path = ROOT / "vllm_ascend/models/qwen4_exp/model.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "_GDNAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_native_delta_rule")
    observed = []
    chunk_module = ModuleType("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
    helper_module = ModuleType("vllm_ascend._310p.ops.fla.gdn_310")

    def chunk(**kwargs):
        observed.append(kwargs)
        if kwargs["wy_prepare"] is not None:
            kwargs["wy_prepare"]()
        return kwargs["v"], kwargs["initial_state"] + 1

    chunk_module.chunk_gated_delta_rule_310 = chunk
    helper_module._cached_chunk_plan = lambda *args: "plan"
    helper_module._cached_recurrent_step_meta = lambda *args, **kwargs: None
    helper_module.npu_recurrent_gated_delta_rule_310 = Mock(return_value=torch.zeros(1, 4, 1, 128))
    monkeypatch.setitem(sys.modules, chunk_module.__name__, chunk_module)
    monkeypatch.setitem(sys.modules, helper_module.__name__, helper_module)
    scope = {"torch": torch, "GDNAttentionMetadata": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    cache = torch.randn(4, 1, 128, 128)
    ids = torch.tensor([2, 1], dtype=torch.int32)
    valid = torch.tensor([False, True])
    expected = cache[ids.long()].clone()
    expected[~valid] = 0
    expected += 1
    state_io = SimpleNamespace(
        gather=Mock(
            side_effect=lambda cache, ids, valid: (
                torch.where(valid[:, None, None, None], cache[ids.long()], 0).transpose(-1, -2).contiguous()
            )
        ),
        scatter=Mock(
            side_effect=lambda cache, ids, valid, state: cache.__setitem__(ids.long(), state.transpose(-1, -2))
        ),
    )
    wy = Mock()
    layer = SimpleNamespace(kv_cache=(None, cache), _gdn_wy_prepare=wy)
    if native:
        layer._gdn_state_io = state_io
    fn = scope["_native_delta_rule"]
    meta = SimpleNamespace(spec_sequence_masks=None, num_prefills=2)
    q = torch.zeros(4, 1, 128)
    fn(layer, q, q, q, torch.zeros(1, 4, 1), torch.ones(1, 4, 1), meta, ids, torch.tensor([0, 2, 4]), valid)
    assert torch.equal(cache[ids.long()], expected)
    assert observed[-1]["state_is_kernel_layout"] is native
    wy.assert_called_once_with()
    assert state_io.gather.call_count == int(native)
    assert state_io.scatter.call_count == int(native)
    # Decode bypasses gather/scatter and WY preparation even when configured.
    meta.num_prefills = 0
    fn(layer, q, q, q, torch.zeros(1, 4, 1), torch.ones(1, 4, 1), meta, ids, torch.tensor([0, 2, 4]), valid)
    assert state_io.gather.call_count == int(native)
    assert wy.call_count == 1


@pytest.mark.parametrize(
    "candidate,attribute,key,namespace",
    [
        ("native_state_layout", "_gdn_state_io", "state_io", "qwen_transfer_v1"),
        ("fused_wy", "_gdn_wy_prepare", "wy", "qwen_prefill_v2"),
    ],
)
def test_resident_hooks_target_actual_qwen_method_and_restore_failures(
    monkeypatch, candidate, attribute, key, namespace
):
    import importlib

    from vllm_ascend.models.qwen4_exp.model import _GDNAttention

    native = object()

    def original(self):
        assert getattr(self, attribute) is native
        raise RuntimeError("kernel")

    monkeypatch.setattr(_GDNAttention, "_native_delta_rule", original)
    factory = importlib.import_module(f"tools.qwen4exp.resident_candidates.{candidate}").replacements
    replacements = factory({namespace: {key: native}})
    assert list(replacements) == ["vllm_ascend.models.qwen4_exp.model:_GDNAttention._native_delta_rule"]
    for existed in (False, True):
        layer = SimpleNamespace()
        if existed:
            setattr(layer, attribute, None)
        with pytest.raises(RuntimeError, match="kernel"):
            next(iter(replacements.values()))(layer)
        assert hasattr(layer, attribute) is existed
        assert getattr(layer, attribute, None) is None


def test_transfer_snapshot_rejects_partial_ranks_restarts_and_decreasing_counts():
    import copy

    from tools.qwen4exp.transfer_snapshot import compare

    first = [
        {
            "rank": rank,
            "pid": rank + 100,
            "transfer_audit": {"prefix_mamba": {}, "runner": TransferLedger().snapshot(), "modules": {}},
        }
        for rank in range(4)
    ]
    last = copy.deepcopy(first)
    for worker in last:
        ledger = TransferLedger(8)
        ledger.record("d2d", "archive", nbytes=64)
        worker["transfer_audit"]["runner"] = ledger.snapshot()
    result = compare(first, last)
    assert all(entry["delta"]["d2d_bytes"] == 64 for entry in result["results"])
    with pytest.raises(ValueError, match="expected ranks"):
        compare(first, last[:-1])
    changed = copy.deepcopy(last)
    changed[0]["pid"] += 1
    with pytest.raises(ValueError, match="worker changed"):
        compare(first, changed)
    with pytest.raises(ValueError, match="restarted"):
        compare(last, first)


@pytest.mark.parametrize("batched", [False, True])
def test_runner_remaps_all_groups_then_stages_tables_without_extra_copies(monkeypatch, batched):
    from vllm_ascend._310p.prefix_mamba_state import prefix_mamba_active_columns

    path = ROOT / "vllm_ascend/_310p/model_runner_310p.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310"
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_remap_compact_mamba_block_tables"
    )
    scope = {
        "np": np,
        "cast": lambda cls, value: value,
        "MultiGroupBlockTable310": object,
        "prefix_mamba_active_columns": prefix_mamba_active_columns,
        "remap_prefix_mamba_rows": remap_prefix_mamba_rows,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    tiers = {g: tier()[0] for g in range(3)}
    drains = []
    for g, t in tiers.items():
        t.remap_table(np.array([[99, 98]], dtype=np.int32), 2)
        monkeypatch.setattr(t, "_synchronize_device_state", lambda g=g: drains.append(g))
    blocks = [
        SimpleNamespace(
            is_mamba_group=True,
            num_blocks_per_row=np.array([2]),
            block_table=SimpleNamespace(np=np.array([[101, 102]], dtype=np.int32), gpu=object()),
        )
        for g in tiers
    ]
    input_batch = SimpleNamespace(
        block_table=SimpleNamespace(block_tables=blocks), num_computed_tokens_cpu=np.array([0]), req_ids=["first"]
    )
    runner = SimpleNamespace(
        supports_compact_mamba_state=True,
        supports_prefix_mamba_state_tier=True,
        _prefix_mamba_tiers=tiers,
        _prefix_phase_batching=batched,
        input_batch=input_batch,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64, num_speculative_blocks=1)) for g in tiers
            ]
        ),
    )
    uploads = []

    def stage(group, mapped, device):
        uploads.append((group, mapped.copy()))
        if batched:
            assert all(set(t._resident) == {101, 102} for t in tiers.values())

    runner._copy_compact_mamba_table = stage
    scope["_remap_compact_mamba_block_tables"](runner, 1, np.array([1]))
    assert len(drains) == (1 if batched else 3)
    assert len(uploads) == 3
    assert len(input_batch._prefix_mamba_postprocess_tables) == 3


def test_fresh_unused_ids_do_not_add_a_batch_drain(monkeypatch):
    t, _ = tier()
    sync = Mock()
    monkeypatch.setattr(t, "_synchronize_device_state", sync)
    apply_prefix_mamba_updates({1: t}, {1: [101]}, {1: []}, batched=True)
    sync.assert_not_called()


@pytest.mark.parametrize(
    "candidate,resource,key",
    [("native_state_layout", "qwen_transfer_v1", "state_io"), ("fused_wy", "qwen_prefill_v2", "wy")],
)
def test_repeated_preparation_unwraps_only_the_previous_gdn_candidate(monkeypatch, candidate, resource, key):
    import importlib

    from vllm_ascend.models.qwen4_exp.model import _GDNAttention

    original = Mock()

    def call(self):
        original()

    monkeypatch.setattr(_GDNAttention, "_native_delta_rule", call)
    factory = importlib.import_module(f"tools.qwen4exp.resident_candidates.{candidate}").replacements
    resources = {resource: {key: object()}}
    first = next(iter(factory(resources).values()))
    monkeypatch.setattr(_GDNAttention, "_native_delta_rule", first)
    second = next(iter(factory(resources).values()))
    second(SimpleNamespace())
    original.assert_called_once_with()
    assert second._qwen_delta_rule_base is call


def test_transfer_manifest_is_host_only_and_rejects_changed_frozen_sources(tmp_path):
    import json

    from tools.glm_perf.resident_native import file_digest
    from tools.qwen4exp.prepare_native_transfer import make_manifest

    binaries = {}
    for name in ("native_state_layout", "native_cached_metadata"):
        binary = tmp_path / f"{name}.bin"
        binary.write_bytes(name.encode())
        binaries[name] = {"path": str(binary), "sha256": file_digest(binary)}
    source = tmp_path / "native_int4_schedule.h"
    source.write_text("// frozen fixture\n")
    library = tmp_path / "bridge.so"
    library.write_bytes(b"fixture")
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "namespace": "qwen_transfer_v1",
                "sources": {source.name: file_digest(source)},
                "bridge": {"path": str(library), "sha256": file_digest(library)},
                "binaries": binaries,
            }
        )
    )
    value = make_manifest(tmp_path, ROOT)
    assert value["operators"] == ["qwen_transfer_v1::launch"]
    assert "load_library=False" in value["validation_source"]
    assert any(entry["path"].endswith("qwen4_exp/model.py") for entry in value["assets"])
    source.write_text("// changed\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        make_manifest(tmp_path, ROOT)


def test_native_metadata_resource_preserves_config_and_rejects_bad_metadata(monkeypatch):
    from tools.qwen4exp.native_cached_metadata import NativeCachedMetadata

    monkeypatch.setattr(native_prefill, "_on_npu", lambda device: True)
    monkeypatch.setattr(native_prefill, "_capturing", lambda device: False)
    bank = SimpleNamespace(
        weight=torch.zeros(128, 64, 64, dtype=torch.int8),
        weight_scale=torch.ones(128, 64, 1, dtype=torch.float16),
        weight_offset=torch.zeros(128, 64, 1, dtype=torch.float16),
        weight_sum=torch.zeros(128, 64, 1, dtype=torch.float16),
    )
    prepared = (
        torch.zeros(129, 64, dtype=torch.int8),
        torch.zeros(129, 64, dtype=torch.int8),
        torch.ones(129, 1, 8),
        torch.zeros(129, 1, 8),
    )
    ends = torch.full((128,), 100, dtype=torch.int64)
    launch = Mock()
    native = NativeCachedMetadata(object(), launch)
    output = native(bank, prepared, ends)
    assert output.shape == (129, 64) and output.dtype == torch.float16
    config = launch.call_args.args[1][-1]
    assert config.tolist() == [129, 128, 64, 128, 8, 0, 1]
    assert launch.call_args.args[2] == 8
    bank.weight_offset = bank.weight_offset.float()
    with pytest.raises(ValueError, match="geometry/dtype"):
        native(bank, prepared, ends)
    assert launch.call_count == 1
