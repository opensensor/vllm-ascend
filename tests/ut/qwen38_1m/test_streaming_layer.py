# SPDX-License-Identifier: Apache-2.0
"""Actual Qwen method seams and offline layer ownership failure cases."""

import ast
import sys
from copy import copy
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tools.qwen4exp.streaming_layer import (
    GDN_TARGET,
    PLE_TARGET,
    CacheOwner,
    LayerComposition,
    LayerPolicy,
    LayerResources,
    StepOwnership,
    accepted_host_counts,
    consume_qsa_selection,
    validate_qsa_cache_owners,
)
from vllm_ascend._310p.host_staging import PinnedHostStaging
from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier, apply_prefix_mamba_updates

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend/models/qwen4_exp/model.py"
RUNNER = ROOT / "vllm_ascend/_310p/model_runner_310p.py"


def extract(path, cls, name, scope):
    node = next(c for c in ast.parse(path.read_text()).body if isinstance(c, ast.ClassDef) and c.name == cls)
    method = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def composition():
    state = SimpleNamespace(gather=Mock(), scatter=Mock())
    return LayerComposition(LayerPolicy(), LayerResources(state, Mock()))


@pytest.mark.parametrize("state_native,wy_native", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("warm", [False, True])
def test_actual_gdn_prefill_uses_composed_resources_and_restores(monkeypatch, state_native, wy_native, warm):
    chunk_module = ModuleType("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
    helper_module = ModuleType("vllm_ascend._310p.ops.fla.gdn_310")
    observed = []

    def chunk(**kwargs):
        observed.append(kwargs)
        if kwargs["wy_prepare"]:
            kwargs["wy_prepare"]()
        return kwargs["v"], kwargs["initial_state"] + 1

    chunk_module.chunk_gated_delta_rule_310 = chunk
    helper_module._cached_chunk_plan = lambda *args: "plan"
    helper_module._cached_recurrent_step_meta = lambda *args, **kwargs: None
    helper_module.npu_recurrent_gated_delta_rule_310 = Mock(return_value=torch.zeros(1, 4, 1, 128))
    monkeypatch.setitem(sys.modules, chunk_module.__name__, chunk_module)
    monkeypatch.setitem(sys.modules, helper_module.__name__, helper_module)
    original = extract(MODEL, "_GDNAttention", "_native_delta_rule", {"torch": torch, "GDNAttentionMetadata": object})
    cache = torch.randn(4, 1, 128, 128)
    slots = torch.tensor([1, 2], dtype=torch.int32)
    valid = torch.tensor([warm, True])
    expected = cache[slots.long()].clone()
    expected[~valid] = 0
    expected += 1
    native = SimpleNamespace(
        gather=Mock(
            side_effect=lambda c, s, valid: (
                torch.where(valid[:, None, None, None], c[s.long()], 0).transpose(-1, -2).contiguous()
            )
        ),
        scatter=Mock(side_effect=lambda c, s, valid, final: c.__setitem__(s.long(), final.transpose(-1, -2))),
    )
    wy = Mock()
    policy = LayerPolicy(native_state_io=state_native, native_wy=wy_native)
    candidate = LayerComposition(policy, LayerResources(native, wy))
    layer = SimpleNamespace(kv_cache=(None, cache), _gdn_state_io=None)
    q = torch.zeros(4, 1, 128)
    meta = SimpleNamespace(spec_sequence_masks=None, num_prefills=2)
    args = (q, q, q, torch.zeros(1, 4, 1), torch.ones(1, 4, 1), meta, slots, torch.tensor([0, 2, 4]), valid)
    candidate.gdn_call(original, layer, *args)
    assert observed[-1]["state_is_kernel_layout"] is state_native
    assert torch.equal(cache[slots.long()], expected)
    assert native.gather.call_count == int(state_native)
    assert wy.call_count == int(wy_native)
    assert layer._gdn_state_io is None and not hasattr(layer, "_gdn_wy_prepare")
    meta.num_prefills = 0
    candidate.gdn_call(original, layer, *args)
    assert native.gather.call_count == int(state_native)
    assert wy.call_count == int(wy_native)
    meta.spec_sequence_masks = torch.ones(1, dtype=torch.bool)
    meta.spec_decode_metadata = SimpleNamespace(spec_causal_conv1d=SimpleNamespace(num_accepted_tokens="accepted"))
    candidate.gdn_call(original, layer, *args)
    assert helper_module.npu_recurrent_gated_delta_rule_310.call_args.kwargs["num_accepted_tokens"] == "accepted"
    assert native.gather.call_count == int(state_native)


def test_atomic_resources_restore_on_error_and_reject_competing_owner():
    candidate = composition()
    layer = SimpleNamespace(_gdn_state_io=None)

    def fail(instance):
        assert instance._gdn_state_io is candidate.resources.state_io
        assert instance._gdn_wy_prepare is candidate.resources.wy
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        candidate.gdn_call(fail, layer)
    assert layer._gdn_state_io is None and not hasattr(layer, "_gdn_wy_prepare")
    layer._gdn_wy_prepare = object()
    with pytest.raises(ValueError, match="competing"):
        candidate.gdn_call(fail, layer)
    assert layer._gdn_state_io is None
    with pytest.raises(FrozenInstanceError):
        candidate.policy.ple_staging_tokens = 10


def test_single_composition_map_refuses_stacked_factories():
    candidate = composition()
    original = lambda *a, **k: "ok"
    methods = candidate.replacements(
        gdn_original=original, prefix_update_original=original, prefix_remap_original=original, ple_original=original
    )
    assert GDN_TARGET in methods and PLE_TARGET in methods and len(methods) == 4
    with pytest.raises(ValueError, match="stacked"):
        candidate.replacements(
            gdn_original=methods[GDN_TARGET],
            prefix_update_original=original,
            prefix_remap_original=original,
            ple_original=original,
        )


def test_prefix_cow_uses_fresh_phase_drain_and_restores_flag(monkeypatch):
    candidate = composition()
    tiers, states, barriers = {}, {}, []
    for group in range(3):
        state = torch.zeros(3, 2)
        tier = PrefixMambaStateTier([(state,)], 3)
        tier.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
        state[tier.slot_for(101)].fill_(7)
        monkeypatch.setattr(tier, "_synchronize_device_state", lambda: barriers.append("completed"))
        tiers[group], states[group] = tier, state
    runner = SimpleNamespace()

    def update(instance):
        apply_prefix_mamba_updates(
            tiers, {g: [] for g in tiers}, {g: [(101, 102)] for g in tiers}, batched=instance._prefix_phase_batching
        )

    candidate.prefix_call(update, runner)
    assert barriers == ["completed"]
    assert not hasattr(runner, "_prefix_phase_batching")
    for group, tier in tiers.items():
        assert torch.equal(states[group][tier.slot_for(102)], torch.tensor([7.0, 7.0]))
    candidate.prefix_call(update, runner)
    assert len(barriers) == 2


def test_actual_mixed_attention_keeps_spec_and_prefill_state_ownership():
    original = extract(MODEL, "_GDNAttention", "_native_mixed_attention", {"torch": torch, "copy": copy})
    seen = []
    layer = SimpleNamespace(
        key_dim=1,
        value_dim=1,
        num_k_heads=1,
        num_v_heads=1,
        params=SimpleNamespace(head_k_dim=1, head_v_dim=1),
        compute_dtype=torch.float32,
        _native_gating=lambda a, b: (torch.ones(1, 4, 1), torch.ones(1, 4, 1)),
        _stateful_short_conv=lambda value, *a, **k: value,
    )

    def delta(q, k, v, g, beta, meta, slots, locs, valid):
        seen.append((meta, slots, valid))
        return v

    layer._native_delta_rule = delta
    meta = SimpleNamespace(
        spec_sequence_masks="spec",
        spec_decode_metadata="accepted",
        num_spec_decode_tokens=2,
        num_spec_decodes=1,
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        spec_token_indx=torch.tensor([0, 1]),
        non_spec_token_indx=torch.tensor([2, 3]),
        spec_state_indices_tensor=torch.tensor([1]),
        non_spec_state_indices_tensor=torch.tensor([2]),
        spec_query_start_loc=torch.tensor([0, 2]),
        non_spec_query_start_loc=torch.tensor([0, 2]),
        has_initial_state=torch.tensor([True]),
    )
    mixed = torch.arange(12).reshape(4, 3).float()
    result = original(layer, mixed, None, None, meta)
    assert torch.equal(result.flatten(), mixed[:, 2])
    assert seen[0][0].spec_decode_metadata == "accepted" and seen[0][2] is None
    assert seen[1][0].spec_sequence_masks is None and seen[1][0].spec_decode_metadata is None
    assert seen[1][2] is meta.has_initial_state
    assert seen[0][1] is not seen[1][1]


def test_qsa_layer_storage_identity_and_direct_selection():
    caches = [torch.empty(4) for _ in range(6)]
    owners = [CacheOwner("layer0", *caches[:3]), CacheOwner("layer1", *caches[3:])]
    assert validate_qsa_cache_owners(iter(owners)) == tuple(owners)
    owners[1] = CacheOwner("layer1", caches[0].view(2, 2), *caches[4:])
    with pytest.raises(ValueError, match="shared"):
        validate_qsa_cache_owners(owners)
    selection = object()
    assert consume_qsa_selection(selection, lambda *, selection: selection) is selection
    tree = ast.parse(MODEL.read_text())
    qsa = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_QSAAttention")
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "copy_group_selection_into"
        for n in ast.walk(qsa)
    )


def test_ple_preserves_actual_completion_guard_and_configuration():
    log = []
    event = SimpleNamespace(
        query=lambda: False, synchronize=lambda: log.append("wait_dma"), record=lambda: log.append("submitted")
    )
    stage = PinnedHostStaging((4, 2), torch.float32, pin_memory=False, event_factory=lambda: event)
    candidate = LayerComposition(
        LayerPolicy(ple_staging_tokens=2), LayerResources(SimpleNamespace(gather=Mock(), scatter=Mock()), Mock())
    )
    layer = SimpleNamespace(
        ple=SimpleNamespace(host_staging_tokens=0, _row_host_stage=stage, num_ngram_heads=2, per_head_dim=2)
    )

    def forward(instance):
        assert instance.ple.host_staging_tokens == 2
        return instance.ple._row_host_stage.copy_to(torch.ones(4, 2), torch.zeros(4, 2))

    candidate.ple_call(forward, layer)
    candidate.ple_call(forward, layer)
    assert log == ["submitted", "wait_dma", "submitted"]
    assert layer.ple.host_staging_tokens == 0 and layer.ple._row_host_stage is stage
    layer.ple.host_staging_tokens = 3
    with pytest.raises(ValueError, match="capacity"):
        candidate.ple_call(forward, layer)


def test_actual_runner_reuses_one_raw_cpu_snapshot_and_clears_on_commit():
    scope = {"replace": replace}
    original = extract(RUNNER, "NPUModelRunner310", "_bookkeeping_sync", scope)
    # Extracted super() methods require their original class closure; inspect
    # the actual call graph instead of substituting a fake runner implementation.
    node = ast.parse(RUNNER.read_text())
    runner = next(c for c in node.body if isinstance(c, ast.ClassDef) and c.name == "NPUModelRunner310")
    bookkeeping = next(m for m in runner.body if isinstance(m, ast.FunctionDef) and m.name == "_bookkeeping_sync")
    assert callable(original)
    assert (
        sum(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "cpu"
            for n in ast.walk(bookkeeping)
        )
        == 1
    )
    commit = next(
        m for m in runner.body if isinstance(m, ast.FunctionDef) and m.name == "_update_states_after_model_execute"
    )
    assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(commit))
    sampled, counts = object(), object()
    snapshot = (sampled, ("a", "b"), counts)
    assert accepted_host_counts(snapshot, sampled, ["a", "b"]) is counts
    assert accepted_host_counts(snapshot, sampled, ["b", "a"]) is None
    assert accepted_host_counts(snapshot, object(), ["a", "b"]) is None


def test_mtp_last_consumers_commit_and_cancellation():
    step = StepOwnership(3)
    target = step.acquire("target", "layer1", ["verification", "accepted_state"])
    draft = step.acquire("draft", "layer1", ["verification"])
    sampled, counts = object(), object()
    args = {"snapshot": (sampled, ("a",), counts), "sampled": sampled, "request_ids": ["a"]}
    with pytest.raises(ValueError, match="last consumers"):
        step.commit_accepted_state(**args)
    with pytest.raises(ValueError, match="submission"):
        step.complete(target, "verification", established_completion=False)
    step.complete(target, "verification", established_completion=True)
    step.complete(target, "accepted_state", established_completion=True)
    step.complete(draft, "verification", established_completion=True)
    assert step.commit_accepted_state(**args) is counts
    with pytest.raises(ValueError):
        step.commit_accepted_state(**args)
    cancelled = StepOwnership(4)
    stale = cancelled.acquire("state", "layer1", ["checkpoint"])
    with pytest.raises(ValueError, match="cancellation"):
        cancelled.cancel(established_completion=False)
    cancelled.cancel(established_completion=True)
    with pytest.raises(ValueError, match="stale"):
        cancelled.complete(stale, "checkpoint", established_completion=True)


@pytest.mark.parametrize("fail_commit", [False, True])
def test_actual_runner_snapshot_publication_uses_raw_counts_and_finally_cleanup(fail_commit):
    node = ast.parse(RUNNER.read_text())
    original = next(c for c in node.body if isinstance(c, ast.ClassDef) and c.name == "NPUModelRunner310")
    methods = [
        m
        for m in original.body
        if isinstance(m, ast.FunctionDef) and m.name in ("_bookkeeping_sync", "_update_states_after_model_execute")
    ]
    seen = []

    class Base:
        def _bookkeeping_sync(self, scheduler, sampler, *args, **kwargs):
            assert sampler.sampled_token_ids.device.type == "cpu"
            return sampler.sampled_token_ids

        def _update_states_after_model_execute(self, tokens, scheduler):
            seen.append(self.input_batch._mamba_accepted_counts_snapshot)
            if fail_commit:
                raise RuntimeError("commit failed")

    cls = ast.ClassDef(
        name="NPUModelRunner310",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    scope = {"Base": Base, "replace": replace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(RUNNER), "exec"), scope)
    runner = scope["NPUModelRunner310"]()
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner._qwen4exp_mtp_ple, runner.use_async_scheduling, runner.need_accepted_tokens = True, False, True
    calls = []
    host = torch.tensor([[2, 3, -1], [4, -1, -1]])
    sampled = SimpleNamespace(cpu=lambda: calls.append("cpu") or host, device=SimpleNamespace(type="npu"))

    @dataclass
    class Sampler:
        sampled_token_ids: object

    assert runner._bookkeeping_sync(None, Sampler(sampled)) is host
    try:
        runner._update_states_after_model_execute(sampled, None)
    except RuntimeError:
        assert fail_commit
    assert calls == ["cpu"]
    assert torch.equal(seen[0], torch.tensor([2, 1]))
    assert runner._mamba_sample_snapshot is None
    assert not hasattr(runner.input_batch, "_mamba_accepted_counts_snapshot")


def test_ple_replay_calls_eager_hook_and_error_keeps_live_stage():
    scope = {"torch": torch, "BreakableCUDAGraphCapture": SimpleNamespace(current=lambda: None)}
    forward = extract(MODEL, "_PLEInjection", "forward", scope)
    candidate = composition()
    created_stage = object()
    layer = SimpleNamespace(
        ple=SimpleNamespace(host_staging_tokens=0, _row_host_stage=None, num_ngram_heads=1, per_head_dim=2)
    )

    def eager(instance, *args, **kwargs):
        assert instance.ple.host_staging_tokens == 2560
        instance.ple._row_host_stage = created_stage
        raise RuntimeError("after submission")

    layer._forward_eager = lambda *args, **kwargs: candidate.ple_call(eager, layer, *args, **kwargs)
    with pytest.raises(RuntimeError, match="submission"):
        forward(layer, None, None)
    assert layer.ple.host_staging_tokens == 0 and layer.ple._row_host_stage is created_stage
    assert PLE_TARGET.endswith(":_PLEInjection._forward_eager")


@pytest.mark.parametrize(
    "kwargs", [{"native_state_io": 1}, {"native_wy": None}, {"ple_staging_tokens": True}, {"ple_staging_tokens": 65537}]
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LayerPolicy(**kwargs)


def test_step_bounds_foreign_leases_and_no_reuse():
    step = StepOwnership(1, max_leases=1)
    first = step.acquire("state", "layer1", ["GDN", "checkpoint"])
    with pytest.raises(ValueError, match="capacity"):
        step.acquire("next", "layer2", ["GDN"])
    foreign = StepOwnership(1).acquire("state", "layer1", ["GDN", "checkpoint"])
    with pytest.raises(ValueError, match="foreign"):
        step.complete(foreign, "GDN", established_completion=True)
    step.complete(first, "GDN", established_completion=True)
    with pytest.raises(ValueError, match="already"):
        step.complete(first, "GDN", established_completion=True)
