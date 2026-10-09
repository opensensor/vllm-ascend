# SPDX-License-Identifier: Apache-2.0
"""Complete CPU callback composition; numerical fixtures are not NPU gates."""

import builtins
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp import streaming_candidate as candidate_module
from tools.qwen4exp.resident_candidates import streaming as resident_factory
from tools.qwen4exp.streaming_candidate import (
    MOE_TARGET,
    CandidateConfig,
    CandidateResources,
    StreamingCandidate,
)
from tools.qwen4exp.streaming_epilogue import WindowPlan
from tools.qwen4exp.streaming_layer import LayerPolicy, LayerResources
from tools.qwen4exp.streaming_schedule import SchedulePlan, SchedulePolicy, streaming_prefill
from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch


def rank_receipts(plan):
    return [
        {
            "rank": rank,
            "pid": rank + 100,
            "generation": plan.generation,
            "execution_namespace": "cpu_composition_fixture",
            "plan_sha256": plan.sha256,
        }
        for rank in range(4)
    ]


def quantize_numpy(value):
    groups = np.asarray(value, dtype=np.float32).reshape(value.shape[0], -1, 128)
    maximum = np.max(np.abs(groups), axis=-1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(np.float32)
    quantized = np.clip(np.rint(groups / scales[..., None]), -127, 127).astype(np.int32)
    return quantized.reshape(value.shape), np.repeat(scales, 128, axis=1)


def packed_tensor(value):
    groups = value.float().reshape(value.shape[0], -1, 128)
    maximum = groups.abs().amax(-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    quantized = (groups / scales.unsqueeze(-1)).round().clamp(-127, 127)
    high = torch.floor(quantized / 16)
    low = quantized - 16 * high - 8

    def pack(codes):
        unsigned = torch.where(codes < 0, codes + 16, codes)
        return (unsigned[..., ::2] + 16 * unsigned[..., 1::2]).to(torch.int8).reshape(value.shape[0], -1).contiguous()

    return (
        pack(low),
        pack(high),
        scales.unsqueeze(-1).expand(-1, -1, 8).contiguous(),
        quantized.sum(-1).unsqueeze(-1).expand(-1, -1, 8).contiguous(),
    )


def decoded_tensor(prepared):
    def codes(packed):
        unsigned = packed.int() & 255
        values = torch.stack((unsigned & 15, unsigned >> 4), -1).flatten(-2)
        return torch.where(values >= 8, values - 16, values)

    quantized = codes(prepared[0]) + 16 * codes(prepared[1]) + 8
    return quantized.float() * prepared[2][:, :, 0].repeat_interleave(128, dim=1)


class ProjectionFixture:
    def __init__(self, log, fail=None):
        self.log, self.fail = log, fail

    def __call__(self, bank, packed, ends):
        self.log.append(("gate_up", packed[0].shape[0]))
        if self.fail == "projection":
            raise RuntimeError("projection failed")
        live = int(ends[-1])  # CPU callback only; production uses device ends.
        output = torch.zeros((packed[0].shape[0], 1280), dtype=torch.float16)
        if live:
            factors = torch.repeat_interleave(
                torch.arange(1, 129), ends.diff(prepend=torch.zeros(1, dtype=torch.int64))
            ).float()
            selected = decoded_tensor(tuple(v[:live] for v in packed))[:, torch.arange(1280) * 2]
            output[:live] = (selected * 0.03 + factors[:, None] * 0.01).half()
        return output

    def columns(self, bank, packed, ends, first, count):
        self.log.append(("columns", first, count))
        if self.fail == "columns":
            raise RuntimeError("window failed")
        live = int(ends[-1])
        output = torch.zeros((packed[0].shape[0], count * 128), dtype=torch.float16)
        if live:
            selected = decoded_tensor(tuple(v[:live] for v in packed))[
                :, torch.arange(first * 128, (first + count) * 128) % 640
            ]
            factors = torch.repeat_interleave(
                torch.arange(1, 129), ends.diff(prepend=torch.zeros(1, dtype=torch.int64))
            ).float()
            output[:live] = (selected * 0.02 + factors[:, None] * 0.001).half()
        return output


def independent_local(inputs, weights, ids, expert_offset=0):
    """Per-token/per-route reference independent of dispatch/gather/window code."""
    values, scales = quantize_numpy(inputs.numpy())
    dequant = values.astype(np.float32) * scales
    result = np.zeros((inputs.shape[0], 2560), np.float32)
    columns = np.arange(2560) % 640
    for token in range(inputs.shape[0]):
        accumulator = np.zeros(2560, np.float32)
        for route in range(ids.shape[1]):
            expert = int(ids[token, route]) - expert_offset
            if not 0 <= expert < 128:
                continue
            factor = np.float32(expert + 1)
            projected = (dequant[token, np.arange(1280) * 2] * np.float32(0.03) + factor * np.float32(0.01)).astype(
                np.float16
            )
            gate, up = projected[:640], projected[640:]
            # Match the fixture's explicit builtin FP16 activation boundary.
            activation = (F.silu(torch.from_numpy(gate)) * torch.from_numpy(up)).numpy()
            hidden_quant, hidden_scale = quantize_numpy(activation[None, :])
            hidden = hidden_quant[0].astype(np.float32) * hidden_scale[0]
            routed = (hidden[columns] * np.float32(0.02) + factor * np.float32(0.001)).astype(np.float16)
            accumulator += routed.astype(np.float32) * np.float32(np.float16(weights[token, route]))
        result[token] = accumulator.astype(np.float16).astype(np.float32)
    return torch.from_numpy(result)


def fixture(tokens=5, top_k=3, *, all_peer=False, chunk=2560, windows=8, placement="tp_sharded", fail=None):
    inputs = (torch.randn(tokens, 2560, generator=torch.Generator().manual_seed(27)) * 0.3).half()
    ids = (torch.arange(tokens * top_k).reshape(tokens, top_k) * 17) % 140
    if all_peer:
        ids.fill_(128)
    weights = (torch.arange(1, top_k + 1).float() / (top_k * (top_k + 1) / 2)).repeat(tokens, 1)
    log = []
    plan = SchedulePlan(SchedulePolicy(chunk_tokens=chunk, shared_policy=placement), 4, "1" * 64, "2" * 64, "3" * 64)
    config = CandidateConfig(
        plan,
        WindowPlan(tiles_per_window=windows),
        LayerPolicy(native_state_io=False, native_wy=False, ple_staging_tokens=0),
    )
    module = SimpleNamespace(
        native_int4=True,
        grouped_routing=True,
        device_routing=True,
        max_routed_rows=30,
        top_k=top_k,
        grouped_activation="cann_builtin_fp16",
        grouped_finalize="cann_v2",
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        expert_tp_size=4,
        num_local_experts=128,
        expert_offset=0,
        grouped_route_count_mode="compare",
        has_shared_expert=placement != "none",
        shared_expert_replicated=placement == "replicated",
        projections={
            "gate_up_proj": SimpleNamespace(weight=SimpleNamespace(shape=(128, 1280, 1280))),
            "down_proj": SimpleNamespace(weight=SimpleNamespace(shape=(128, 2560, 320))),
        },
    )
    shared = lambda chunk: chunk.float() * 0.0125
    module._forward_shared = shared

    def route(_module, tensor):
        assert tensor is inputs
        log.append(("route", len(tensor)))
        return weights, ids

    def pack(tensor):
        log.append(("pack", tensor.shape[0], tensor.shape[1]))
        if fail == "pack":
            raise RuntimeError("pack failed")
        return packed_tensor(tensor)

    def gather(prepared, sorted_tokens, ends):
        live = int(ends[-1])
        log.append(("gather", live, sorted_tokens.numel()))
        result = [torch.full((sorted_tokens.numel(), *value.shape[1:]), 37, dtype=value.dtype) for value in prepared]
        for out, value in zip(result, prepared):
            out[:live] = value.index_select(0, sorted_tokens[:live].long())
        return tuple(result)

    def activation(projected):
        log.append(("activation", len(projected)))
        gate, up = projected.chunk(2, dim=1)
        return (F.silu(gate) * up).contiguous()

    def finalize(routed, dispatch, same_weights, dtype, policy):
        assert dtype == torch.float32 and policy == "cann_v2"
        log.append(("finalize", routed.shape[1]))
        ordered = routed.index_select(0, dispatch.inverse_order).reshape(same_weights.shape[0], top_k, -1)
        result = torch.zeros((same_weights.shape[0], routed.shape[1]))
        for slot in range(top_k):
            result += ordered[:, slot].float() * same_weights[:, slot, None].half().float()
        return result.half().float()

    def schedule(_module, tensor, agreed, receipts, *, route, local, context):
        assert _module is module and tensor is inputs

        def submit(value, slot):
            log.append(("reduce", slot, value.shape[0]))
            if fail == "reduce":
                raise RuntimeError("collective unknown")
            # Four independent identical CPU rank contributions for the fixture.
            reduced = value.clone()
            for _ in range(3):
                reduced += value
            return reduced, object()

        return streaming_prefill(
            tensor,
            agreed,
            receipts,
            route=route,
            local=local,
            shared=shared if placement != "none" else None,
            submit_reduce=submit,
            wait=lambda event: log.append(("wait",)),
            store=lambda dst, src: dst.copy_(src),
            allocate_output=lambda x: torch.empty(x.shape, dtype=torch.float32),
            allocate_slot=lambda x, count: torch.empty((count, x.shape[1]), dtype=torch.float32),
        )

    resources = CandidateResources(
        "4" * 64,
        ProjectionFixture(log, fail),
        gather,
        pack,
        activation,
        finalize,
        route,
        build_grouped_expert_dispatch,
        schedule,
        None,
        LayerResources(),
    )
    candidate = StreamingCandidate(config, resources, rank_receipts(plan))
    return candidate, module, inputs, weights, ids, log


@pytest.mark.parametrize("tokens,top_k,all_peer,windows", [(1, 2, False, 8), (5, 3, False, 3), (7, 2, True, 8)])
def test_complete_local_pipeline_matches_independent_per_route_reference(tokens, top_k, all_peer, windows):
    candidate, module, inputs, weights, ids, log = fixture(tokens, top_k, all_peer=all_peer, windows=windows)
    output = candidate.local(module, inputs, weights, ids)
    torch.testing.assert_close(output, independent_local(inputs, weights, ids), rtol=0, atol=0)
    assert [entry for entry in log if entry[:1] == ("pack",) and entry[2] == 2560] == [("pack", tokens, 2560)]
    assert [entry for entry in log if entry[:1] == ("pack",) and entry[2] == 640] == [("pack", tokens * top_k, 640)]
    columns = [entry for entry in log if entry[0] == "columns"]
    assert sum(entry[2] * 128 for entry in columns) == 2560
    assert output.shape == (tokens, 2560)
    if all_peer:
        assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("placement", ["tp_sharded", "replicated", "none"])
def test_full_candidate_route_once_and_complete_shared_rank_schedule(placement):
    candidate, module, inputs, weights, ids, log = fixture(129, 2, chunk=128, placement=placement)
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    output = candidate.forward(fallback, module, inputs, capturing=False)
    routed = independent_local(inputs, weights, ids)
    shared = module._forward_shared(inputs)
    expected = routed + shared if placement == "tp_sharded" else routed
    reduced = expected.clone()
    for _ in range(3):
        reduced += expected
    if placement == "replicated":
        reduced += shared
    torch.testing.assert_close(output, reduced.half(), rtol=0, atol=0)
    assert [entry for entry in log if entry[0] == "route"] == [("route", 129)]
    assert [entry[1] for entry in log if entry[0] == "pack" and entry[2] == 2560] == [128, 1]
    assert [entry[2] for entry in log if entry[0] == "reduce"] == [128, 1]
    assert not candidate.poisoned


@pytest.mark.parametrize("placement", ["tp_sharded", "replicated", "none"])
def test_distinct_ep4_expert_shards_match_independent_complete_reference(placement):
    values = [fixture(129, 3, chunk=128, windows=3, placement=placement) for _ in range(4)]
    shared_by_rank, local_by_rank = [], []
    global_ids = (torch.arange(129 * 3).reshape(129, 3) * 37) % 512
    for rank, (candidate, module, inputs, weights, ids, log) in enumerate(values):
        module.expert_offset = rank * 128
        ids.copy_(global_ids)
        shared_scale = 0.025 if placement == "replicated" else (rank + 1) * 0.0125
        shared_by_rank.append(inputs.float() * shared_scale)
        local_by_rank.append(independent_local(inputs, weights, ids, module.expert_offset))
    contributions = [
        local + shared if placement == "tp_sharded" else local for local, shared in zip(local_by_rank, shared_by_rank)
    ]
    reference = contributions[0].clone()
    for rank in range(1, 4):
        reference += contributions[rank]
    if placement == "replicated":
        reference += shared_by_rank[0]

    for rank, (candidate, module, inputs, weights, ids, log) in enumerate(values):
        submissions = []
        peer_inputs = contributions

        def schedule(
            _module,
            tensor,
            agreed,
            receipts,
            *,
            route,
            local,
            context,
            _rank=rank,
            _shared=shared_by_rank[rank],
            _peers=peer_inputs,
            _submissions=submissions,
        ):
            chunk_index = 0
            shared_index = 0

            def shared(chunk):
                nonlocal shared_index
                start, stop = agreed.chunks(len(tensor))[shared_index]
                shared_index += 1
                return _shared[start:stop]

            def submit(local_value, slot):
                nonlocal chunk_index
                start, stop = agreed.chunks(len(tensor))[chunk_index]
                chunk_index += 1
                torch.testing.assert_close(local_value, _peers[_rank][start:stop], rtol=0, atol=0)
                result = _peers[0][start:stop].clone()
                for peer in range(1, 4):
                    result += _peers[peer][start:stop]
                _submissions.append((start, stop, slot))
                return result, object()

            return streaming_prefill(
                tensor,
                agreed,
                receipts,
                route=route,
                local=local,
                shared=shared if placement != "none" else None,
                submit_reduce=submit,
                wait=lambda event: None,
                store=lambda dst, src: dst.copy_(src),
                allocate_output=lambda x: torch.empty(x.shape, dtype=torch.float32),
                allocate_slot=lambda x, count: torch.empty((count, x.shape[1]), dtype=torch.float32),
            )

        composed = StreamingCandidate(
            candidate.config, replace(candidate.resources, schedule=schedule), candidate.rank_receipts
        )
        output = composed.forward(
            Mock(side_effect=AssertionError("unexpected fallback")), module, inputs, capturing=False
        )
        torch.testing.assert_close(output, reference.half(), rtol=0, atol=0)
        assert submissions == [(0, 128, 0), (128, 129, 1)]
        assert [entry for entry in log if entry[0] == "route"] == [("route", 129)]


@pytest.mark.parametrize(
    "reason",
    [
        "capture",
        "sparse",
        "w8",
        "width",
        "activation",
        "finalizer",
        "tp",
        "gate_shape",
        "down_shape",
        "experts",
        "shared_policy",
        "input_dtype",
        "params_dtype",
    ],
)
def test_unsupported_dispatch_falls_back_without_touching_candidate_resources(reason):
    candidate, module, inputs, weights, ids, log = fixture(16, 3)
    capturing = False
    if reason == "capture":
        capturing = True
    elif reason == "sparse":
        module.max_routed_rows = 100
    elif reason == "w8":
        module.native_int4 = False
    elif reason == "width":
        inputs = inputs[:, :2559]
    elif reason == "activation":
        module.grouped_activation = "torch"
    elif reason == "finalizer":
        module.grouped_finalize = "torch"
    elif reason == "tp":
        module.expert_tp_size = 2
    elif reason == "gate_shape":
        module.projections["gate_up_proj"].weight.shape = (128, 1280, 640)
    elif reason == "down_shape":
        module.projections["down_proj"].weight.shape = (128, 1280, 320)
    elif reason == "experts":
        module.num_local_experts = 64
    elif reason == "shared_policy":
        module.shared_expert_replicated = True
    elif reason == "input_dtype":
        inputs = inputs.float()
    else:
        module.params_dtype = torch.bfloat16
    expected = object()
    original = Mock(return_value=expected)
    assert candidate.forward(original, module, inputs, capturing=capturing) is expected
    original.assert_called_once_with(module, inputs)
    assert log == [] and not candidate.poisoned


@pytest.mark.parametrize("failure", ["pack", "projection", "columns", "reduce"])
def test_partial_failure_poisoned_and_never_falls_back(failure):
    candidate, module, inputs, *rest = fixture(16, 3, fail=failure)
    fallback = Mock()
    with pytest.raises(RuntimeError):
        candidate.forward(fallback, module, inputs, capturing=False)
    assert candidate.poisoned and not fallback.called
    with pytest.raises(RuntimeError, match="poisoned"):
        candidate.forward(fallback, module, inputs, capturing=True)


class FakeAdmission:
    def __init__(self, candidate):
        self.configuration = candidate.config.sha256
        self.resources = candidate.resources.inventory_sha256
        self.plan = candidate.config.schedule.sha256
        self.rank_plan_receipts = candidate.rank_receipts
        self.calls = 0

    @property
    def plan_receipts(self):
        return self.rank_plan_receipts

    def require_execution(self, *, configuration_sha256, resources_sha256, plan_sha256):
        self.calls += 1
        if (configuration_sha256, resources_sha256, plan_sha256) != (self.configuration, self.resources, self.plan):
            raise ValueError("fake admission identity mismatch")


def originals():
    original = lambda *a, **k: None
    return {
        "moe_original": original,
        "capturing": lambda inputs: False,
        "gdn_original": original,
        "prefix_update_original": original,
        "prefix_remap_original": original,
        "ple_original": original,
    }


@pytest.mark.parametrize("change", ["missing", "configuration", "resource", "plan", "rank"])
def test_installation_admission_binds_every_policy_resource_and_rank(change):
    candidate, *rest = fixture()
    admission = FakeAdmission(candidate)
    if change != "missing":
        candidate.admission = admission
    if change == "configuration":
        admission.configuration = "0" * 64
    elif change == "resource":
        admission.resources = "0" * 64
    elif change == "plan":
        admission.plan = "0" * 64
    elif change == "rank":
        admission.rank_plan_receipts = tuple(dict(r, pid=r["pid"] + 1) for r in admission.rank_plan_receipts)
    with pytest.raises(ValueError):
        candidate.replacements(**originals())


def test_composition_map_one_owner_and_no_stacking():
    candidate, *rest = fixture()
    candidate.admission = FakeAdmission(candidate)
    hooks = candidate.replacements(**originals())
    assert len(hooks) == 5 and hooks[MOE_TARGET]._qwen_streaming_owner is candidate
    with pytest.raises(ValueError, match="original"):
        candidate.replacements(**dict(originals(), moe_original=hooks[MOE_TARGET]))
    with pytest.raises(ValueError, match="stacked"):
        candidate.replacements(
            **dict(originals(), gdn_original=next(v for k, v in hooks.items() if "_native_delta_rule" in k))
        )


@pytest.mark.parametrize(
    "target_suffix", ["_native_delta_rule", "_update_states", "_remap_compact_mamba_block_tables", "_forward_eager"]
)
def test_layer_hook_partial_failure_poisons_entire_candidate_owner(target_suffix):
    candidate, module, inputs, *rest = fixture()
    candidate.admission = FakeAdmission(candidate)

    def fail(instance, *args, **kwargs):
        raise RuntimeError("layer submission failed")

    bindings = originals()
    for key in ("gdn_original", "prefix_update_original", "prefix_remap_original", "ple_original"):
        bindings[key] = fail
    hooks = candidate.replacements(**bindings)
    target = next(hook for name, hook in hooks.items() if name.endswith(target_suffix))
    with pytest.raises(RuntimeError, match="layer submission"):
        target(SimpleNamespace())
    assert candidate.poisoned
    fallback = Mock()
    with pytest.raises(RuntimeError, match="poisoned"):
        candidate.forward(fallback, module, inputs, capturing=True)
    assert not fallback.called


def test_resident_factory_rejects_admission_before_runtime_imports(monkeypatch):
    candidate, *rest = fixture()
    imported = []
    normal = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("torch", "torch_npu") or name.startswith("vllm_ascend"):
            imported.append(name)
            raise AssertionError("runtime imported before admission")
        return normal(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="pending"):
        resident_factory.replacements({"fixture": {"candidate": candidate}})
    assert imported == []


def test_native_loader_rejects_before_device_library_or_kernel_loading(monkeypatch, tmp_path):
    from tools.qwen4exp import build_streaming

    candidate, *rest = fixture()
    monkeypatch.setattr(build_streaming, "verify_bundle", lambda *a, **k: {"resources_sha256": "0" * 64})
    admission = FakeAdmission(candidate)
    forbidden = Mock(side_effect=AssertionError("native library loaded"))
    monkeypatch.setattr(torch.ops, "load_library", forbidden)
    normal = builtins.__import__
    imports = []

    def guarded(name, *args, **kwargs):
        if name == "torch_npu":
            imports.append(name)
            raise AssertionError("device runtime loaded")
        return normal(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="identity mismatch"):
        candidate_module.prepare_native_resources(tmp_path, candidate.config, admission)
    assert not forbidden.called and not imports


def test_invalid_resource_and_partial_rank_identity_rejected():
    candidate, *rest = fixture()
    with pytest.raises(ValueError, match="incomplete"):
        replace(candidate.resources, gather=None)
    with pytest.raises(ValueError, match="ranks"):
        StreamingCandidate(candidate.config, candidate.resources, candidate.rank_receipts[:3])
