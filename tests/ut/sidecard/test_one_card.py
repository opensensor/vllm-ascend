# SPDX-License-Identifier: Apache-2.0
import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.ut.sidecard.test_native_expert_probe import native_fixture
from tools.glm_perf.deployment_profiles import one_card_sidecard_profile, validate_profile
from tools.sidecard.one_card import (
    npu_range,
    prepare_banks,
    stage_checkpoint,
    validate_device_pair,
    validate_packed_gate,
)
from tools.sidecard.service_adapter import local_routes, one_card_routed, service_placement
from tools.sidecard.service_prepare import prepare
from tools.sidecard.service_validate import check_metrics


def test_one_card_placement_covers_all_experts_with_two_npu_ranks():
    placement = service_placement(one_card=True)
    assert placement.owners == ("cuda:0", "npu:0", "npu:1")
    assert npu_range(0) == (144, 216)
    assert npu_range(1) == (216, 288)
    for layer in placement.layers:
        assert layer == ("cuda:0",) * 144 + ("npu:0",) * 72 + ("npu:1",) * 72
    selected = one_card_sidecard_profile()
    assert selected.chips == 2 and selected.expert_partition_size == 4
    assert validate_device_pair((0, 1)) == (0, 1)
    assert validate_device_pair((2, 3)) == (2, 3)
    assert validate_profile(selected.name, tensor_parallel_size=2, pipeline_parallel_size=1, mtp_tokens=0) == selected
    with pytest.raises(ValueError, match="topology"):
        validate_profile(selected.name, tensor_parallel_size=4, pipeline_parallel_size=1, mtp_tokens=0)


@pytest.mark.parametrize("rank", [2, 3, True, -1])
def test_released_or_invalid_rank_is_rejected(rank):
    with pytest.raises(ValueError):
        npu_range(rank)


def test_new_cuda_boundary_accepts_143_and_rejects_144():
    routes, _ = local_routes([[143] + [0] * 7], [[1.0] + [0.0] * 7], cuda_experts=144)
    assert routes[0].expert == 143
    with pytest.raises(ValueError, match="route"):
        local_routes([[144] + [0] * 7], [[1.0] + [0.0] * 7], cuda_experts=144)


def model_fixture(rank):
    owners = []
    for layer in range(3, 45):
        bank = SimpleNamespace(
            local_expert_offset=rank * 144, num_local_experts=144, offload_to_cpu=False, grouped_ready=False
        )
        owners.append((f"model.layers.{layer}.mlp_w2", SimpleNamespace(w2_experts=bank)))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
        speculative_config=None,
        model_config=SimpleNamespace(enforce_eager=True),
    )
    return SimpleNamespace(vllm_config=config, named_modules=lambda: iter(owners)), owners


@pytest.mark.parametrize("rank", [0, 1])
def test_bank_partition_is_set_before_loading(rank):
    model, owners = model_fixture(rank)
    prepare_banks(model, rank)
    assert all(
        (owner.w2_experts.local_expert_offset, owner.w2_experts.num_local_experts) == (npu_range(rank)[0], 72)
        for _, owner in owners
    )
    owners[-1][1].w2_experts.grouped_ready = True
    with pytest.raises(ValueError, match="allocations"):
        prepare_banks(model, rank)


def test_one_card_routing_preserves_slots_weights_and_sum_across_both_ranks(monkeypatch):
    distributed = ModuleType("vllm.distributed")
    distributed.tensor_model_parallel_all_reduce = lambda x: x
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    ids = torch.tensor([[0, 71, 72, 143, 144, 215, 216, 287], [143, 144, 215, 216, 0, 72, 287, 71]])
    weights = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8) / 32
    calls = []

    class Method:
        def apply(self, layer, hidden, route_weights, route_ids, shared, shared_input):
            calls.append((route_ids.clone(), route_weights.clone()))
            return (route_ids.float() * route_weights).sum(1, keepdim=True).expand_as(hidden).clone()

    hidden = torch.zeros(2, 4)
    parts = []
    for rank in (0, 1):
        bank = SimpleNamespace(local_expert_offset=npu_range(rank)[0], num_local_experts=72)
        module = SimpleNamespace(w2_experts=bank)
        parts.append(
            one_card_routed(Method(), Method() if rank == 0 else None)(
                module, hidden, ids, weights, reduce_results=False
            )
        )
    torch.testing.assert_close(
        sum(parts), (ids.float() * weights).sum(1, keepdim=True).expand_as(hidden), rtol=0, atol=0
    )
    native0, cuda, native1 = calls
    assert torch.equal(native0[1], weights * ((ids >= 144) & (ids < 216)))
    assert torch.equal(cuda[1], weights * (ids < 144))
    assert torch.equal(native1[1], weights * (ids >= 216))


def test_streamed_loader_uses_admitted_bank_range_instead_of_default_tp_range(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next_w2/model.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_place_streamed_expert"
    )
    package = ModuleType("sidecard_bank_stub")
    package.__path__ = []
    mapping_module = ModuleType("sidecard_bank_stub.weight_mapping")
    mapping_module.map_expert_tensor = lambda name, geometry: SimpleNamespace(
        block="layers.3", expert_id=int(name), slot="w1", kind="codes"
    )
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, mapping_module.__name__, mapping_module)
    scope = {
        "__package__": package.__name__,
        "torch": torch,
        "_PackedW2ExpertBank": object,
        "_SLOT_KIND_TO_ATTR": {("w1", "codes"): "gate_packed"},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    loaded = []

    class Bank(SimpleNamespace):
        def __getitem__(self, key):
            return SimpleNamespace(offload_to_cpu=False)

    bank = Bank(
        local_expert_offset=144,
        num_local_experts=72,
        place_resident_tensor=lambda *args, **kwargs: loaded.append(args[0]),
    )
    for expert in (0, 143, 144, 215, 216, 287):
        scope["_place_streamed_expert"]({"layers.3": bank}, str(expert), torch.zeros(1), {})
    assert loaded == [144, 215]


def test_one_card_requires_packed_transport_before_touching_files(tmp_path):
    with pytest.raises(ValueError, match="packed"):
        prepare(*([tmp_path] * 7), one_card=True)


def test_checkpoint_staging_preserves_original_metadata_and_tensor_inodes(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    native_fixture(source)
    manifest_path = source / "native-layout.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.update(deployment_profile="two-card-selective-w2", world_size=4, num_experts=288)
    manifest_path.write_text(json.dumps(manifest))
    original = manifest_path.read_bytes()
    output = tmp_path / "staged"
    receipt = stage_checkpoint(source, output)
    assert manifest_path.read_bytes() == original
    assert source.joinpath("native.safetensors").stat().st_ino == output.joinpath("native.safetensors").stat().st_ino
    assert manifest_path.stat().st_ino != output.joinpath("native-layout.json").stat().st_ino
    assert (
        json.loads(output.joinpath("config.json").read_bytes())["ascend_glm_deployment_profile"]
        == "one-card-sidecard-w2"
    )
    assert receipt["npu_workers"] == 2
    with pytest.raises(ValueError, match="fresh"):
        stage_checkpoint(source, output)


def test_packed_gate_requires_all_real_weight_cases_and_pinned_identities(tmp_path):
    path = tmp_path / "gate.json"
    report = dict(
        passed=True,
        candidate_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        candidate_weight_layout="k32_nibbles_n_v1",
        cuda_experts_per_layer=144,
        candidate_partial_reduce=True,
        cases=[
            dict(layer=layer, tokens=tokens, routes=tokens * routes, bitwise_equal=True)
            for layer in (3, 8, 33)
            for tokens in (128, 512, 640)
            for routes in (1, 2, 8)
        ],
    )
    path.write_text(json.dumps(report))
    assert validate_packed_gate(path, "a" * 64, "b" * 64)["cases"] == 27
    with pytest.raises(ValueError, match="pinned"):
        validate_packed_gate(path, "c" * 64, "b" * 64)
    report["cases"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="pinned"):
        validate_packed_gate(path, "a" * 64, "b" * 64)


@pytest.mark.parametrize("devices", [(0, 2), (1, 2), (0, 1, 2, 3), (True, 1), [0, 1], ()])
def test_physical_selection_rejects_cross_card_and_multiple_card_allocations(devices):
    with pytest.raises(ValueError, match="physical NPU pair"):
        validate_device_pair(devices)


def test_overlap_validation_requires_decode_and_bulk_prefill_shadows_in_all_layers():
    report = dict(
        one_card=True,
        overlaps_native_cuda=True,
        retains_npu_shadow_banks=False,
        cuda_worker_pid=123,
        cuda_attention=False,
        profile="ascend-backbone-cuda-experts",
        packed_cuda_gate=dict(passed=True, cases=27, cuda_experts_per_layer=144),
        layers={
            str(layer): dict(
                real_calls=4,
                real_cuda_routes=10,
                shadow_checks=0,
                shadow_max_absolute_error=0,
                shadow_max_nrmse=0,
                overlap_decode_shadow_checks=3,
                overlap_prefill_shadow_checks=1,
            )
            for layer in range(3, 45)
        },
    )
    assert check_metrics(report)["serial_overlap_shadow_checks"] == 168
    report["layers"]["33"]["overlap_prefill_shadow_checks"] = 0
    with pytest.raises(AssertionError, match="decode/prefill"):
        check_metrics(report)
