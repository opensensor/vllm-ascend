# SPDX-License-Identifier: Apache-2.0
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.deployment_profiles import SELECTIVE_W3_LAYERS, W4_LAYERS
from tools.sidecard.device_backend import CUDA_COMPACT_PLANES
from tools.sidecard.one_card_partition import PRECISION_PARTITION, one_card_partition
from tools.sidecard.resident import required_cuda_bytes
from tools.sidecard.service_adapter import local_routes, one_card_routed, service_placement


def test_precision_placement_covers_all_experts_and_balances_remaining_w4_work():
    partition = one_card_partition(PRECISION_PARTITION)
    placement = service_placement(one_card=True, cuda_partition=partition.name)
    assert partition.resident_experts == 9632
    for layer in range(3, 45):
        owners = placement.layer(layer)
        assert len(owners) == 288
        if layer in W4_LAYERS:
            assert owners.count("cuda:0") == 176
            assert owners.count("npu:0") == owners.count("npu:1") == 56
            assert owners[159:161] == ("cuda:0", "npu:0")
            assert owners[215:217] == ("npu:0", "cuda:0")
            assert owners[231:233] == ("cuda:0", "npu:1")
        else:
            assert set(owners) == {"cuda:0"}
            assert not partition.native_enabled(layer)


def test_precision_capacity_uses_native_widths_and_the_largest_w3_staging_bank():
    partition = one_card_partition(PRECISION_PARTITION)
    bits = {str(layer): 4 if layer in W4_LAYERS else 3 if layer in SELECTIVE_W3_LAYERS else 2 for layer in range(3, 45)}
    plan = SimpleNamespace(
        by_layer={layer: partition.cuda_ids(layer) for layer in range(3, 45)}, manifest={"layer_bits": bits}
    )
    expected = (22 * 176 * 12 + 8 * 288 * 9 + 12 * 288 * 6 + 1728 + 256) * 1024**2 + 9632 * 49152
    assert required_cuda_bytes(plan, CUDA_COMPACT_PLANES) == expected
    partition.validate_layer_bits(bits)
    bits["8"] = 4
    with pytest.raises(ValueError, match="W4 layers"):
        partition.validate_layer_bits(bits)


@pytest.mark.parametrize("expert", (160, 180, 215, 232, 287))
def test_w4_cuda_boundary_rejects_native_owned_positive_routes(expert):
    allowed = one_card_partition(PRECISION_PARTITION).cuda_ids(3)
    with pytest.raises(ValueError, match="foreign"):
        local_routes([[expert] + [0] * 7], [[1.0] + [0.0] * 7], cuda_experts=len(allowed), allowed_experts=allowed)


def test_noncontiguous_cuda_boundary_preserves_global_ids():
    allowed = one_card_partition(PRECISION_PARTITION).cuda_ids(3)
    routes, _ = local_routes(
        [[159, 216, 231] + [0] * 5], [[0.25, 0.25, 0.5] + [0.0] * 5], cuda_experts=len(allowed), allowed_experts=allowed
    )
    assert [route.expert for route in routes] == [159, 216, 231]


@pytest.mark.parametrize("name", ("unknown", True, 216))
def test_unknown_partition_is_rejected(name):
    with pytest.raises(ValueError, match="unsupported"):
        one_card_partition(name)


class WeightedMethod:
    """Reuse output storage so the placement shadow must protect aliases."""

    overlap_native = False

    def __init__(self):
        self.output = None
        self.calls = []
        self.state = {}
        self.session = SimpleNamespace(flush=lambda: None)

    def apply(self, layer, hidden, weights, ids, shared, shared_input, *, rank_one_hidden=None):
        self.calls.append((ids.clone(), weights.clone()))
        value = ((ids.float() + 1) * weights).sum(1, keepdim=True).expand_as(hidden)
        if self.output is None:
            self.output = torch.empty_like(value)
        self.output.copy_(value)
        return self.output


@pytest.mark.parametrize("layer", (3, 8, 33))
def test_real_route_composition_without_shadow_skips_all_lowbit_native_calls(monkeypatch, layer):
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            tensor_model_parallel_all_reduce=lambda x: x,
            get_tp_group=lambda: SimpleNamespace(broadcast=lambda value, src: value),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata=None)),
    )
    ids = torch.tensor([[0, 143, 144, 159, 160, 215, 216, 231], [159, 160, 215, 216, 231, 232, 286, 287]])
    weights = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8) / 32
    hidden = torch.zeros((2, 4))
    parts = []
    for rank in (0, 1):
        native, cuda = WeightedMethod(), WeightedMethod() if rank == 0 else None
        module = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=144 + rank * 72, num_local_experts=72))
        method = one_card_routed(native, cuda, cuda_partition=PRECISION_PARTITION, layer_id=layer)
        parts.append(method(module, hidden, ids, weights, reduce_results=False))
        assert len(native.calls) == (1 if layer == 3 else 0)
    expected = ((ids.float() + 1) * weights).sum(1, keepdim=True).expand_as(hidden)
    torch.testing.assert_close(sum(parts), expected, rtol=0, atol=0)


@pytest.mark.parametrize("layer", (3, 8, 33))
def test_placement_shadow_compares_both_ranks_with_legacy_ownership_and_protects_workspace(monkeypatch, layer):
    context = threading.local()
    condition = threading.Condition()
    inputs, results = {}, {}

    def all_reduce(value):
        call = context.call
        context.call += 1
        with condition:
            inputs.setdefault(call, {})[context.rank] = value.clone()
            if len(inputs[call]) == 2:
                results[call] = inputs[call][0] + inputs[call][1]
                condition.notify_all()
            assert condition.wait_for(lambda: call in results, timeout=5), "TP collective order diverged"
            return results[call].clone()

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            tensor_model_parallel_all_reduce=all_reduce,
            get_tp_group=lambda: SimpleNamespace(broadcast=lambda value, src: value),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata={})),
    )
    ids = torch.tensor([[0, 143, 144, 159, 160, 215, 216, 231], [159, 160, 215, 216, 231, 232, 286, 287]])
    weights = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8) / 32

    def forward(rank):
        context.rank, context.call = rank, 0
        native, cuda = WeightedMethod(), WeightedMethod() if rank == 0 else None
        module = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=144 + rank * 72, num_local_experts=72))
        method = one_card_routed(native, cuda, cuda_partition=PRECISION_PARTITION, layer_id=layer)
        result = method(module, torch.zeros((2, 4)), ids, weights)
        if cuda is not None:
            assert cuda.state["placement_decode_shadow_checks"] == 1
            assert cuda.state["placement_shadow_max_absolute_error"] == 0
        assert len(native.calls) == (2 if layer == 3 else 1)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(forward, (0, 1)))
    expected = ((ids.float() + 1) * weights).sum(1, keepdim=True).expand(2, 4)
    for value in result:
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layer", (3, 8, 33))
def test_moved_rank_one_experts_keep_rank_one_activations(monkeypatch, layer):
    hidden = (torch.ones((1, 4), dtype=torch.float16), torch.full((1, 4), 1.0009765625, dtype=torch.float16))
    active_rank = [0]
    broadcasts = []

    def broadcast(value, src):
        assert src == 1 and value.data_ptr() != hidden[active_rank[0]].data_ptr()
        broadcasts.append(active_rank[0])
        value.copy_(hidden[1])
        return value

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            tensor_model_parallel_all_reduce=lambda value: value,
            get_tp_group=lambda: SimpleNamespace(broadcast=broadcast),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata=None)),
    )

    class InputSensitiveMethod:
        overlap_native = False

        def apply(self, module, x, weights, ids, shared, shared_input, *, rank_one_hidden=None):
            rows = x[:, None, :].float().expand(-1, ids.shape[1], -1)
            if rank_one_hidden is not None:
                rows = torch.where((ids >= 216)[..., None], rank_one_hidden[:, None, :].float(), rows)
            return (rows * (weights * (ids.float() + 1))[..., None]).sum(1)

    ids = torch.tensor([[0, 144, 159, 160, 215, 216, 231, 287]])
    weights = torch.full((1, 8), 0.125)
    parts = []
    for rank in (0, 1):
        active_rank[0] = rank
        module = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=144 + rank * 72, num_local_experts=72))
        method = one_card_routed(
            InputSensitiveMethod(),
            InputSensitiveMethod() if rank == 0 else None,
            cuda_partition=PRECISION_PARTITION,
            layer_id=layer,
        )
        parts.append(method(module, hidden[rank], ids, weights, reduce_results=False))
    expected = (
        torch.where((ids >= 216)[..., None], hidden[1][:, None].float(), hidden[0][:, None].float())
        * (weights * (ids.float() + 1))[..., None]
    ).sum(1)
    torch.testing.assert_close(sum(parts), expected, rtol=0, atol=0)
    assert broadcasts == [0, 1]
    assert hidden[0].eq(1).all() and hidden[1].eq(1.0009765625).all()
