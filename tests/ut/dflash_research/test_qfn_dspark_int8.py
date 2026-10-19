# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
import torch

from tools.dflash_research import qfn_dspark_int8 as candidate


class UnquantizedMethod:
    pass


def worker():
    def linear():
        return NS(weight=torch.ones(16, 16).half(), quant_method=UnquantizedMethod(), tp_size=1, bias=None)

    layers = [NS(self_attn=NS(qkv_proj=linear()), mlp=NS(gate_up_proj=linear(), down_proj=linear())) for _ in range(5)]
    model = NS(
        model=NS(layers=layers, markov_head=NS(markov_w2=linear())),
        config=NS(hidden_size=2560),
        draft_id_to_target_id=None,
        lm_head=linear(),
    )
    return NS(
        rank=0,
        model_runner=NS(
            speculative_config=NS(method="dspark", draft_tensor_parallel_size=1, enforce_eager=True),
            drafter=NS(use_cuda_graph=False, model=NS(unwrap=lambda: model)),
            use_async_scheduling=False,
        ),
    )


@pytest.fixture
def prepared(monkeypatch):
    class Private:
        def __init__(self, weight):
            self.shape = tuple(weight.shape)
            self.weight = torch.zeros_like(weight, dtype=torch.int8)
            self.scale = torch.ones(weight.shape[0])

    monkeypatch.setattr(candidate, "PreparedLinear", Private)


def test_switch_restores_methods_and_preserves_original_weight_storage(prepared):
    owner = candidate.DenseInt8Control(worker())
    assert len(owner.entries) == 17
    owner.private_checks = [{} for _ in range(34)]
    pointers = [layer.weight.data_ptr() for _, layer, _, _ in owner.entries]
    owner.configure(True)
    assert all(layer.quant_method is new for _, layer, _, new in owner.entries)
    owner.configure(False)
    assert all(layer.quant_method is old for _, layer, old, _ in owner.entries)
    assert pointers == [layer.weight.data_ptr() for _, layer, _, _ in owner.entries]


def test_owner_conflict_rejects_without_partially_switching(prepared):
    owner = candidate.DenseInt8Control(worker())
    owner.private_checks = [{} for _ in range(34)]
    owner.entries[-1][1].quant_method = object()
    with pytest.raises(RuntimeError, match="owner changed"):
        owner.configure(True)
    assert all(layer.quant_method is old for _, layer, old, _ in owner.entries[:-1])


@pytest.mark.parametrize(
    "field,value", [("method", "mtp"), ("draft_tensor_parallel_size", 6), ("enforce_eager", False)]
)
def test_rejects_unsupported_runtime(prepared, field, value):
    item = worker()
    setattr(item.model_runner.speculative_config, field, value)
    with pytest.raises(ValueError, match="requires eager"):
        candidate.DenseInt8Control(item)


def test_rejects_nonboolean_switch(prepared):
    owner = candidate.DenseInt8Control(worker())
    with pytest.raises(ValueError, match="boolean"):
        owner.configure(1)


def test_cannot_enable_before_private_checks(prepared):
    owner = candidate.DenseInt8Control(worker())
    with pytest.raises(RuntimeError, match="private projection checks"):
        owner.configure(True)
    assert all(layer.quant_method is old for _, layer, old, _ in owner.entries)
