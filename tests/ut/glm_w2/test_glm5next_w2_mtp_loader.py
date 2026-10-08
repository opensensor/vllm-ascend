# SPDX-License-Identifier: Apache-2.0
"""Packed loader + shipped dense loader integration on CPU-sized modules.

The shipped loader class is compiled from its source AST to avoid importing
its NPU model dependencies. Its load/remap/ownership methods are unmodified.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import regex as re
import torch
from torch import nn

from vllm_ascend.models.glm5next_w2 import moe
from vllm_ascend.models.glm5next_w2.model import Glm5NextW2MTP, _PackedW2ExpertBank


@pytest.fixture
def draft(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/mtp.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Glm5NextMTP")
    module = ModuleType("vllm_ascend.models.glm5next.mtp")

    def spec_layer(config, name):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if (
            match
            and config.num_hidden_layers <= int(match[1]) < config.num_hidden_layers + config.num_nextn_predict_layers
        ):
            return int(match[1])
        return None

    module.__dict__.update(
        nn=nn,
        torch=torch,
        DeepseekV2MixtureOfExperts=type("MoE", (), {}),
        fused_moe_make_expert_params_mapping=lambda *a, **k: [],
        get_spec_layer_idx_from_weight_name=spec_layer,
        _try_load_fp8_indexer_wk=lambda *a: False,
        _try_load_fp8_attn_proj=lambda *a: False,
        maybe_remap_kv_scale_name=lambda name, params: name,
        default_weight_loader=lambda param, value: param.data.copy_(value),
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    subset = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(subset, str(path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(moe, "_ep_rank_size", lambda: (0, 1))
    place = _PackedW2ExpertBank.place_resident_tensor
    monkeypatch.setattr(
        _PackedW2ExpertBank,
        "place_resident_tensor",
        lambda self, expert, attr, tensor, **kw: place(self, expert, attr, tensor, device="cpu"),
    )
    model = Glm5NextW2MTP.__new__(Glm5NextW2MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        n_routed_experts=2,
        first_k_dense_replace=0,
        hidden_size=32,
        moe_intermediate_size=32,
        mla_nope=False,
    )
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 1
    model.model.num_mtp_layers = 1
    model.model.embed_tokens = nn.Embedding(8, 32)
    layer = nn.Module()
    layer.enorm = nn.Linear(32, 1, bias=False)
    layer.eh_proj = nn.Linear(64, 32, bias=False)
    layer.shared_head = nn.Module()
    layer.shared_head.norm = nn.Linear(32, 1, bias=False)
    layer.shared_head.head = nn.Linear(32, 8, bias=False)
    layer.mtp_block = nn.Module()
    layer.mtp_block.mlp = nn.Module()
    layer.mtp_block.mlp.shared_experts = nn.Module()
    shared = layer.mtp_block.mlp.shared_experts
    shared.gate_up_proj = nn.Linear(32, 64, bias=False)
    shared.down_proj = nn.Linear(32, 32, bias=False)
    shared.gate_up_proj.weight.weight_loader = lambda p, w, shard: p.data[shard * 32 : (shard + 1) * 32].copy_(w)
    bank = _PackedW2ExpertBank(
        32, 32, 2, layer_key="layers.1", local_expert_offset=0, num_local_experts=2, offload_to_cpu=False
    )
    layer.mtp_block.mlp_w2 = SimpleNamespace(w2_experts=bank)
    model.model.layers = nn.ModuleDict({"1": layer})
    return model


def weights():
    prefix = "model.language_model.layers.1."
    result = [
        (prefix + "enorm.weight", torch.ones(1, 32)),
        (prefix + "eh_proj.weight", torch.ones(32, 64)),
        (prefix + "shared_head.norm.weight", torch.ones(1, 32)),
    ]
    for proj in ("gate", "up", "down"):
        result.append((prefix + f"mlp.shared_experts.{proj}_proj.weight", torch.full((32, 32), 2.0)))
        for expert in range(2):
            # Mixed W3/W4 shapes exercise the existing packed mapper.
            width = 12 if proj in ("gate", "up") else 16
            result.extend(
                [
                    (prefix + f"mlp.experts.{expert}.{proj}_proj_codes", torch.ones(32, width, dtype=torch.uint8)),
                    (prefix + f"mlp.experts.{expert}.{proj}_proj_scale", torch.ones(1, 1)),
                ]
            )
    return result


def test_draft_loads_packed_banks_and_shared_fp16_without_backbone(draft):
    loaded = draft.load_weights(iter([("model.language_model.layers.0.junk.weight", torch.zeros(1)), *weights()]))
    layer = draft.model.layers["1"]
    assert layer.mtp_block.mlp_w2.w2_experts.grouped_ready
    assert layer.mtp_block.mlp_w2.w2_experts.gate_packed_bank.shape == (2, 32, 12)
    assert layer.mtp_block.mlp_w2.w2_experts.down_packed_bank.shape == (2, 32, 16)
    assert not draft.has_own_lm_head and layer.shared_head.head is None
    assert draft.model.embed_tokens is None
    assert "model.layers.1.eh_proj.weight" in loaded
    assert all("layers.0." not in name for name in loaded)
    assert torch.equal(layer.mtp_block.mlp.shared_experts.gate_up_proj.weight, torch.full((64, 32), 2.0))


@pytest.mark.parametrize(
    "missing", ["enorm.weight", "mlp.shared_experts.up_proj.weight", "mlp.experts.1.down_proj_scale"]
)
def test_incomplete_draft_checkpoint_fails(draft, missing):
    with pytest.raises(ValueError, match="Incomplete|incomplete"):
        draft.load_weights((n, v) for n, v in weights() if not n.endswith(missing))


def test_duplicate_tensor_rejected(draft):
    values = weights()
    with pytest.raises(ValueError, match="duplicate"):
        draft.load_weights(iter([*values, values[-1]]))


def test_owned_checkpoint_head_is_retained(draft):
    draft.load_weights(iter([*weights(), ("model.language_model.layers.1.shared_head.head.weight", torch.ones(8, 32))]))
    assert draft.has_own_lm_head and draft.model.layers["1"].shared_head.head is not None
