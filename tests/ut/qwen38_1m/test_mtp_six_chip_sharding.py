# SPDX-License-Identifier: Apache-2.0
"""Exercise actual MTP constructors/loaders with device-free dependency shells."""

import ast
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp.head_partition import vocab_partition_padding
from vllm_ascend.models.qwen4_exp.shared_partition import place_uneven_shared_expert_tensor, shared_expert_range
from vllm_ascend.models.qwen4_exp.weight_mapping import local_expert_range

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def mtp_classes(monkeypatch):
    path = ROOT / "vllm_ascend/models/qwen4_exp/mtp.py"
    parsed = ast.parse(path.read_text())
    nodes = [
        n
        for n in parsed.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in ("_MTPFP16MoE", "_MTPPredictor", "AscendQwen4ExpMTP", "_select_local_mtp_routes")
    ]
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            node.bases = [ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())]
    distributed = ModuleType("vllm.distributed")
    distributed.tensor_model_parallel_all_reduce = lambda tensor: tensor
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    class Vocabulary(torch.nn.Module):
        def __init__(self, rows, width, *, padding_size=64, **kwargs):
            super().__init__()
            padded = math.ceil(rows / padding_size) * padding_size
            assert padded % 6 == 0
            self.padding_size = padding_size
            self.weight = torch.nn.Parameter(torch.zeros(padded // 6, width, dtype=torch.float16))

    class Layer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.ple = None

    class Residual(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def prepare_norm_affine(self):
            pass

    scope = {
        "torch": torch,
        "nn": torch.nn,
        "DEFAULT_VOCAB_PADDING_SIZE": 64,
        "VocabParallelEmbedding": Vocabulary,
        "ParallelLMHead": Vocabulary,
        "vocab_partition_padding": vocab_partition_padding,
        "local_expert_range": local_expert_range,
        "shared_expert_range": shared_expert_range,
        "place_uneven_shared_expert_tensor": place_uneven_shared_expert_tensor,
        "_resolve_expert_sharding": lambda config: (config.rank, 6),
        "AscendQwen4ExpDecoderLayer": Layer,
        "_GatedResidual": Residual,
        "maybe_prefix": lambda prefix, name: name,
        "copy": lambda config: SimpleNamespace(**vars(config)),
        "make_empty_intermediate_tensors_factory": lambda *args: None,
        "_format_eager_linear_weights_npu": lambda *args: None,
        "w4_config": lambda config: None,
        "LogitsProcessor": lambda *args: None,
        "_linear": lambda x, w, dtype: F.linear(x.to(dtype), w.to(dtype)),
        "swiglu_gate_up": lambda tensor: F.silu(tensor.chunk(2, -1)[0]) * tensor.chunk(2, -1)[1],
        "route_topk": lambda logits, top_k, **kwargs: (
            torch.softmax(logits, -1).topk(top_k, -1).values,
            torch.softmax(logits, -1).topk(top_k, -1).indices,
        ),
        "AscendQwen4ExpForCausalLM": SimpleNamespace(
            packed_modules_mapping={},
            _place_qsa_q_gate_tensor=lambda *args: None,
            _place_qsa_head_tensor=lambda *args: None,
        ),
        "_remap_non_expert": lambda *args: None,
    }
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope


def configuration(experts=13, shared=17, flag=True):
    return SimpleNamespace(
        num_experts=experts,
        num_experts_per_tok=3,
        hidden_size=8,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=shared,
        ascend_expert_quantization={"mtp_uneven_sharding": flag, "gdn_head_partition": "padded_compact"},
        vocab_size=248320,
        hc_count=4,
        hc_lowrank=4,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        mtp_num_hidden_layers=1,
    )


def policy():
    return SimpleNamespace(main_dtype=torch.float16, accumulation_dtype=torch.float32, router_dtype=torch.float32)


@pytest.mark.parametrize("experts,shared", [(13, 17), (512, 640)])
def test_actual_bank_partitions_cover_each_expert_and_shared_channel_once(mtp_classes, experts, shared):
    banks = [mtp_classes["_MTPFP16MoE"](configuration(experts, shared), policy(), (rank, 6)) for rank in range(6)]
    expert_ids = [i for bank in banks for i in range(bank.expert_offset, bank.expert_offset + bank.num_local_experts)]
    channels = [i for bank in banks for i in range(bank.shared_start, bank.shared_stop)]
    assert expert_ids == list(range(experts))
    assert channels == list(range(shared))
    assert all(bank.shared_gate_up.shape[0] == 2 * bank.local_shared_intermediate for bank in banks)
    if experts == 512:
        assert [bank.num_local_experts for bank in banks] == [86, 86, 85, 85, 85, 85]
        assert [bank.local_shared_intermediate for bank in banks] == [107, 107, 107, 107, 106, 106]


@pytest.mark.parametrize("flag", [False, "true", 1])
def test_uneven_mtp_requires_explicit_boolean_opt_in(mtp_classes, flag):
    with pytest.raises(ValueError, match="divisible|boolean"):
        mtp_classes["_MTPFP16MoE"](configuration(flag=flag), policy(), (0, 6))


@pytest.mark.parametrize("rank", range(6))
def test_actual_loader_reads_global_expert_offset_and_uneven_shared_columns(mtp_classes, rank):
    # The rank's second local interval differs from rank*local_count; index-coded
    # checkpoint values detect missing or duplicate experts and shared columns.
    config = configuration()
    bank = mtp_classes["_MTPFP16MoE"](config, policy(), (rank, 6))
    owner = mtp_classes["AscendQwen4ExpMTP"].__new__(mtp_classes["AscendQwen4ExpMTP"])
    torch.nn.Module.__init__(owner)
    owner.config, owner.dtype_policy = config, policy()
    owner.model = torch.nn.Module()
    owner.model.expert_sharding = (rank, 6)
    layer = torch.nn.Module()
    layer.mlp = bank
    owner.model.layers = torch.nn.ModuleList([layer])
    owner.model.prepare_norm_affines = lambda: None
    expert_up = torch.arange(13).half()[:, None, None].expand(13, 8, 8).clone()
    expert_down = (100 + torch.arange(13)).half()[:, None, None].expand(13, 8, 4).clone()
    shared_gate = torch.arange(17).half()[:, None].expand(17, 8).clone()
    shared_up = shared_gate + 100
    shared_down = (200 + torch.arange(17)).half()[None, :].expand(8, 17).clone()
    weights = [
        ("mtp.layers.0.mlp.experts.gate_up_proj", expert_up),
        ("mtp.layers.0.mlp.experts.down_proj", expert_down),
        ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", shared_gate),
        ("mtp.layers.0.mlp.shared_expert.up_proj.weight", shared_up),
        ("mtp.layers.0.mlp.shared_expert.down_proj.weight", shared_down),
    ]
    loaded = owner.load_weights(weights)
    for local in range(bank.num_local_experts):
        assert torch.equal(bank.gate_up_proj[local], expert_up[bank.expert_offset + local])
        assert torch.equal(bank.down_proj[local], expert_down[bank.expert_offset + local])
    first, stop = bank.shared_start, bank.shared_stop
    assert torch.equal(bank.shared_gate_up, torch.cat([shared_gate[first:stop], shared_up[first:stop]]))
    assert torch.equal(bank.shared_down, shared_down[:, first:stop])
    assert "model.layers.0.mlp.shared_gate_up" in loaded
    assert "model.layers.0.mlp.shared_down" in loaded


def test_actual_draft_embedding_and_head_use_same_tp_padding_as_target(mtp_classes):
    config = configuration()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=SimpleNamespace(tensor_parallel_size=6, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        device_config=SimpleNamespace(device=SimpleNamespace(type="cpu")),
        rank=5,
    )
    p = policy()
    p.embedding_dtype = p.lm_head_dtype = torch.float16
    mtp_classes["Qwen4ExpDtypePolicy"] = SimpleNamespace(from_vllm_config=lambda _: p)
    model = mtp_classes["AscendQwen4ExpMTP"](vllm_config=vllm_config)
    assert model.model.embed_tokens.padding_size == model.lm_head.padding_size == 192
    assert model.model.embed_tokens.weight.shape == model.lm_head.weight.shape == (41408, 8)
