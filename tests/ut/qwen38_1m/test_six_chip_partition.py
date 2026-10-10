# SPDX-License-Identifier: Apache-2.0
"""Padded head/load/state semantics, independent FP64 recurrence and real geometry."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.gdn_reference import gdn_delta_rule_recurrent
from tools.qwen4exp.three_card_profile import checkpoint_overlay, make_profile
from vllm_ascend.models.qwen4_exp.head_partition import (
    gdn_execution_shard,
    gdn_head_shard,
    gdn_partition_policy,
    place_padded_gdn_tensor,
    shard_gdn_tensor,
    vocab_partition_padding,
)
from vllm_ascend.models.qwen4_exp.qsa_head_sharding import qsa_head_shard
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import Qwen4ExpGDNParams
from vllm_ascend.models.qwen4_exp.weight_mapping import local_expert_range


@pytest.mark.parametrize("tp", [1, 2, 4, 6, 8, 16])
def test_real_head_expert_and_qsa_ownership(tp):
    policy = "padded" if tp == 6 else "strict"
    shards = [gdn_head_shard(16, 48, 128, 128, rank, tp, policy) for rank in range(tp)]
    assert sum(s.live_key_heads for s in shards) == 16
    assert sum(s.live_value_heads for s in shards) == 48
    assert len({s.conv_dim for s in shards}) == 1
    covered = [h for s in shards for h in range(s.key_start, s.key_start + s.live_key_heads)]
    assert covered == list(range(16))
    ranges = [local_expert_range(512, tp, r) for r in range(tp)]
    assert [e for start, stop in ranges for e in range(start, stop)] == list(range(512))
    if tp == 6:
        assert [s.live_key_heads for s in shards] == [3, 3, 3, 3, 3, 1]
        assert {(stop - start) for start, stop in ranges} == {85, 86}
        qsa = [qsa_head_shard(24, 2, 256, rank, tp) for rank in range(tp)]
        assert [s.query_start for s in qsa] == [0, 4, 8, 12, 16, 20]
        assert [s.kv_start for s in qsa] == [0, 0, 0, 1, 1, 1]


def test_strict_default_and_bad_partitions_rejected():
    assert gdn_partition_policy(SimpleNamespace()) == "strict"
    with pytest.raises(ValueError, match="not divisible"):
        gdn_head_shard(16, 48, 128, 128, 0, 6)
    for args in (
        (16, 47, 128, 128, 0, 6, "padded"),
        (16, 48, 128, 128, 6, 6, "padded"),
        (16, 48, 128, 128, 0, 0, "padded"),
    ):
        with pytest.raises(ValueError):
            gdn_head_shard(*args)


@pytest.mark.parametrize(
    "kind,shape",
    [
        ("qkv", (80, 7)),
        ("conv", (80, 1, 4)),
        ("z", (48, 7)),
        ("out", (7, 48)),
        ("a", (48, 7)),
        ("b", (48, 7)),
        ("A_log", (48,)),
        ("dt_bias", (48,)),
        ("norm", (1,)),
    ],
)
def test_loader_padding_and_reassembly_preserves_trained_values(kind, shape):
    source = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    original = source.clone()
    parts = [shard_gdn_tensor(source, kind, gdn_head_shard(16, 48, 1, 1, r, 6, "padded"), 16, 48) for r in range(6)]
    if kind in ("qkv", "conv"):
        rebuilt = torch.cat(
            [
                torch.cat([p[:3] for p in parts])[:16],
                torch.cat([p[3:6] for p in parts])[:16],
                torch.cat([p[6:] for p in parts])[:48],
            ]
        )
        assert torch.count_nonzero(parts[-1][1:3]) == 0
        assert torch.count_nonzero(parts[-1][4:6]) == 0
        assert torch.count_nonzero(parts[-1][9:]) == 0
        assert torch.equal(rebuilt, source.squeeze(1) if kind == "conv" else source)
    elif kind == "out":
        assert torch.equal(torch.cat(parts, dim=1)[:, :48], source)
        assert torch.count_nonzero(parts[-1][:, 3:]) == 0
    elif kind == "norm":
        assert all(torch.equal(p, source) for p in parts)
    else:
        assert torch.equal(torch.cat(parts)[:48], source)
        assert torch.count_nonzero(parts[-1][3:]) == 0
    assert torch.equal(source, original)


def test_reload_clears_padding_and_preserves_ba_order():
    shard = gdn_head_shard(16, 48, 1, 1, 5, 6, "padded")
    target = torch.full((18, 2), 999.0)
    params = {"model.layers.0.attention.in_proj_ba": target}
    for kind, value in (("b", 7.0), ("a", 11.0)):
        place_padded_gdn_tensor(
            params, f"model.layers.0.linear_attn.in_proj_{kind}.weight", torch.full((48, 2), value), shard, 16, 48
        )
    assert torch.equal(target[:3], torch.full((3, 2), 7.0))
    assert torch.equal(target[9:12], torch.full((3, 2), 11.0))
    assert torch.count_nonzero(target[3:9]) == torch.count_nonzero(target[12:]) == 0
    with pytest.raises(ValueError, match="shape"):
        place_padded_gdn_tensor(
            params, "model.layers.0.linear_attn.in_proj_a.weight", torch.zeros(47, 2), shard, 16, 48
        )


@pytest.mark.parametrize("tokens", [1, 3, 65])
def test_padded_recurrent_outputs_and_initial_final_states_match_full_reference(tokens):
    gen = torch.Generator().manual_seed(tokens)
    q = torch.randn(tokens, 16, 4, generator=gen).double() * 0.1
    k = torch.randn(tokens, 16, 4, generator=gen).double() * 0.1
    v = torch.randn(tokens, 48, 4, generator=gen).double()
    g = -torch.rand(tokens, 48, generator=gen).double()
    beta = torch.rand(tokens, 48, generator=gen).double()
    state = torch.randn(48, 4, 4, generator=gen).double() * 0.01
    original = state.clone()
    full, final = gdn_delta_rule_recurrent(q.repeat_interleave(3, 1), k.repeat_interleave(3, 1), v, g, beta, state)
    outputs, states = [], []
    for rank in range(6):
        shard = gdn_head_shard(16, 48, 4, 4, rank, 6, "padded")
        local_q, local_k = [torch.zeros(tokens, 3, 4, dtype=torch.float64) for _ in range(2)]
        local_q[:, : shard.live_key_heads] = q[:, shard.key_start : shard.key_start + shard.live_key_heads]
        local_k[:, : shard.live_key_heads] = k[:, shard.key_start : shard.key_start + shard.live_key_heads]
        local_v = torch.zeros(tokens, 9, 4, dtype=torch.float64)
        local_g = torch.zeros(tokens, 9, dtype=torch.float64)
        local_beta = torch.zeros_like(local_g)
        local_state = torch.zeros(9, 4, 4, dtype=torch.float64)
        live = shard.live_value_heads
        start = shard.value_start
        local_v[:, :live], local_g[:, :live], local_beta[:, :live] = (
            v[:, start : start + live],
            g[:, start : start + live],
            beta[:, start : start + live],
        )
        local_state[:live] = state[start : start + live]
        out, end = gdn_delta_rule_recurrent(
            local_q.repeat_interleave(3, 1), local_k.repeat_interleave(3, 1), local_v, local_g, local_beta, local_state
        )
        assert torch.count_nonzero(out[:, live:]) == torch.count_nonzero(end[live:]) == 0
        outputs.append(out[:, :live])
        states.append(end[:live])
    torch.testing.assert_close(torch.cat(outputs, 1), full, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(states), final, rtol=0, atol=0)
    assert torch.equal(state, original)


def test_saved_checkpoint_profile_keeps_vision_and_trained_heads_and_reports_no_admission():
    path = Path(__file__).resolve().parents[3] / "artifacts/qwen38-memory-audit-20261008/model-config.json"
    original = json.loads(path.read_text())
    config, receipt = make_profile(original)
    assert config["vision_config"] == original["vision_config"]
    assert config["text_config"]["linear_num_key_heads"] == 16
    assert config["text_config"]["linear_num_value_heads"] == 48
    assert "gdn_head_partition" not in original["text_config"]["ascend_expert_quantization"]
    assert sum(r["expert_count"] for r in receipt["ranks"]) == 512
    assert not receipt["hardware_admission"] and not receipt["mtp_enabled"]
    assert receipt["mm_encoder_tp_mode"] == "data" and receipt["images_enabled"]
    assert {r["gdn_allocated_value_heads"] for r in receipt["ranks"]} == {9}
    assert {r["gdn_conv_channels"] for r in receipt["ranks"]} == {1920}
    assert [r["gdn_execution_value_heads"] for r in receipt["ranks"]] == [9, 9, 9, 9, 6, 6]
    assert [r["shared_channel_count"] for r in receipt["ranks"]] == [107, 107, 107, 107, 106, 106]
    assert receipt["shared_expert_execution"] == "tp_sharded_uneven"
    assert receipt["gdn_uniform_cache_reserve_retained"] and not receipt["gdn_dummy_heads_executed"]


@pytest.fixture
def actual_model_methods():
    # Compile actual method bodies with host dependencies, without importing
    # unrelated vLLM platform/attention backends on this CPU-only host.

    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/model.py"
    parsed = ast.parse(path.read_text())
    params = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "_gdn_params_from_config")
    gdn = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "_GDNAttention")
    gdn.bases, gdn.keywords = (
        [ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
        [],
    )
    gdn.body = [n for n in gdn.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    owner = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "AscendQwen4ExpForCausalLM")
    owner.bases, owner.keywords = [], []
    owner.body = [
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name in ("get_gdn_mamba_state_shape_from_config", "_place_gdn_tensor")
    ]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), params, gdn, owner],
        type_ignores=[],
    )
    scope = dict(
        torch=torch,
        nn=torch.nn,
        Qwen4ExpGDNParams=Qwen4ExpGDNParams,
        gdn_head_shard=gdn_head_shard,
        gdn_execution_shard=gdn_execution_shard,
        gdn_partition_policy=gdn_partition_policy,
        place_padded_gdn_tensor=place_padded_gdn_tensor,
        _register_in_static_forward_context=lambda *_: None,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope


def test_actual_constructor_loader_and_cache_descriptor_use_same_padded_geometry(actual_model_methods):
    config = SimpleNamespace(
        hidden_size=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        ascend_expert_quantization={"gdn_head_partition": "padded"},
    )
    policy = SimpleNamespace(
        accumulation_dtype=torch.float32,
        main_dtype=torch.float16,
        mamba_conv_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float32,
    )
    gdn = actual_model_methods["_GDNAttention"](config=config, dtype_policy=policy, expert_sharding=(5, 6))
    assert (gdn.num_k_heads, gdn.num_v_heads, gdn.conv_dim) == (3, 9, 1920)
    owner_cls = actual_model_methods["AscendQwen4ExpForCausalLM"]
    vconfig = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=SimpleNamespace(tensor_parallel_size=6),
        num_speculative_tokens=0,
    )
    assert owner_cls.get_gdn_mamba_state_shape_from_config(vconfig) == ((3, 1920), (9, 128, 128))
    owner = SimpleNamespace(model=SimpleNamespace(config=config))
    params = {"model.layers.0.attention.in_proj_qkv": gdn.in_proj_qkv}
    source = torch.ones(10240, 4)
    owner_cls._place_gdn_tensor(owner, params, "model.layers.0.linear_attn.in_proj_qkv.weight", source, 5, 6)
    assert torch.count_nonzero(gdn.in_proj_qkv[128:384]) == 0
    assert torch.count_nonzero(gdn.in_proj_qkv[512:768]) == 0
    assert torch.count_nonzero(gdn.in_proj_qkv[1152:]) == 0
    assert torch.count_nonzero(gdn.in_proj_qkv[:128]) == 128 * 4


def test_embedding_and_lm_head_pad_real_vocabulary_to_six_equal_rank_ranges():
    config = SimpleNamespace(ascend_expert_quantization={"gdn_head_partition": "padded"})
    alignment = vocab_partition_padding(config, 6, 64)
    assert alignment == 192
    padded = (248320 + alignment - 1) // alignment * alignment
    assert padded == 248448 and padded // 6 == 41408
    assert vocab_partition_padding(SimpleNamespace(), 4, 64) == 64
    assert vocab_partition_padding(SimpleNamespace(), 6, 64) == 64


def test_canonical_checkpoint_overlay_is_separate_append_only_and_preserves_weight_bytes(tmp_path):
    real_config = Path(__file__).resolve().parents[3] / "artifacts/qwen38-memory-audit-20261008/model-config.json"
    source = tmp_path / "checkpoint"
    source.mkdir()
    original = real_config.read_bytes()
    (source / "config.json").write_bytes(original)
    (source / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    (source / "model-00001.safetensors").write_bytes(b"unmodified packed weights fixture")
    target = tmp_path / "tp6"
    receipt = checkpoint_overlay(source, target)
    assert not receipt["hardware_admission"]
    assert (target / "model-00001.safetensors").is_symlink()
    assert (target / "model-00001.safetensors").read_bytes() == b"unmodified packed weights fixture"
    assert (source / "config.json").read_bytes() == original
    assert (
        json.loads((target / "config.json").read_text())["text_config"]["ascend_expert_quantization"][
            "gdn_head_partition"
        ]
        == "padded_compact"
    )
    with pytest.raises(FileExistsError):
        checkpoint_overlay(source, target)
    with pytest.raises(ValueError, match="separate"):
        checkpoint_overlay(source, source / "tp6")
