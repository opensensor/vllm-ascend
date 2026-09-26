# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the Qwen4Exp FP16 MTP draft head."""

from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.test_qwen4exp_assembly import (
    _single_rank_tp,
    _tiny_text_config,
    _vllm_config,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.mtp import AscendQwen4ExpMTP, _MTPFP16MoE


def _build(*, expert_sharding: tuple[int, int] = (0, 1), qsa: bool = False) -> AscendQwen4ExpMTP:
    config = _tiny_text_config(num_layers=1, qsa=qsa, moe=True, ple_layer_ids=())
    config.mtp_num_hidden_layers = 1
    vllm_config = _vllm_config(config)
    with (
        _single_rank_tp(),
        patch("vllm_ascend.models.qwen4_exp.mtp._resolve_expert_sharding", return_value=expert_sharding),
    ):
        return AscendQwen4ExpMTP(vllm_config=vllm_config)


def test_mtp_uses_full_attention_without_ple_or_int8_experts():
    with torch.device("meta"):
        model = _build(qsa=True)
    layer = model.model.layers[0]
    assert layer.ple is None
    assert layer.uses_qsa
    assert layer.mlp.gate_up_proj[0].dtype == model.dtype_policy.main_dtype
    assert tuple(layer.mlp.gate_up_proj[0].shape) == (64, 64)
    assert tuple(layer.mlp.down_proj[0].shape) == (64, 32)
    assert not any(parameter.dtype == torch.int8 for parameter in model.parameters())


def test_mtp_recycles_multi_stream_state_and_applies_embedding_to_each_stream():
    model = _build()
    hidden_size = model.config.hidden_size
    with torch.no_grad():
        model.model.fc_hidden.copy_(torch.eye(hidden_size))
        model.model.fc_embedding.copy_(torch.eye(hidden_size))
    hidden = torch.randn(2, model.config.hc_count, hidden_size, dtype=model.dtype_policy.main_dtype)
    embedding = torch.randn(2, hidden_size, dtype=model.dtype_policy.main_dtype)
    with _single_rank_tp(), torch.no_grad():
        sample, recycled = model(None, torch.arange(2), hidden.flatten(-2), inputs_embeds=embedding)
        logits = model.compute_logits(sample)
    assert sample.shape == (2, hidden_size)
    assert recycled.shape == hidden.flatten(-2).shape
    assert logits.shape == (2, model.config.vocab_size)
    assert torch.isfinite(recycled).all()
    assert torch.isfinite(sample).all()
    hidden32 = hidden.float()
    embedding32 = embedding.float()
    hidden_norm = hidden32 * torch.rsqrt(hidden32.square().mean(dim=-1, keepdim=True) + model.config.rms_norm_eps)
    embedding_norm = embedding32 * torch.rsqrt(
        embedding32.square().mean(dim=-1, keepdim=True) + model.config.rms_norm_eps
    )
    expected = (hidden_norm + embedding_norm[:, None, :]).flatten(-2).half()
    torch.testing.assert_close(recycled, expected, atol=0.001, rtol=0.001)
    with pytest.raises(ValueError, match="multi-stream"):
        model(None, torch.arange(2), hidden[:, 0], inputs_embeds=embedding)


def test_checkpoint_weights_load_into_fp16_experts_and_projection():
    model = _build()
    bank = model.model.layers[0].mlp
    gate_up = torch.arange(bank.num_experts * 64 * 64, dtype=model.dtype_policy.main_dtype).reshape(4, 64, 64)
    down = torch.arange(bank.num_experts * 64 * 32, dtype=model.dtype_policy.main_dtype).reshape(4, 64, 32)
    fc = torch.eye(64, dtype=model.dtype_policy.main_dtype)
    norm = torch.full((128,), 0.25, dtype=model.dtype_policy.main_dtype)
    with _single_rank_tp():
        loaded = model.load_weights(
            [
                ("mtp.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("mtp.layers.0.mlp.experts.down_proj", down),
                ("mtp.fc_hidden.weight", fc),
                ("mtp.pre_fc_norm_hidden.weight", norm),
            ]
        )
    assert "model.fc_hidden" in loaded
    assert "model.pre_fc_norm_hidden" in loaded
    assert "model.layers.0.mlp.gate_up_proj.3" in loaded
    torch.testing.assert_close(bank.gate_up_proj[2], gate_up[2])
    torch.testing.assert_close(bank.down_proj[3], down[3])
    torch.testing.assert_close(model.model.fc_hidden, fc)
    torch.testing.assert_close(model.model.pre_fc_norm_hidden, norm)


def test_checkpoint_experts_are_sliced_by_tp_rank_and_invalid_weights_fail():
    model = _build(expert_sharding=(1, 2))
    bank = model.model.layers[0].mlp
    gate_up = torch.arange(bank.num_experts * 64 * 64, dtype=model.dtype_policy.main_dtype).reshape(4, 64, 64)
    model.load_weights([("mtp.layers.0.mlp.experts.gate_up_proj", gate_up)])
    torch.testing.assert_close(bank.gate_up_proj[0], gate_up[2])
    torch.testing.assert_close(bank.gate_up_proj[1], gate_up[3])
    with pytest.raises(ValueError, match="expected"):
        model.load_weights([("mtp.layers.0.mlp.experts.gate_up_proj", gate_up.float())])
    with pytest.raises(ValueError, match="expected"):
        model.load_weights([("mtp.layers.0.mlp.experts.gate_up_proj", gate_up[:3])])


def test_mtp_qsa_checkpoint_heads_are_sharded_for_tp4():
    model = _build(expert_sharding=(2, 4), qsa=True)
    attention = model.model.layers[0].attention
    config = model.config
    hidden = config.hidden_size
    head_dim = config.head_dim
    q_gate = torch.arange(config.num_attention_heads * 2 * head_dim * hidden, dtype=torch.float16).reshape(
        config.num_attention_heads * 2 * head_dim, hidden
    )
    k = torch.arange(config.num_key_value_heads * head_dim * hidden, dtype=torch.float16).reshape(
        config.num_key_value_heads * head_dim, hidden
    )
    v = k + 1
    o = torch.arange(hidden * config.num_attention_heads * head_dim, dtype=torch.float16).reshape(
        hidden, config.num_attention_heads * head_dim
    )
    with _single_rank_tp():
        loaded = model.load_weights(
            [
                ("mtp.layers.0.self_attn.q_proj.weight", q_gate),
                ("mtp.layers.0.self_attn.k_proj.weight", k),
                ("mtp.layers.0.self_attn.v_proj.weight", v),
                ("mtp.layers.0.self_attn.o_proj.weight", o),
            ]
        )
    prefix = "model.layers.0.attention"
    assert loaded == {f"{prefix}.{name}" for name in ("q_proj", "gate_proj", "k_proj", "v_proj", "o_proj")}
    assert attention.num_heads == 1
    assert attention.num_kv_heads == 1
    assert attention.get_kv_cache_spec(None).num_kv_heads == 1
    source_heads = q_gate.reshape(config.num_attention_heads, 2, head_dim, hidden)
    torch.testing.assert_close(attention.q_proj, source_heads[2, 0])
    torch.testing.assert_close(attention.gate_proj, source_heads[2, 1])
    torch.testing.assert_close(attention.k_proj, k[head_dim:])
    torch.testing.assert_close(attention.v_proj, v[head_dim:])
    torch.testing.assert_close(attention.o_proj, o[:, 2 * head_dim : 3 * head_dim])


def test_mtp_dense_attention_weights_bypass_qsa_head_slicing():
    model = _build(qsa=False)
    target = model.model.layers[0].attention.k_proj
    source = torch.arange(target.numel(), dtype=target.dtype).reshape_as(target)
    with _single_rank_tp():
        loaded = model.load_weights([("mtp.layers.0.self_attn.k_proj.weight", source)])
    assert loaded == {"model.layers.0.attention.k_proj"}
    torch.testing.assert_close(target, source)


def test_tp_expert_partials_sum_to_unsharded_fp16_moe():
    config = _tiny_text_config(num_layers=1, qsa=False, moe=True, ple_layer_ids=())
    config.shared_expert_intermediate_size = 0
    policy = Qwen4ExpDtypePolicy.for_310p()
    full = _MTPFP16MoE(config, policy, (0, 1))
    left = _MTPFP16MoE(config, policy, (0, 2))
    right = _MTPFP16MoE(config, policy, (1, 2))
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        full.gate.copy_(torch.randn(full.gate.shape, generator=generator).half() * 0.1)
        for index in range(config.num_experts):
            full.gate_up_proj[index].copy_(
                torch.randn(full.gate_up_proj[index].shape, generator=generator).half() * 0.1
            )
            full.down_proj[index].copy_(torch.randn(full.down_proj[index].shape, generator=generator).half() * 0.1)
        for rank, shard in enumerate((left, right)):
            shard.gate.copy_(full.gate)
            for local in range(shard.num_local_experts):
                global_id = rank * shard.num_local_experts + local
                shard.gate_up_proj[local].copy_(full.gate_up_proj[global_id])
                shard.down_proj[local].copy_(full.down_proj[global_id])
    left._tp_reduce = lambda tensor: tensor
    right._tp_reduce = lambda tensor: tensor
    x = torch.randn((3, config.hidden_size), generator=generator).half()
    torch.testing.assert_close(left(x).float() + right(x).float(), full(x).float(), atol=0.002, rtol=0.002)
