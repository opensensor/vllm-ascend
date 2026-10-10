# SPDX-License-Identifier: Apache-2.0
"""Actual shared/GDN methods: disjoint channels, live state and cache isolation."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from tests.ut.qwen38_1m.test_w4_moe import config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.head_partition import (
    gdn_execution_caches,
    gdn_execution_shard,
    gdn_head_shard,
    shard_gdn_tensor,
)
from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM, _GDNAttention
from vllm_ascend.models.qwen4_exp.shared_partition import place_uneven_shared_expert_tensor, shared_expert_range
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE


@pytest.mark.parametrize("channels", [640, 24, 6])
def test_shared_uneven_ranges_cover_every_trained_channel_once(channels):
    spans = [shared_expert_range(channels, r, 6) for r in range(6)]
    assert [c for start, stop in spans for c in range(start, stop)] == list(range(channels))
    assert max(b - a for a, b in spans) - min(b - a for a, b in spans) <= 1


@pytest.mark.parametrize("args", [(5, 0, 6), (640, 6, 6), (640, -1, 6), (640, 0, 0)])
def test_invalid_shared_geometry_rejected(args):
    with pytest.raises(ValueError):
        shared_expert_range(*args)


@pytest.mark.parametrize("tokens", [1, 3, 65])
def test_actual_shared_loader_and_forward_reconstruct_full_expert_without_replication(tokens):
    torch.manual_seed(614)
    cfg = config(num_layers=1, num_experts=12, top_k=2, shared_inter=640)
    cfg.ascend_expert_quantization["shared_expert_execution"] = "tp_sharded_uneven"
    hidden = cfg.hidden_size
    weights = {
        p: torch.randn((hidden, 640) if p == "down" else (640, hidden), dtype=torch.float64) * 0.05
        for p in ("gate", "up", "down")
    }
    router = torch.randn(1, hidden, dtype=torch.float64) * 0.05
    inputs = torch.randn(tokens, hidden, dtype=torch.float64)
    expected = F.linear(
        F.silu(F.linear(inputs, weights["gate"])) * F.linear(inputs, weights["up"]), weights["down"]
    ) * torch.sigmoid(F.linear(inputs, router))
    outputs = []
    owner = SimpleNamespace(config=cfg)
    widths = []
    for rank in range(6):
        layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(rank, 6)).double()
        layer.compute_dtype = torch.float64
        layer.params_dtype = torch.float64
        params = {
            "model.layers.0.mlp.shared_gate_up": layer.shared_gate_up,
            "model.layers.0.mlp.shared_down": layer.shared_down,
        }
        for projection, weight in weights.items():
            name = f"model.layers.0.mlp.shared_expert.{projection}_proj.weight"
            assert (
                AscendQwen4ExpForCausalLM._place_shared_expert_tensor(owner, params, name, weight, rank, 6) is not None
            )
        layer.shared_expert_gate.data.copy_(router)
        widths.append(layer.local_shared_inter)
        assert not layer.shared_expert_replicated
        outputs.append(layer._forward_shared(inputs))
    assert widths == [107, 107, 107, 107, 106, 106]
    torch.testing.assert_close(sum(outputs), expected, rtol=1e-12, atol=1e-12)


def test_shared_load_failure_preserves_destination_and_reload_overwrites_all_channels():
    target = torch.full((212, 8), 17.0)
    params = {"m.mlp.shared_gate_up": target}
    name = "m.mlp.shared_expert.gate_proj.weight"
    with pytest.raises(ValueError, match="shape"):
        place_uneven_shared_expert_tensor(params, name, torch.ones(639, 8), 640, 5, 6)
    assert torch.all(target == 17)
    place_uneven_shared_expert_tensor(params, name, torch.ones(640, 8), 640, 5, 6)
    assert torch.all(target[:106] == 1) and torch.all(target[106:] == 17)


def test_shared_uneven_partial_joins_existing_routed_collective_once():
    cfg = config(num_layers=1, num_experts=12, top_k=1, shared_inter=640)
    cfg.ascend_expert_quantization["shared_expert_execution"] = "tp_sharded_uneven"
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(5, 6))
    inputs = torch.zeros(2, cfg.hidden_size, dtype=torch.float16)
    routed = torch.full_like(inputs, 2.0, dtype=torch.float32)
    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.route_topk", return_value=(torch.ones(2, 1), torch.zeros(2, 1))),
        patch.object(layer, "_forward_host_routed", return_value=routed),
        patch.object(layer, "_forward_shared", return_value=torch.full_like(routed, 3.0)),
        patch.object(layer, "_tp_reduce", side_effect=lambda t: t * 6) as reduce,
    ):
        torch.testing.assert_close(layer(inputs), torch.full_like(inputs, 30.0))
    reduce.assert_called_once()


def test_compact_cache_views_preserve_real_page_stride_and_do_not_copy():
    allocated = gdn_head_shard(16, 48, 4, 4, 5, 6, "padded_compact")
    execution = gdn_execution_shard(allocated)
    # Actual cache pages can have alignment slack as well as head padding.
    conv_storage = torch.full((3, 256), -91.0)
    state_storage = torch.full((3, 192), -93.0)
    conv = conv_storage[:, :180].view(3, 3, 60)
    state = state_storage[:, :144].view(3, 9, 4, 4)
    live_conv, live_state = gdn_execution_caches((conv, state), allocated, execution)
    assert live_conv.shape == (3, 3, 40) and live_state.shape == (3, 6, 4, 4)
    assert live_conv.stride() == (256, 40, 1) and live_state.stride() == (192, 16, 4, 1)
    assert live_conv.data_ptr() == conv.data_ptr() and live_state.data_ptr() == state.data_ptr()
    owner = SimpleNamespace(
        _compact_gdn=True, kv_cache=(conv, state), head_shard=allocated, execution_head_shard=execution
    )
    assert _GDNAttention._execution_caches(owner)[0].data_ptr() == conv.data_ptr()
    replacement = (conv.clone(), state.clone())
    owner.kv_cache = replacement
    assert _GDNAttention._execution_caches(owner)[0].data_ptr() == replacement[0].data_ptr()
    live_conv[1].fill_(7)
    live_state[2].fill_(11)
    assert torch.all(conv_storage[1, :120] == 7) and torch.all(conv_storage[1, 120:] == -91)
    assert torch.all(state_storage[2, :96] == 11) and torch.all(state_storage[2, 96:] == -93)
    with pytest.raises(ValueError, match="dense"):
        gdn_execution_caches((conv[:, :, ::2], state), allocated, execution)


@pytest.mark.parametrize("cold", [False, True])
def test_actual_compact_gdn_prefill_decode_matches_full_state_and_leaves_slack_untouched(monkeypatch, cold):
    torch.manual_seed(619)
    cfg = SimpleNamespace(
        hidden_size=8,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        ascend_expert_quantization={"gdn_head_partition": "strict"},
    )
    policy = SimpleNamespace(
        main_dtype=torch.float64,
        accumulation_dtype=torch.float64,
        mamba_conv_cache_dtype=torch.float64,
        mamba_ssm_cache_dtype=torch.float64,
    )
    full = _GDNAttention(config=cfg, dtype_policy=policy, prefix="full")
    for p in full.parameters():
        p.data.normal_(0, 0.05)
    full.A_log.data.fill_(-1)
    initial_conv = torch.randn(3, 3, 320, dtype=torch.float64) * 0.05
    initial_state = torch.randn(3, 48, 4, 4, dtype=torch.float64) * 0.05
    full.kv_cache = (initial_conv.clone(), initial_state.clone())
    layers = []
    cfg.ascend_expert_quantization = {"gdn_head_partition": "padded_compact"}
    suffixes = {
        "in_proj_qkv": "in_proj_qkv.weight",
        "in_proj_z": "in_proj_z.weight",
        "conv_weight": "conv1d.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "norm_weight": "norm.weight",
        "out_proj": "out_proj.weight",
    }
    for rank in range(6):
        layer = _GDNAttention(config=cfg, dtype_policy=policy, expert_sharding=(rank, 6), prefix=f"r{rank}")
        layer._tp_reduce = lambda value: value
        params = {f"x.attention.{n}": p for n, p in layer.named_parameters()}
        owner = SimpleNamespace(model=SimpleNamespace(config=cfg))
        for param, suffix in suffixes.items():
            AscendQwen4ExpForCausalLM._place_gdn_tensor(
                owner, params, f"x.linear_attn.{suffix}", getattr(full, param), rank, 6
            )
        for gate, source in (("b", full.in_proj_ba[:48]), ("a", full.in_proj_ba[48:])):
            AscendQwen4ExpForCausalLM._place_gdn_tensor(
                owner, params, f"x.linear_attn.in_proj_{gate}.weight", source, rank, 6
            )
        layer.kv_cache = (
            torch.full((3, 3, 60), float("nan"), dtype=torch.float64),
            torch.full((3, 9, 4, 4), float("nan"), dtype=torch.float64),
        )
        live_conv, live_state = layer._execution_caches()
        for slot in range(3):
            live_conv[slot].copy_(shard_gdn_tensor(initial_conv[slot].T, "conv", layer.execution_head_shard, 16, 48).T)
        start = layer.head_shard.value_start
        count = layer.num_v_heads
        live_state.copy_(initial_state[:, start : start + count])
        vconfig = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=cfg),
            parallel_config=SimpleNamespace(tensor_parallel_size=6),
            num_speculative_tokens=0,
        )
        assert layer.get_state_shape() == AscendQwen4ExpForCausalLM.get_gdn_mamba_state_shape_from_config(vconfig)
        assert layer.get_state_shape() == ((3, 60), (9, 4, 4))
        # Projection/gating GEMM batch geometry can differ at FP64 last bits;
        # the state comparison below uses a fixed 1e-12 CPU reference bound.
        layers.append(layer)
    assert [layer.num_v_heads for layer in layers] == [9, 9, 9, 9, 6, 6]
    assert sum(layer.conv_dim for layer in layers) == 320
    assert sum(layer.num_v_heads for layer in layers) == 48
    import vllm_ascend.models.qwen4_exp.model as model

    monkeypatch.setattr(model, "GDNAttentionMetadata", SimpleNamespace)
    monkeypatch.setattr(model, "is_forward_context_available", lambda: True)
    for phase, lengths in (("prefill", [2, 3]), ("decode", [1, 1])):
        starts = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        md = SimpleNamespace(
            num_actual_tokens=sum(lengths),
            spec_sequence_masks=None,
            num_prefills=2 if phase == "prefill" else 0,
            num_decodes=2 if phase == "decode" else 0,
            non_spec_state_indices_tensor=torch.tensor([0, 2], dtype=torch.int32),
            non_spec_query_start_loc=starts,
            has_initial_state=torch.tensor([True, not cold] if phase == "prefill" else [True, True]),
            query_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
        )
        monkeypatch.setattr(
            model,
            "get_forward_context",
            lambda md=md: SimpleNamespace(attn_metadata={layer.prefix: md for layer in [full, *layers]}),
        )
        inputs = torch.randn(sum(lengths), 8, dtype=torch.float64)
        expected = full(inputs, torch.zeros(sum(lengths)))
        partials = [layer(inputs, torch.zeros(sum(lengths))) for layer in layers]
        torch.testing.assert_close(sum(partials), expected, rtol=1e-12, atol=1e-12)
        for layer in layers:
            conv, state = layer._execution_caches()
            for slot in range(3):
                torch.testing.assert_close(
                    conv[slot],
                    shard_gdn_tensor(full.kv_cache[0][slot].T, "conv", layer.execution_head_shard, 16, 48).T,
                    rtol=1e-12,
                    atol=1e-12,
                )
            start = layer.head_shard.value_start
            torch.testing.assert_close(
                state, full.kv_cache[1][:, start : start + layer.num_v_heads], rtol=1e-12, atol=1e-12
            )
            if layer.num_v_heads < 9:
                assert torch.isnan(layer.kv_cache[1][:, layer.num_v_heads :]).all()
                assert torch.isnan(layer.kv_cache[0].flatten(1)[:, 3 * layer.conv_dim :]).all()


@pytest.mark.parametrize("prefill", [False, True])
def test_actual_native_gdn_handoff_keeps_strided_page_and_only_live_heads(monkeypatch, prefill):
    helper = ModuleType("vllm_ascend._310p.ops.fla.gdn_310")
    chunk_module = ModuleType("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
    allocated = gdn_head_shard(16, 48, 4, 4, 5, 6, "padded_compact")
    execution = gdn_execution_shard(allocated)
    cache = torch.full((3, 9, 4, 4), float("nan"))
    cache[:, :6] = 3
    conv = torch.zeros(3, 3, 60)
    live = gdn_execution_caches((conv, cache), allocated, execution)[1]
    layer = SimpleNamespace(
        _compact_gdn=True,
        head_shard=allocated,
        num_v_heads=6,
        kv_cache=(conv, cache),
        _execution_caches=lambda: (None, live),
    )
    ids = torch.tensor([0, 2], dtype=torch.int32)

    def recurrent(**kwargs):
        assert kwargs["state"].data_ptr() == cache.data_ptr()
        assert kwargs["state"].shape == (3, 6, 4, 4)
        assert kwargs["state"].stride(0) == cache.stride(0)
        kwargs["state"][ids] = kwargs["state"][ids] + 1
        return kwargs["v"]

    def chunk(**kwargs):
        assert kwargs["initial_state"].shape == (2, 6, 4, 4)
        assert kwargs["initial_state"].is_contiguous()
        return kwargs["v"], kwargs["initial_state"] + 1

    helper.npu_recurrent_gated_delta_rule_310 = Mock(side_effect=recurrent)
    helper._cached_recurrent_step_meta = lambda *args, **kwargs: None
    helper._cached_chunk_plan = lambda *args: None
    chunk_module.chunk_gated_delta_rule_310 = Mock(side_effect=chunk)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, chunk_module.__name__, chunk_module)
    md = SimpleNamespace(spec_sequence_masks=None, num_prefills=int(prefill))
    args = (
        torch.zeros(2, 2, 4),
        torch.zeros(2, 2, 4),
        torch.zeros(2, 6, 4),
        torch.zeros(2, 6),
        torch.zeros(2, 6),
        md,
        ids,
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.ones(2, dtype=torch.bool),
    )
    _GDNAttention._native_delta_rule(layer, *args)
    assert torch.all(cache[ids, :6] == 4) and torch.all(cache[1, :6] == 3)
    assert torch.isnan(cache[:, 6:]).all()
    if prefill:
        layer._gdn_state_io = object()
        with pytest.raises(ValueError, match="strided"):
            _GDNAttention._native_delta_rule(layer, *args)
        assert chunk_module.chunk_gated_delta_rule_310.call_count == 1
