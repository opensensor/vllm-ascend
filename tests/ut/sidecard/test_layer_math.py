# SPDX-License-Identifier: Apache-2.0
"""Independent arithmetic, causal cache, and serving admission regressions."""

import pytest
import torch

from tools.sidecard.layer_math import TextLayer, mhc_post, mhc_pre, recurrent_kda, rms_norm, validate_config
from tools.sidecard.layer_server import request_tokens
from tools.sidecard.layer_weights import private_tensor, shard_dense_tensor


def supported_config():
    return {
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "hc_mult": 4,
        "qk_rope_head_dim": 0,
        "n_group": 1,
        "scoring_func": "sigmoid",
        "index_topk": 2048,
        "index_kpool": 4,
        "swiglu_limit": 10.0,
    }


@pytest.mark.parametrize("limit", [1, 2048])
def test_exact_all_candidate_admission(limit):
    validate_config(supported_config(), limit)


@pytest.mark.parametrize("limit", [0, 1048577, True, 1.5])
def test_reject_context_outside_trained_range(limit):
    with pytest.raises(ValueError):
        validate_config(supported_config(), limit)


@pytest.mark.parametrize("field,value", [("qk_rope_head_dim", 64), ("n_group", 8), ("hc_mult", 2)])
def test_reject_unsupported_checkpoint_semantics(field, value):
    config = supported_config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config, 2048)


def test_mhc_stream_orientation_and_residual_range():
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) * 10000
    comb = torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]]).float()
    x = torch.arange(6).half()[None, :]
    post = torch.tensor([1, 2, 3, 4]).float().reshape(1, 4, 1)
    expected = (torch.einsum("tij,tih->tjh", comb, residual) + post * x[:, None, :]).bfloat16().float()
    actual = mhc_post(x, residual, post, comb, torch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.max() > torch.finfo(torch.float16).max
    assert torch.isfinite(actual).all()


def test_mhc_pre_against_double_precision_scalar_reference():
    torch.manual_seed(19)
    residual = torch.randn(2, 4, 8)
    weights = {
        "hc_attn_fn": torch.randn(24, 32),
        "hc_attn_scale": torch.tensor([0.3, 0.2, 0.1]),
        "hc_attn_base": torch.randn(24),
        "norm": torch.randn(8).half(),
    }
    config = {"rms_norm_eps": 1e-5, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20}
    post, comb, value = mhc_pre(residual, weights, "attn", "norm", config, torch)
    flat = residual.double().flatten(1)
    mixes = flat @ weights["hc_attn_fn"].double().t()
    mixes *= (flat.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    pre = torch.sigmoid(mixes[:, :4] * 0.3 + weights["hc_attn_base"][:4].double()) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[:, 4:8] * 0.2 + weights["hc_attn_base"][4:8].double())
    logits = (mixes[:, 8:] * 0.1 + weights["hc_attn_base"][8:].double()).reshape(-1, 4, 4)
    expected_comb = logits.softmax(-1) + 1e-6
    expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        expected_comb /= expected_comb.sum(-1, keepdim=True) + 1e-6
        expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    expected_value = (pre[..., None] * residual.double()).sum(1).bfloat16().float()
    expected_value = (
        expected_value * (expected_value.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * weights["norm"].float()
    ).half()
    torch.testing.assert_close(post, expected_post[..., None].bfloat16().float(), rtol=0, atol=0)
    torch.testing.assert_close(comb, expected_comb.bfloat16().float(), rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)


def test_recurrent_kda_matches_independent_outer_product_reference_and_chunks():
    torch.manual_seed(71)
    q, k, v = [torch.randn(5, 2, 8).half() for _ in range(3)]
    gate = -torch.rand(5, 2, 8)
    beta = torch.rand(5, 2)
    initial = torch.randn(2, 8, 8)
    state = initial.clone()
    actual = recurrent_kda(q, k, v, gate, beta, state, torch)
    reference = initial.double()
    expected = []
    for row in range(5):
        per_head = []
        for head in range(2):
            qr, kr, vr = q[row, head].double(), k[row, head].double(), v[row, head].double()
            qr /= (qr.square().sum() + 1e-6).sqrt() * 8**0.5
            kr /= (kr.square().sum() + 1e-6).sqrt()
            reference[head] *= gate[row, head].double().exp()[None, :]
            innovation = (vr - reference[head] @ kr) * beta[row, head].double()
            reference[head] += torch.outer(innovation, kr)
            per_head.append(reference[head] @ qr)
        expected.append(torch.stack(per_head))
    torch.testing.assert_close(actual, torch.stack(expected).half(), rtol=0.001, atol=0.0005)
    torch.testing.assert_close(state, reference.float(), rtol=1e-5, atol=1e-6)
    chunk_state = initial.clone()
    chunked = torch.cat(
        [
            recurrent_kda(q[:2], k[:2], v[:2], gate[:2], beta[:2], chunk_state, torch),
            recurrent_kda(q[2:], k[2:], v[2:], gate[2:], beta[2:], chunk_state, torch),
        ]
    )
    torch.testing.assert_close(actual, chunked, rtol=0, atol=0)
    torch.testing.assert_close(state, chunk_state, rtol=0, atol=0)


def dsa_layer():
    torch.manual_seed(13)
    layer = object.__new__(TextLayer)
    layer.torch, layer.device, layer.length, layer.max_context = torch, "cpu", 0, 12
    layer.config = {
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
    }
    layer.weights = {
        "self_attn.q_a_proj.weight": torch.randn(6, 8).half(),
        "self_attn.q_a_layernorm.weight": torch.ones(6).half(),
        "self_attn.q_b_proj.weight": torch.randn(8, 6).half(),
        "self_attn.kv_a_proj_with_mqa.weight": torch.randn(8, 8).half(),
        "self_attn.kv_a_layernorm.weight": torch.ones(8).half(),
        "self_attn.kv_b_proj.weight": torch.randn(16, 8).half(),
        "self_attn.o_proj.weight": torch.randn(8, 8).half(),
    }
    layer.cache = torch.empty(12, 8).half()
    return layer


def test_dsa_chunking_is_causal_and_preserves_owner_cache():
    layer = dsa_layer()
    other = dsa_layer()
    x = torch.randn(5, 8).half()
    complete = layer.dsa(x)
    chunked = torch.cat([other.dsa(x[:2]), other.dsa(x[2:])])
    torch.testing.assert_close(complete, chunked, rtol=0, atol=0)
    torch.testing.assert_close(layer.cache[:5], other.cache[:5], rtol=0, atol=0)
    changed = dsa_layer()
    x[3:] += 50
    torch.testing.assert_close(complete[:3], changed.dsa(x)[:3], rtol=0, atol=0)
    with pytest.raises(ValueError, match="capacity"):
        layer.dsa(torch.zeros(8, 8).half())


class Tokenizer:
    def encode(self, text):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(messages[0]["content"])


def test_context_includes_output_reservation():
    tokenizer = Tokenizer()
    assert request_tokens({"prompt": "abc", "max_tokens": 5}, tokenizer, 8, chat=False) == ([0, 1, 2], 5)
    with pytest.raises(ValueError, match="fit"):
        request_tokens({"prompt": "abc", "max_tokens": 6}, tokenizer, 8, chat=False)


@pytest.mark.parametrize(
    "extra", [{"temperature": 1}, {"n": 2}, {"tools": [{}]}, {"stop": "END"}, {"model": "other"}, {"max_tokens": True}]
)
def test_unsupported_serving_features_are_explicitly_rejected(extra):
    with pytest.raises(ValueError):
        request_tokens({"prompt": "abc", "max_tokens": 1, **extra}, Tokenizer(), 2048, chat=False)


def test_tp2_keeps_complete_kv_head_blocks_and_replicated_gate_projections():
    weight = torch.arange(120).reshape(20, 6)
    shards = [shard_dense_tensor("self_attn.kv_b_proj.weight", weight, {}, rank) for rank in range(2)]
    torch.testing.assert_close(torch.cat(shards), weight)
    assert torch.equal(shards[0], weight[:10])
    assert torch.equal(shards[1], weight[10:])
    for name in ("self_attn.q_a_proj.weight", "self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "hc_attn_fn"):
        assert shard_dense_tensor(name, weight, {}, 0) is weight
        assert shard_dense_tensor(name, weight, {}, 1) is weight


def test_shared_mlp_tp2_contributions_sum_to_complete_projection():
    torch.manual_seed(86)
    hidden = torch.randn(3, 8)
    gate, up, down = torch.randn(6, 8), torch.randn(6, 8), torch.randn(8, 6)
    activation = torch.nn.functional.silu(hidden @ gate.t()) * (hidden @ up.t())
    complete = activation @ down.t()
    parts = []
    for rank in range(2):
        local_gate = shard_dense_tensor("mlp.shared_experts.gate_proj.weight", gate, {}, rank)
        local_up = shard_dense_tensor("mlp.shared_experts.up_proj.weight", up, {}, rank)
        local_down = shard_dense_tensor("mlp.shared_experts.down_proj.weight", down, {}, rank)
        local_activation = torch.nn.functional.silu(hidden @ local_gate.t()) * (hidden @ local_up.t())
        parts.append(local_activation @ local_down.t())
    torch.testing.assert_close(parts[0] + parts[1], complete)


def test_chat_accepts_tokenizer_batchencoding_result():
    class Backend(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    tokens, limit = request_tokens(
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}, Backend(), 8, chat=True
    )
    assert tokens == [1, 2, 3]
    assert limit == 2


def test_final_contraction_preserves_residual_range_until_norm(monkeypatch):
    layer = object.__new__(TextLayer)
    layer.torch, layer.device, layer.index, layer.linear, layer.tp_rank = torch, "cpu", 44, False, None
    layer.native_sinkhorn = None
    layer.config = {"num_hidden_layers": 45, "routed_scaling_factor": 2.5}
    layer.weights = {"mlp.gate.weight": torch.zeros(288, 6), "mlp.gate.e_score_correction_bias": torch.zeros(288)}
    hidden = torch.zeros(1, 6).half()
    residual = torch.full((1, 4, 6), 100000.0)
    post, comb = torch.zeros(1, 4, 1), torch.eye(4)[None]
    monkeypatch.setattr("tools.sidecard.layer_math.mhc_pre", lambda *args: (post, comb, hidden))
    layer.dsa = lambda x: x
    layer.dense_mlp = lambda x, prefix: x
    layer.experts = lambda index, x, ids, weights: x.float()
    contracted, *deferred = layer.forward(hidden, residual, post, comb)
    assert contracted.dtype == torch.float32
    assert contracted.min() > torch.finfo(torch.float16).max
    assert deferred == [None, None, None]
    normalized = rms_norm(contracted, torch.ones(6).half(), 1e-5, torch)
    torch.testing.assert_close(normalized, torch.ones(1, 6).half(), rtol=0, atol=0)


def test_private_tensor_never_aliases_checkpoint_file(tmp_path):
    original = torch.arange(12).half()
    raw = original.numpy().tobytes()
    path = tmp_path / "weights.bin"
    path.write_bytes(b"header!!" + raw)
    descriptor = {"dtype": "F16", "shape": [3, 4], "data_offsets": [0, len(raw)]}
    with path.open("rb") as handle:
        tensor = private_tensor(handle, descriptor, 8, torch)
        tensor.add_(100)
        reread = private_tensor(handle, descriptor, 8, torch)
    torch.testing.assert_close(reread.flatten(), original, rtol=0, atol=0)
    assert path.read_bytes() == b"header!!" + raw
    path.write_bytes(b"header!!")
    with path.open("rb") as handle, pytest.raises(ValueError, match="truncated"):
        private_tensor(handle, descriptor, 8, torch)
