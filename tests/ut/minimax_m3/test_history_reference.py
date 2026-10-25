# SPDX-License-Identifier: Apache-2.0
"""Gates for the independent full-history oracle, without importing serving ops."""

import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from tools.minimax_m3.history_reference import SourceWeights, run, sparse_attention


def test_full_budget_matches_dense_causal_attention():
    generator = torch.Generator().manual_seed(173)
    query = torch.randn(9, 4, 8, generator=generator).half()
    key, value = [torch.randn(9, 2, 8, generator=generator).half() for _ in range(2)]
    iq = torch.randn(9, 2, 8, generator=generator).half()
    ik = torch.randn(9, 8, generator=generator).half()
    config = {"sparse_block_size": 4, "sparse_topk_blocks": 3, "sparse_local_block": 1, "sparse_init_block": 0}
    actual, _ = sparse_attention(query, key, value, iq, ik, config, tile=3)
    expected = torch.empty_like(query)
    for position in range(9):
        for head in range(4):
            scores = key[: position + 1, head // 2].float() @ query[position, head].float() / 8**0.5
            expected[position, head] = scores.softmax(0) @ value[: position + 1, head // 2].float()
    torch.testing.assert_close(actual, expected.flatten(1), rtol=0, atol=0)


def test_sparse_heads_select_different_history_and_exclude_future():
    # Final block is always selected. Head 0 additionally selects block 0;
    # head 1 selects block 1. Zero queries make attention a visible-token mean.
    query = torch.zeros(6, 2, 1).half()
    key = torch.zeros(6, 2, 1).half()
    value = torch.arange(6).half()[:, None, None].expand(6, 2, 1)
    iq = torch.tensor([1.0, -1.0]).half()[None, :, None].expand(6, 2, 1)
    ik = torch.tensor([10.0, 10.0, -10.0, -10.0, 0.0, 0.0]).half()[:, None]
    config = {"sparse_block_size": 2, "sparse_topk_blocks": 2, "sparse_local_block": 1, "sparse_init_block": 0}
    actual, selected = sparse_attention(query, key, value, iq, ik, config, tile=2)
    assert selected[-1].sort().values.tolist() == [[0, 2], [1, 2]]
    torch.testing.assert_close(actual[-1], torch.tensor([2.5, 3.5]).half())
    # Adding an extreme future value cannot affect the prefix.
    changed = value.clone()
    changed[-1] = 60000
    modified, _ = sparse_attention(query, key, changed, iq, ik, config, tile=3)
    torch.testing.assert_close(modified[:-1], actual[:-1], rtol=0, atol=0)
    appended, last_selected = sparse_attention(query[-1:], key, value, iq[-1:], ik, config)
    torch.testing.assert_close(appended, actual[-1:], rtol=0, atol=0)
    torch.testing.assert_close(last_selected, selected[-1:], rtol=0, atol=0)


def test_source_weight_decode_known_codes_and_exponents(tmp_path):
    tensors = {
        "expert.weight_packed": torch.tensor([[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F]], dtype=torch.uint8),
        "expert.weight_scale": torch.tensor([[2.0]]).to(torch.float8_e4m3fn),
        "expert.weight_global_scale": torch.tensor(4.0),
        "dense.weight": torch.ones(1, 64).to(torch.float8_e4m3fn),
        "dense.weight_scale_inv": torch.tensor([[126, 128]], dtype=torch.uint8),
    }
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "weights.safetensors" for k in tensors}})
    )
    source = SourceWeights(tmp_path, "cpu")
    expected = torch.tensor([[0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6, 0]]).half() / 2
    torch.testing.assert_close(source.weight("expert.weight"), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        source.weight("dense.weight"), torch.tensor([[0.5] * 32 + [2.0] * 32]).half(), rtol=0, atol=0
    )


def test_whole_history_resume_matches_uninterrupted_and_rejects_changed_identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    generator = torch.Generator().manual_seed(42)
    tensors = {}

    def weight(name, shape, scale=0.05):
        tensors[name] = torch.randn(*shape, generator=generator) * scale

    config = dict(
        num_hidden_layers=2,
        vocab_size=16,
        head_dim=8,
        rotary_dim=4,
        rope_theta=10000,
        rms_norm_eps=1e-6,
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
        moe_layer_freq=[0, 1],
        num_local_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=1,
        sparse_attention_config=dict(
            sparse_block_size=4, sparse_topk_blocks=2, sparse_local_block=1, sparse_init_block=0
        ),
    )
    weight("language_model.model.embed_tokens.weight", (16, 32))
    weight("language_model.model.norm.weight", (32,))
    weight("language_model.lm_head.weight", (16, 32))
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weight(prefix + name + ".weight", (32,))
        for name, heads in (("q", 4), ("k", 2), ("v", 2), ("index_q", 2), ("index_k", 1)):
            weight(prefix + f"self_attn.{name}_proj.weight", (heads * 8, 32))
            if name != "v":
                weight(prefix + f"self_attn.{name}_norm.weight", (8,))
        weight(prefix + "self_attn.o_proj.weight", (32, 32))
        if layer == 0:
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight(prefix + f"mlp.{name}.weight", (32, 32))
        else:
            prefix += "block_sparse_moe."
            weight(prefix + "gate.weight", (4, 32))
            weight(prefix + "e_score_correction_bias", (4,))
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight(prefix + f"shared_experts.{name}.weight", (32, 32))
            for expert in range(4):
                for name in ("w1", "w2", "w3"):
                    weight(prefix + f"experts.{expert}.{name}.weight", (32, 32))
    save_file(tensors, model / "weights.safetensors")
    (model / "config.json").write_text(json.dumps({"text_config": config}))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"tokens": list(range(9))}))
    args = Namespace(
        model=model,
        tokens=tokens,
        output=tmp_path / "full",
        observed=None,
        device="cpu",
        tp=2,
        layers=2,
        resume=False,
        save_kv=True,
        activation_format="fp16",
    )
    with torch.inference_mode():
        run(args)
        expected = torch.load(args.output / "logits.pt", weights_only=True)
        args.output = tmp_path / "resumed"
        args.layers = 1
        run(args)
        args.layers, args.resume = 2, True
        run(args)
        actual = torch.load(args.output / "logits.pt", weights_only=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(json.loads((args.output / "layers.json").read_text())) == 2
        assert torch.load(args.output / "kv-01.pt", weights_only=True)["k"].shape == (9, 2, 8)
        args.tp = 1
        with pytest.raises(ValueError, match="identity"):
            run(args)
        args.tp, args.resume = 2, False
        tokens.write_text(json.dumps({"tokens": list(range(8))}))
        args.output = tmp_path / "prefix"
        run(args)
        args.prefix_cache = args.output
        args.output = tmp_path / "decode"
        tokens.write_text(json.dumps({"tokens": list(range(9))}))
        run(args)
        appended = torch.load(args.output / "logits.pt", weights_only=True)
        # Single-row GEMV and multi-row GEMM can land on opposite FP16
        # rounding boundaries. Require sub-milliscale logit agreement.
        torch.testing.assert_close(appended, expected, rtol=0, atol=torch.finfo(torch.float16).eps / 2)
        for layer in range(2):
            before = torch.load(args.prefix_cache / f"kv-{layer:02d}.pt", weights_only=True)
            after = torch.load(args.output / f"kv-{layer:02d}.pt", weights_only=True)
            for name in before:
                torch.testing.assert_close(after[name][:-1], before[name], rtol=0, atol=0)
        args.tp = 1
        with pytest.raises(ValueError, match="Prefix cache"):
            run(args)
        args.tp, args.prefix_cache, args.tail_logits = 2, None, 2
        tokens.write_text(json.dumps({"tokens": list(range(10))}))
        args.output = tmp_path / "full-ten"
        run(args)
        expected_suffix = torch.load(args.output / "logits.pt", weights_only=True)
        args.prefix_cache, args.output = tmp_path / "prefix", tmp_path / "suffix"
        run(args)
        actual_suffix = torch.load(args.output / "logits.pt", weights_only=True)
        assert actual_suffix.shape == (2, 16)
        torch.testing.assert_close(actual_suffix, expected_suffix, rtol=0, atol=torch.finfo(torch.float16).eps / 2)
        args.resume, args.prefix_cache = True, None
        with pytest.raises(ValueError, match="identity"):
            run(args)
