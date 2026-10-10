# SPDX-License-Identifier: Apache-2.0
"""Execute actual GLM pipeline forward/factory without importing NPU plugins."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.deployment_profiles import profile
from tools.glm_perf.native_checkpoint import selected_weights

MODEL = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/model.py"


class Intermediates(dict):
    pass


def methods():
    tree = ast.parse(MODEL.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextModel")
    chosen = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("forward", "make_empty_intermediate_tensors")
    ]
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    scope = dict(
        torch=torch, IntermediateTensors=Intermediates, PPMissingLayer=type("Missing", (), {}), get_pp_group=lambda: pp
    )
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(MODEL), "exec"), scope)
    return scope, pp


class DeferredLayer:
    """Small deterministic mHC recurrence: post/comb materially affect output."""

    def __init__(self, ordinal, final=False):
        self.ordinal, self.final = ordinal, final

    def __call__(self, positions, hidden, residual, post, comb):
        if post is None:
            residual = hidden.float().unsqueeze(1).repeat(1, 4, 1)
        else:
            residual = torch.einsum("tij,tih->tjh", comb, residual) + post * hidden[:, None].float()
        hidden = (residual.mean(1) * (1 + 0.01 * self.ordinal)).half()
        post = torch.full((hidden.shape[0], 4, 1), 0.125 + self.ordinal * 0.001)
        comb = torch.eye(4).expand(hidden.shape[0], 4, 4) * 0.75
        if self.final:
            return (
                (torch.einsum("tij,tih->tjh", comb, residual) + post * hidden[:, None]).mean(1).half(),
                None,
                None,
                None,
            )
        return hidden, residual, post, comb


@pytest.mark.parametrize("tokens", [1, 17, 640])
def test_three_stage_forward_matches_unsplit_and_receive_buffers(tokens):
    scope, pp = methods()
    x = torch.linspace(-1, 1, tokens * 8).reshape(tokens, 8).half()
    config = SimpleNamespace(hidden_size=8, mhc=True, mhc_num_residual_streams=4)
    layers = [DeferredLayer(i, final=i == 8) for i in range(9)]
    model = SimpleNamespace(
        config=config,
        is_sequence_parallel=False,
        norm=lambda value: value,
        embed_input_ids=lambda ids: x,
        _active_layers=layers,
    )
    positions = torch.arange(tokens)
    expected = scope["forward"](model, positions, positions, None)
    incoming = None
    for stage in range(3):
        pp.is_first_rank, pp.is_last_rank = stage == 0, stage == 2
        model._active_layers = layers[stage * 3 : (stage + 1) * 3]
        output = scope["forward"](model, positions if stage == 0 else None, positions, incoming)
        if stage < 2:
            incoming = scope["make_empty_intermediate_tensors"](model, tokens, torch.float16, "cpu")
            assert set(incoming) == set(output) == {"hidden_states", "residual", "post", "comb"}
            for key in output:
                assert incoming[key].shape == output[key].shape and incoming[key].dtype == output[key].dtype
                incoming[key].copy_(output[key])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_non_mhc_intermediate_layout_retains_residual():
    scope, _ = methods()
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=8, mhc=False))
    tensors = scope["make_empty_intermediate_tensors"](model, 3, torch.float16, "cpu")
    assert set(tensors) == {"hidden_states", "residual"}
    assert all(value.shape == (3, 8) and value.dtype == torch.float16 for value in tensors.values())


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_checkpoint_selection_is_stage_local_and_expert_local(stage):
    weights = {}
    for layer in range(45):
        weights[f"model.language_model.layers.{layer}.self_attn.weight"] = "dense"
        for expert in (0, 143, 144, 287):
            weights[f"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj_codes"] = "native"
    weights.update(
        {
            "model.language_model.embed_tokens.weight": "dense",
            "model.language_model.norm.weight": "dense",
            "lm_head.weight": "dense",
            "model.language_model.layers.45.shared_head.weight": "draft",
        }
    )
    first, stop = stage * 15, (stage + 1) * 15
    keys = tuple(f"layers.{layer}" for layer in range(first, stop))
    chosen = list(selected_weights(weights, (144, 288), keys, pipeline_partition=(first, stop, stage == 0, stage == 2)))
    assert len([n for n in chosen if ".self_attn." in n]) == 15
    assert len([n for n in chosen if ".experts." in n]) == 30
    assert ("model.language_model.embed_tokens.weight" in chosen) == (stage == 0)
    assert ("lm_head.weight" in chosen) == (stage == 2)
    assert ("model.language_model.norm.weight" in chosen) == (stage == 2)
    assert not any("layers.45." in n for n in chosen)


def test_profiles_preserve_w2_and_use_w3_for_three_cards():
    two, three = profile(2), profile(3)
    assert (two.chips, two.tensor_parallel_size, two.pipeline_parallel_size, two.mtp_tokens) == (4, 4, 1, 1)
    assert two.default_expert_bits == 2 and two.w3_layers
    assert (three.chips, three.tensor_parallel_size, three.pipeline_parallel_size, three.mtp_tokens) == (6, 2, 3, 0)
    assert three.default_expert_bits == 3 and three.expert_partition_size == 2
    for candidate in (two, three):
        command = candidate.launch_args("/models/test")
        assert "--enable-prompt-tokens-details" in command
        assert command[command.index("--max-num-batched-tokens") + 1] == "640"
    assert "--enforce-eager" in three.launch_args("/models/test")
    assert "--speculative-config" not in three.launch_args("/models/test")


@pytest.mark.parametrize("cards", [1, 4, True, "3"])
def test_profiles_reject_ambiguous_card_counts(cards):
    with pytest.raises(ValueError):
        profile(cards)


def test_w2_adapter_hooks_visit_only_stage_owned_layers():
    path = MODEL.parent.parent / "glm5next_w2/model.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_iter_model_layers")
    scope = {"Any": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    placeholders = [object()] * 15
    active = [SimpleNamespace(layer_idx=i) for i in range(15, 30)]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=placeholders + active + placeholders, start_layer=15, end_layer=30)
    )
    assert scope["_iter_model_layers"](model) == active


def test_hub_resolution_fetches_native_binaries_and_reuses_local_directory(tmp_path):
    path = MODEL.parents[2] / "model_loader/glm_native_int4.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_native_folder")
    calls = []
    scope = {"Path": Path, "snapshot_download": lambda **kw: calls.append(kw) or str(tmp_path)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    resolve = scope["resolve_native_folder"]
    assert resolve(str(tmp_path)) == tmp_path and not calls
    assert resolve("opensensor/glm-candidate", "revision-pin", "/tmp/cache") == tmp_path
    assert calls[0]["revision"] == "revision-pin"
    assert "native-kernels/**/*" in calls[0]["allow_patterns"]
    with pytest.raises(FileNotFoundError):
        resolve(str(tmp_path / "missing"))


def test_profile_admission_rejects_tp6_mtp_and_precision_mismatches():
    from tools.glm_perf.deployment_profiles import validate_profile

    assert (
        validate_profile(
            "three-card-w3",
            tensor_parallel_size=2,
            pipeline_parallel_size=3,
            mtp_tokens=0,
            layer_bits={"3": 4, "33": 3},
        ).cards
        == 3
    )
    with pytest.raises(ValueError, match="topology"):
        validate_profile("three-card-w3", tensor_parallel_size=6, pipeline_parallel_size=1, mtp_tokens=0)
    with pytest.raises(ValueError, match="topology"):
        validate_profile("three-card-w3", tensor_parallel_size=2, pipeline_parallel_size=3, mtp_tokens=1)
    with pytest.raises(ValueError, match="precision"):
        validate_profile(
            "three-card-w3", tensor_parallel_size=2, pipeline_parallel_size=3, mtp_tokens=0, layer_bits={"33": 2}
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_profile("bad-profile", tensor_parallel_size=2, pipeline_parallel_size=3, mtp_tokens=0)


def test_parameter_ledger_uses_stage_ownership_and_excludes_disabled_draft():
    from tools.glm_perf.card_memory_plan import parameter_ledger

    config = {"num_hidden_layers": 3, "n_routed_experts": 4}
    tensors = {
        "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 8]},
        "lm_head.weight": {"dtype": "BF16", "shape": [4, 8]},
        "model.language_model.layers.3.extra.weight": {"dtype": "BF16", "shape": [2, 8]},
    }
    for layer in range(3):
        for expert in range(4):
            tensors[f"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"] = {
                "dtype": "F8_E4M3",
                "shape": [256, 256],
            }
    report = parameter_ledger(config, dict.fromkeys(tensors, "source"), {"source": tensors}, profile(3))
    assert report["hardware_admission"] is False
    assert [s["layers"] for s in report["stages"]] == [[0, 1], [1, 2], [2, 3]]
    assert [s["dense_parameter_fp32_replicated_upper_bound_bytes"] for s in report["stages"]] == [128, 0, 128]
    assert all(
        s["routed_codes_and_fp16_scales_bytes"] == (256 * 256 * 3 // 8 + 8 * 8 * 2) * 2 for s in report["stages"]
    )


def test_export_cli_does_not_drop_card_binding(tmp_path, monkeypatch, capsys):
    from tools.glm_perf import export_native_cpu

    calls = []
    monkeypatch.setattr(export_native_cpu, "export", lambda *args, **kw: calls.append(kw) or {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "export",
            "--source",
            "source",
            "--output",
            str(tmp_path),
            "--bundle",
            "bundle",
            "--world-size",
            "2",
            "--cards",
            "3",
        ],
    )
    export_native_cpu.main()
    assert calls == [{"world_size": 2, "cards": 3, "resume": False}]


def test_split_native_keeps_decode_through_16_tokens_and_rejects_scale_mismatch():
    from tools.glm_perf.native_checkpoint import SplitNativeMoE

    class Backend:
        required_weight_scale_layout = "fp16_storage_fp32_compute_v1"

        def __init__(self, offset):
            self.offset = offset

        def __call__(self, x):
            return x + self.offset

    prefill, decode = Backend(10), Backend(20)
    combined = SplitNativeMoE(prefill, decode)
    for rows in (1, 8, 16):
        assert torch.equal(combined(torch.zeros(rows, 2)), torch.full((rows, 2), 20.0))
    for rows in (17, 640):
        assert torch.equal(combined(torch.zeros(rows, 2)), torch.full((rows, 2), 10.0))
    decode.required_weight_scale_layout = "different"
    with pytest.raises(ValueError, match="scale contracts"):
        SplitNativeMoE(prefill, decode)


def test_actual_decoder_mhc_forward_survives_two_pipeline_boundaries():
    from types import MethodType

    from vllm_ascend.models.glm5next.ops.mhc_ops import hc_contract, hc_expand
    from vllm_ascend.models.glm5next_w2.moe import hc_post, hc_pre

    tree = ast.parse(MODEL.read_text())
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm5NextDecoderLayer")
    method = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    decoder_scope = dict(torch=torch, hc_expand=hc_expand, hc_contract=hc_contract)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODEL), "exec"), decoder_scope)
    scope, pp = methods()
    generator = torch.Generator().manual_seed(19)
    tokens, width, streams, depth = 17, 8, 4, 9

    class Layer:
        def __init__(self, index):
            self.layer_idx, self.num_hidden_layers = index, depth
            self.mhc, self.is_mtp_layer, self.is_sequence_parallel, self._mlp_is_moe = True, False, False, False
            self.n = streams
            self.hc_attn_fn = torch.randn(24, width * streams, generator=generator) * 0.02
            self.hc_ffn_fn = torch.randn(24, width * streams, generator=generator) * 0.02
            self.hc_attn_scale = self.hc_ffn_scale = torch.tensor([0.1, 0.2, 0.1])
            self.hc_attn_base = self.hc_ffn_base = torch.linspace(-0.2, 0.2, 24)
            self.input_layernorm = self.post_attention_layernorm = SimpleNamespace(
                weight=torch.ones(width).half(), variance_epsilon=1e-5
            )
            self.self_attn = lambda hidden_states, positions: (hidden_states.float() * 0.7).half()
            self.mlp = lambda x: torch.tanh(x.float()).half()
            self.forward = MethodType(decoder_scope["forward"], self)

        def __call__(self, *args):
            return self.forward(*args)

        def hc_pre(self, state, fn, scale, base, norm_weight, norm_eps):
            post, comb, x = hc_pre(
                state,
                fn,
                scale,
                base,
                rms_eps=norm_eps,
                hc_pre_eps=1e-5,
                hc_sinkhorn_eps=1e-5,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=3,
            )
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
            return post, comb, (x * norm_weight.float()).half()

        def hc_post(self, x, residual, post, comb):
            return hc_post(x, residual, post, comb)

        def hc_fused_post_pre(self, x, residual, post, comb, *args, **kw):
            state = self.hc_post(x, residual, post, comb)
            post, comb, x = self.hc_pre(state, *args, **kw)
            return state, post, comb, x

    layers = [Layer(i) for i in range(depth)]
    x = torch.randn(tokens, width, generator=generator).half()
    model = SimpleNamespace(
        config=SimpleNamespace(mhc=True, mhc_num_residual_streams=streams, hidden_size=width),
        is_sequence_parallel=False,
        embed_input_ids=lambda ids: x,
        _active_layers=layers,
        norm=SimpleNamespace(weight=torch.ones(width).half(), variance_epsilon=1e-5),
    )
    positions = torch.arange(tokens)
    expected = scope["forward"](model, positions, positions, None)
    incoming = None
    for stage in range(3):
        pp.is_first_rank, pp.is_last_rank = stage == 0, stage == 2
        model._active_layers = layers[stage * 3 : (stage + 1) * 3]
        actual = scope["forward"](model, positions if stage == 0 else None, positions, incoming)
        if not pp.is_last_rank:
            incoming = scope["make_empty_intermediate_tensors"](model, tokens, torch.float16, "cpu")
            for key, value in actual.items():
                assert value.dtype == incoming[key].dtype
                incoming[key].copy_(value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
