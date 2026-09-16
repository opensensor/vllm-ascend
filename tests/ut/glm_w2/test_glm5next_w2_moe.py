# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3-Flash W2 routed-MoE + weight mapping (plan G6).

Everything runs host-side with NO NPU and NO Triton. The E1.3 W2 fused-MoE
*kernel* (``AscendW2DynamicFusedMoEMethod310``) needs ``torch_npu`` and is
already covered by the DeepSeek E1.2/E1.3 suite, which this module REUSES
verbatim (``eager_moe_combine``, the packed-W2 format, the active-expert
unpack). So these tests exercise the GLM-specific seams:

* the host router ``glm_route_topk`` (sigmoid ``noaux_tc`` top-8 of 288,
  ``norm_topk_prob``, deferred ``routed_scaling_factor``, and the softmax /
  no-bias / single-group reduction to the shared DeepSeek W2 router),
* the eager ``routed * 2.5 + shared`` combine,
* the multi-head hyper-connection host ops (``hc_pre`` / ``hc_post`` /
  ``hc_expand`` / ``hc_contract``),
* ``Glm5NextW2MoE`` construction / ``from_config`` / an injected-method forward,
* the ``weight_mapping`` classification + expert fusion + coverage validation
  (against a tiny synthetic geometry and, when mounted, the real artifact).

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/glm_w2/test_glm5next_w2_moe.py
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from vllm_ascend.models.glm5next_w2 import moe as M
from vllm_ascend.models.glm5next_w2 import weight_mapping as WM

_REAL_ARTIFACT = Path("/run/media/matteius/20TB-drive/models/GLM-5.3-Flash-W2-310p")

# Tiny synthetic geometry for fast, readable coverage/shape tests. Keeps the GLM
# structure: first_k_dense_replace dense layers, then MoE, plus one MTP layer.
_TINY_GEOMETRY = {
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_hidden_layers": 5,
    "first_k_dense_replace": 2,
    "num_nextn_predict_layers": 1,
    "mtp_layer_index": 5,
}

# Real GLM geometry (matches the converted artifact + manifest).
_REAL_GEOMETRY = {
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 288,
    "num_hidden_layers": 45,
    "first_k_dense_replace": 3,
    "num_nextn_predict_layers": 1,
    "mtp_layer_index": 45,
}


# ---------------------------------------------------------------------------
# Router: glm_route_topk
# ---------------------------------------------------------------------------


def test_route_topk_shapes_and_norm():
    torch.manual_seed(0)
    tokens, num_experts, top_k = 7, 288, 8
    logits = torch.randn(tokens, num_experts)
    ids, weights = M.glm_route_topk(logits, top_k, scoring_func="sigmoid", renormalize=True)
    assert ids.shape == (tokens, top_k)
    assert weights.shape == (tokens, top_k)
    assert ids.dtype == torch.int64
    # top-k ids are unique per token and in range.
    for row in ids:
        assert len(set(row.tolist())) == top_k
        assert int(row.min()) >= 0 and int(row.max()) < num_experts
    # norm_topk_prob -> weights sum to 1 per token.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(tokens, dtype=weights.dtype), atol=1e-9)


def test_route_topk_no_renorm_keeps_raw_scores():
    torch.manual_seed(1)
    logits = torch.randn(3, 16)
    ids, weights = M.glm_route_topk(logits, 4, scoring_func="sigmoid", renormalize=False)
    expected = torch.sigmoid(logits.double()).gather(1, ids)
    assert torch.allclose(weights, expected)
    # Un-normalized sigmoid weights do NOT generally sum to 1.
    assert not torch.allclose(weights.sum(dim=-1), torch.ones(3, dtype=weights.dtype))


def test_route_topk_selection_bias_shifts_choice_not_weight():
    # e_score_correction_bias (noaux_tc) changes WHICH experts are selected, but
    # the returned weight is the *un-biased* score at the chosen expert.
    logits = torch.zeros(1, 4)
    logits[0, 0] = 0.1  # expert 0 has the highest raw score
    bias = torch.tensor([0.0, 0.0, 0.0, 10.0])  # bias pushes expert 3 into the top-1
    ids, weights = M.glm_route_topk(logits, 1, scoring_func="sigmoid", renormalize=False, e_score_correction_bias=bias)
    assert int(ids[0, 0]) == 3  # biased selection picks expert 3
    # weight is the raw sigmoid(0.0) = 0.5 at expert 3, NOT sigmoid(0.0)+10.
    assert torch.allclose(weights[0, 0], torch.tensor(0.5, dtype=weights.dtype))


def test_route_topk_softmax_no_bias_matches_deepseek_anchor():
    # The GLM router in the softmax / no-bias / n_group=1 config must reduce
    # bit-for-bit to the shared DeepSeek E1.2 W2 router (the parity anchor).
    from vllm_ascend.models.deepseek_v41.w2_unpack import route_topk_w2

    torch.manual_seed(2)
    logits = torch.randn(5, 32)
    ids_g, w_g = M.glm_route_topk(logits, 6, scoring_func="softmax", renormalize=True, n_group=1)
    ids_d, w_d = route_topk_w2(logits, 6, renormalize=True)
    assert torch.equal(ids_g.to(ids_d.dtype), ids_d)
    assert torch.allclose(w_g.to(w_d.dtype), w_d, atol=1e-6)


def test_route_topk_rejects_unknown_scoring():
    with pytest.raises(ValueError):
        M.glm_route_topk(torch.randn(1, 4), 2, scoring_func="relu")


# ---------------------------------------------------------------------------
# Eager combine: routed * routed_scaling_factor + shared
# ---------------------------------------------------------------------------


def test_eager_combine_scales_routed_and_adds_shared():
    routed = torch.ones(3, 8)
    shared = torch.full((3, 8), 2.0)
    out = M.eager_moe_combine(routed, shared, 2.5, accumulation_dtype=torch.float32)
    assert torch.allclose(out, torch.full((3, 8), 2.5 + 2.0))


def test_eager_combine_without_shared():
    routed = torch.ones(2, 4)
    out = M.eager_moe_combine(routed, None, 2.5, accumulation_dtype=torch.float32)
    assert torch.allclose(out, torch.full((2, 4), 2.5))


# ---------------------------------------------------------------------------
# Multi-head hyper-connection host ops
# ---------------------------------------------------------------------------


def test_hc_expand_contract_roundtrip():
    x = torch.randn(6, 16)
    n = 4
    expanded = M.hc_expand(x, n)
    assert expanded.shape == (6, n, 16)
    # All streams are replicas -> contract (mean) returns the original.
    assert torch.allclose(M.hc_contract(expanded, n), x, atol=1e-6)


def test_hc_pre_post_shapes_and_residual_mixing():
    torch.manual_seed(3)
    tokens, n, hidden = 5, 4, 16
    residual = torch.randn(tokens, n, hidden)
    fn = torch.randn((2 + n) * n, n * hidden) * 0.02
    hc_scale = torch.ones(3)
    hc_base = torch.zeros((2 + n) * n)
    post_mix, comb_mix, layer_input = M.hc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-5,
        hc_pre_eps=0.0,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=1,
    )
    assert layer_input.shape == (tokens, hidden)
    assert post_mix.shape[0] == tokens and comb_mix.shape[0] == tokens
    moe_out = torch.randn(tokens, hidden)
    updated = M.hc_post(moe_out, residual, post_mix, comb_mix)
    # Post re-expands back across the n residual streams.
    assert updated.shape == (tokens, n, hidden)


# ---------------------------------------------------------------------------
# Glm5NextW2MoE construction / from_config / injected-method forward
# ---------------------------------------------------------------------------


def test_moe_defaults_are_glm_geometry():
    moe = M.Glm5NextW2MoE()
    assert moe.num_experts == 288
    assert moe.top_k == 8
    assert moe.renormalize is True
    assert moe.routed_scaling_factor == 2.5
    assert moe.scoring_func == "sigmoid"


def test_moe_from_config_reads_geometry():
    cfg = types.SimpleNamespace(
        n_routed_experts=16,
        num_experts_per_token=4,
        norm_topk_prob=False,
        routed_scaling_factor=1.5,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
    )
    moe = M.Glm5NextW2MoE.from_config(cfg)
    assert moe.num_experts == 16
    assert moe.top_k == 4
    assert moe.renormalize is False
    assert moe.routed_scaling_factor == 1.5


def test_moe_forward_routes_experts_then_combines_with_injected_method():
    # Inject a fake E1.3 method so the full forward runs on CPU (no torch_npu):
    # it returns a per-token sum of the selected top-k weights, so we can assert
    # combine applied `routed_scaling_factor` and added the shared output.
    class _FakeMethod:
        def apply(self, layer, hidden, topk_weights, topk_ids, a, b):
            # routed[t] = sum_k topk_weights[t,k], broadcast over hidden.
            per_token = topk_weights.to(hidden.dtype).sum(dim=-1, keepdim=True)
            return per_token.expand_as(hidden).contiguous()

    tokens, hidden, num_experts, top_k = 4, 8, 12, 3
    moe = M.Glm5NextW2MoE(
        num_experts=num_experts,
        top_k=top_k,
        renormalize=True,
        routed_scaling_factor=2.5,
        method=_FakeMethod(),
        w2_experts=[object()],  # non-None so routed_experts_forward proceeds
        shared_expert=lambda h: torch.full_like(h, 0.25),
    )
    torch.manual_seed(4)
    hidden_states = torch.randn(tokens, hidden)
    router_logits = torch.randn(tokens, num_experts)
    out = moe.forward(hidden_states, router_logits)
    # With renormalize=True the top-k weights sum to 1 per token, so routed==1;
    # combine = 1 * 2.5 + 0.25 shared = 2.75 everywhere.
    assert out.shape == (tokens, hidden)
    assert torch.allclose(out.float(), torch.full((tokens, hidden), 2.75), atol=1e-5)


def test_moe_routed_experts_forward_requires_bank():
    moe = M.Glm5NextW2MoE(method=object())
    with pytest.raises(ValueError):
        moe.routed_experts_forward(torch.randn(2, 8), torch.zeros(2, 3, dtype=torch.int64), torch.ones(2, 3))


# ---------------------------------------------------------------------------
# weight_mapping: classification
# ---------------------------------------------------------------------------


def test_classify_expert_router_shared_dense_hc_vision():
    L = "model.language_model.layers"
    assert WM.classify_tensor(f"{L}.3.mlp.experts.0.gate_proj_codes") is WM.WeightClass.W2_EXPERT
    assert WM.classify_tensor(f"{L}.3.mlp.experts.0.down_proj_scale") is WM.WeightClass.W2_EXPERT
    assert WM.classify_tensor(f"{L}.3.mlp.gate.weight") is WM.WeightClass.FP16
    assert WM.classify_tensor(f"{L}.3.mlp.shared_experts.up_proj.weight") is WM.WeightClass.FP16
    assert WM.classify_tensor(f"{L}.0.mlp.down_proj.weight") is WM.WeightClass.FP16  # dense layer
    assert WM.classify_tensor(f"{L}.3.hc_ffn_fn") is WM.WeightClass.FP16
    assert WM.classify_tensor(f"{L}.3.self_attn.o_proj.weight") is WM.WeightClass.FP16
    assert WM.classify_tensor("model.visual.blocks.0.attn.qkv.weight") is WM.WeightClass.EXCLUDE


# ---------------------------------------------------------------------------
# weight_mapping: expert fusion (gate->w13 off0, up->w13 off inter, down->w2)
# ---------------------------------------------------------------------------


def test_map_expert_tensor_fusion_offsets():
    g = _REAL_GEOMETRY
    L = "model.language_model.layers"
    gate = WM.map_expert_tensor(f"{L}.3.mlp.experts.5.gate_proj_codes", g)
    assert gate.target_param == "w13_codes" and gate.row_offset == 0 and gate.fuses_into_w13
    up_codes = WM.map_expert_tensor(f"{L}.3.mlp.experts.5.up_proj_codes", g)
    assert up_codes.target_param == "w13_codes" and up_codes.row_offset == g["moe_intermediate_size"]
    up_scale = WM.map_expert_tensor(f"{L}.3.mlp.experts.5.up_proj_scale", g)
    assert up_scale.target_param == "w13_scale" and up_scale.row_offset == g["moe_intermediate_size"] // 32
    down = WM.map_expert_tensor(f"{L}.3.mlp.experts.5.down_proj_codes", g)
    assert down.target_param == "w2_codes" and down.row_offset == 0 and not down.fuses_into_w13
    assert down.block == "layers.3" and down.expert_id == 5


def test_map_expert_tensor_rejects_non_expert():
    with pytest.raises(ValueError):
        WM.map_expert_tensor("model.language_model.layers.3.mlp.gate.weight", _REAL_GEOMETRY)


def test_expected_expert_shape_matches_converted_artifact():
    g = _REAL_GEOMETRY
    # gate/up (w1/w3): codes [2048, 1024], scale [64, 128].
    assert WM.expected_expert_shape("w1", "codes", g) == (2048, 1024)
    assert WM.expected_expert_shape("w1", "scale", g) == (64, 128)
    # down (w2): codes [4096, 512], scale [128, 64].
    assert WM.expected_expert_shape("w2", "codes", g) == (4096, 512)
    assert WM.expected_expert_shape("w2", "scale", g) == (128, 64)


def test_validate_expert_tensor_shape_and_dtype():
    g = _REAL_GEOMETRY
    L = "model.language_model.layers"
    good = WM.TensorMeta(f"{L}.3.mlp.experts.0.gate_proj_codes", "U8", (2048, 1024))
    WM.validate_expert_tensor(good, g)  # no raise
    bad_shape = WM.TensorMeta(f"{L}.3.mlp.experts.0.gate_proj_codes", "U8", (2048, 999))
    with pytest.raises(WM.ShapeMismatchError):
        WM.validate_expert_tensor(bad_shape, g)
    bad_dtype = WM.TensorMeta(f"{L}.3.mlp.experts.0.gate_proj_scale", "F16", (64, 128))
    with pytest.raises(WM.DtypeMismatchError):
        WM.validate_expert_tensor(bad_dtype, g)


# ---------------------------------------------------------------------------
# weight_mapping: MoE block ids + coverage validation
# ---------------------------------------------------------------------------


def test_moe_block_ids_skip_dense_include_mtp():
    ids = WM.moe_block_ids(_TINY_GEOMETRY)
    # dense layers 0,1 skipped; MoE layers 2,3,4; MTP layer 5.
    assert ids == [2, 3, 4, 5]
    real = WM.moe_block_ids(_REAL_GEOMETRY)
    assert real[0] == 3 and real[-1] == 45 and len(real) == 43


def _all_expert_names(geometry):
    names = []
    for layer in WM.moe_block_ids(geometry):
        for e in range(geometry["n_routed_experts"]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for kind in ("codes", "scale"):
                    names.append(f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}_{kind}")
    return names


def test_validate_weight_map_full_coverage_tiny():
    names = _all_expert_names(_TINY_GEOMETRY)
    # add some non-expert names (ignored by the router coverage check).
    names += [
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.2.mlp.gate.weight",
        "model.language_model.embed_tokens.weight",
    ]
    summary = WM.validate_weight_map(names, _TINY_GEOMETRY)
    # 4 MoE blocks x 4 experts x 6 tensors = 96.
    assert summary["expert_blocks"] == 4
    assert summary["expert_tensors"] == 4 * 4 * 6


def test_validate_weight_map_missing_rejected():
    names = _all_expert_names(_TINY_GEOMETRY)[:-1]  # drop one
    with pytest.raises(WM.MissingTensorError):
        WM.validate_weight_map(names, _TINY_GEOMETRY)


def test_validate_weight_map_extra_dense_expert_rejected():
    names = _all_expert_names(_TINY_GEOMETRY)
    # a dense layer (0) must NOT carry experts -> ExtraTensorError.
    names += [f"model.language_model.layers.0.mlp.experts.0.gate_proj_{k}" for k in ("codes", "scale")]
    names += [f"model.language_model.layers.0.mlp.experts.0.{p}_{k}"
              for p in ("up_proj", "down_proj") for k in ("codes", "scale")]
    with pytest.raises(WM.ExtraTensorError):
        WM.validate_weight_map(names, _TINY_GEOMETRY)


def test_validate_weight_map_duplicate_rejected():
    names = _all_expert_names(_TINY_GEOMETRY)
    names.append(names[0])
    with pytest.raises(WM.DuplicateTensorError):
        WM.validate_weight_map(names, _TINY_GEOMETRY)


# ---------------------------------------------------------------------------
# Real artifact (skipped when the volume is not mounted)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_REAL_ARTIFACT / "model.safetensors.index.json").is_file(),
    reason="GLM W2 artifact volume not mounted",
)
def test_real_artifact_full_coverage_and_placement():
    names = list(WM.iter_artifact_tensor_metas(_REAL_ARTIFACT))
    summary = WM.validate_weight_map((m.name for m in names), _REAL_GEOMETRY)
    assert summary["expert_blocks"] == 43
    assert summary["expert_tensors"] == 288 * 43 * 6  # 74304
    placement = WM.estimate_per_chip_bytes(names, world_size=4)
    d = placement.as_dict()
    # GLM is light: well under the 40 GiB/chip ceiling. The converter already
    # dropped the vision tower (G2: 347 excluded), so the *converted* artifact
    # carries no vision tensors -- exclusion is exercised by test_classify_*.
    assert d["within_budget"]
    assert d["excluded_count"] == 0
    assert d["w2_expert_count"] == 74304


# ---------------------------------------------------------------------------
# Import hygiene: no triton on the moe / weight_mapping path
# ---------------------------------------------------------------------------


def test_moe_and_weight_mapping_have_no_triton_import():
    for mod in (M, WM):
        src = Path(mod.__file__).read_text()
        for lineno, line in enumerate(src.splitlines(), start=1):
            code = line.split("#", 1)[0]
            assert "import triton" not in code, f"{mod.__file__}:{lineno}"
            assert "from triton" not in code, f"{mod.__file__}:{lineno}"
            assert "ops.triton" not in code, f"{mod.__file__}:{lineno}"
