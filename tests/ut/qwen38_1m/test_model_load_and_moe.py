# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Close the two Qwen4Exp production stubs on the Ascend 310P host path (T3.x).

Two things are validated, both CPU-only (NO NPU, NO Triton, NO 224 GB load):

1. ``AscendQwen4ExpForCausalLM.load_weights`` -- the real checkpoint loader.
   * A **synthetic real-shaped checkpoint** built from the manifest geometry
     (48 layers x 512 experts x 3 projections) is validated purely as
     *metadata* (name -> {dtype, shape}); no expert payload is ever allocated,
     so the 224 GB bank is never materialized. Missing / extra / wrong-dtype /
     wrong-shape expert indices are rejected via the T3.1 mapper.
   * On a **tiny config** with real payloads, per-expert W8A8 tensors land in
     the fused ``w13_*``/``w2_*`` layout at the right expert row-slice, and
     non-expert F16 tensors load by name. Missing / extra / dtype / shape
     mismatches raise.

2. The MoE block forward -- the real W8A8 fused-expert path. The routed grouped
   QDQ math is compared against the independent T3.3-validated eager/fused
   reference (``test_moe_w8a8_parity``) within the frozen W8A8 GEMM tolerances,
   including router renormalization and the symmetric-offset ``(q - offset) *
   scale`` dequant.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_model_load_and_moe.py
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.reference.tolerances import (
    W8A8_GEMM_ATOL,
    W8A8_GEMM_RTOL,
)
from tests.ut.qwen38_1m.test_moe_w8a8_parity import (
    _build_experts,
    _moe_fused,
    _route,
)
from vllm_ascend.models.qwen4_exp import moe as moe_mod
from vllm_ascend.models.qwen4_exp.weight_mapping import (
    ExtraTensorError,
    MissingTensorError,
    TensorDtypeError,
    TensorShapeError,
    expected_expert_tensor_names,
    map_expert_tensor,
    validate_expert_weight_map,
)

_MANIFEST_PATH = Path(__file__).parents[3] / "artifacts" / "qwen38-1m" / "checkpoint-manifest.json"


# ---------------------------------------------------------------------------
# Config + build helpers (mirrors the sibling assembly test's shims)
# ---------------------------------------------------------------------------
def _tiny_moe_config(
    *,
    num_layers: int = 2,
    num_experts: int = 6,
    top_k: int = 3,
    hidden: int = 32,
    moe_inter: int = 16,
    shared_inter: int = 16,
) -> SimpleNamespace:
    """A tiny all-MoE Qwen4Exp text config (every layer is a routed-MoE layer)."""
    return SimpleNamespace(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=48,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        eos_token_id=0,
        # All full_attention (dense eager attn), no QSA/indexer.
        layer_types=["full_attention"] * num_layers,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[],
        ple_embed_dim=hidden,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=True,
    )


def _vllm_config(cfg: SimpleNamespace) -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_text_config=cfg,
        hf_config=SimpleNamespace(text_config=cfg, vision_config=None),
        dtype=torch.float16,
        head_dtype=None,
        multimodal_config=None,
    )
    return SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="align", mamba_ssm_cache_dtype="float32"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@contextlib.contextmanager
def _single_rank_tp():
    vmod = "vllm.model_executor.layers.vocab_parallel_embedding"
    lmod = "vllm.model_executor.layers.logits_processor"
    avmod = "vllm_ascend.ops.vocab_parallel_embedding"
    qmod = "vllm_ascend.models.qwen4_exp.model"
    with (
        patch(f"{vmod}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{vmod}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{vmod}.tensor_model_parallel_all_reduce", side_effect=lambda x: x),
        # These helpers moved out of logits_processor in newer vLLM snapshots.
        # ``create=True`` keeps the host fixture compatible with both layouts;
        # older snapshots consume the patched globals, newer ones ignore them.
        patch(f"{lmod}.get_tensor_model_parallel_world_size", return_value=1, create=True),
        patch(f"{lmod}.tensor_model_parallel_gather", side_effect=lambda x: x, create=True),
        patch(f"{lmod}.tensor_model_parallel_all_gather", side_effect=lambda x, dim=-1: x, create=True),
        patch(f"{avmod}.lmhead_tp_enable", return_value=False),
        patch(f"{avmod}.embedding_tp_enable", return_value=False),
        patch(
            f"{avmod}.get_tp_group",
            return_value=SimpleNamespace(world_size=1, rank_in_group=0),
        ),
        patch(f"{qmod}._resolve_attn_backend", return_value=object),
    ):
        yield


def _build(cfg: SimpleNamespace):
    from vllm.config import set_current_vllm_config

    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    vllm_config = _vllm_config(cfg)
    with _single_rank_tp(), set_current_vllm_config(vllm_config):
        return AscendQwen4ExpForCausalLM(vllm_config=vllm_config)


# ---------------------------------------------------------------------------
# Synthetic real-shaped checkpoint builders (metadata + tiny payloads)
# ---------------------------------------------------------------------------
def _expert_meta_index(geometry: dict[str, int]) -> dict[str, dict[str, object]]:
    """Full metadata index (name -> {dtype, shape}) for every expert tensor.

    Pure metadata: no payload is allocated, so the real 48x512x3 geometry costs
    nothing (never the 224 GB bank).
    """
    hidden = geometry["hidden_size"]
    moe = geometry["moe_intermediate_size"]
    index: dict[str, dict[str, object]] = {}
    for name in expected_expert_tensor_names(geometry):
        # Derive dtype/shape from the name (weight=int8 [out,in]; scale/offset=f32 [out,1]).
        proj = name.rsplit(".", 2)[-2]
        kind = name.rsplit(".", 1)[-1]
        is_w13 = proj in ("gate_proj", "up_proj")
        out_dim = moe if is_w13 else hidden
        in_dim = hidden if is_w13 else moe
        if kind == "weight":
            index[name] = {"dtype": torch.int8, "shape": (out_dim, in_dim)}
        else:
            index[name] = {"dtype": torch.float32, "shape": (out_dim, 1)}
    return index


def _real_geometry() -> dict[str, int]:
    manifest = json.loads(_MANIFEST_PATH.read_text())
    g = manifest["geometry"]
    return {
        "num_hidden_layers": int(g["num_hidden_layers"]),
        "num_experts": int(g["num_experts"]),
        "moe_intermediate_size": int(g["moe_intermediate_size"]),
        "hidden_size": int(g["hidden_size"]),
    }


def _synth_expert_payloads(
    geometry: dict[str, int],
    *,
    seed: int = 0,
) -> list[tuple[str, torch.Tensor]]:
    """Real-shaped per-expert W8A8 tensors (tiny geometry) as a stream of pairs.

    Symmetric quant (offset == 0), matching the real 300i scheme.
    """
    layers = geometry["num_hidden_layers"]
    experts = geometry["num_experts"]
    hidden = geometry["hidden_size"]
    moe = geometry["moe_intermediate_size"]
    gen = torch.Generator().manual_seed(seed)
    out: list[tuple[str, torch.Tensor]] = []
    for layer in range(layers):
        for e in range(experts):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
            for proj, out_dim, in_dim in (
                ("gate_proj", moe, hidden),
                ("up_proj", moe, hidden),
                ("down_proj", hidden, moe),
            ):
                w = torch.randint(-127, 128, (out_dim, in_dim), generator=gen, dtype=torch.int8)
                s = (torch.rand(out_dim, 1, generator=gen) * 0.02 + 0.01).to(torch.float32)
                o = torch.zeros(out_dim, 1, dtype=torch.float32)
                out.append((f"{base}.{proj}.weight", w))
                out.append((f"{base}.{proj}.weight_scale", s))
                out.append((f"{base}.{proj}.weight_offset", o))
    return out


def _non_expert_payloads(model) -> list[tuple[str, torch.Tensor]]:
    """Deterministic F16 payloads for every non-expert named parameter.

    Named parameters that belong to the fused expert bank (``w13_*``/``w2_*``) are
    excluded -- those arrive through the per-expert stream.
    """
    gen = torch.Generator().manual_seed(123)
    fused = {"w13_weight", "w2_weight", "w13_weight_scale", "w13_weight_offset", "w2_weight_scale", "w2_weight_offset"}
    payloads: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if any(component in fused for component in name.split(".")):
            continue
        payloads.append((name, (torch.randn(param.shape, generator=gen) * 0.02).to(param.dtype)))
    return payloads


# ===========================================================================
# 1a. Metadata validation on the REAL geometry (no 224 GB load)
# ===========================================================================
def test_place_expert_tensor_targets_per_expert_weight_parameter():
    """Weight placement uses the local ParameterList slot; scales stay fused."""
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    geometry = {
        "num_hidden_layers": 1,
        "num_experts": 2,
        "moe_intermediate_size": 4,
        "hidden_size": 3,
    }
    weight_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = f"{weight_name}_scale"
    weight_mapping = map_expert_tensor(weight_name, geometry)
    scale_mapping = map_expert_tensor(scale_name, geometry)
    expert_weight = torch.nn.Parameter(torch.zeros(8, 3, dtype=torch.int8), requires_grad=False)
    weight_scale = torch.nn.Parameter(torch.zeros(2, 8, 1), requires_grad=False)
    params = {
        "model.layers.0.mlp.w13_weight.0": expert_weight,
        "model.layers.0.mlp.w13_weight_scale": weight_scale,
    }
    source_weight = torch.arange(12, dtype=torch.int8).reshape(4, 3)
    source_scale = torch.arange(4, dtype=torch.float32).reshape(4, 1)

    weight_target = AscendQwen4ExpForCausalLM._place_expert_tensor(None, params, weight_mapping, source_weight)
    scale_target = AscendQwen4ExpForCausalLM._place_expert_tensor(None, params, scale_mapping, source_scale)

    assert weight_target == "model.layers.0.mlp.w13_weight"
    assert scale_target == "model.layers.0.mlp.w13_weight_scale"
    assert torch.equal(expert_weight[:4], source_weight)
    assert torch.equal(expert_weight[4:], torch.zeros_like(expert_weight[4:]))
    assert torch.equal(weight_scale[0, :4], source_scale)


def test_real_geometry_metadata_validates_without_materializing_bank():
    geometry = _real_geometry()
    assert geometry == {
        "num_hidden_layers": 48,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "hidden_size": 2560,
    }
    index = _expert_meta_index(geometry)
    # 48 * 512 * 3 projections * 3 kinds (weight/scale/offset).
    assert len(index) == 48 * 512 * 3 * 3

    mapping = validate_expert_weight_map(index, geometry)
    # Every projection mapped: weight + scale + offset entries each cover 48*512*3.
    assert len(mapping.weight_entries) == 48 * 512 * 3
    assert len(mapping.scale_entries) == 48 * 512 * 3
    assert len(mapping.offset_entries) == 48 * 512 * 3


def test_real_geometry_metadata_rejects_missing_and_extra():
    geometry = _real_geometry()
    index = _expert_meta_index(geometry)

    # Missing: drop one expert tensor.
    missing = dict(index)
    dropped = next(iter(missing))
    del missing[dropped]
    with pytest.raises(MissingTensorError):
        validate_expert_weight_map(missing, geometry)

    # Extra: add an out-of-geometry expert tensor.
    extra = dict(index)
    extra["model.language_model.layers.0.mlp.experts.512.gate_proj.weight"] = {
        "dtype": torch.int8,
        "shape": (640, 2560),
    }
    with pytest.raises(ExtraTensorError):
        validate_expert_weight_map(extra, geometry)


# ===========================================================================
# 1b. load_weights on a tiny config with real payloads
# ===========================================================================
def test_load_weights_places_experts_and_non_experts():
    cfg = _tiny_moe_config()
    model = _build(cfg)
    geometry = model._expert_geometry()

    expert_stream = _synth_expert_payloads(geometry, seed=1)
    non_expert = _non_expert_payloads(model)
    # Preserve a reference to check exact placement afterwards.
    ref = {name: t for name, t in expert_stream}

    loaded = model.load_weights(iter(expert_stream + non_expert))

    # Fused expert params + every non-expert param landed.
    assert "model.layers.0.mlp.w13_weight" in loaded
    assert "model.layers.0.mlp.w2_weight" in loaded
    for name, _ in non_expert:
        assert name in loaded

    # Expert tensors land in the W8A8 fused layout at the correct row slice.
    layer0 = model.model.layers[0].mlp
    moe = geometry["moe_intermediate_size"]
    for e in range(geometry["num_experts"]):
        base = f"model.language_model.layers.0.mlp.experts.{e}"
        gate = ref[f"{base}.gate_proj.weight"]
        up = ref[f"{base}.up_proj.weight"]
        down = ref[f"{base}.down_proj.weight"]
        assert torch.equal(layer0.w13_weight[e][:moe], gate)
        assert torch.equal(layer0.w13_weight[e][moe:], up)
        assert torch.equal(layer0.w2_weight[e], down)
        # Scales (gate rows first, then up rows) and symmetric zero offsets.
        assert torch.allclose(layer0.w13_weight_scale[e, :moe], ref[f"{base}.gate_proj.weight_scale"])
        assert torch.allclose(layer0.w13_weight_scale[e, moe:], ref[f"{base}.up_proj.weight_scale"])
        assert torch.count_nonzero(layer0.w13_weight_offset[e]) == 0
        assert torch.count_nonzero(layer0.w2_weight_offset[e]) == 0

    # A non-expert F16 param loaded by name (router gate is not quantized).
    gate_ref = dict(non_expert)["model.layers.0.mlp.gate"]
    assert torch.allclose(layer0.gate.float(), gate_ref.float())


def test_load_weights_round_trip_by_name_is_supported():
    """A state-dict round-trip (fused params by name, no per-expert tensors)."""
    cfg = _tiny_moe_config()
    model = _build(cfg)
    gen = torch.Generator().manual_seed(9)
    weights = [(name, (torch.randn(p.shape, generator=gen) * 0.02).to(p.dtype)) for name, p in model.named_parameters()]
    loaded = model.load_weights(weights)
    assert loaded == {name for name, _ in weights}


def test_load_weights_rejects_missing_expert():
    cfg = _tiny_moe_config()
    model = _build(cfg)
    geometry = model._expert_geometry()
    stream = _synth_expert_payloads(geometry, seed=2)
    # Drop the last expert tensor.
    stream = stream[:-1]
    with pytest.raises(MissingTensorError):
        model.load_weights(iter(stream + _non_expert_payloads(model)))


def test_load_weights_rejects_extra_expert():
    cfg = _tiny_moe_config()
    model = _build(cfg)
    geometry = model._expert_geometry()
    stream = _synth_expert_payloads(geometry, seed=3)
    # Add an out-of-range expert (index == num_experts).
    bad = f"model.language_model.layers.0.mlp.experts.{geometry['num_experts']}.gate_proj.weight"
    stream.append((bad, torch.zeros(geometry["moe_intermediate_size"], geometry["hidden_size"], dtype=torch.int8)))
    with pytest.raises(Exception) as exc:
        model.load_weights(iter(stream + _non_expert_payloads(model)))
    from vllm_ascend.models.qwen4_exp.weight_mapping import WeightMappingError

    assert isinstance(exc.value, WeightMappingError)


def test_load_weights_rejects_wrong_dtype_and_shape():
    cfg = _tiny_moe_config()
    model = _build(cfg)
    geometry = model._expert_geometry()
    hidden = geometry["hidden_size"]
    moe = geometry["moe_intermediate_size"]

    # Wrong dtype: expert weight arrives as float instead of int8.
    stream = _synth_expert_payloads(geometry, seed=4)
    stream[0] = (stream[0][0], stream[0][1].to(torch.float32))
    with pytest.raises(TensorDtypeError):
        model.load_weights(iter(stream))

    # Wrong shape: a gate_proj weight with a transposed shape.
    stream2 = _synth_expert_payloads(geometry, seed=5)
    stream2[0] = (stream2[0][0], torch.zeros(hidden, moe, dtype=torch.int8))
    with pytest.raises(TensorShapeError):
        model.load_weights(iter(stream2))


# ===========================================================================
# 2. MoE forward parity vs the T3.3-validated eager/fused W8A8 reference
# ===========================================================================
def _stack_experts(experts, hidden: int, moe: int):
    """Stack a list of T3.3 ``_ExpertWeights`` into the fused param layout."""
    w13_w = torch.stack([e.w13_q for e in experts]).contiguous()
    w13_s = torch.stack([e.w13_scale for e in experts]).contiguous()
    w13_o = torch.stack([e.w13_offset for e in experts]).contiguous()
    w2_w = torch.stack([e.w2_q for e in experts]).contiguous()
    w2_s = torch.stack([e.w2_scale for e in experts]).contiguous()
    w2_o = torch.stack([e.w2_offset for e in experts]).contiguous()
    assert w13_w.shape[1:] == (2 * moe, hidden)
    assert w2_w.shape[1:] == (hidden, moe)
    return w13_w, w13_s, w13_o, w2_w, w2_s, w2_o


@pytest.mark.parametrize("num_experts, top_k", [(8, 3), (16, 4), (512, 10)])
@pytest.mark.parametrize("seed", [0, 1])
def test_moe_grouped_matches_t33_reference(num_experts, top_k, seed):
    """The model W8A8 grouped forward equals the T3.3 fused reference (routed)."""
    hidden, moe, num_tokens = 16, 8, 20
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=gen)
    router_logits = torch.randn(num_tokens, num_experts, generator=gen)

    topk_weights, topk_ids = _route(router_logits, top_k, renormalize=True)
    experts = _build_experts(num_experts, hidden, moe, seed=seed + 100)
    w13_w, w13_s, w13_o, w2_w, w2_s, w2_o = _stack_experts(experts, hidden, moe)

    y_model = moe_mod.w8a8_grouped_experts(x, topk_weights, topk_ids, w13_w, w13_s, w13_o, w2_w, w2_s, w2_o)
    # T3.3 fused reference with no shared expert (routed-only).
    y_ref = _moe_fused(x, topk_weights, topk_ids, experts, [])

    torch.testing.assert_close(y_model, y_ref, rtol=W8A8_GEMM_RTOL, atol=W8A8_GEMM_ATOL)


def test_moe_route_matches_reference_renorm_and_scaling():
    """Router renorm + routed scaling factor match the T3.3 ``_route``."""
    num_tokens, num_experts, top_k = 12, 32, 6
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(num_tokens, num_experts, generator=gen)

    for scaling in (1.0, 1.5):
        w_ref, ids_ref = _route(logits, top_k, renormalize=True, routed_scaling_factor=scaling)
        w_mod, ids_mod = moe_mod.route_topk(logits, top_k, renormalize=True, routed_scaling_factor=scaling)
        assert torch.equal(ids_mod, ids_ref)
        torch.testing.assert_close(w_mod, w_ref, rtol=0, atol=1e-6)
        # Renormalized weights sum to the scaling factor.
        per_token_sum = w_mod.sum(dim=-1)
        torch.testing.assert_close(per_token_sum, torch.full_like(per_token_sum, scaling), rtol=0, atol=1e-5)


def test_moe_symmetric_offset_dequant_is_noop():
    """Real experts are symmetric (offset == 0): (q - offset)*scale == q*scale."""
    gen = torch.Generator().manual_seed(11)
    out_dim, in_dim = 8, 16
    q = torch.randint(-127, 128, (out_dim, in_dim), generator=gen, dtype=torch.int8)
    scale = (torch.rand(out_dim, 1, generator=gen) * 0.02 + 0.01).to(torch.float32)
    zero_offset = torch.zeros(out_dim, 1, dtype=torch.float32)

    deq = moe_mod.dequantize_weight_perchannel(q, scale, zero_offset)
    torch.testing.assert_close(deq, q.to(torch.float32) * scale)


def test_moe_block_forward_finite_and_uses_shared_expert():
    """The assembled MoE module forwards finite F16 and adds the F16 shared expert."""
    from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
    from vllm_ascend.models.qwen4_exp.model import _EagerSparseMoE

    cfg = _tiny_moe_config(num_experts=8, top_k=3, hidden=16, moe_inter=8, shared_inter=8)
    block = _EagerSparseMoE(config=cfg, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)
    assert block.has_shared_expert is True
    assert block.num_experts == 8 and block.top_k == 3

    gen = torch.Generator().manual_seed(0)
    # Give the experts non-trivial symmetric W8A8 weights + F16 router/shared.
    experts = _build_experts(8, 16, 8, seed=2)
    w13_w, w13_s, w13_o, w2_w, w2_s, w2_o = _stack_experts(experts, 16, 8)
    with torch.no_grad():
        for target, source in zip(block.w13_weight, w13_w):
            target.copy_(source)
        block.w13_weight_scale.copy_(w13_s)
        block.w13_weight_offset.copy_(w13_o)
        for target, source in zip(block.w2_weight, w2_w):
            target.copy_(source)
        block.w2_weight_scale.copy_(w2_s)
        block.w2_weight_offset.copy_(w2_o)
        block.gate.copy_((torch.randn(8, 16, generator=gen) * 0.1).to(block.gate.dtype))
        block.shared_gate_up.copy_((torch.randn(16, 16, generator=gen) * 0.1).to(block.shared_gate_up.dtype))
        block.shared_down.copy_((torch.randn(16, 8, generator=gen) * 0.1).to(block.shared_down.dtype))

    x = (torch.randn(6, 16, generator=gen) * 0.5).to(block.params_dtype)
    out = block(x)
    assert out.shape == (6, 16)
    assert out.dtype == block.params_dtype
    assert torch.isfinite(out.float()).all()

    # Zeroing the shared expert changes the output (proves it is added).
    with torch.no_grad():
        block.shared_gate_up.zero_()
        block.shared_down.zero_()
    out_no_shared = block(x)
    assert (out.float() - out_no_shared.float()).abs().max().item() > 1e-4
