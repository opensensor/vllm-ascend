# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-expert weight remap coverage against the real checkpoint.

For every non-expert tensor name in the real checkpoint index, ``_remap_non_expert``
must either map it to an existing model parameter (with a correct fusion/split
shape) or return ``None`` for one of the documented gaps (derived n-gram buffers,
lazy PLE shards, and the eager stand-in params the Ascend modules do not yet own).
Skips cleanly when the checkpoint mount is absent (CI-safe).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.qwen4_exp.model import _remap_non_expert

_CKPT = Path(
    "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
)

# Documented gaps: tensors with no eager Ascend param yet.
_DOCUMENTED_SKIP_SUFFIXES = (
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)


def _is_documented_skip(name: str) -> bool:
    # The PLE n-gram table is derived (buffers) or lazy-shard mmap'd (128 shards).
    if ".ple_embedding." in name:
        return True
    if ".linear_attn." in name or ".mlp.shared_expert." in name:
        return True  # dedicated TP-aware loader paths
    if name.endswith(".self_attn.q_proj.weight"):
        return True  # dedicated per-head query/gate deinterleave
    return any(name.endswith(s) for s in _DOCUMENTED_SKIP_SUFFIXES)


def _model_param_shapes() -> dict[str, tuple[int, ...]]:
    with open(_CKPT / "config.json") as fh:
        text_cfg = SimpleNamespace(**json.load(fh)["text_config"])
    model_config = SimpleNamespace(
        hf_text_config=text_cfg,
        hf_config=SimpleNamespace(text_config=text_cfg, vision_config=None),
        dtype=torch.float16,
        multimodal_config=None,
        model=str(_CKPT),
        download_dir=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        quant_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="align", mamba_ssm_cache_dtype="float32"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
        load_config=SimpleNamespace(download_dir=None),
    )
    vmod = "vllm.model_executor.layers.vocab_parallel_embedding"
    lmod = "vllm.model_executor.layers.logits_processor"
    with (
        torch.device("meta"),
        patch(f"{vmod}.get_tensor_model_parallel_rank", return_value=0),
        patch(f"{vmod}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{vmod}.tensor_model_parallel_all_reduce", side_effect=lambda x: x),
        patch(f"{lmod}.get_tensor_model_parallel_world_size", return_value=1),
        patch(f"{lmod}.tensor_model_parallel_gather", side_effect=lambda x: x),
        patch(f"{lmod}.tensor_model_parallel_all_gather", side_effect=lambda x, dim=-1: x),
    ):
        from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

        model = AscendQwen4ExpForCausalLM(vllm_config=vllm_config)
    return {n: tuple(p.shape) for n, p in model.named_parameters()}


@pytest.fixture(scope="module")
def param_shapes():
    if not (_CKPT / "config.json").exists():
        pytest.skip("checkpoint not mounted")
    return _model_param_shapes()


def _non_expert_names() -> list[str]:
    with open(_CKPT / "quant_model_weights.safetensors.index.json") as fh:
        wm = json.load(fh)["weight_map"]
    return [
        n for n in wm if ".mlp.experts." not in n and not n.startswith("mtp.") and not n.startswith("model.visual.")
    ]


def _rewrite_prefix(name: str) -> str:
    return name.replace("model.language_model.", "model.", 1)


def _real_config() -> SimpleNamespace:
    with open(_CKPT / "config.json") as fh:
        return SimpleNamespace(**json.load(fh)["text_config"])


def test_every_non_expert_tensor_maps_or_is_documented_skip(param_shapes):
    cfg = _real_config()
    unmapped: list[str] = []
    for name in _non_expert_names():
        vllm_name = _rewrite_prefix(name)
        if vllm_name in ("model.embed_tokens.weight", "lm_head.weight"):
            continue  # handled via weight_loader (vocab TP shard)
        placements = _remap_non_expert(vllm_name, cfg)
        if placements is None:
            if not _is_documented_skip(name):
                unmapped.append(name)
            continue
        for target, _slice, _offset in placements:
            if target not in param_shapes:
                unmapped.append(f"{name} -> missing target {target}")
    assert unmapped == [], f"{len(unmapped)} non-expert tensors not mapped:\n" + "\n".join(unmapped[:20])


def test_fusion_targets_have_correct_dim0(param_shapes):
    cfg = _real_config()
    hidden = int(cfg.hidden_size)
    hc_hidden = int(getattr(cfg, "hc_count", 2)) * hidden
    shared_inter = int(getattr(cfg, "shared_expert_intermediate_size", 0) or 0)
    gdn_num_v = int(getattr(cfg, "linear_num_value_heads", 0) or 0)

    assert param_shapes["model.layers.0.mlp.shared_gate_up"][0] == 2 * shared_inter
    assert param_shapes["model.layers.0.attention.in_proj_ba"][0] == 2 * gdn_num_v
    assert param_shapes["model.layers.1.ple.ple.kv_proj_weight"][0] == hc_hidden + hidden


def test_indexer_split_geometry(param_shapes):
    cfg = _real_config()
    index_rows = int(getattr(cfg, "indexer_n_heads", 4)) * int(getattr(cfg, "indexer_head_dim", 128))
    # Layer 3 is the first full_attention (QSA) layer.
    assert param_shapes["model.layers.3.attention.iq_proj"][0] == index_rows
    assert param_shapes["model.layers.3.attention.ik_proj"][0] == int(getattr(cfg, "indexer_head_dim", 128))


def test_q_proj_deinterleaves_query_and_gate_per_head():
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    config = SimpleNamespace(num_attention_heads=2, head_dim=3, hidden_size=2)
    owner = SimpleNamespace(model=SimpleNamespace(config=config))
    q = torch.empty(6, 2)
    gate = torch.empty(6, 2)
    params = {
        "model.layers.3.attention.q_proj": q,
        "model.layers.3.attention.gate_proj": gate,
    }
    source = torch.arange(24).reshape(12, 2)
    loaded = AscendQwen4ExpForCausalLM._place_qsa_q_gate_tensor(
        owner,
        params,
        "model.layers.3.self_attn.q_proj.weight",
        source,
    )
    per_head = source.reshape(2, 2, 3, 2)
    assert loaded == (
        "model.layers.3.attention.q_proj",
        "model.layers.3.attention.gate_proj",
    )
    assert torch.equal(q, per_head[:, 0].reshape_as(q))
    assert torch.equal(gate, per_head[:, 1].reshape_as(gate))


def test_gdn_ba_loader_preserves_checkpoint_packing_order():
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

    config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        head_dim=4,
        partial_rotary_factor=0.5,
    )
    owner = SimpleNamespace(model=SimpleNamespace(config=config))
    target = torch.zeros(4, 3)
    params = {"model.layers.0.attention.in_proj_ba": target}
    b_weight = torch.full((2, 3), 7.0)
    a_weight = torch.full((2, 3), 11.0)

    AscendQwen4ExpForCausalLM._place_gdn_tensor(
        owner,
        params,
        "model.layers.0.linear_attn.in_proj_b.weight",
        b_weight,
        0,
        1,
    )
    AscendQwen4ExpForCausalLM._place_gdn_tensor(
        owner,
        params,
        "model.layers.0.linear_attn.in_proj_a.weight",
        a_weight,
        0,
        1,
    )

    assert torch.equal(target[:2], b_weight)
    assert torch.equal(target[2:], a_weight)


def test_renames_reach_expected_targets(param_shapes):
    cfg = _real_config()
    cases = {
        "model.hyper_connection_mixer.hc_norm.weight": "model.hyper_connection_mixer.hc_norm_weight",
        "model.layers.0.mlp.gate.weight": "model.layers.0.mlp.gate",
        # Layer 0 is linear_attention (GDN), layer 3 is full_attention (QSA),
        # layer 1 carries the PLE injection.
        "model.layers.3.self_attn.k_proj.weight": "model.layers.3.attention.k_proj",
        "model.layers.3.self_attn.q_norm.weight": "model.layers.3.attention.attn.q_norm_weight",
        "model.layers.1.ple.norm_query.weight": "model.layers.1.ple.ple.norm_query_weight",
    }
    for src, expected in cases.items():
        placements = _remap_non_expert(_rewrite_prefix(src), cfg)
        assert placements is not None, src
        assert placements[0][0] == expected, (src, placements)
        assert placements[0][0] in param_shapes
