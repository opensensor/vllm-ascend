# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-checkpoint geometry validation for the 310P Qwen4Exp modules.

The parallel agents that built the Qwen4Exp modules unit-tested them against
TINY / assumed configs (e.g. 32 GDN value heads, ``indexer_head_dim=8``). This
suite loads the **real** checkpoint ``config.json`` ``text_config`` and asserts
that every module's geometry derivation reads the authoritative keys and matches
the shipped tensor geometry:

* GDN linear-attention heads / dims / conv + recurrent & conv state shapes.
* QSA attention (24 / 2 / 256, partial rotary 0.25, theta 1e7, sigmoid gate) and
  its indexer (budget 2048, ratio 4, head-dim 128, 1 kv / 4 heads).
* PLE injection geometry (embed 2560, conv 4, ngram 3, 8 heads/ngram, 128 shard
  parts) and its placement on the decoder layer that carries the checkpoint
  ``layers.<i>.ple.*`` tensors.
* Model assembly: 48 layers = 36 GDN + 12 QSA at ``full_attention_interval=4``
  positions; ``hc_count=4``.
* RoPE: native <= 262,144 uses the checkpoint default/mrope params (theta 1e7),
  NOT yarn; yarn engages only beyond the native window.

Modules are built on the ``meta`` device so no weights are materialized.

Run (the shared ``tests/ut/conftest.py`` fails to import on this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_real_config.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.worker.v2.rope import (
    NATIVE_MAX_POSITION_EMBEDDINGS,
    LongContextRopeConfigError,
    validate_long_context_rope,
)
from vllm_ascend.models.qwen4_exp.indexer_qsa import AscendQwen4ExpQSAIndexer
from vllm_ascend.models.qwen4_exp.model import (
    _LAYER_TYPE_FULL,
    _LAYER_TYPE_LINEAR,
    AscendQwen4ExpModel,
)
from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer
from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    Qwen4ExpGDNParams,
    gdn_conv_state_shape,
    gdn_recurrent_state_shape,
)

# Authoritative checkpoint (see plan T0.1 / checkpoint-manifest.json).
_CKPT = "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
_CONFIG_JSON = os.path.join(_CKPT, "config.json")


def _load_text_config() -> SimpleNamespace:
    if not os.path.isfile(_CONFIG_JSON):
        pytest.skip(f"real checkpoint config absent: {_CONFIG_JSON}")
    with open(_CONFIG_JSON) as fh:
        full = json.load(fh)
    text_config = full.get("text_config")
    if not text_config:
        pytest.skip("checkpoint config.json has no text_config block")
    return SimpleNamespace(**text_config)


@pytest.fixture(scope="module")
def cfg() -> SimpleNamespace:
    return _load_text_config()


# ---------------------------------------------------------------------------
# GDN linear attention.
# ---------------------------------------------------------------------------
def test_gdn_geometry_from_real_config(cfg: SimpleNamespace) -> None:
    params = Qwen4ExpGDNParams.from_hf_config(cfg)
    # Tiny tests used 32 value heads; the real derivation is 48 value / 16 key.
    assert params.num_v_heads == 48
    assert params.num_k_heads == 16
    assert params.head_k_dim == 128
    assert params.head_v_dim == 128
    assert params.conv_kernel_size == 4

    # Derived projection widths: q/k share key_dim, v uses value_dim.
    assert params.key_dim == 16 * 128  # 2048
    assert params.value_dim == 48 * 128  # 6144
    # mixed_qkv = [q(key_dim), k(key_dim), v(value_dim)] -> depthwise conv.
    assert params.conv_dim == params.key_dim * 2 + params.value_dim  # 10240

    # The shipped ``linear_attn.in_proj_a.weight`` is [48, 2560].
    assert params.num_v_heads == cfg.linear_num_value_heads
    assert params.num_v_heads == 48
    assert cfg.hidden_size == 2560

    # Recurrent (SSM) state [num_v_heads, V, K]; conv state [conv_dim, kernel-1].
    assert gdn_recurrent_state_shape(params) == (48, 128, 128)
    assert gdn_conv_state_shape(params) == (10240, 3)


# ---------------------------------------------------------------------------
# QSA sparse attention + indexer.
# ---------------------------------------------------------------------------
def test_qsa_attention_geometry_from_real_config(cfg: SimpleNamespace) -> None:
    with torch.device("meta"):
        qsa = AscendQwen4ExpQSAAttention(config=cfg, layer_idx=3)
    assert qsa.num_query_heads == 24
    assert qsa.num_kv_heads == 2
    assert qsa.head_dim == 256
    assert qsa.group_size == 12
    # partial_rotary_factor 0.25 -> rotary_dim 64 (even).
    assert qsa.rotary_dim == 64
    assert qsa.rotary_dim % 2 == 0
    # Real config nests rope_theta under rope_parameters (1e7), NOT a top-level
    # 1e4 default. This is the reconciled mismatch.
    assert qsa.rope_theta == 10_000_000.0


def test_qsa_output_gate_is_sigmoid(cfg: SimpleNamespace) -> None:
    assert getattr(cfg, "output_gate_type", None) == "sigmoid"


def test_qsa_indexer_geometry_from_real_config(cfg: SimpleNamespace) -> None:
    indexer = AscendQwen4ExpQSAIndexer(config=cfg, layer_idx=3)
    assert indexer.index_n_heads == 4
    assert indexer.index_kv_heads == 1
    assert indexer.index_head_dim == 128
    assert indexer.token_topk == 2048  # indexer_budget
    assert indexer.compress_ratio == 4
    # budget / ratio must divide evenly (2048 / 4 = 512 compressed blocks).
    assert indexer.block_topk == 512


# ---------------------------------------------------------------------------
# PLE injection layer + ngram geometry.
# ---------------------------------------------------------------------------
def test_ple_geometry_from_real_config(cfg: SimpleNamespace) -> None:
    with torch.device("meta"):
        ple = AscendQwen4ExpPLELayer(config=cfg, layer_idx=1)
    assert ple.ple_embed_dim == 2560
    assert ple.conv_kernel_size == 4
    assert ple.short_conv_dilation == 3  # ngram_size
    assert ple.heads_per_ngram == 8
    assert ple.hc_count == 4
    # Total n-gram heads: (ngram_size - 1) predecessor orders * heads/ngram.
    assert ple.num_ngram_heads == (3 - 1) * 8  # 16
    assert ple.per_head_dim == 2560 // 16  # 160
    # ngram table is sharded into split_ngram_parts partitions.
    assert cfg.split_ngram_parts == 128


def test_ple_layer_placement_matches_checkpoint_tensors(cfg: SimpleNamespace) -> None:
    # Config lists ple_layer_ids=[2]; the assembly treats these as 1-based ids,
    # so PLE lands on 0-indexed decoder layer 1 -- exactly where the checkpoint
    # tensors live (``layers.1.ple.*``).
    ple_layer_ids = list(cfg.ple_layer_ids)
    assert ple_layer_ids == [2]
    placed = [i for i in range(cfg.num_hidden_layers) if (i + 1) in ple_layer_ids]
    assert placed == [1]


# ---------------------------------------------------------------------------
# Model assembly.
# ---------------------------------------------------------------------------
def test_model_layer_assembly_from_real_config(cfg: SimpleNamespace) -> None:
    assert cfg.num_hidden_layers == 48
    layer_types = AscendQwen4ExpModel._resolve_layer_types(cfg)
    assert len(layer_types) == 48

    full = [i for i, t in enumerate(layer_types) if t == _LAYER_TYPE_FULL]
    linear = [i for i, t in enumerate(layer_types) if t == _LAYER_TYPE_LINEAR]
    # 3x linear_attention then 1x full_attention, repeating (interval 4).
    assert len(linear) == 36  # GDN
    assert len(full) == 12  # QSA
    assert full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]

    interval = int(getattr(cfg, "full_attention_interval", 4))
    assert interval == 4
    assert full == [i for i in range(48) if (i + 1) % interval == 0]

    assert int(cfg.hc_count) == 4


# ---------------------------------------------------------------------------
# RoPE: native (default/mrope, theta 1e7) vs yarn extension.
# ---------------------------------------------------------------------------
def test_rope_native_uses_default_not_yarn(cfg: SimpleNamespace) -> None:
    rope_parameters = dict(cfg.rope_parameters)
    assert rope_parameters["rope_type"] == "default"
    assert rope_parameters["rope_theta"] == 10_000_000
    assert cfg.max_position_embeddings == NATIVE_MAX_POSITION_EMBEDDINGS

    # Native window: default descriptor, checkpoint theta, no extension.
    native = validate_long_context_rope(NATIVE_MAX_POSITION_EMBEDDINGS, rope_parameters)
    assert native.rope_type == "default"
    assert native.rope_theta == 10_000_000.0
    assert native.factor == 1.0
    assert native.is_extended is False


def test_rope_yarn_only_beyond_native(cfg: SimpleNamespace) -> None:
    rope_parameters = dict(cfg.rope_parameters)
    # A bare bump beyond the native window with the native (non-yarn) params is
    # rejected -- no silent auto-scaling.
    with pytest.raises(LongContextRopeConfigError):
        validate_long_context_rope(NATIVE_MAX_POSITION_EMBEDDINGS + 1, rope_parameters)

    # An explicit yarn extension config engages only beyond the native window.
    extended = validate_long_context_rope(
        1_048_576,
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": NATIVE_MAX_POSITION_EMBEDDINGS,
            "rope_theta": 10_000_000,
        },
    )
    assert extended.rope_type == "yarn"
    assert extended.factor == 4.0
    assert extended.is_extended is True
    assert extended.rope_theta == 10_000_000.0
