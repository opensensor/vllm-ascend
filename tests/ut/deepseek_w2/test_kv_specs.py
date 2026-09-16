# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 MLA latent KV-cache specs + per-layer plan
(plan E2.2). Everything runs host-side with NO NPU / Triton.

The module builds the DeepSeek MLA **latent** KV-cache spec (compressed
``kv_lora_rank`` latent + decoupled-RoPE key), the ratio-{0,1,2} per-layer plan,
hybrid KV-cache groups (reusing the Qwen4Exp KV plumbing), and the exact 8K
per-chip byte table.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest \
        tests/ut/deepseek_w2/test_kv_specs.py
"""

import pytest
import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig, MLAAttentionSpec

from vllm_ascend.models.deepseek_v41.dtype_policy import ASCEND_DEEPSEEKV41_DTYPE_POLICY
from vllm_ascend.models.deepseek_v41.kv import (
    CONTEXT_8K,
    DEEPSEEKV41_C8_DTYPE,
    DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS,
    DEEPSEEKV41_NUM_HIDDEN_LAYERS,
    INDEXER_HEAD_DIM,
    MLA_KV_LORA_RANK,
    MLA_LATENT_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    build_deepseekv41_kv_cache_config,
    build_deepseekv41_kv_cache_groups,
    build_deepseekv41_layer_plan,
    deepseekv41_kv_bytes_per_chip,
    deepseekv41_kv_bytes_table,
    deepseekv41_scheduler_block_size,
    layer_compress_ratio,
    make_indexer_compressed_spec,
    make_mla_latent_spec,
)

ATTN_BLOCK = 128
FP16_BYTES = 2


class _TinyV41Config:
    """A tiny V4.1 text config: 6 layers exercising every ratio in {0,1,2}."""

    def __init__(self, compress_ratios, num_hidden_layers=None):
        self.compress_ratios = list(compress_ratios)
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else len(compress_ratios)


def _tiny_config():
    return _TinyV41Config([0, 1, 2, 0, 1, 2])


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
def test_mla_latent_geometry_constants():
    assert MLA_KV_LORA_RANK == 512
    assert QK_ROPE_HEAD_DIM == 64
    # One MLA latent per token = kv_lora latent + decoupled RoPE key.
    assert MLA_LATENT_HEAD_DIM == MLA_KV_LORA_RANK + QK_ROPE_HEAD_DIM == 576
    assert INDEXER_HEAD_DIM == 128
    assert DEEPSEEKV41_NUM_HIDDEN_LAYERS == 40
    # V4.1 draws its per-layer ratios from {0,1,2} (shipped V4 used {0,4,128}).
    assert DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS == (0, 1, 2)


# ---------------------------------------------------------------------------
# MLA latent spec
# ---------------------------------------------------------------------------
def test_mla_latent_spec_shape_and_page():
    spec = make_mla_latent_spec(attention_block_size=ATTN_BLOCK)
    assert isinstance(spec, MLAAttentionSpec)
    assert spec.num_kv_heads == 1  # single shared latent, TP-replicated
    assert spec.head_size == MLA_LATENT_HEAD_DIM == 576
    assert spec.dtype == ASCEND_DEEPSEEKV41_DTYPE_POLICY.kv_cache_dtype == torch.float16
    assert spec.compress_ratio == 1  # main latent is uncompressed
    assert spec.storage_block_size == ATTN_BLOCK
    # MLA stores ONE latent per token (no K/V doubling): 128 * 576 * 2 bytes.
    assert spec.page_size_bytes == ATTN_BLOCK * MLA_LATENT_HEAD_DIM * FP16_BYTES == 147456


# ---------------------------------------------------------------------------
# Indexer compressed-history spec
# ---------------------------------------------------------------------------
def test_indexer_spec_ratio1_and_ratio2():
    r1 = make_indexer_compressed_spec(1, attention_block_size=ATTN_BLOCK)
    r2 = make_indexer_compressed_spec(2, attention_block_size=ATTN_BLOCK)
    for spec in (r1, r2):
        assert spec.num_kv_heads == 1
        assert spec.head_size == INDEXER_HEAD_DIM == 128
        # storage_block_size == attention block: one latent per ratio tokens.
        assert spec.storage_block_size == ATTN_BLOCK
    assert r1.block_size == ATTN_BLOCK * 1 == 128
    assert r1.compress_ratio == 1
    assert r2.block_size == ATTN_BLOCK * 2 == 256
    assert r2.compress_ratio == 2
    # Both pages are storage_block * head * dtype (compression is in block_size).
    assert r1.page_size_bytes == ATTN_BLOCK * INDEXER_HEAD_DIM * FP16_BYTES == 32768
    assert r2.page_size_bytes == 32768


def test_indexer_spec_c8_halves_page():
    r1 = make_indexer_compressed_spec(1, attention_block_size=ATTN_BLOCK, c8=True)
    assert r1.dtype == DEEPSEEKV41_C8_DTYPE
    assert get_dtype_size(DEEPSEEKV41_C8_DTYPE) == 1
    assert r1.page_size_bytes == ATTN_BLOCK * INDEXER_HEAD_DIM * 1 == 16384


def test_indexer_spec_rejects_unsupported_ratio():
    # V4.1 indexer ratios are {1,2}; ratio 0 is dense and ratio 4 is the old V4.
    with pytest.raises(ValueError):
        make_indexer_compressed_spec(0)
    with pytest.raises(ValueError):
        make_indexer_compressed_spec(4)


# ---------------------------------------------------------------------------
# Per-layer plan (ratio-{0,1,2})
# ---------------------------------------------------------------------------
def test_layer_compress_ratio_reads_config_and_defaults_dense():
    ratios = [0, 1, 2]
    assert layer_compress_ratio(ratios, 0) == 0
    assert layer_compress_ratio(ratios, 1) == 1
    assert layer_compress_ratio(ratios, 2) == 2
    # Out-of-range (e.g. MTP layers) treated as dense, mirroring V4.
    assert layer_compress_ratio(ratios, 9) == 0


def test_layer_plan_classification():
    plan = build_deepseekv41_layer_plan(_tiny_config())
    assert len(plan) == 6
    expected = [
        (0, 0, True, False),
        (1, 1, False, True),
        (2, 2, False, True),
        (3, 0, True, False),
        (4, 1, False, True),
        (5, 2, False, True),
    ]
    for entry, (idx, ratio, dense, has_idx) in zip(plan, expected):
        assert entry.layer_idx == idx
        assert entry.compress_ratio == ratio
        assert entry.is_dense is dense
        assert entry.has_indexer is has_idx
        # Every layer carries the MLA latent; only sparse layers have an indexer.
        assert entry.mla_latent_spec.head_size == MLA_LATENT_HEAD_DIM
        if dense:
            assert entry.indexer_spec is None
        else:
            assert entry.indexer_spec is not None
            assert entry.indexer_spec.compress_ratio == ratio


def test_layer_plan_accepts_raw_ratio_sequence():
    plan = build_deepseekv41_layer_plan([0, 2])
    assert [p.compress_ratio for p in plan] == [0, 2]


def test_layer_plan_rejects_unknown_ratio():
    with pytest.raises(ValueError):
        build_deepseekv41_layer_plan([0, 4])  # 4 is the shipped-V4 ratio


def test_full_40_layer_plan_all_ratios_valid():
    # A representative 40-layer pattern: dense first, then repeating {0,1,2}.
    ratios = [0] + [(i % 3) for i in range(DEEPSEEKV41_NUM_HIDDEN_LAYERS - 1)]
    cfg = _TinyV41Config(ratios, num_hidden_layers=DEEPSEEKV41_NUM_HIDDEN_LAYERS)
    plan = build_deepseekv41_layer_plan(cfg)
    assert len(plan) == 40
    assert all(p.compress_ratio in (0, 1, 2) for p in plan)
    assert all(p.mla_latent_spec is not None for p in plan)


# ---------------------------------------------------------------------------
# Hybrid KV-cache groups / config
# ---------------------------------------------------------------------------
def test_kv_cache_groups_merge_by_spec():
    plan = build_deepseekv41_layer_plan(_tiny_config())
    groups = build_deepseekv41_kv_cache_groups(plan)
    # One MLA-latent group (all identical) + one group per distinct indexer ratio.
    assert len(groups) == 3
    by_head = {g.kv_cache_spec.head_size: g for g in groups}
    mla_group = by_head[MLA_LATENT_HEAD_DIM]
    assert sorted(mla_group.layer_names) == [f"mla.{i}" for i in range(6)]

    idx_groups = [g for g in groups if g.kv_cache_spec.head_size == INDEXER_HEAD_DIM]
    assert len(idx_groups) == 2
    by_block = {g.kv_cache_spec.block_size: sorted(g.layer_names) for g in idx_groups}
    assert by_block[128] == ["indexer.1", "indexer.4"]  # ratio 1
    assert by_block[256] == ["indexer.2", "indexer.5"]  # ratio 2


def test_build_kv_cache_config_valid():
    plan = build_deepseekv41_layer_plan(_tiny_config())
    num_blocks = 4
    config = build_deepseekv41_kv_cache_config(plan, num_blocks=num_blocks)
    assert isinstance(config, KVCacheConfig)
    assert config.num_blocks == num_blocks
    assert len(config.kv_cache_groups) == 3
    assert len(config.kv_cache_tensors) == len(config.kv_cache_groups)
    # Every layer appears exactly once across all groups.
    all_layers = [name for g in config.kv_cache_groups for name in g.layer_names]
    assert len(all_layers) == len(set(all_layers))
    assert set(all_layers) == {f"mla.{i}" for i in range(6)} | {
        "indexer.1",
        "indexer.2",
        "indexer.4",
        "indexer.5",
    }
    # Tensor sizing follows page_size_bytes * num_blocks.
    for group, tensor in zip(config.kv_cache_groups, config.kv_cache_tensors):
        assert tensor.size == group.kv_cache_spec.page_size_bytes * num_blocks


def test_scheduler_block_size_is_lcm():
    plan = build_deepseekv41_layer_plan(_tiny_config())
    # lcm(128 attn, 128 ratio-1, 256 ratio-2) == 256.
    assert deepseekv41_scheduler_block_size(plan) == 256
    # Dense-only plan collapses to the attention block.
    dense_plan = build_deepseekv41_layer_plan([0, 0, 0])
    assert deepseekv41_scheduler_block_size(dense_plan) == 128


# ---------------------------------------------------------------------------
# 8K byte math (exact) + long-context C8
# ---------------------------------------------------------------------------
def test_8k_byte_math_exact():
    cfg = _tiny_config()  # [0,1,2,0,1,2] -> 6 MLA + (2x r1) + (2x r2)
    b = deepseekv41_kv_bytes_per_chip(cfg, max_model_len=CONTEXT_8K)

    # MLA latent: 576 * 2 = 1152 B/token; 64 blocks * 147456 = 9,437,184/layer.
    assert b["mla_latent_bytes_per_token"] == 1152
    assert b["mla_latent_page_bytes"] == 147456
    assert b["mla_num_blocks"] == CONTEXT_8K // ATTN_BLOCK == 64
    assert b["mla_per_layer_bytes"] == 64 * 147456 == 9_437_184
    assert b["num_layers"] == 6
    assert b["mla_total_bytes"] == 9_437_184 * 6 == 56_623_104

    # Indexer: r1 -> 64*32768 = 2,097,152; r2 -> 32*32768 = 1,048,576. Two each.
    r1_bytes = 64 * 32768
    r2_bytes = 32 * 32768
    assert b["indexer_bytes"] == 2 * r1_bytes + 2 * r2_bytes == 6_291_456
    assert b["indexer_layers"] == 4

    assert b["total_bytes"] == 56_623_104 + 6_291_456 == 62_914_560
    # PP=1, latent TP-replicated -> per-chip == aggregate.
    assert b["per_chip_bytes"] == b["total_bytes"]
    assert b["pipeline_parallel_size"] == 1


def test_8k_c8_indexer_halves_indexer_only():
    cfg = _tiny_config()
    b16 = deepseekv41_kv_bytes_per_chip(cfg, max_model_len=CONTEXT_8K, c8_indexer=False)
    b8 = deepseekv41_kv_bytes_per_chip(cfg, max_model_len=CONTEXT_8K, c8_indexer=True)
    # MLA latent unchanged (precision-critical); indexer halved.
    assert b8["mla_total_bytes"] == b16["mla_total_bytes"]
    assert b8["indexer_bytes"] == b16["indexer_bytes"] // 2 == 3_145_728
    assert b8["indexer_layout"] == "C8"


def test_mla_much_smaller_than_full_mha():
    cfg = _tiny_config()
    b = deepseekv41_kv_bytes_per_chip(cfg, max_model_len=CONTEXT_8K, num_attention_heads=128, qk_nope_head_dim=128)
    ref = b["full_mha_reference"]
    # 2 * 128 heads * (128 nope + 64 rope) * 2 bytes = 98304 B/token.
    assert ref["mha_bytes_per_token"] == 2 * 128 * 192 * FP16_BYTES == 98304
    # MLA latent (1152 B/token) is ~85x smaller than materialized MHA.
    assert ref["mla_shrink_factor"] == pytest.approx(98304 / 1152, rel=1e-9)
    assert ref["mla_shrink_factor"] > 50


def test_byte_table_bf16_vs_c8():
    cfg = _tiny_config()
    table = deepseekv41_kv_bytes_table(cfg, max_model_len=CONTEXT_8K)
    assert set(table) == {"BF16", "C8"}
    assert table["BF16"]["total_bytes"] == 62_914_560
    assert table["C8"]["indexer_bytes"] == 3_145_728
    # C8 total is strictly smaller than BF16 (indexer shrinks, MLA fixed).
    assert table["C8"]["total_bytes"] < table["BF16"]["total_bytes"]
