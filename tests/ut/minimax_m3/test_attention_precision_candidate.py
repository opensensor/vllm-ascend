# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from tools.minimax_m3.attention_precision_candidate import dense_replacements, normalized_rotary, prepare
from vllm_ascend._310p.ops.minimax_m3 import ATTENTION_QUERY_TILE, paged_sparse_attention


def test_fused_normalization_rotary_preserves_fp32_intermediate():
    x = torch.tensor([[0.7, 0.3, 0.1, -0.2]]).half()
    norm = SimpleNamespace(weight=torch.tensor([0.001, 0.002, 0.003, 0.004]).half(), variance_epsilon=1e-6)
    cosine, sine = torch.tensor([[[0.8]]]), torch.tensor([[[0.6]]])
    actual = normalized_rotary(x, norm, cosine, sine, 4, 2)
    normalized = x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + norm.weight.float())
    expected = normalized.clone()
    expected[:, 0] = normalized[:, 0] * 0.8 - normalized[:, 1] * 0.6
    expected[:, 1] = normalized[:, 1] * 0.8 + normalized[:, 0] * 0.6
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert not torch.equal(actual[:, :2], actual[:, :2].half().float())


def test_separate_and_fused_projection_paths_agree_and_quantize_only_history_and_index():
    generator = torch.Generator().manual_seed(91)
    projected = torch.randn(3, 24, generator=generator).half()
    norm = SimpleNamespace(weight=torch.zeros(4).half(), variance_epsilon=1e-6)
    rope = SimpleNamespace(
        is_neox_style=True, rotary_dim=2, cos_sin_cache=torch.tensor([[1, 0], [0.8, 0.6], [0.6, 0.8]]).half()
    )
    common = dict(
        q_size=8,
        kv_size=4,
        index_q_size=4,
        idx_head_dim=4,
        head_dim=4,
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
        rotary_emb=rope,
    )
    positions = torch.arange(3)
    quantized = []

    def quantize(value):
        quantized.append(value.clone())
        return value.clamp(-448, 448).to(torch.float8_e4m3fn).float()

    fused = SimpleNamespace(**common, qkv_proj=lambda x: (projected, None), indexer_proj=None)
    expected = prepare(fused, positions, None, quantize)
    assert len(quantized) == 4
    split = SimpleNamespace(
        **common, qkv_proj=lambda x: (projected[:, :16], None), indexer_proj=lambda x: (projected[:, 16:], None)
    )
    actual = prepare(split, positions, None, quantize)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.dtype == torch.float16
    torch.testing.assert_close(actual[2], projected[:, 12:16].to(torch.float8_e4m3fn).half())


def test_dense_ablation_includes_all_visible_blocks_without_changing_global_tile():
    q = torch.zeros(1, 4, 128).half()
    key = torch.zeros(1, 2, 128).half()
    value = torch.full_like(key, 383)
    iq, ik = torch.zeros(1, 2, 128).half(), torch.zeros(1, 128).half()
    key_cache = torch.zeros(3, 128, 2, 128).half()
    value_cache = torch.arange(384).half().reshape(3, 128, 1, 1).expand_as(key_cache).clone()
    index_cache = torch.zeros(3, 128, 128).half()
    arguments = (q, key, value, iq, ik, key_cache, value_cache, index_cache)
    metadata = dict(
        query_starts=(0, 1),
        seq_lens=torch.tensor([384]),
        block_table=torch.tensor([[0, 1, 2]]),
        index_block_table=torch.tensor([[0, 1, 2]]),
        slot_mapping=torch.tensor([383]),
        index_slot_mapping=torch.tensor([383]),
        max_seq_len=384,
        topk_blocks=1,
        scale=128**-0.5,
    )
    sparse = paged_sparse_attention(*arguments, **metadata)
    candidate = next(iter(dense_replacements(None).values()))
    dense = candidate(*arguments, **metadata)
    torch.testing.assert_close(sparse, torch.full_like(sparse, 319.5), rtol=0, atol=0)
    torch.testing.assert_close(dense, torch.full_like(dense, 191.5), rtol=0, atol=0)
    assert paged_sparse_attention.__globals__["ATTENTION_QUERY_TILE"] == ATTENTION_QUERY_TILE
