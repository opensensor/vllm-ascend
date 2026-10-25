# SPDX-License-Identifier: Apache-2.0
"""Resident diagnostic for fused FP32 QK math and identity-scaled FP8 history.

Emulates rounding in FP16 storage; it does not compress the cache or reproduce
the complete Hopper BF16/Marlin execution path.
"""

from types import FunctionType

import torch

from tools.minimax_m3.activation_reference import round_e4m3

DENSE_ABLATION_QUERY_TILE = 8


def normalized_rotary(value, norm, cosine, sine, head_size, rotary_dim):
    shaped = value.float().reshape(len(value), -1, head_size)
    shaped = shaped * torch.rsqrt((shaped * shaped).mean(-1, keepdim=True) + norm.variance_epsilon)
    shaped = shaped * (1 + norm.weight.float())
    half = rotary_dim // 2
    left, right = shaped[..., :half], shaped[..., half:rotary_dim]
    return torch.cat(
        (left * cosine - right * sine, right * cosine + left * sine, shaped[..., rotary_dim:]), -1
    ).reshape(value.shape)


def prepare(self, positions, hidden_states, quantize):
    qkv, _ = self.qkv_proj(hidden_states)
    main_size = self.q_size + 2 * self.kv_size
    if self.indexer_proj is None:
        q, k, v, iq, ik = qkv.split([self.q_size, self.kv_size, self.kv_size, self.index_q_size, self.idx_head_dim], -1)
    else:
        q, k, v = qkv[..., :main_size].split([self.q_size, self.kv_size, self.kv_size], -1)
        indexed, _ = self.indexer_proj(hidden_states)
        iq, ik = indexed.split([self.index_q_size, self.idx_head_dim], -1)
    if not self.rotary_emb.is_neox_style:
        raise ValueError("Attention candidate requires NeoX rotary")
    cosine, sine = self.rotary_emb.cos_sin_cache.index_select(0, positions).float().chunk(2, -1)
    cosine, sine = cosine[:, None], sine[:, None]

    def transform(value, norm, head_size):
        return normalized_rotary(value, norm, cosine, sine, head_size, self.rotary_emb.rotary_dim)

    q = transform(q, self.q_norm, self.head_dim).to(qkv.dtype)
    k = quantize(transform(k, self.k_norm, self.head_dim)).to(qkv.dtype)
    iq = quantize(transform(iq, self.index_q_norm, self.idx_head_dim)).to(qkv.dtype)
    ik = quantize(transform(ik, self.index_k_norm, self.idx_head_dim)).to(qkv.dtype)
    return q, k, quantize(v.float()).to(qkv.dtype).contiguous(), iq, ik


def replacements(native_resources):
    # Reuse exact powers without introducing a late CANN Pow allocation.
    owned = torch.tensor([2.0**i for i in range(-126, 128)], dtype=torch.float32)
    table = owned.to("npu")
    torch.npu.synchronize()

    def power(exponents):
        return table.index_select(0, (exponents.clamp(-126, 127) + 126).long().flatten()).reshape(exponents.shape)

    def candidate(self, positions, hidden_states):
        return prepare(self, positions, hidden_states, lambda value: round_e4m3(value, power))

    return {"vllm_ascend.models.minimax_m3.minimax_m3:MiniMaxM3SparseAttention._sparse_prepare": candidate}


def dense_replacements(native_resources):
    """Diagnostic ablation only: retain every visible block instead of top 16.

    Smaller query tiles bound gather scratch. This changes the model's trained
    attention pattern and must not be promoted as a source-equivalent fix.
    """
    # Lazy import keeps the reference helpers usable without serving dependencies.
    from vllm_ascend._310p.ops.minimax_m3 import paged_sparse_attention

    tiled = FunctionType(
        paged_sparse_attention.__code__,
        {**paged_sparse_attention.__globals__, "ATTENTION_QUERY_TILE": DENSE_ABLATION_QUERY_TILE},
        argdefs=paged_sparse_attention.__defaults__,
        closure=paged_sparse_attention.__closure__,
    )
    tiled.__kwdefaults__ = paged_sparse_attention.__kwdefaults__

    def all_blocks(*args, **kwargs):
        block_size = args[5].shape[1]
        kwargs["topk_blocks"] = (kwargs["max_seq_len"] + block_size - 1) // block_size
        return tiled(*args, **kwargs)

    return {"vllm_ascend.models.minimax_m3.minimax_m3:paged_sparse_attention": all_blocks}
