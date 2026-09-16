# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P GLM-5.3-Flash W2 DeepSeek-Sparse-Attention (DSA) -- eager (G5).

The 11 ``full_attn_layers`` of GLM-5.3-Flash (``glm5_next``) are
``deepseek_sparse_attention`` (DSA) layers: a **Lightning-Indexer top-k token
selection** feeding a **full attention core**. This module is the host-lane,
Triton-free, ``torch_npu``-free 310P implementation of that path. It exists
because the shipped ``glm5next.sparse_attn_indexer_kpool.SparseAttnIndexerKpool``
raises ``NotImplementedError`` on Ascend (its scoring path is a CUDA-only
DeepGEMM/radix-topk stack), and the shipped ``glm5next.attention`` MLA wrapper
pulls ``FusedMoEFactory`` + device MLA kernels that do not import on the CPU host.

REUSE vs. GLM adapter (what this module does NOT fork)
-----------------------------------------------------
GLM's DSA is the same *family* as DeepSeek V4.1's sparse attention, so the
**indexer selection math is imported verbatim** from
:mod:`vllm_ascend.models.deepseek_v41.indexer` -- not re-derived:

* :func:`~...indexer.mean_pool_compress`   -- kpool key compression (CSA2 pool).
* :func:`~...indexer.lightning_indexer_scores`
      ``score[t,m] = sum_h w[t,h] * relu(scale * (q_{t,h} . k_pool_m))``.
* :func:`~...indexer.select_topk_blocks` / :func:`~...indexer.block_topk_for_ratio`
      deterministic, causal, descending-score top-k block selection.

The thin GLM adapter this module adds:

* **kpool ratio.** GLM uses ``index_kpool`` (=4 in the real config); the
  deepseek_v41 *module* only gates ratios {0,1,2}, but its *functions* work for
  any ratio >= 1, so we reuse the functions and drive them with ``index_kpool``.
* **GLM projections.** GLM's own indexer projections (``wq_b`` q-LoRA -> index
  queries; the fused ``wk_weights_proj`` hidden -> ``[head_dim, n_head]`` giving
  K + per-head weights; ``k_norm`` LayerNorm) and the softmax/head weight scaling
  the shipped ``Indexer`` applies.
* **GLM MLA-NoPE attention core.** GLM-5.3-Flash MLA is the **NoPE** variant
  (``qk_rope_head_dim == 0``): there is no decoupled RoPE tail. So, unlike
  ``deepseek_v41.mla`` (which always rotates a rope tail), the core here is a
  pure low-rank latent attention (q down/up + kv down/up via ``kv_b_proj``) with
  **no rope**, restricted to the indexer-selected tokens.

Dtypes come from the G3 policy (:data:`ASCEND_GLM5NEXT_W2_DTYPE_POLICY`): fp16
IO (``dsa`` / ``indexer`` sites), fp32 accumulation (``dsa_accumulation``) for
RMSNorm / softmax / matmul reduction. No dtype literals are spelled at compute
sites. A ``dtype=torch.float64`` override runs the whole module in float64 for
the parity harness (selection reduction is float64 regardless, matching the
deepseek_v41 host indexer, so selection equivalence is exact).

Sparsity + local window
-----------------------
Query ``t`` attends to the union of (a) the tokens of the top-k **fully-formed
visible** kpool blocks the indexer selected, and (b) its **current partial pool**
(the local causal window ``[floor(t/kpool)*kpool, t]``). The local window
guarantees non-empty, self-inclusive attention for the first ``kpool`` tokens
(before any block is formed) and mirrors the "always attend recent tokens"
behavior of sparse attention. Both sets are causal by construction (a visible
block is fully-formed => all its tokens are <= t), so the DSA output at ``t`` is
independent of tokens > ``t``.

NOTE (device parity follow-up): GLM's real kpool indexer also adds an absolute
position embedding (``index_kpool_compress_ape``) to the pooled keys and a gate
(``index_kpool_compress_gate``) before scoring. This host module runs the plain
mean-pool reuse (``use_compress_ape=False`` by default) which is the exact
deepseek_v41 selection; ``compress_ape`` is exposed as an opt-in adapter so a
later device-parity task can wire the GLM APE/gate without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.indexer import (
    block_topk_for_ratio,
    lightning_indexer_scores,
    mean_pool_compress,
    select_topk_blocks,
)

from .dtype_policy import (
    ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
    Glm5NextW2DtypePolicy,
)

# GLM-5.3-Flash DSA is the NoPE MLA variant: no decoupled RoPE tail.
GLM_QK_ROPE_HEAD_DIM = 0
# LayerNorm eps the shipped indexer uses on the compressed index-K.
INDEXER_KNORM_EPS = 1e-6


# ===========================================================================
# Small eager building blocks (formula-identical across fp16/fp32/fp64)
# ===========================================================================
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, accum_dtype: torch.dtype) -> torch.Tensor:
    """``x * rsqrt(mean(x^2)+eps) * weight`` over the last dim, in ``accum_dtype``."""
    orig_dtype = x.dtype
    xa = x.to(accum_dtype)
    w = weight.to(accum_dtype)
    variance = xa.square().mean(dim=-1, keepdim=True)
    normed = xa * torch.rsqrt(variance + eps) * w
    return normed.to(orig_dtype)


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    accum_dtype: torch.dtype,
) -> torch.Tensor:
    """Affine LayerNorm over the last dim, in ``accum_dtype`` (indexer k_norm)."""
    orig_dtype = x.dtype
    xa = x.to(accum_dtype)
    mean = xa.mean(dim=-1, keepdim=True)
    var = xa.var(dim=-1, keepdim=True, unbiased=False)
    normed = (xa - mean) * torch.rsqrt(var + eps) * weight.to(accum_dtype) + bias.to(accum_dtype)
    return normed.to(orig_dtype)


# ===========================================================================
# Indexer selection result
# ===========================================================================
@dataclass(frozen=True)
class DsaIndexerResult:
    """Result of one GLM DSA indexer forward pass.

    Attributes:
        blocks_per_token: ``[T]`` lists of selected kpool-block ids (descending
            score; causal visible range only) -- from the reused
            :func:`select_topk_blocks`.
        token_mask: ``[T, S]`` bool; ``True`` where query ``t`` may attend key
            ``j`` (selected-block tokens + local current-pool window, causal).
        scores: ``[T, N]`` block scores (accumulation dtype).
        block_topk: the per-token block budget used.
    """

    blocks_per_token: list[list[int]]
    token_mask: torch.Tensor
    scores: torch.Tensor
    block_topk: int


@dataclass(frozen=True)
class DsaSelection:
    """Selection surfaced from :meth:`AscendGlm5NextW2DSA.forward`."""

    blocks_per_token: list[list[int]]
    token_mask: torch.Tensor
    block_topk: int


# ===========================================================================
# GLM DSA kpool Lightning-Indexer (reuses deepseek_v41 selection functions)
# ===========================================================================
class Glm5NextW2DsaIndexer(nn.Module):
    """Host-eager kpool Lightning-Indexer for the 310P GLM DSA path.

    Projections mirror the shipped ``glm5next.attention.Indexer`` (``wq_b`` /
    fused ``wk_weights_proj`` / ``k_norm``); scoring + top-k selection are the
    reused deepseek_v41 functions. Selection accumulates in float64 (matching
    the deepseek_v41 host indexer) so reuse-equivalence is exact.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_kpool: int,
        layernorm_eps: float = INDEXER_KNORM_EPS,
        dtype_policy: Glm5NextW2DtypePolicy = ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        use_compress_ape: bool = False,
    ) -> None:
        super().__init__()
        self.dtype_policy = dtype_policy
        self.dtype = dtype if dtype is not None else dtype_policy.cast_site("indexer")
        self.eps = layernorm_eps
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.n_head = index_n_heads
        self.head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        # Scores + selection reduce in float64 to match the deepseek_v41 host
        # indexer exactly (a strict superset of the fp32 device accumulation).
        self.select_accum_dtype = torch.float64

        self.softmax_scale = self.head_dim**-0.5
        # weights_proj scaling the shipped Indexer applies.
        self.weight_scale = self.softmax_scale * self.n_head**-0.5
        self.block_topk = block_topk_for_ratio(self.index_topk, self.index_kpool)

        factory = {"dtype": self.dtype, "device": device}
        # wq_b: q-LoRA -> index queries [n_head*head_dim].
        self.wq_b = nn.Parameter(torch.empty(self.n_head * self.head_dim, q_lora_rank, **factory))
        # Fused wk + weights_proj: hidden -> [head_dim (K) ; n_head (weights)].
        self.wk_weights_proj = nn.Parameter(torch.empty(self.head_dim + self.n_head, hidden_size, **factory))
        # k_norm affine LayerNorm over the compressed index-K head dim.
        self.k_norm_weight = nn.Parameter(torch.empty(self.head_dim, **factory))
        self.k_norm_bias = nn.Parameter(torch.empty(self.head_dim, **factory))
        # Optional GLM kpool absolute-position embedding (device-parity opt-in).
        self.use_compress_ape = use_compress_ape
        if use_compress_ape:
            self.compress_ape: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.index_kpool, self.head_dim, **factory)
            )
        else:
            self.compress_ape = None
        self.reset_parameters()

    def reset_parameters(self, seed: int | None = None) -> None:
        gen = torch.Generator().manual_seed(0 if seed is None else seed)

        def fill(p: nn.Parameter, scale: float, bias: float = 0.0) -> None:
            vals = torch.randn(p.shape, generator=gen, dtype=torch.float64) * scale + bias
            with torch.no_grad():
                p.copy_(vals.to(p.dtype))

        fill(self.wq_b, 0.05)
        fill(self.wk_weights_proj, 0.05)
        fill(self.k_norm_weight, 0.0, bias=1.0)
        fill(self.k_norm_bias, 0.0)
        if self.compress_ape is not None:
            fill(self.compress_ape, 0.02)

    # -- projections --------------------------------------------------------
    def project_query(self, qr: torch.Tensor) -> torch.Tensor:
        """q-LoRA ``[T, q_lora_rank]`` -> index queries ``[T, n_head, head_dim]``."""
        q = qr.to(self.dtype) @ self.wq_b.t()
        return q.view(-1, self.n_head, self.head_dim)

    def project_k_and_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden ``[T, hidden]`` -> (normed K ``[T, head_dim]``, weights ``[T, n_head]``)."""
        kw = hidden_states.to(self.dtype) @ self.wk_weights_proj.t()
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]
        k = _layer_norm(k, self.k_norm_weight, self.k_norm_bias, self.eps, self.select_accum_dtype)
        weights = weights * self.weight_scale
        return k, weights

    # -- compression + scoring + selection (reused functions) --------------
    def _compress(self, raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
        compressed = mean_pool_compress(raw_keys, compress_ratio, accum_dtype=self.select_accum_dtype)
        if self.compress_ape is not None and compress_ratio == self.index_kpool:
            # Faithful GLM adapter: fold the per-pool-position APE into the pool
            # mean (kept off by default so the plain reuse path is exact).
            ape = self.compress_ape.to(self.select_accum_dtype).mean(dim=0, keepdim=True)
            compressed = compressed + ape
        return compressed

    def select_blocks(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
        compress_ratio: int | None = None,
    ) -> list[list[int]]:
        """Reused deepseek_v41 selection: compress -> lightning score -> top-k.

        Uses ``vllm_ascend.models.deepseek_v41.indexer`` verbatim so the GLM DSA
        selection is bit-identical to the DeepSeek indexer on shared inputs.
        """
        ratio = self.index_kpool if compress_ratio is None else compress_ratio
        compressed = self._compress(raw_keys, ratio)
        scores = lightning_indexer_scores(
            query, weights, compressed, self.softmax_scale, accum_dtype=self.select_accum_dtype
        )
        block_topk = block_topk_for_ratio(self.index_topk, ratio)
        return select_topk_blocks(scores, positions, ratio, block_topk)

    def score(self, query: torch.Tensor, weights: torch.Tensor, raw_keys: torch.Tensor) -> torch.Tensor:
        compressed = self._compress(raw_keys, self.index_kpool)
        return lightning_indexer_scores(
            query, weights, compressed, self.softmax_scale, accum_dtype=self.select_accum_dtype
        )

    # -- token mask (selected blocks + local current-pool window) ----------
    def token_mask_from_blocks(
        self,
        blocks_per_token: list[list[int]],
        positions: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Build the ``[T, S]`` bool attend-mask from per-token block selections.

        Allowed keys for query ``t`` = tokens of the selected fully-formed kpool
        blocks + the current partial pool ``[floor(t/kpool)*kpool, t]`` (local
        window). All causal by construction.
        """
        num_q = len(blocks_per_token)
        mask = torch.zeros(num_q, seq_len, dtype=torch.bool)
        kpool = self.index_kpool
        for t in range(num_q):
            pos = int(positions[t].item())
            # Selected fully-formed blocks -> their member tokens.
            for b in blocks_per_token[t]:
                lo = b * kpool
                hi = min(lo + kpool, seq_len)
                mask[t, lo:hi] = True
            # Local current-pool window (guarantees self-attention, no empty row).
            pool_start = (pos // kpool) * kpool
            mask[t, pool_start : pos + 1] = True
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
    ) -> DsaIndexerResult:
        """Project -> compress -> score -> select -> build attend-mask."""
        seq_len = hidden_states.shape[0]
        query = self.project_query(qr)
        raw_keys, weights = self.project_k_and_weights(hidden_states)
        compressed = self._compress(raw_keys, self.index_kpool)
        scores = lightning_indexer_scores(
            query, weights, compressed, self.softmax_scale, accum_dtype=self.select_accum_dtype
        )
        blocks_per_token = select_topk_blocks(scores, positions, self.index_kpool, self.block_topk)
        token_mask = self.token_mask_from_blocks(blocks_per_token, positions, seq_len)
        return DsaIndexerResult(
            blocks_per_token=blocks_per_token,
            token_mask=token_mask,
            scores=scores,
            block_topk=self.block_topk,
        )


# ===========================================================================
# GLM DSA layer: NoPE MLA-latent core, restricted to the indexer selection
# ===========================================================================
class AscendGlm5NextW2DSA(nn.Module):
    """Eager 310P GLM-5.3-Flash DSA layer (NoPE MLA core + reused indexer).

    Projection layout mirrors the shipped ``glm5next.attention.Glm5NextMLAAttention``
    (low-rank q down/up, shared kv down + ``kv_b_proj`` up into per-head
    ``[qk_nope_head_dim | v_head_dim]``, ``o_proj``), with ``qk_rope_head_dim==0``
    (NoPE) so there is no rope tail. Attention is restricted to the indexer's
    selected tokens (+ local pool window).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_kpool: int,
        rms_norm_eps: float = 1e-5,
        dtype_policy: Glm5NextW2DtypePolicy = ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dtype_policy = dtype_policy
        self.param_dtype = dtype if dtype is not None else dtype_policy.cast_site("dsa")
        is_fp64 = self.param_dtype == torch.float64
        # fp64 params keep fp64 accumulation so the parity harness is exact.
        self.accum_dtype = self.param_dtype if is_fp64 else dtype_policy.cast_site("dsa_accumulation")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = GLM_QK_ROPE_HEAD_DIM  # NoPE
        self.qk_head_dim = qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.eps = rms_norm_eps
        self.softmax_scale = self.qk_head_dim**-0.5

        H = self.num_heads
        factory = {"dtype": self.param_dtype, "device": device}
        # q low-rank down + norm + up (NoPE: up dim == qk_nope_head_dim).
        self.w_dq = nn.Parameter(torch.empty(q_lora_rank, hidden_size, **factory))
        self.q_a_norm = nn.Parameter(torch.empty(q_lora_rank, **factory))
        self.w_uq = nn.Parameter(torch.empty(H * self.qk_head_dim, q_lora_rank, **factory))
        # shared kv down + norm (NoPE: no rope tail, so kv_lora_rank only).
        self.w_dkv = nn.Parameter(torch.empty(kv_lora_rank, hidden_size, **factory))
        self.kv_a_norm = nn.Parameter(torch.empty(kv_lora_rank, **factory))
        # kv_b_proj: latent -> per-head [qk_nope_head_dim (K) | v_head_dim (V)].
        self.w_ukv = nn.Parameter(torch.empty(H * (qk_nope_head_dim + v_head_dim), kv_lora_rank, **factory))
        # output projection.
        self.w_o = nn.Parameter(torch.empty(hidden_size, H * v_head_dim, **factory))

        # Reused-selection kpool indexer (shares hidden + q-LoRA rank).
        indexer_dtype = self.param_dtype if is_fp64 else None
        self.indexer = Glm5NextW2DsaIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
            dtype_policy=dtype_policy,
            dtype=indexer_dtype,
            device=device,
        )
        self.reset_parameters()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        dtype_policy: Glm5NextW2DtypePolicy = ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> AscendGlm5NextW2DSA:
        """Build from an HF-style GLM-5.3-Flash text config (G7 entry point)."""

        def g(name: str, default: object = None) -> object:
            return getattr(config, name, default)

        return cls(
            hidden_size=int(g("hidden_size")),
            num_attention_heads=int(g("num_attention_heads")),
            q_lora_rank=int(g("q_lora_rank")),
            kv_lora_rank=int(g("kv_lora_rank")),
            qk_nope_head_dim=int(g("qk_nope_head_dim")),
            v_head_dim=int(g("v_head_dim")),
            index_n_heads=int(g("index_n_heads")),
            index_head_dim=int(g("index_head_dim")),
            index_topk=int(g("index_topk")),
            index_kpool=int(g("index_kpool")),
            rms_norm_eps=float(g("rms_norm_eps", 1e-5)),
            dtype_policy=dtype_policy,
            dtype=dtype,
            device=device,
        )

    def reset_parameters(self, seed: int | None = None) -> None:
        gen = torch.Generator().manual_seed(0 if seed is None else seed)

        def fill(p: nn.Parameter, scale: float, bias: float = 0.0) -> None:
            vals = torch.randn(p.shape, generator=gen, dtype=torch.float64) * scale + bias
            with torch.no_grad():
                p.copy_(vals.to(p.dtype))

        fill(self.w_dq, 0.05)
        fill(self.q_a_norm, 0.0, bias=1.0)
        fill(self.w_uq, 0.05)
        fill(self.w_dkv, 0.05)
        fill(self.kv_a_norm, 0.0, bias=1.0)
        fill(self.w_ukv, 0.05)
        fill(self.w_o, 0.05)
        # Reseed the indexer deterministically from the same seed.
        self.indexer.reset_parameters(seed=None if seed is None else seed + 1)

    # -- projections --------------------------------------------------------
    def _project_q(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden -> (q [T,H,qk_head_dim], q_latent [T,q_lora_rank]) after down/up+norm.

        The normed q-latent is also the indexer's ``qr`` input (GLM projects the
        index query off the shared q-LoRA rank).
        """
        q_latent = _rms_norm(hidden @ self.w_dq.t(), self.q_a_norm, self.eps, self.accum_dtype)
        q = (q_latent @ self.w_uq.t()).view(-1, self.num_heads, self.qk_head_dim)
        return q, q_latent

    def _project_kv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden -> (k_nope [T,H,nope], v [T,H,v_head]) via shared latent + kv_b_proj."""
        c = _rms_norm(hidden @ self.w_dkv.t(), self.kv_a_norm, self.eps, self.accum_dtype)
        kv = (c.to(self.accum_dtype) @ self.w_ukv.t().to(self.accum_dtype)).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        return k_nope, v

    # -- attention ----------------------------------------------------------
    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked softmax attention. ``mask`` is ``[T,S]`` bool (True = attend)."""
        q = q.to(self.accum_dtype)
        k = k.to(self.accum_dtype)
        v = v.to(self.accum_dtype)
        scores = torch.einsum("thd,shd->ths", q, k) * self.softmax_scale
        additive = torch.where(mask, 0.0, float("-inf")).to(self.accum_dtype)
        scores = scores + additive[:, None, :]
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum("ths,shv->thv", probs, v)  # [T,H,v]
        num_q = q.shape[0]
        out = context.reshape(num_q, -1) @ self.w_o.t().to(self.accum_dtype)
        return out

    def attend_with_mask(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the MLA-NoPE core under an explicit ``[T,S]`` bool mask.

        Exposed so tests (and G7) can drive the exact attention core with a
        chosen mask -- e.g. the indexer selection mask vs. a dense causal mask.
        """
        q, _ = self._project_q(hidden_states)
        k_nope, v = self._project_kv(hidden_states)
        out = self._attend(q, k_nope, v, mask)
        return out.to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        return_selection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DsaSelection]:
        """DSA forward: indexer top-k selection -> restricted MLA-NoPE attention.

        Args:
            hidden_states: ``[T, hidden]`` activations.
            positions: ``[T]`` integer positions (defaults to ``arange(T)``).
            return_selection: also return the :class:`DsaSelection` (block ids +
                token mask) the attention was restricted to.
        """
        num_q = hidden_states.shape[0]
        if positions is None:
            positions = torch.arange(num_q, device=hidden_states.device)

        q, q_latent = self._project_q(hidden_states)
        k_nope, v = self._project_kv(hidden_states)

        selection = self.indexer(hidden_states, q_latent, positions)
        out = self._attend(q, k_nope, v, selection.token_mask)
        out = out.to(hidden_states.dtype)

        if return_selection:
            return out, DsaSelection(
                blocks_per_token=selection.blocks_per_token,
                token_mask=selection.token_mask,
                block_topk=selection.block_topk,
            )
        return out


__all__ = [
    "GLM_QK_ROPE_HEAD_DIM",
    "INDEXER_KNORM_EPS",
    "AscendGlm5NextW2DSA",
    "DsaIndexerResult",
    "DsaSelection",
    "Glm5NextW2DsaIndexer",
]
