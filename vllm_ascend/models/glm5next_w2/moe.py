# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash routed-expert MoE on the 2-bit (W2) 310P path (plan G6).

This is the GLM / 310P *adaptation* of the shipped ``Glm5NextMoE`` routed path
(``vllm_ascend/models/glm5next/model.py``). The shipped module builds a
``FusedMoEFactory`` (which owns the router, the routed experts, the shared
expert and the ``routed_scaling_factor`` combine) and is driven, at the decoder
layer, by GLM's multi-head hyper-connection (``mhc``) pre/post ops. Neither the
``FusedMoEFactory`` nor the tilelang/Triton ``mhc`` kernels are available on the
Triton-free 310P lane, so this module keeps the exact same *math* while swapping
each piece for eager, host-parity-verified primitives:

1. **Router (host-side).** GLM scores the ``288`` routed-expert logits with a
   ``sigmoid`` (``scoring_func``), applies the ``noaux_tc`` selection bias
   (``e_score_correction_bias``) and group-limited routing (``n_group`` /
   ``topk_group``; a no-op for GLM's ``n_group=1``), keeps top-``8``, and
   renormalizes the kept *scores* (``norm_topk_prob=True``). The router runs
   *here* and produces ``topk_ids`` / ``topk_weights`` -- it does **not** run
   inside a ``FusedMoEFactory`` internal router. In the ``softmax`` /
   no-bias / single-group configuration :func:`glm_route_topk` reduces
   bit-for-bit to the DeepSeek E1.2
   :func:`~vllm_ascend.models.deepseek_v41.w2_unpack.route_topk_w2` (the shared
   W2 router), which is the host-parity anchor.

2. **Routed experts -> E1.3 W2 method.** The ``topk_ids`` / ``topk_weights`` are
   handed to :class:`AscendW2DynamicFusedMoEMethod310`
   (``_310p/quantization/methods/w2_dynamic.py``, resolved through the 310P
   registry), **not** a ``FusedMoEFactory``. Only the ``<= top_k`` active
   experts are widened from packed W2 -> INT8 (E1.2 ``unpack_active_experts``),
   never the full 288-expert bank. The shared expert is deliberately kept **out**
   of the method call so the routed output comes back un-scaled and the eager
   combine below owns ``routed_scaling_factor``.

3. **Eager MoE combine.** ``routed * routed_scaling_factor + shared`` via the
   reused DeepSeek :func:`eager_moe_combine` (a plain ``torch.addcmul``). GLM's
   ``routed_scaling_factor = 2.5``; the **shared expert stays FP16**
   (``policy.shared_expert_dtype``) and rides the residual un-scaled.

4. **Multi-head hyper-connection (``mhc``).** GLM wraps the FFN/MoE in a
   residual-stream hyper-connection: :func:`hc_pre` collapses the ``n`` residual
   streams into the single ``layer_input`` the MoE consumes, and :func:`hc_post`
   re-expands the MoE output back across the streams
   (``out_j = post_j * moe_out + sum_i comb_ij * residual_i``). These are a
   Triton-free, host-testable re-expression of upstream ``mhc_pre_torch`` /
   ``mhc_post_torch`` (``vllm.model_executor.kernels.mhc``); :func:`hc_expand` /
   :func:`hc_contract` are the residual-stream width helpers (mirroring the
   shipped ``glm5next/ops/mhc_ops.py``). The full ``mhc`` sinkhorn machinery and
   its per-layer parameters live on the decoder layer (wired at G7); this module
   owns the FFN-side pre/post application.

All dtypes are read from the authoritative G3 policy
(:data:`~vllm_ascend.models.glm5next_w2.dtype_policy.ASCEND_GLM5NEXT_W2_DTYPE_POLICY`),
never spelled as literals here.

Import hygiene (Triton-free)
----------------------------
Importing this module pulls in **only** ``torch``, the host-clean DeepSeek E1.2
``w2_unpack`` router + the reused eager combine / method resolver, and the G3
dtype policy -- never the shipped ``FusedMoEFactory``, the ``mhc`` tilelang
kernels, or the heavy 310P quantization-methods stack (which imports
``torch_npu``). The E1.3 method is resolved **lazily** at wire time only.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

# Reuse the DeepSeek W2 host router, eager combine and E1.3 method resolver: the
# plan mandates GLM reuse the DeepSeek W2 kernel + combine verbatim. All three
# are host-clean and Triton-free.
from vllm_ascend.models.deepseek_v41.moe import (
    eager_moe_combine,
    resolve_w2_moe_method,
)

from .dtype_policy import (
    ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
    Glm5NextW2DtypePolicy,
)

if TYPE_CHECKING:  # keep runtime imports light (no heavy vLLM / Triton pull)
    from vllm_ascend.quantization.methods.base import AscendMoEScheme

# ---------------------------------------------------------------------------
# GLM-5.3-Flash routed-MoE geometry. The authoritative values live on the HF
# text config (and mirror the shape contract in ``glm5next_w2/model.py`` and the
# W2 manifest ``artifacts/glm-5.3-flash-w2/manifest.json``); these named
# constants are the defaults used when a config is not supplied, so no bare
# magic numbers reach the routing / combine call sites (AGENTS.md).
# ---------------------------------------------------------------------------
GLM5NEXT_N_ROUTED_EXPERTS = 288
GLM5NEXT_NUM_EXPERTS_PER_TOK = 8  # top-8
GLM5NEXT_N_SHARED_EXPERTS = 1
GLM5NEXT_MOE_INTERMEDIATE_SIZE = 2048
GLM5NEXT_HIDDEN_SIZE = 4096
# GLM scales the routed contribution by 2.5 *outside* the router path (normalize
# top-k weights, then scale the routed output), matching the shipped
# ``Glm5NextMoE.routed_scaling_factor``.
GLM5NEXT_ROUTED_SCALING_FACTOR = 2.5
# ``norm_topk_prob``: renormalize the kept top-k router weights to sum to 1.
GLM5NEXT_NORM_TOPK_PROB = True
# GLM gates the router with a sigmoid (DeepSeek-style ``noaux_tc``) rather than a
# softmax; the selection adds ``e_score_correction_bias`` and is group-limited.
GLM5NEXT_SCORING_FUNC = "sigmoid"
GLM5NEXT_N_GROUP = 1
GLM5NEXT_TOPK_GROUP = 1
# Number of group-score contributors summed per group in DeepSeek-style
# group-limited routing (top-2 scores per group). Trivial for GLM (n_group=1).
_GROUP_SCORE_TOPK = 2

# GLM multi-head hyper-connection: number of residual streams. Derived from the
# checkpoint's ``hc_*_base`` width (``(2 + n) * n = 24`` -> ``n = 4``).
GLM5NEXT_MHC_NUM_RESIDUAL_STREAMS = 4

__all__ = [
    "GLM5NEXT_N_ROUTED_EXPERTS",
    "GLM5NEXT_NUM_EXPERTS_PER_TOK",
    "GLM5NEXT_N_SHARED_EXPERTS",
    "GLM5NEXT_MOE_INTERMEDIATE_SIZE",
    "GLM5NEXT_HIDDEN_SIZE",
    "GLM5NEXT_ROUTED_SCALING_FACTOR",
    "GLM5NEXT_NORM_TOPK_PROB",
    "GLM5NEXT_SCORING_FUNC",
    "GLM5NEXT_N_GROUP",
    "GLM5NEXT_TOPK_GROUP",
    "GLM5NEXT_MHC_NUM_RESIDUAL_STREAMS",
    "glm_route_topk",
    "resolve_w2_moe_method",
    "eager_moe_combine",
    "hc_expand",
    "hc_contract",
    "hc_pre",
    "hc_post",
    "Glm5NextW2MoE",
]


# ===========================================================================
# Router: sigmoid/softmax scoring -> noaux_tc bias + group-limited top-k -> renorm
# ===========================================================================


def glm_route_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    scoring_func: str = GLM5NEXT_SCORING_FUNC,
    renormalize: bool = GLM5NEXT_NORM_TOPK_PROB,
    e_score_correction_bias: torch.Tensor | None = None,
    n_group: int = GLM5NEXT_N_GROUP,
    topk_group: int = GLM5NEXT_TOPK_GROUP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GLM router: score -> ``noaux_tc`` bias + group-limited top-``k`` -> renorm.

    The GLM-5.3-Flash routed MoE gates its ``288`` experts with a ``sigmoid``
    (``scoring_func``); selection adds the ``noaux_tc`` correction bias
    (``e_score_correction_bias``) and is group-limited (``n_group`` /
    ``topk_group``). The returned *weights* are the original (un-biased) scores
    gathered at the selected experts, renormalized to sum to 1 when
    ``norm_topk_prob`` is set. The ``routed_scaling_factor`` is applied later, at
    the eager combine -- not here.

    In the ``scoring_func="softmax"`` / ``e_score_correction_bias=None`` /
    ``n_group=1`` configuration this reduces bit-for-bit to the DeepSeek E1.2
    :func:`~vllm_ascend.models.deepseek_v41.w2_unpack.route_topk_w2` (the shared
    W2 router), the host-parity anchor.

    Returns:
        ``(topk_ids[int64, [T, top_k]], topk_weights[float64, [T, top_k]])``.
    """
    logits = router_logits.double()
    if scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported scoring_func {scoring_func!r} (expected 'sigmoid' or 'softmax')")

    num_tokens, num_experts = scores.shape
    scores_for_choice = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.double().view(1, -1)

    if n_group > 1:
        # DeepSeek-style group-limited routing: rank groups by the sum of their
        # top-2 selection scores, keep ``topk_group`` groups, mask the rest.
        if num_experts % n_group:
            raise ValueError(f"num_experts {num_experts} not divisible by n_group {n_group}")
        experts_per_group = num_experts // n_group
        grouped = scores_for_choice.view(num_tokens, n_group, experts_per_group)
        group_contrib = min(_GROUP_SCORE_TOPK, experts_per_group)
        group_scores = grouped.topk(group_contrib, dim=-1).values.sum(dim=-1)
        keep_groups = torch.topk(group_scores, topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, keep_groups, 1.0)
        expert_mask = group_mask.unsqueeze(-1).expand(num_tokens, n_group, experts_per_group).reshape(
            num_tokens, num_experts
        )
        scores_for_choice = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))

    topk_ids = torch.topk(scores_for_choice, top_k, dim=-1).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids, topk_weights


# ===========================================================================
# Multi-head hyper-connection (mhc) FFN-side ops -- Triton-free host math
# ===========================================================================


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """``[s, hidden]`` -> ``[s, n, hidden]`` by replication (residual streams)."""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """``[s, n, hidden]`` -> ``[s, hidden]`` by averaging the residual streams."""
    return x.mean(dim=1)


def hc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre block: collapse residual streams and emit the post/comb mixes.

    A Triton-free, dtype-flexible re-expression of upstream
    ``vllm.model_executor.kernels.mhc.torch.mhc_pre_torch``. From the RMS-scaled
    mix logits ``residual @ fn.T`` it derives ``pre_mix`` (used to collapse the
    ``n`` residual streams into ``layer_input``), ``post_mix`` and the
    sinkhorn-normalized ``comb_mix`` (consumed by :func:`hc_post`).

    Args:
        residual: ``[..., n, hidden]`` residual streams.
        fn / hc_scale / hc_base: the layer's hyper-connection parameters
            (``fn: [(2 + n) * n, n * hidden]``, ``hc_scale: [3]``,
            ``hc_base: [(2 + n) * n]``).

    Returns:
        ``(post_mix[..., n, 1], comb_mix[..., n, n], layer_input[..., hidden])``.
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(compute_dtype)
    fn_c = fn.to(compute_dtype)
    hc_scale_c = hc_scale.to(compute_dtype)
    hc_base_c = hc_base.to(compute_dtype)

    mixes = torch.matmul(x, fn_c.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale_c[0] + hc_base_c[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale_c[1] + hc_base_c[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(num_tokens, hc_mult, hc_mult) * hc_scale_c[2] + hc_base_c[
        2 * hc_mult :
    ].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(compute_dtype), dim=1)
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size).to(residual.dtype),
    )


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """mHC post block: re-expand the layer output across the residual streams.

    ``out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i`` -- a
    Triton-free, dtype-flexible re-expression of upstream ``mhc_post_torch``.

    Args:
        x: ``[..., hidden]`` layer (MoE) output.
        residual: ``[..., n, hidden]`` incoming residual streams.
        post_layer_mix / comb_res_mix: the ``[..., n, 1]`` / ``[..., n, n]``
            mixes from :func:`hc_pre`.

    Returns:
        ``[..., n, hidden]`` updated residual streams (in ``residual.dtype``).
    """
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(compute_dtype),
        residual.to(compute_dtype),
    )
    post_term = post_layer_mix.to(compute_dtype) * x.unsqueeze(-2).to(compute_dtype)
    return (mixed_residual + post_term).to(residual.dtype)


# ===========================================================================
# GLM-5.3-Flash W2 routed-expert MoE
# ===========================================================================


class Glm5NextW2MoE(nn.Module):
    """GLM routed-expert MoE: host router -> E1.3 W2 method -> eager combine.

    The Triton-free 310P adaptation of the shipped ``Glm5NextMoE`` routed path.
    It owns the router (GLM sigmoid ``noaux_tc`` top-``8`` of ``288`` +
    ``norm_topk_prob``), dispatches the routed experts through the E1.3
    :class:`AscendW2DynamicFusedMoEMethod310` (W2 packed weights + INT8 dynamic
    activations, active-expert unpack only), and combines with an eager
    ``routed * 2.5 + shared`` in place of the ``FusedMoEFactory``. The
    hyper-connection wrapping (:func:`hc_pre` / :func:`hc_post`) is applied by
    :meth:`forward_hyper_connection`.

    The packed W2 expert bank (``w2_experts``) and FP16 ``shared_expert`` are
    populated by the streamed W2 loader; the E1.3 method is resolved lazily on
    first use (or injected for tests).
    """

    def __init__(
        self,
        *,
        num_experts: int = GLM5NEXT_N_ROUTED_EXPERTS,
        top_k: int = GLM5NEXT_NUM_EXPERTS_PER_TOK,
        renormalize: bool = GLM5NEXT_NORM_TOPK_PROB,
        routed_scaling_factor: float = GLM5NEXT_ROUTED_SCALING_FACTOR,
        scoring_func: str = GLM5NEXT_SCORING_FUNC,
        n_group: int = GLM5NEXT_N_GROUP,
        topk_group: int = GLM5NEXT_TOPK_GROUP,
        method: AscendMoEScheme | None = None,
        w2_experts: list[Any] | None = None,
        shared_expert: Callable[[torch.Tensor], torch.Tensor] | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        dtype_policy: Glm5NextW2DtypePolicy | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.scoring_func = str(scoring_func)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.dtype_policy = dtype_policy or ASCEND_GLM5NEXT_W2_DTYPE_POLICY
        # The packed W2 bank (streamed loader) and always-on FP16 shared expert.
        self.w2_experts = w2_experts
        # ``shared_expert`` may be an nn.Module (the shipped Glm5NextMLP, FP16) or
        # any callable; nn.Module stores a non-module callable as a plain attr.
        self.shared_expert = shared_expert
        # The ``noaux_tc`` selection bias (gate.e_score_correction_bias), FP32.
        self.e_score_correction_bias = e_score_correction_bias
        # Resolved lazily (see :attr:`method`) so import stays Triton-free.
        self._method = method

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        method: AscendMoEScheme | None = None,
        w2_experts: list[Any] | None = None,
        shared_expert: Callable[[torch.Tensor], torch.Tensor] | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        dtype_policy: Glm5NextW2DtypePolicy | None = None,
    ) -> Glm5NextW2MoE:
        """Build from an HF config, reading the GLM routed-MoE geometry.

        Reads ``n_routed_experts`` / ``num_experts_per_token`` / ``moe_renormalize``
        / ``routed_scaling_factor`` / ``scoring_func`` / ``n_group`` / ``topk_group``
        off ``config`` (falling back to the GLM defaults), so the G7 assembly can
        wire straight from the shipped ``Glm5NextMoE``'s config without
        re-spelling the geometry.
        """
        # GLM configs spell top-k as ``num_experts_per_token`` (the shipped
        # ``Glm5NextMoE`` reads that); accept ``num_experts_per_tok`` too.
        top_k = getattr(config, "num_experts_per_token", None)
        if top_k is None:
            top_k = getattr(config, "num_experts_per_tok", GLM5NEXT_NUM_EXPERTS_PER_TOK)
        renormalize = getattr(config, "moe_renormalize", None)
        if renormalize is None:
            renormalize = getattr(config, "norm_topk_prob", GLM5NEXT_NORM_TOPK_PROB)
        return cls(
            num_experts=getattr(config, "n_routed_experts", GLM5NEXT_N_ROUTED_EXPERTS),
            top_k=top_k,
            renormalize=renormalize,
            routed_scaling_factor=getattr(config, "routed_scaling_factor", GLM5NEXT_ROUTED_SCALING_FACTOR),
            scoring_func=getattr(config, "scoring_func", GLM5NEXT_SCORING_FUNC),
            n_group=getattr(config, "n_group", GLM5NEXT_N_GROUP),
            topk_group=getattr(config, "topk_group", GLM5NEXT_TOPK_GROUP),
            method=method,
            w2_experts=w2_experts,
            shared_expert=shared_expert,
            e_score_correction_bias=e_score_correction_bias,
            dtype_policy=dtype_policy,
        )

    @property
    def method(self) -> AscendMoEScheme:
        """The E1.3 W2 fused-MoE method (resolved lazily on first access)."""
        if self._method is None:
            self._method = resolve_w2_moe_method()
        return self._method

    # -- router -------------------------------------------------------------

    def route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GLM host router (see :func:`glm_route_topk`).

        Returns:
            ``(topk_ids[int64, [T, top_k]], topk_weights[float64, [T, top_k]])``.
        """
        return glm_route_topk(
            router_logits,
            self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            e_score_correction_bias=self.e_score_correction_bias,
            n_group=self.n_group,
            topk_group=self.topk_group,
        )

    # -- routed experts (E1.3 method) --------------------------------------

    def routed_experts_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        experts: list[Any] | None = None,
    ) -> torch.Tensor:
        """Dispatch the routed experts through the E1.3 W2 method.

        Hands the pre-selected ``topk_ids`` / ``topk_weights`` to
        :meth:`AscendW2DynamicFusedMoEMethod310.apply`. The shared expert is held
        back (``w2_shared_expert=None``) so the routed output is *un-scaled* and
        the eager combine owns ``routed_scaling_factor``. The method unpacks only
        the ``<= top_k`` active experts and takes its host path on CPU (device
        INT8 grouped matmul re-expressed via the E1.2 active-expert unpack).
        """
        experts = experts if experts is not None else self.w2_experts
        if experts is None:
            raise ValueError(
                "Glm5NextW2MoE.routed_experts_forward requires the packed W2 expert bank; "
                "set `w2_experts` (populated by the streamed W2 loader) or pass `experts=`."
            )
        method_layer = types.SimpleNamespace(w2_experts=experts, w2_shared_expert=None)
        return self.method.apply(method_layer, hidden_states, topk_weights, topk_ids, None, None)

    # -- eager combine ------------------------------------------------------

    def combine(self, routed: torch.Tensor, shared_output: torch.Tensor | None) -> torch.Tensor:
        """Eager ``routed * routed_scaling_factor + shared`` (see :func:`eager_moe_combine`)."""
        return eager_moe_combine(
            routed,
            shared_output,
            self.routed_scaling_factor,
            accumulation_dtype=self.dtype_policy.accumulation_dtype,
        )

    # -- full forward -------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_output: torch.Tensor | None = None,
        *,
        experts: list[Any] | None = None,
    ) -> torch.Tensor:
        """Routed-expert MoE forward: route -> W2 experts -> eager combine.

        Args:
            hidden_states: ``[T, hidden]`` activations.
            router_logits: ``[T, num_experts]`` gate logits (from the shipped
                gate linear, upstream).
            shared_output: optional pre-computed FP16 shared-expert output. When
                ``None`` and :attr:`shared_expert` is set, it is computed here
                (FP16, ``policy.shared_expert_dtype``).
            experts: optional explicit packed W2 bank (else :attr:`w2_experts`).

        Returns:
            ``[T, hidden]`` combined output in the promoted accumulation
            precision (the caller casts to ``main_dtype``).
        """
        topk_ids, topk_weights = self.route(router_logits)
        routed = self.routed_experts_forward(hidden_states, topk_ids, topk_weights, experts=experts)
        if shared_output is None and self.shared_expert is not None:
            # Shared expert stays FP16 and rides the residual un-scaled.
            shared_output = self.shared_expert(hidden_states)
        return self.combine(routed, shared_output)

    # -- hyper-connection-wrapped forward ----------------------------------

    def forward_hyper_connection(
        self,
        residual_streams: torch.Tensor,
        router_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        experts: list[Any] | None = None,
    ) -> torch.Tensor:
        """FFN-side multi-head hyper-connection around the W2 MoE.

        Collapses the ``n`` residual streams into the MoE's ``layer_input``
        (:func:`hc_pre`), computes ``router_logits = router_fn(layer_input)``,
        runs the W2 routed+shared MoE, then re-expands the output back across the
        residual streams (:func:`hc_post`). This is the FFN half of GLM's
        ``mhc``; the decoder layer (G7) supplies the per-layer ``hc_*`` params.

        Returns:
            ``[..., n, hidden]`` updated residual streams.
        """
        post_mix, comb_mix, layer_input = hc_pre(
            residual_streams,
            fn,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        router_logits = router_fn(layer_input)
        moe_out = self.forward(layer_input, router_logits, experts=experts)
        return hc_post(moe_out, residual_streams, post_mix, comb_mix)
