# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch eager reference for GLM-5.3-Flash KDA linear attention.

This module is the **parity oracle** for the 310P Triton-free KDA port
(plan task G4 of ``glm53-flash-w2-310p-plan.md``). It reimplements the exact
math of the shipped 910/Triton path -- ``vllm_ascend/models/glm5next/kda.py``
and the kernels under ``vllm_ascend/ops/triton/kda/`` -- as a transparent,
sequential recurrence in plain fp32 PyTorch. NO NPU, NO Triton, NO custom ops:
just ``torch``. It is not performance-oriented; it is meant to be obviously
correct so G4 can diff its (fast, fused) kernels against it.

KDA = Kimi Delta Attention: a gated delta-rule linear attention (chunked /
recurrent linear attn with a data-dependent forget gate and a delta-rule state
update), closely related to Gated DeltaNet (GDN). GLM-5.3-Flash config:
``num_heads=64``, ``head_dim=128``, ``short_conv_kernel_size=4``, bounded
("safe") per-head-per-channel gate with ``gate_lower_bound=-5.0``.

Correspondence to the shipped code (source of truth), step by step:

1. Short depthwise causal conv1d + SiLU over q/k/v
   -> ``kda.py`` ``causal_conv1d_fn``/``causal_conv1d_update`` calls, whose torch
      semantics live in ``vllm_ascend/ops/causal_conv1d.py::causal_conv1d_ref``
      (``F.conv1d`` with ``padding=width-1`` truncated to seqlen, then
      ``F.silu``). Implemented here by :func:`short_conv1d_causal`.
2. Bounded "safe" decay gate
   -> ``vllm_ascend/ops/triton/kda/gate.py::apply_kda_gate`` with
      ``safe_gate=True``: ``g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``.
      Implemented here by :func:`kda_safe_gate`.
3. beta = sigmoid(beta_raw)
   -> ``kda.py`` passes ``sigmoid_beta=True`` to ``fused_recurrent_kda`` (decode)
      and ``_cast_sigmoid(beta)`` to the chunk path (prefill). Scalar per head
      (``IS_BETA_HEADWISE = beta.ndim == v.ndim`` is False for GLM).
      Implemented here inside :func:`kda_recurrent_reference` (``beta_is_raw``).
4. The gated delta-rule recurrence itself, per head, fp32 state ``S`` of shape
   ``[V, K] = [head_dim, head_dim]``
   -> ``vllm_ascend/ops/triton/kda/fused_recurrent_kda.py`` inner loop
      (``IS_KDA=True`` branch). Per timestep:
        q, k   <- L2-normalize over the last dim (``USE_QK_L2NORM_IN_KERNEL``)
        q      <- q * scale          (scale = K ** -0.5, from kda wrapper)
        S      <- S * exp(g)[None, :]         # decay along K columns
        u      <- v - (S * k[None, :]).sum(-1)  # delta rule: v - S @ k
        u      <- u * beta                      # per-head scalar gate
        S      <- S + u[:, None] * k[None, :]   # rank-1 state update
        o      <- (S * q[None, :]).sum(-1)      # read-out: S @ q
      Implemented here by :func:`kda_recurrent_reference`.
5. Output gated RMSNorm + o_proj
   -> ``kda.py`` ``self.o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")``
      then ``o_proj`` (RowParallelLinear). ``FusedRMSNormGated.forward_native``
      (vllm ``model_executor/layers/fla/ops/kda.py``) is
      ``rmsnorm(x)*weight * sigmoid(g)``. Implemented here by
      :func:`gated_rmsnorm` and tied together in :func:`kda_layer_reference`.

Assumptions / ambiguities for G4 to reconcile (documented per the task):
  * L2-norm epsilon: the kernel uses ``1e-6`` (``b_q / sqrt(sum(b_q^2) + 1e-6)``).
    Matched here.
  * ``scale`` defaults to ``K ** -0.5`` (=head_dim**-0.5) in the kda wrapper
    (``fused_recurrent_kda_fwd``/``chunk_kda`` when ``scale is None``). Matched.
  * o_norm eps defaults to ``1e-5`` (``FusedRMSNormGated`` default). Matched.
  * Gate is applied as ``S *= exp(g)`` where ``g`` is the already-bounded decay
    (negative). The chunked prefill kernel computes cumulative sums in log space
    and may reorder fp32 adds vs. this strict left-to-right recurrence; small
    numeric drift there is expected -- this sequential form is the reference.
  * ``beta`` is a per-head scalar for GLM (not head-wise per channel). If a
    future checkpoint ships head-wise beta, pass ``beta`` of shape [T, H, V].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Defaults mirrored from the shipped code (cited above).
KDA_GATE_LOWER_BOUND = -5.0          # config.linear_lower_bound / gate.py default
L2NORM_EPS = 1e-6                    # fused_recurrent_kda.py inner loop
O_NORM_EPS = 1e-5                    # FusedRMSNormGated default eps
GLM_NUM_HEADS = 64                   # config.linear_num_heads
GLM_HEAD_DIM = 128                   # config.linear_head_dim
GLM_SHORT_CONV_KERNEL = 4           # config.linear_conv_kernel_dim


def short_conv1d_causal(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Depthwise causal conv1d + optional SiLU over the time axis.

    Mirrors ``vllm_ascend/ops/causal_conv1d.py::causal_conv1d_ref`` (which is
    what ``kda.py`` routes to on Ascend): ``F.conv1d`` with ``groups=dim`` and
    ``padding=width-1``, truncated to the original seqlen so output at time t
    only sees inputs <= t (causal), then SiLU.

    Args:
        x: [T, C] channel-last time series (T tokens, C channels).
        weight: [C, width] depthwise kernel (one filter of length ``width`` per
            channel). ``width`` == ``short_conv_kernel_size`` (4 for GLM).
        bias: optional [C] bias.
        activation: "silu"/"swish" or None.

    Returns:
        [T, C] fp32 convolved+activated sequence.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    x = x.float()
    seqlen, dim = x.shape
    kdim, width = weight.shape
    if kdim != dim:
        raise ValueError(f"conv weight channels {kdim} != x channels {dim}")
    # [T, C] -> [1, C, T] for depthwise conv1d.
    x_ct = x.transpose(0, 1).unsqueeze(0)
    w = weight.float().unsqueeze(1)  # [C, 1, width]
    b = None if bias is None else bias.float()
    out = F.conv1d(x_ct, w, b, padding=width - 1, groups=dim)
    out = out[..., :seqlen]                # causal truncation
    out = out.squeeze(0).transpose(0, 1)   # back to [T, C]
    if activation is not None:
        out = F.silu(out)
    return out


def kda_safe_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float = KDA_GATE_LOWER_BOUND,
) -> torch.Tensor:
    """Bounded ("safe") KDA decay gate in log space.

    Reference: ``vllm_ascend/ops/triton/kda/gate.py::apply_kda_gate`` with
    ``safe_gate=True`` -- ``lower_bound * sigmoid(exp(a_log) * (raw_g + dt_bias))``.
    The result is a per-(token, head, channel) *log-decay* in
    ``(lower_bound, 0)`` (negative), consumed by the recurrence as ``exp(g)``.

    Args:
        raw_g: [..., H, D] raw gate projection (``f_b_proj(f_a)`` in kda.py).
        a_log: broadcastable to [H, 1] (kda.py stores it [1,1,H,1]); ``exp(a_log)``
            is the per-head input scale ``A``.
        dt_bias: optional per-(H, D) or [H*D] additive bias (``self.dt_bias``).
        lower_bound: gate floor (``config.linear_lower_bound``, -5.0).

    Returns:
        fp32 tensor shaped like ``raw_g`` holding the log-decay.
    """
    g = raw_g.float()
    heads = a_log.numel()
    if g.shape[-2] != heads:
        raise ValueError(
            f"KDA gate head mismatch: raw_g {tuple(raw_g.shape)} vs A_log numel {heads}"
        )
    a = a_log.reshape(*([1] * (g.dim() - 2)), heads, 1).to(dtype=torch.float32, device=g.device)
    if dt_bias is not None:
        bias = dt_bias.to(dtype=torch.float32, device=g.device).reshape(
            *([1] * (g.dim() - 2)), heads, g.shape[-1]
        )
        g = g + bias
    return lower_bound * torch.sigmoid(torch.exp(a) * g)


def gated_rmsnorm(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = O_NORM_EPS,
) -> torch.Tensor:
    """Sigmoid-gated RMSNorm over the last dim.

    Reference: ``FusedRMSNormGated(activation="sigmoid").forward_native`` in
    vllm ``model_executor/layers/fla/ops/kda.py`` -- normalize x by its RMS over
    the last dim, scale by ``weight``, then multiply by ``sigmoid(g)``.

    Args:
        x: [..., D] core attention output.
        g: [..., D] gate (``g_b_proj(g_a)`` in kda.py).
        weight: optional [D] affine weight.
        eps: RMS epsilon (1e-5).
    """
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.float()
    return x_normed * torch.sigmoid(g.float())


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    beta_is_raw: bool = True,
    l2norm_eps: float = L2NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit sequential KDA gated-delta-rule recurrence (the parity oracle).

    Mirrors the ``IS_KDA=True`` inner loop of
    ``vllm_ascend/ops/triton/kda/fused_recurrent_kda.py`` exactly, done eagerly
    in fp32. Processes one sequence (batch dim optional but must be 1). Per head
    ``h`` the recurrent state ``S`` is ``[V, K] = [head_dim, head_dim]``.

    Args:
        q, k: [T, H, K] post-conv query/key (K = head_dim).
        v: [T, H, V] post-conv value (V = head_dim).
        g: [T, H, K] **log-decay** gate already produced by :func:`kda_safe_gate`.
        beta: [T, H] per-head gate. Raw (pre-sigmoid) if ``beta_is_raw`` (matches
            kda.py ``sigmoid_beta=True`` / ``_cast_sigmoid``); else already in (0,1).
        scale: query scale; defaults to ``K ** -0.5`` (kda wrapper default).
        initial_state: optional [H, V, K] carry-in state (zeros if None).
        use_qk_l2norm: L2-normalize q and k over the last dim before use.
        beta_is_raw: apply ``sigmoid`` to beta inside (see above).
        l2norm_eps: epsilon for the in-kernel L2 norm (1e-6).

    Returns:
        (out, final_state):
          out: [T, H, V] fp32 core attention output (fp16-castable).
          final_state: [H, V, K] fp32 recurrent state after the last token.
    """
    # Accept an optional leading batch dim of size 1 (kda.py uses [1, T, H, D]).
    squeezed = False
    if q.dim() == 4:
        if q.shape[0] != 1:
            raise ValueError("batched recurrence expects batch size 1")
        q, k, v, g = q[0], k[0], v[0], g[0]
        beta = beta[0] if beta.dim() == 3 else beta
        squeezed = True

    T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    if beta_is_raw:
        beta = torch.sigmoid(beta)

    if use_qk_l2norm:
        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + l2norm_eps)
        k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + l2norm_eps)
    q = q * scale

    if initial_state is None:
        state = torch.zeros(H, V, K, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().clone()

    out = torch.empty(T, H, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        q_t = q[t]            # [H, K]
        k_t = k[t]            # [H, K]
        v_t = v[t]            # [H, V]
        g_t = g[t]            # [H, K]  (log-decay)
        beta_t = beta[t]      # [H]

        # S <- S * exp(g)[None, :]   (decay along K columns, broadcast over V)
        state = state * torch.exp(g_t).unsqueeze(1)          # [H, V, K]
        # u <- v - (S * k[None, :]).sum(-1)   == v - S @ k
        Sk = (state * k_t.unsqueeze(1)).sum(dim=-1)          # [H, V]
        u = v_t - Sk                                         # [H, V]
        # u <- u * beta   (per-head scalar)
        u = u * beta_t.unsqueeze(-1)                         # [H, V]
        # S <- S + u[:, None] * k[None, :]   (rank-1 update)
        state = state + u.unsqueeze(-1) * k_t.unsqueeze(1)   # [H, V, K]
        # o <- (S * q[None, :]).sum(-1)   == S @ q
        out[t] = (state * q_t.unsqueeze(1)).sum(dim=-1)      # [H, V]

    if squeezed:
        out = out.unsqueeze(0)
    return out, state


def kda_chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """State-carrying chunked driver over :func:`kda_recurrent_reference`.

    This is NOT the parallel intra-chunk algorithm of ``chunk_kda`` -- it simply
    runs the exact sequential recurrence one chunk at a time, threading the
    recurrent state across chunk boundaries. Its purpose is a consistency oracle:
    a correct chunked/streaming implementation (like G4's) must equal the whole-
    sequence recurrence, which this proves the recurrence supports. Returns the
    same ``(out, final_state)`` as the recurrent form and must match it exactly.
    """
    if q.dim() == 4:
        q, k, v, g = q[0], k[0], v[0], g[0]
        if beta.dim() == 3:
            beta = beta[0]
    T = q.shape[0]
    state = initial_state
    outs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        b_chunk = beta[start:end]
        o, state = kda_recurrent_reference(
            q[start:end], k[start:end], v[start:end], g[start:end], b_chunk,
            initial_state=state, **kwargs,
        )
        outs.append(o)
    return torch.cat(outs, dim=0), state


def kda_layer_reference(
    hidden_or_qkv,
    *,
    conv_weight: torch.Tensor,
    raw_g: torch.Tensor,
    beta_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    g_out: torch.Tensor,
    o_norm_weight: torch.Tensor | None = None,
    conv_bias: torch.Tensor | None = None,
    lower_bound: float = KDA_GATE_LOWER_BOUND,
    num_heads: int = GLM_NUM_HEADS,
    head_dim: int = GLM_HEAD_DIM,
    apply_conv: bool = True,
) -> torch.Tensor:
    """End-to-end KDA core (conv -> gate -> recurrence -> gated RMSNorm).

    Ties the pieces together in the order of ``kda.py``'s forward, stopping just
    before ``o_proj`` (a plain RowParallelLinear the caller applies). Returns the
    normalized/gated core output [T, H, head_dim], fp16-castable.

    Args:
        hidden_or_qkv: fused q|k|v projection [T, 3*H*head_dim] (the ``qkv`` split
            in kda.py, before the short conv).
        conv_weight: merged q|k|v depthwise conv weight [3*H*head_dim, width].
        raw_g: gate projection [T, H*head_dim] (``f_b_proj(f_a)``).
        beta_raw: raw beta [T, H] (``b`` shard, pre-sigmoid).
        a_log: [H] or broadcastable (``self.A_log``).
        dt_bias: [H*head_dim] or None (``self.dt_bias``).
        g_out: output gate [T, H*head_dim] (``g_b_proj(g_a)``).
        o_norm_weight: [head_dim] o_norm affine weight (or None).
        conv_bias: optional conv bias.
        apply_conv: run the short conv (set False if q|k|v are already convolved).
    """
    qkv = hidden_or_qkv.float()
    T = qkv.shape[0]
    proj = num_heads * head_dim
    if apply_conv:
        qkv = short_conv1d_causal(qkv, conv_weight, conv_bias, activation="silu")
    q, k, v = qkv.split(proj, dim=-1)
    q = q.reshape(T, num_heads, head_dim)
    k = k.reshape(T, num_heads, head_dim)
    v = v.reshape(T, num_heads, head_dim)

    g_log = kda_safe_gate(
        raw_g.reshape(T, num_heads, head_dim), a_log, dt_bias, lower_bound=lower_bound
    )
    core, _ = kda_recurrent_reference(q, k, v, g_log, beta_raw, beta_is_raw=True)

    g2 = g_out.reshape(T, num_heads, head_dim)
    return gated_rmsnorm(core, g2, o_norm_weight)
