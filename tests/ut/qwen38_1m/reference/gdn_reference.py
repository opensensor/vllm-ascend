# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for Qwen4Exp Gated DeltaNet (GDN).

Ports the *formulas* (not Triton) from:
  * ``vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py``
    (``fused_sigmoid_gating_delta_rule_update_kernel``) -- the gating math and
    the token-recurrent delta rule.
  * ``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`` and the CPU
    op ``.../mamba/ops/cpu/gdn_attention.py`` -- how conv + gating + recurrence
    compose.

Two mathematically-identical forms are provided:
  * :func:`gdn_delta_rule_recurrent` -- the exact token-by-token recurrence used
    by the fused CUDA/CPU decode kernel (the golden reference).
  * :func:`gdn_delta_rule_chunked` -- a chunk-parallel UT/WY-style block form,
    the structure the prefill (chunked) path uses.

State convention matches the fused kernel: ``S[h, v, k]`` (value-major), so
``o = S @ q`` and the delta update is ``S += beta*(v - S@k) outer k``.

The gating exactly mirrors the kernel:
    x          = a + dt_bias
    softplus_x = softplus(x)  (linear above ``threshold`` when beta==1)
    g          = -exp(A_log) * softplus_x        # per (token, head), <= 0
    beta_gate  = sigmoid(b)
"""

from __future__ import annotations

import torch

# Kernel constants (fused_sigmoid_gating_delta_rule_update defaults).
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0
_L2NORM_EPS = 1e-6


def gdn_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    softplus_beta: float = _SOFTPLUS_BETA,
    softplus_threshold: float = _SOFTPLUS_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-(token, head) log-decay ``g`` and gate ``beta_gate``.

    Args:
        a, b: ``[T, H]`` selective projections.
        A_log, dt_bias: ``[H]`` per-head parameters.

    Returns:
        ``(g, beta_gate)`` each ``[T, H]``. ``g`` is the log decay (<= 0).
    """
    a = a.double()
    b = b.double()
    A_log = A_log.double()
    dt_bias = dt_bias.double()
    x = a + dt_bias
    bx = softplus_beta * x
    softplus_x = torch.where(
        bx <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log1p(torch.exp(bx)),
        x,
    )
    g = -torch.exp(A_log) * softplus_x
    beta_gate = torch.sigmoid(b)
    return g, beta_gate


def _l2norm(x: torch.Tensor, eps: float = _L2NORM_EPS) -> torch.Tensor:
    """L2 normalize the last dim as the kernel does: ``x * rsqrt(sum(x^2)+eps)``."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def preprocess_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply optional L2 norm then the query scale, matching the kernel order.

    ``q`` and ``k`` are ``[T, H, K]``. ``scale`` defaults to ``K ** -0.5``.
    """
    q = q.double()
    k = k.double()
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    return q, k


def gdn_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_gate: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token gated delta rule (golden form).

    Args:
        q, k: ``[T, H, K]`` (already preprocessed via :func:`preprocess_qk`).
        v: ``[T, H, V]``.
        g, beta_gate: ``[T, H]`` from :func:`gdn_gating`.
        initial_state: ``[H, V, K]`` or ``None`` (zeros).

    Returns:
        ``(o, final_state)`` with ``o`` = ``[T, H, V]`` and state ``[H, V, K]``.
    """
    q, k, v = q.double(), k.double(), v.double()
    g, beta_gate = g.double(), beta_gate.double()
    seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(num_heads, v_dim, k_dim, dtype=torch.float64)
    else:
        state = initial_state.double().clone()

    outputs = torch.empty(seq_len, num_heads, v_dim, dtype=torch.float64)
    for t in range(seq_len):
        decay = torch.exp(g[t]).view(num_heads, 1, 1)
        state = state * decay
        # u = beta * (v - S @ k)
        s_k = torch.einsum("hvk,hk->hv", state, k[t])
        u = (v[t] - s_k) * beta_gate[t].unsqueeze(-1)
        state = state + torch.einsum("hv,hk->hvk", u, k[t])
        outputs[t] = torch.einsum("hvk,hk->hv", state, q[t])
    return outputs, state


def gdn_delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_gate: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel gated delta rule (UT/WY block form).

    Mathematically identical to :func:`gdn_delta_rule_recurrent`; it reassociates
    the recurrence into per-chunk triangular solves, mirroring the prefill
    (chunked) kernel. See module docstring for the derivation shapes.
    """
    q, k, v = q.double(), k.double(), v.double()
    g, beta_gate = g.double(), beta_gate.double()
    seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(num_heads, v_dim, k_dim, dtype=torch.float64)
    else:
        state = initial_state.double().clone()

    outputs = torch.empty(seq_len, num_heads, v_dim, dtype=torch.float64)
    eye = torch.eye(chunk_size, dtype=torch.float64)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        c = end - start
        qc = q[start:end]  # [c, H, K]
        kc = k[start:end]
        vc = v[start:end]  # [c, H, V]
        gc = g[start:end]  # [c, H]
        bc = beta_gate[start:end]  # [c, H]

        cumsum_g = torch.cumsum(gc, dim=0)  # [c, H] inclusive
        cumsum_h = cumsum_g.transpose(0, 1)  # [H, c]
        gamma = torch.exp(cumsum_h)  # [H, c]
        # decay[h, i, j] = exp(G_i - G_j)
        decay = torch.exp(cumsum_h[:, :, None] - cumsum_h[:, None, :])  # [H, c, c]

        kk = torch.einsum("ihk,jhk->hij", kc, kc)  # [H, c, c]
        beta_h = bc.transpose(0, 1)  # [H, c]

        strict_lower = torch.tril(torch.ones(c, c, dtype=torch.float64), diagonal=-1)
        a_mat = beta_h[:, :, None] * decay * kk * strict_lower  # [H, c, c]
        m_mat = eye[:c, :c].unsqueeze(0) + a_mat
        t_mat = torch.linalg.solve_triangular(m_mat, eye[:c, :c].unsqueeze(0).expand(num_heads, c, c), upper=False)

        bv = (beta_h.transpose(0, 1)[:, :, None] * vc).transpose(0, 1)  # [H, c, V]
        s_k_in = torch.einsum("hvk,ihk->hiv", state, kc)  # [H, c, V]
        state_term = (beta_h * gamma)[:, :, None] * s_k_in  # [H, c, V]
        rhs = bv - state_term  # [H, c, V]
        u_mat = torch.matmul(t_mat, rhs)  # [H, c, V]

        s_q_in = torch.einsum("hvk,ihk->hiv", state, qc)  # [H, c, V]
        qk = torch.einsum("ihk,jhk->hij", qc, kc)  # [H, c, c]
        lower_incl = torch.tril(torch.ones(c, c, dtype=torch.float64), diagonal=0)
        b_mat = decay * qk * lower_incl  # [H, c, c]
        o_intra = torch.matmul(b_mat, u_mat)  # [H, c, V]
        o_chunk = gamma[:, :, None] * s_q_in + o_intra  # [H, c, V]
        outputs[start:end] = o_chunk.transpose(0, 1)

        g_last = cumsum_h[:, -1]  # [H]
        wj = torch.exp(g_last[:, None] - cumsum_h)  # [H, c]
        uw = u_mat * wj[:, :, None]  # [H, c, V]
        s_term = torch.einsum("hjv,jhk->hvk", uw, kc)  # [H, V, K]
        state = torch.exp(g_last)[:, None, None] * state + s_term

    return outputs, state


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Causal depthwise 1D convolution over a single sequence.

    Mirrors the GDN ``causal_conv1d`` applied to ``mixed_qkv`` before the
    q/k/v split: left-padded by ``kernel_size - 1`` so output length == input
    length, one filter per channel, optional SiLU.

    Args:
        x: ``[T, C]`` sequence-major activations.
        weight: ``[C, kernel_size]`` depthwise filters.
        bias: ``[C]`` or ``None``.
        activation: ``"silu"`` or ``None``.

    Returns:
        ``[T, C]``.
    """
    x = x.double()
    weight = weight.double()
    seq_len, channels = x.shape
    kernel_size = weight.shape[-1]
    x_t = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
    x_pad = torch.nn.functional.pad(x_t, (kernel_size - 1, 0))
    out = torch.nn.functional.conv1d(
        x_pad,
        weight.unsqueeze(1),
        bias=bias.double() if bias is not None else None,
        groups=channels,
    )
    out = out[..., :seq_len].squeeze(0).transpose(0, 1)  # [T, C]
    if activation == "silu":
        out = out * torch.sigmoid(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")
    return out


__all__ = [
    "gdn_gating",
    "preprocess_qk",
    "gdn_delta_rule_recurrent",
    "gdn_delta_rule_chunked",
    "causal_depthwise_conv1d",
]
