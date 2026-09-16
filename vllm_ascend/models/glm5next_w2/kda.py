# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-free 310P GLM-5.3-Flash KDA (Kimi Delta Attention) linear attention (G4).

This is the Ascend 310P host-lane replacement for the shipped 910/Triton KDA
path (``vllm_ascend/models/glm5next/kda.py``, which pulls the Triton KDA op
under ``vllm_ascend/ops/triton/kda/``). It reimplements the *exact* Kimi-Delta
gated-delta-rule math in plain ``torch`` -- with NO Triton dependency and NO
``vllm_ascend/ops/triton`` on the load path -- and is numerically validated
against the eager parity oracle ``tools/glm_w2/kda_reference.py`` (task G-ref).

Why eager torch and NOT the ``vllm_ascend/_310p/ops/fla`` GDN kernels
--------------------------------------------------------------------
The 310P Gated-DeltaNet (Qwen GDN) Triton-free kernels under ``_310p/ops/fla``
share the *delta-rule skeleton* but differ from KDA in three load-bearing ways,
so they cannot be reused as-is:

1. **Decay shape.** ``fused_recurrent_gated_delta_rule_pytorch`` applies a
   *per-head scalar* forget gate (``exp(g).view(HV, 1, 1)`` -- decays the whole
   ``[V, K]`` state uniformly). KDA applies a *per-K-channel* decay
   (``S *= exp(g)[None, :]`` -- one decay per key channel). These are different
   operators.
2. **Gate law.** ``fused_gdn_gating_pytorch`` uses the unbounded softplus gate
   ``g = -exp(A_log) * softplus(a + dt_bias)``. KDA uses the bounded "safe"
   sigmoid gate ``g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``.
3. **Host runnability.** ``l2norm_310p`` hard-imports ``torch_npu`` (absent on
   the CPU host), so it cannot sit on the parity path.

The delta-rule read/update itself (``u = (v - S@k) * beta``; ``S += u k^T``;
``o = S@q``) and the per-head *scalar* beta DO match GDN, but given (1)-(3) an
eager torch implementation is both simpler and exact against the oracle. The
contractions use batched ``torch.bmm`` (matmul-friendly on the NPU) instead of
the oracle's elementwise-``sum`` so the shape is production-appropriate; this
reassociates the fp32 K-reduction, giving parity to ~1e-4 (see the G4 tests).

torch_npu hygiene
-----------------
``torch_npu`` is imported behind a runtime guard and never required: the whole
module imports and runs its parity path on plain ``torch`` (CPU host). The only
optional device fast-path is an ``npu_rms_norm``-backed L2 norm, off by default.

Dtypes flow through the authoritative G3 policy
:data:`ASCEND_GLM5NEXT_W2_DTYPE_POLICY`: the recurrence accumulates in the
``kda_accumulation`` dtype (float32) and the layer boundary casts to the ``kda``
dtype (float16). No bare dtype literals here.

G7 wiring note
--------------
``forward`` mirrors ``kda_layer_reference`` (conv -> safe gate -> recurrence ->
sigmoid-gated RMSNorm), stopping before ``o_proj`` unless an ``o_proj`` callable
is supplied. G7 attaches this module in place of ``Glm5NextLinearAttention`` for
the 34 ``KDA_LAYERS`` (the DSA ``full_attn_layers`` are owned by G5); the module
carries recurrent state across chunks via ``initial_state`` for prefill and a
one-token step for decode.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dtype_policy import ASCEND_GLM5NEXT_W2_DTYPE_POLICY, Glm5NextW2DtypePolicy

# ---- torch_npu is optional and guarded: the module must import + run on CPU. -
try:  # pragma: no cover - exercised only on Ascend hardware
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False

# Defaults mirrored from the shipped code / G-ref oracle (single source of truth
# for the *values* is the shipped config; these are the documented constants).
KDA_GATE_LOWER_BOUND = -5.0  # config.linear_lower_bound
L2NORM_EPS = 1e-6            # fused_recurrent_kda in-kernel L2 norm eps
O_NORM_EPS = 1e-5           # FusedRMSNormGated default eps
GLM_NUM_HEADS = 64          # config.linear_num_heads
GLM_HEAD_DIM = 128          # config.linear_head_dim
GLM_SHORT_CONV_KERNEL = 4   # config.linear_conv_kernel_dim

__all__ = ["Glm5NextW2KDA"]


class Glm5NextW2KDA(nn.Module):
    """Triton-free 310P KDA linear-attention core (parity: G-ref oracle).

    The module is deliberately *functional*: it holds only scalar config +
    epsilons + the dtype policy, and every method takes the projected tensors /
    weights explicitly. That keeps it constructible and testable on a plain CPU
    host (no vLLM config, no NPU, no loaded checkpoint) and lets G7 drive it from
    the shipped layer's already-projected q|k|v|beta|gate tensors.

    Args:
        num_heads: KDA linear-attention heads (64 for GLM-5.3-Flash).
        head_dim: per-head K == V dim (128).
        short_conv_kernel_size: depthwise causal conv width (4).
        lower_bound: safe-gate floor (-5.0).
        scale: query scale; defaults to ``head_dim ** -0.5`` (kda wrapper default).
        l2norm_eps: in-kernel L2-norm epsilon (1e-6).
        o_norm_eps: output gated-RMSNorm epsilon (1e-5).
        dtype_policy: authoritative G3 dtype table (accum=float32, kda=float16).
        use_npu_l2norm: opt-in ``torch_npu`` L2-norm fast-path on device (off by
            default; ignored when ``torch_npu`` is unavailable).
    """

    def __init__(
        self,
        *,
        num_heads: int = GLM_NUM_HEADS,
        head_dim: int = GLM_HEAD_DIM,
        short_conv_kernel_size: int = GLM_SHORT_CONV_KERNEL,
        lower_bound: float = KDA_GATE_LOWER_BOUND,
        scale: float | None = None,
        l2norm_eps: float = L2NORM_EPS,
        o_norm_eps: float = O_NORM_EPS,
        dtype_policy: Glm5NextW2DtypePolicy = ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
        use_npu_l2norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.short_conv_kernel_size = short_conv_kernel_size
        self.lower_bound = lower_bound
        self.scale = head_dim ** -0.5 if scale is None else scale
        self.l2norm_eps = l2norm_eps
        self.o_norm_eps = o_norm_eps
        self.dtype_policy = dtype_policy
        # Read dtypes from the policy (never spell literals at call sites).
        self.compute_dtype = dtype_policy.cast_site("kda_accumulation")  # float32
        self.io_dtype = dtype_policy.cast_site("kda")                    # float16
        self.use_npu_l2norm = bool(use_npu_l2norm) and _HAS_TORCH_NPU

    # -- pieces ------------------------------------------------------------- #
    def short_conv1d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
    ) -> torch.Tensor:
        """Depthwise causal conv1d + optional SiLU over the time axis.

        Mirrors ``causal_conv1d_ref`` (what the shipped ``kda.py`` routes to):
        ``F.conv1d`` with ``groups=dim`` and ``padding=width-1`` truncated to the
        original seqlen (so output at t sees only inputs <= t), then SiLU.

        Args:
            x: ``[T, C]`` channel-last time series.
            weight: ``[C, width]`` depthwise kernel.
            bias: optional ``[C]`` bias.
            activation: ``"silu"``/``"swish"`` or ``None``.
        """
        if activation not in (None, "silu", "swish"):
            raise NotImplementedError("activation must be None, silu, or swish")
        x = x.to(self.compute_dtype)
        seqlen, dim = x.shape
        kdim, width = weight.shape
        if kdim != dim:
            raise ValueError(f"conv weight channels {kdim} != x channels {dim}")
        x_ct = x.transpose(0, 1).unsqueeze(0)  # [1, C, T]
        w = weight.to(self.compute_dtype).unsqueeze(1)  # [C, 1, width]
        b = None if bias is None else bias.to(self.compute_dtype)
        out = F.conv1d(x_ct, w, b, padding=width - 1, groups=dim)
        out = out[..., :seqlen]  # causal truncation
        out = out.squeeze(0).transpose(0, 1)  # [T, C]
        if activation is not None:
            out = F.silu(out)
        return out

    def safe_gate(
        self,
        raw_g: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Bounded ("safe") KDA decay gate in log space.

        ``g = lower_bound * sigmoid(exp(a_log) * (raw_g + dt_bias))`` -- a
        per-(token, head, channel) log-decay in ``(lower_bound, 0)`` consumed by
        the recurrence as ``exp(g)``. Matches
        ``vllm_ascend/ops/triton/kda/gate.py::apply_kda_gate`` (safe_gate=True).

        Args:
            raw_g: ``[..., H, D]`` raw gate projection (``f_b_proj(f_a)``).
            a_log: ``[H]`` (or broadcastable); ``exp(a_log)`` is the per-head scale.
            dt_bias: optional ``[H*D]`` / ``[H, D]`` additive bias.
        """
        g = raw_g.to(self.compute_dtype)
        heads = a_log.numel()
        if g.shape[-2] != heads:
            raise ValueError(
                f"KDA gate head mismatch: raw_g {tuple(raw_g.shape)} vs A_log numel {heads}"
            )
        a = a_log.reshape(*([1] * (g.dim() - 2)), heads, 1).to(dtype=self.compute_dtype, device=g.device)
        if dt_bias is not None:
            bias = dt_bias.to(dtype=self.compute_dtype, device=g.device).reshape(
                *([1] * (g.dim() - 2)), heads, g.shape[-1]
            )
            g = g + bias
        return self.lower_bound * torch.sigmoid(torch.exp(a) * g)

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize the last dim (eps ``l2norm_eps``).

        Default eager path: ``x / sqrt(sum(x^2) + eps)`` -- bit-identical to the
        oracle. The optional ``torch_npu`` device fast-path uses an
        ``npu_rms_norm`` with a unit/sqrt(dim) weight (same identity as
        ``l2norm_310p``); it is only taken on real hardware and is off by default.
        """
        if self.use_npu_l2norm and torch_npu is not None and x.is_npu:  # pragma: no cover
            import math

            dim = x.shape[-1]
            weight = torch.full((dim,), 1.0 / math.sqrt(dim), dtype=x.dtype, device=x.device)
            y, _ = torch_npu.npu_rms_norm(x.reshape(-1, dim).contiguous(), weight, self.l2norm_eps / dim)
            return y.reshape(x.shape)
        return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + self.l2norm_eps)

    def gated_rmsnorm(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sigmoid-gated RMSNorm over the last dim (``FusedRMSNormGated`` native).

        ``rmsnorm(x) * weight * sigmoid(g)``, RMS over the last dim with
        ``o_norm_eps``. Computed in the accumulation (fp32) dtype.
        """
        x_float = x.to(self.compute_dtype)
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.o_norm_eps)
        if weight is not None:
            x_normed = x_normed * weight.to(self.compute_dtype)
        return x_normed * torch.sigmoid(g.to(self.compute_dtype))

    # -- core recurrence ---------------------------------------------------- #
    def recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        use_qk_l2norm: bool = True,
        beta_is_raw: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated delta-rule KDA recurrence (parity: ``kda_recurrent_reference``).

        Per head ``h`` the state ``S`` is ``[V, K] = [head_dim, head_dim]``. Per
        timestep (batched over heads via ``bmm``)::

            S <- S * exp(g)[:, None, :]            # per-K-channel decay
            u <- (v - S @ k) * beta                # delta rule, per-head scalar beta
            S <- S + u k^T                          # rank-1 update
            o <- S @ q

        Args:
            q, k: ``[T, H, K]`` post-conv query/key (or ``[1, T, H, K]``).
            v: ``[T, H, V]`` post-conv value.
            g: ``[T, H, K]`` **log-decay** gate (from :meth:`safe_gate`).
            beta: ``[T, H]`` per-head gate; raw (pre-sigmoid) iff ``beta_is_raw``.
            initial_state: optional ``[H, V, K]`` carry-in state (zeros if None).
            use_qk_l2norm: L2-normalize q, k before use.
            beta_is_raw: apply sigmoid to beta inside (matches ``sigmoid_beta``).

        Returns:
            ``(out [T, H, V], final_state [H, V, K])`` in the accumulation dtype.
        """
        squeezed = False
        if q.dim() == 4:
            if q.shape[0] != 1:
                raise ValueError("batched recurrence expects batch size 1")
            q, k, v, g = q[0], k[0], v[0], g[0]
            beta = beta[0] if beta.dim() == 3 else beta
            squeezed = True

        T, H, K = q.shape
        V = v.shape[-1]
        cdt = self.compute_dtype

        q = q.to(cdt)
        k = k.to(cdt)
        v = v.to(cdt)
        g = g.to(cdt)
        beta = beta.to(cdt)
        if beta_is_raw:
            beta = torch.sigmoid(beta)

        if use_qk_l2norm:
            q = self._l2norm(q)
            k = self._l2norm(k)
        q = q * self.scale

        if initial_state is None:
            state = torch.zeros(H, V, K, dtype=cdt, device=q.device)
        else:
            state = initial_state.to(cdt).clone()

        # Precompute the per-K-channel decay once for the whole sequence.
        decay = torch.exp(g)  # [T, H, K]

        out = torch.empty(T, H, V, dtype=cdt, device=q.device)
        for t in range(T):
            q_t = q[t]                       # [H, K]
            k_t = k[t].unsqueeze(-1)         # [H, K, 1]
            v_t = v[t]                       # [H, V]
            beta_t = beta[t].unsqueeze(-1)   # [H, 1]

            state = state * decay[t].unsqueeze(1)             # [H, V, K]
            sk = torch.bmm(state, k_t).squeeze(-1)            # [H, V] == S @ k
            u = (v_t - sk) * beta_t                            # [H, V]
            state = state + torch.bmm(u.unsqueeze(-1), k_t.transpose(-1, -2))  # [H,V,1]x[H,1,K]
            out[t] = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)           # [H, V] == S @ q

        if squeezed:
            out = out.unsqueeze(0)
        return out, state

    def chunked_recurrence(
        self,
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
        """State-carrying chunked driver over :meth:`recurrence` (prefill lane).

        Threads the recurrent state across chunk boundaries; mathematically
        identical to the whole-sequence recurrence (proven equal in the G4 UTs).
        This is the shape G7 uses for prefill (chunk) + decode (chunk_size=1).
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
            o, state = self.recurrence(
                q[start:end], k[start:end], v[start:end], g[start:end], beta[start:end],
                initial_state=state, **kwargs,
            )
            outs.append(o)
        return torch.cat(outs, dim=0), state

    # -- end-to-end layer core --------------------------------------------- #
    def forward(
        self,
        hidden_or_qkv: torch.Tensor,
        *,
        conv_weight: torch.Tensor,
        raw_g: torch.Tensor,
        beta_raw: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor | None,
        g_out: torch.Tensor,
        o_norm_weight: torch.Tensor | None = None,
        conv_bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        apply_conv: bool = True,
        o_proj=None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """End-to-end KDA core: conv -> safe gate -> recurrence -> gated RMSNorm.

        Mirrors ``kda_layer_reference`` (and the shipped ``kda.py`` forward
        order), stopping before ``o_proj`` unless an ``o_proj`` callable is
        given. Returns the normalized/gated core output ``[T, H, head_dim]``
        cast to ``output_dtype`` (defaults to the policy ``kda`` dtype, float16;
        pass ``torch.float32`` for tight fp32 parity).

        Args:
            hidden_or_qkv: fused ``q|k|v`` projection ``[T, 3*H*head_dim]``.
            conv_weight: merged ``q|k|v`` depthwise conv weight ``[3*H*head_dim, width]``.
            raw_g: gate projection ``[T, H*head_dim]`` (``f_b_proj(f_a)``).
            beta_raw: raw beta ``[T, H]`` (pre-sigmoid).
            a_log: ``[H]`` (or broadcastable) ``A_log``.
            dt_bias: ``[H*head_dim]`` or None.
            g_out: output gate ``[T, H*head_dim]`` (``g_b_proj(g_a)``).
            o_norm_weight: ``[head_dim]`` o_norm affine weight, or None.
            conv_bias: optional conv bias.
            initial_state: optional ``[H, V, K]`` carry-in recurrent state.
            apply_conv: run the short conv (False if q|k|v are already convolved).
            o_proj: optional callable applied to the flattened ``[T, H*head_dim]``
                core output (e.g. a RowParallelLinear); returns its result as-is.
            output_dtype: cast dtype for the core output (default: policy kda).
        """
        num_heads = self.num_heads
        head_dim = self.head_dim
        out_dt = self.io_dtype if output_dtype is None else output_dtype

        qkv = hidden_or_qkv.to(self.compute_dtype)
        T = qkv.shape[0]
        proj = num_heads * head_dim
        if apply_conv:
            qkv = self.short_conv1d(qkv, conv_weight, conv_bias, activation="silu")
        q, k, v = qkv.split(proj, dim=-1)
        q = q.reshape(T, num_heads, head_dim)
        k = k.reshape(T, num_heads, head_dim)
        v = v.reshape(T, num_heads, head_dim)

        g_log = self.safe_gate(raw_g.reshape(T, num_heads, head_dim), a_log, dt_bias)
        core, _ = self.recurrence(q, k, v, g_log, beta_raw, initial_state=initial_state, beta_is_raw=True)

        g2 = g_out.reshape(T, num_heads, head_dim)
        core = self.gated_rmsnorm(core, g2, o_norm_weight)  # fp32
        core = core.to(out_dt)
        if o_proj is not None:
            return o_proj(core.reshape(T, num_heads * head_dim))
        return core
