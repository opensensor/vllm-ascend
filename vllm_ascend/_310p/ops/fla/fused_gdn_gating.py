import torch
import torch.nn.functional as F

# Softplus is evaluated in fp32: the gate feeds an exponential decay term in the
# delta-rule recurrence, where fp16 rounding of `softplus(a + dt_bias)` is
# visible in the accumulated state.
_GATING_COMPUTE_DTYPE = torch.float32


def gdn_gating_constants(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the activation-independent half of the GDN gate.

    ``-exp(A_log)`` and the fp32 copy of ``dt_bias`` only depend on loaded
    weights, so a caller that holds them across steps turns four NPU kernels
    per layer per token into zero. See
    :func:`~vllm_ascend._310p.ops.fla.gdn_310.AscendGatedDeltaNetAttention310._gating_constants`.

    Returns:
        ``(neg_exp_A_log, dt_bias_f32)``, both shaped ``[num_heads]``.
    """
    A_log_f = A_log.to(_GATING_COMPUTE_DTYPE)
    return -torch.exp(A_log_f), dt_bias.to(_GATING_COMPUTE_DTYPE)


def fused_gdn_gating_pytorch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    constants: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of fused_gdn_gating.
    This is a fallback implementation for 310P without Triton support.

    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: a parameter, shape [batch, num_heads]
        b: b parameter, shape [batch, num_heads]
        dt_bias: dt bias, shape [num_heads]
        beta: softplus beta parameter
        threshold: softplus threshold parameter
        constants: optional ``(neg_exp_A_log, dt_bias_f32)`` from
            :func:`gdn_gating_constants`, reused across decode steps so the
            constant folding does not re-run as NPU kernels every token.

    Returns:
        g: gating parameter, shape [1, batch, num_heads]
        beta_output: sigmoid(b), shape [1, batch, num_heads]
    """
    if constants is None:
        constants = gdn_gating_constants(A_log, dt_bias)
    neg_exp_A_log, dt_bias_f = constants

    # x = a + dt_bias, broadcasting [num_heads] against [batch, num_heads].
    x = a.to(_GATING_COMPUTE_DTYPE) + dt_bias_f

    # F.softplus is the single-kernel form of the guarded
    # `where(beta * x <= threshold, log1p(exp(beta * x)) / beta, x)` expression
    # that the Triton kernel implements, including the linear tail.
    softplus_x = F.softplus(x, beta=beta, threshold=threshold)

    # g = -exp(A_log) * softplus(x), with the leading sequence dimension.
    g = (neg_exp_A_log * softplus_x).unsqueeze(0)

    # Match Triton kernel: sigmoid in fp32, then cast to input b dtype.
    beta_output = torch.sigmoid(b.to(_GATING_COMPUTE_DTYPE)).to(b.dtype).unsqueeze(0)

    return g, beta_output
