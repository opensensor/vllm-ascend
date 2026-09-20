import math
import os

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


# Must match TileRowsForHeads() in
# csrc/moe/gdn_gating_v310/op_host/gdn_gating_v310_tiling.cpp. The transfers need
# tile_rows * num_heads to be a multiple of 16 elements so both the fp16 and fp32
# DataCopy stay 32-byte aligned; take the smallest such value so decode pads a
# 1-token call out to 4 rows (H=12) or 1 row (H=48) rather than to 32.
_GATING_ALIGN_ELEMS = 16


def _gating_tile_rows(num_heads: int) -> int:
    return _GATING_ALIGN_ELEMS // math.gcd(num_heads, _GATING_ALIGN_ELEMS)


_GATING_OP: object | None = None
_GATING_OP_RESOLVED = False


def _gating_op():
    """The fused ``npu_gdn_gating_310`` AscendC kernel if it was built.

    OPT-IN ONLY: the kernel currently FAILS device parity against
    :func:`fused_gdn_gating_pytorch` (relative error ~1.0 on every shape), so it
    must never be selected implicitly just because the vendor library happens to
    be installed. Set ``VLLM_ASCEND_ENABLE_GDN_GATING_KERNEL=1`` to opt in while
    debugging; the op-chain path stays the default and is the correctness oracle.
    """
    global _GATING_OP, _GATING_OP_RESOLVED
    if not _GATING_OP_RESOLVED:
        _GATING_OP_RESOLVED = True
        if os.environ.get("VLLM_ASCEND_ENABLE_GDN_GATING_KERNEL", "0") == "1":
            try:
                _GATING_OP = torch.ops._C_ascend.npu_gdn_gating_310
            except (AttributeError, RuntimeError):
                _GATING_OP = None
    return _GATING_OP


def gdn_gating_tiled_constants(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``gdn_gating_constants`` broadcast to ``[TILE_ROWS, num_heads]``.

    The kernel takes these pre-tiled: a GDN head count of ``48 // tp`` gives
    rows of ``4 * H`` bytes, which is not the 32-byte multiple ``DataCopy``
    needs, while a ``[TILE_ROWS, H]`` block always is. They depend only on
    loaded weights, so this runs once per layer rather than once per token.
    """
    neg_exp_A_log, dt_bias_f32 = gdn_gating_constants(A_log, dt_bias)
    tile_rows = _gating_tile_rows(A_log.shape[0])
    return (
        neg_exp_A_log.unsqueeze(0).expand(tile_rows, -1).contiguous(),
        dt_bias_f32.unsqueeze(0).expand(tile_rows, -1).contiguous(),
    )


def fused_gdn_gating_310(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    constants: tuple[torch.Tensor, torch.Tensor] | None = None,
    tiled_constants: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel GDN gate, falling back to the op-chain implementation.

    The kernel collapses the seven framework ops of
    :func:`fused_gdn_gating_pytorch` into one launch, which is what matters on
    310P: decode is bound by kernel count, and this gate runs in 48 of 64
    layers every token.
    """
    op = _gating_op()
    if op is None or tiled_constants is None or a.dim() != 2:
        return fused_gdn_gating_pytorch(A_log, a, b, dt_bias, beta, threshold, constants=constants)
    neg_exp_tiled, dt_bias_tiled = tiled_constants

    # 310P DataCopyPad does not service sub-32-byte GM->UB reads, so the kernel
    # needs whole tiles. Padding the token axis to get them is a net LOSS at
    # decode: F.pad allocates, fills and copies (~83us measured), swamping the
    # ~50us the fused kernel saves. Measured at H=12:
    #   T=12 (exact multiple)  op-chain 84.8us -> fused 34.7us   +2.41ms/token
    #   T=10 (pads to 12)      op-chain 90.1us -> fused 117.7us  -1.32ms/token
    # So take the kernel only when the shape already tiles exactly, and leave the
    # proven op chain to handle everything else. Prefill hits this often; decode
    # with MTP k=4 and max_num_seqs=2 gives T=5 or 10 and correctly does not.
    num_tokens = a.shape[0]
    tile_rows = neg_exp_tiled.shape[0]
    if num_tokens % tile_rows:
        return fused_gdn_gating_pytorch(A_log, a, b, dt_bias, beta, threshold, constants=constants)

    g, beta_output = op(a.contiguous(), b.contiguous(), neg_exp_tiled, dt_bias_tiled, float(beta))
    return g[:num_tokens].unsqueeze(0), beta_output[:num_tokens].unsqueeze(0)
