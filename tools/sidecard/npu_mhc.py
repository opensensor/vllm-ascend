# SPDX-License-Identifier: Apache-2.0
"""Unbound 310P mHC candidates; importing this module never imports a device SDK.

T11 registration (_C_ascend namespace, PrivateUse1 implementations):
  sidecard_mhc_pre_310(Tensor residual, Tensor fn, Tensor scale, Tensor base,
                       Tensor norm_weight, float norm_eps, float hc_eps)
                       -> (Tensor, Tensor, Tensor)
  sidecard_mhc_post_310(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor

Pre returns post FP32 [R,4,1] already BF16-rounded, *unrounded* logits FP32
[R,4,4], and normalized FP16 hidden [R,4096]. The existing qualified Sinkhorn
must consume logits and its result must then round through BF16 to FP32.
Neither operator needs GM workspace. The pre UB requirement is 118,816 bytes;
post uses 9,216 bytes. Supports 1..32 rows; qualification is per operator/row.

No rows are enabled by default. A T12 receipt must justify explicit row sets;
unsupported or slower shapes retain the caller's original implementation. A
failed native call propagates: retrying a possibly partially issued device
operation through eager execution is unsafe.
"""

import math
from dataclasses import dataclass

HIDDEN = 4096
STREAMS = 4
MIXES = 24
SUPPORTED_ROWS = frozenset(range(1, 33))
# Fixed before hardware testing. Rounded outputs must preserve qualified bytes.
# A different reduction tree cannot silently weaken that requirement.
RAW_LOGIT_ATOL = 2e-4
RAW_LOGIT_RTOL = 2e-5
NUMERICAL_CONTRACT = {
    "raw_logits": {"atol": RAW_LOGIT_ATOL, "rtol": RAW_LOGIT_RTOL},
    "post": "bitwise equality after BF16 rounding into FP32",
    "comb": "bitwise equality after qualified Sinkhorn and BF16 rounding",
    "hidden": "bitwise FP16 equality after FP32 RMS norm",
    "residual": "bitwise FP32 equality after BF16 rounding",
    "nonfinite": "reject on either side for finite test fixtures",
    "post_reduction": "input streams 0,1,2,3, then x/post; separate FP32 multiply and add",
    "pre_reduction": "ascending 1024-element tiles; pairwise halving then 64-lane FP32 vector sum",
}


def validate_eps(norm_eps, hc_eps):
    """Reject underflow/overflow that would change the C++ float attributes."""
    import struct

    for value in (norm_eps, hc_eps):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("mHC epsilons must be finite positive FP32 values")
        try:
            converted = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as error:
            raise ValueError("mHC epsilon overflows FP32") from error
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError("mHC epsilons must be finite positive FP32 values")


def pre_reference(residual, fn, scale, base, norm_weight, norm_eps, hc_eps, torch):
    """Qualified NPU eager pre math up to the unchanged Sinkhorn boundary."""
    validate_eps(norm_eps, hc_eps)
    flat = residual.flatten(1).float()
    mixes = (flat[:, None, :] * fn.float()[None, :, :]).sum(-1)
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes[:, :STREAMS] * scale.float()[0] + base.float()[:STREAMS]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, STREAMS : 2 * STREAMS] * scale.float()[1] + base.float()[STREAMS:8])
    logits = (mixes[:, 8:] * scale.float()[2] + base.float()[8:]).reshape(-1, STREAMS, STREAMS)
    hidden = (pre[..., None] * residual).sum(1).to(torch.bfloat16).float()
    hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + norm_eps)
    hidden = (hidden * norm_weight.float()).to(norm_weight.dtype)
    return post[..., None].to(torch.bfloat16).float(), logits, hidden


def post_reference(x, residual, post, comb, torch):
    """Qualified broadcast reduction; never cast the residual through FP16."""
    mixed = (residual[:, :, None, :] * comb[:, :, :, None]).sum(1)
    return (mixed + post * x[:, None, :].float()).to(torch.bfloat16).float()


def ordered_post_reference(x, residual, post, comb, torch):
    """Bounded ordered math mirrored by the candidate (not an equality claim)."""
    mixed = residual[:, 0, None, :] * comb[:, 0, :, None]
    for stream in range(1, STREAMS):
        mixed = mixed + residual[:, stream, None, :] * comb[:, stream, :, None]
    return (mixed + post * x[:, None, :].float()).to(torch.bfloat16).float()


def resolve_ops(namespace="_C_ascend"):
    """Resolve only in an already initialized NPU owner after T11 registration."""
    import torch

    ops = getattr(torch.ops, namespace)
    try:
        return ops.sidecard_mhc_pre_310, ops.sidecard_mhc_post_310
    except AttributeError as error:
        raise RuntimeError("sidecard mHC candidates require T11 native registration and build") from error


def admissible(residual, rows):
    return (
        residual.device.type == "npu"
        and residual.is_contiguous()
        and len(residual.shape) == 3
        and tuple(residual.shape[1:]) == (STREAMS, HIDDEN)
        and residual.shape[0] in rows
        and str(residual.dtype) == "torch.float32"
    )


@dataclass(frozen=True)
class NativeMHC:
    """Explicit owner-local policy; empty defaults keep the production baseline."""

    pre_op: object
    post_op: object
    sinkhorn: object
    pre_rows: frozenset = frozenset()
    post_rows: frozenset = frozenset()

    def __post_init__(self):
        for rows in (self.pre_rows, self.post_rows):
            if not isinstance(rows, frozenset) or any(type(row) is not int for row in rows):
                raise ValueError("qualified row sets must be immutable integer sets")
            if not rows <= SUPPORTED_ROWS:
                raise ValueError("unsupported native mHC row count")
        if any(not callable(operation) for operation in (self.pre_op, self.post_op, self.sinkhorn)):
            raise ValueError("native pre/post and qualified Sinkhorn callables are required")

    def pre(self, residual, prepared_weights, norm_eps, hc_eps, iterations, torch, fallback):
        if not admissible(residual, self.pre_rows):
            return fallback()
        validate_eps(norm_eps, hc_eps)
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Sinkhorn iteration count must be a positive integer")
        post, logits, hidden = self.pre_op(residual, *prepared_weights, norm_eps, hc_eps)
        comb = self.sinkhorn(logits.contiguous(), iterations, hc_eps).to(torch.bfloat16).float()
        return post, comb, hidden

    def post(self, x, residual, post, comb, fallback):
        if not admissible(residual, self.post_rows):
            return fallback()
        return self.post_op(x, residual, post, comb)


def prepare_weights(weights, kind, norm_name):
    """Call once at owner construction, never convert big fn weights per token."""
    if kind not in ("attn", "ffn"):
        raise ValueError("mHC mixer kind must be attn or ffn")
    return (
        weights[f"hc_{kind}_fn"].float().contiguous(),
        weights[f"hc_{kind}_scale"].float().contiguous(),
        weights[f"hc_{kind}_base"].float().contiguous(),
        weights[norm_name].contiguous(),
    )
