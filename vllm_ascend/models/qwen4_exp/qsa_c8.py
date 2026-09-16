# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate A prototype: C8 (signed-INT8) main QSA K/V cache for Ascend 310P
(plan T8.1).

This is a **host-measurement prototype** for the 1M cache decision (PRD §R8
Candidate A). It pre-decides nothing -- D4 decides on hardware. The module is
intentionally *host-safe*: it imports only ``torch`` and the Triton-free QSA ops
(``ops/qsa_attention``, ``ops/qsa_cache``) plus the authoritative dtype policy;
it pulls **no** ``torch_npu`` / Triton / CUDA module, so it materializes and
unit-tests on the 310P host lane.

What Candidate A stores
-----------------------
The *main* QSA K/V cache -- the full-context key/value store that sparse
attention reads -- is quantized to signed symmetric INT8 (1 byte/element),
exactly half the fp16 main dtype (2 bytes). The compressed *indexer* history and
the raw index-key ring stay at their T1.4 dtypes; per PRD §R8 the indexer is
kept BF16 initially and C8-vs-BF16 is compared at every gate.

Scale granularity (DOCUMENTED DECISION)
---------------------------------------
**Per-(token, head)**: one INT8 scale per stored K/V head-vector, the amax over
the ``head_dim`` axis (``scale = amax(|row_head|) / 127``, symmetric, no offset).

Rationale:

* *Streaming-computable at write time.* A KV cache is written incrementally, one
  step's rows at a time; a per-(token, head) scale depends only on the row being
  written, so the fused cache-write path computes it with no cross-token
  statistics. Per-channel (per ``head_dim`` index) scales would need statistics
  accumulated across all tokens -- unavailable to a paged streaming write.
* *Independent per head.* Different KV heads carry different magnitude
  distributions; scaling each head-vector independently preserves each head's
  dynamic range instead of letting one hot head clip the others (which a single
  per-token scale over ``Hkv*D`` would cause).
* *Established asc pattern.* It mirrors the 310P W8A8 dynamic activation scheme
  (``tests/ut/qwen38_1m/reference/w8a8_reference.py`` ``quantize_per_token_int8``:
  symmetric, ``round(x/scale)`` clamped to ``[-128, 127]``, reduction over the
  last dim), reused here with the last dim = ``head_dim`` per head.

The per-(token, head) scales ride an out-of-band fp32 companion cache; their
footprint is ``1 / head_dim`` of the INT8 payload (negligible), so the 1M byte
math still reports the INT8 payload as exactly half BF16 (see
:mod:`vllm_ascend.models.qwen4_exp.kv_cache` ``qsa_main_kv_bytes_table``).

Fused write/read paths
----------------------
:func:`make_c8_kv_quant_hook` builds the quant hook wired through T6.2's
``qsa_write_kv_to_cache(quant_hook=...)`` seam (this module never edits
``ops/qsa_attention.py``). :class:`C8QSAMainKVCache` owns the INT8 payload caches
and the fp16 scale caches and provides the fused write (quant on scatter) and
read (dequant on gather) paths end to end.
"""

from __future__ import annotations

import torch

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .kv_cache import QSA_C8_MAIN_DTYPE
from .ops.qsa_attention import QSAKVQuantHook, qsa_write_kv_to_cache
from .ops.qsa_cache import PAD_SLOT_ID, qsa_gather_rows, qsa_scatter_rows

# --- INT8 grid (signed symmetric; mirrors w8a8_reference) -----------------
INT8_MIN = -128
INT8_MAX = 127
# Symmetric grid uses the positive half-range (127 levels), no offset.
_SYM_LEVELS = 127.0

# --- Pre-declared acceptance bars (PRD §8.1: named constants before asserts)
# Round-trip: dequant(quant(x)) recovers x only up to half a grid step. The hard,
# correct bound is |err| <= scale/2 per element; this eps is the small float
# slack for the round + fp32 arithmetic on top of that hard bound.
C8_ROUNDTRIP_EPS = 1e-4
# Selection-set agreement (C8 vs BF16 keys) bar: mean fraction of BF16-selected
# tokens retained by C8 selection, over the probe query tokens. Measured on the
# T0.4-seeded short probes at ~0.985+ with worst-probe-mean ~0.985 and worst
# per-token ~0.94; 0.90 fails a real selection regression while clearing the
# observed spread with head-room.
C8_SELECTION_AGREEMENT_BAR = 0.90
# Short-context attention-logit closeness (C8 K/V vs BF16 K/V): Frobenius
# relative error of the sparse-attention output. Per-(token, head) quant noise
# (~scale/2) averages down over the softmax-weighted sum, keeping the whole
# output well under 5% (measured ~0.008).
C8_LOGITS_REL = 0.05

# Type of the per-(token, head) scale companion produced/consumed here.
C8ScaleRecorder = dict


def _split_heads(rows: torch.Tensor, num_kv_heads: int) -> tuple[torch.Tensor, int]:
    """View flat ``[T, Hkv*D]`` (or ``[T, Hkv, D]``) rows as ``[T, Hkv, D]``."""
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if rows.ndim == 3:
        if rows.shape[1] != num_kv_heads:
            raise ValueError("rows head dim does not match num_kv_heads")
        return rows, rows.shape[2]
    if rows.ndim != 2:
        raise ValueError("QSA K/V rows must be [T, Hkv*D] or [T, Hkv, D]")
    channels = rows.shape[1]
    if channels % num_kv_heads:
        raise ValueError("row width not divisible by num_kv_heads")
    head_dim = channels // num_kv_heads
    return rows.view(rows.shape[0], num_kv_heads, head_dim), head_dim


def quantize_kv_int8(
    rows: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(token, head) signed symmetric INT8 quantization of K/V rows.

    Args:
        rows: ``[T, Hkv*D]`` (flat) or ``[T, Hkv, D]`` float K/V rows.
        num_kv_heads: number of KV heads packed in the row width.

    Returns:
        ``(q, scale)`` where ``q`` is int8 in the *same layout* as ``rows``
        (flat in -> flat out) and ``scale`` is ``[T, Hkv]`` float. All-zero
        head-vectors get scale ``1.0`` (they quantize to zero regardless).
    """
    flat_in = rows.ndim == 2
    heads, head_dim = _split_heads(rows, num_kv_heads)
    x = heads.float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _SYM_LEVELS, torch.ones_like(amax))
    q = torch.round(x / scale).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    scale = scale.squeeze(-1)
    if flat_in:
        return q.reshape(rows.shape[0], num_kv_heads * head_dim), scale
    return q, scale


def dequantize_kv_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Inverse of :func:`quantize_kv_int8` (per-(token, head), no offset).

    ``q`` is ``[T, Hkv*D]`` or ``[T, Hkv, D]`` int8; ``scale`` is ``[T, Hkv]``.
    Returns a float tensor in the same layout as ``q``.
    """
    flat_in = q.ndim == 2
    heads, head_dim = _split_heads(q, num_kv_heads)
    deq = heads.float() * scale.unsqueeze(-1)
    if flat_in:
        return deq.reshape(q.shape[0], num_kv_heads * head_dim)
    return deq


def make_c8_kv_quant_hook(
    num_kv_heads: int,
    recorder: C8ScaleRecorder,
) -> QSAKVQuantHook:
    """Build the C8 write-path quant hook for ``qsa_write_kv_to_cache``.

    The returned hook quantizes ``(key_rows, value_rows)`` to INT8 (per-(token,
    head)) *before* the paged scatter and records the per-(token, head) scales
    into ``recorder`` under keys ``"key_scale"`` / ``"value_scale"`` (in the same
    row order as ``slot_mapping``), so the caller can scatter them into the
    companion scale cache. This is the only wiring into T6.2's reserved seam --
    ``ops/qsa_attention.py`` is not modified.
    """

    def hook(key_rows: torch.Tensor, value_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_q, key_scale = quantize_kv_int8(key_rows, num_kv_heads)
        value_q, value_scale = quantize_kv_int8(value_rows, num_kv_heads)
        recorder["key_scale"] = key_scale
        recorder["value_scale"] = value_scale
        return key_q, value_q

    return hook


class C8QSAMainKVCache:
    """Host prototype of the Candidate A C8 main QSA K/V cache.

    Owns the INT8 payload caches (``[num_slots, Hkv*D]``) and the fp16 companion
    scale caches (``[num_slots, Hkv]``). Writes quantize on scatter through the
    T6.2 hook; reads dequantize on gather. Everything is pure torch on CPU/host.
    """

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_slots <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_slots = num_slots
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.width = num_kv_heads * head_dim
        # Scales ride an out-of-band fp32 companion cache: full precision keeps
        # the QDQ round-trip at the exact |err| <= scale/2 bound, and the
        # companion is negligible (1/head_dim of the INT8 payload), so the 1M
        # byte table still reports the payload as exactly half BF16.
        self.scale_dtype = dtype_policy.accumulation_dtype
        dev = torch.device(device)
        self.key_int8 = torch.zeros(num_slots, self.width, dtype=QSA_C8_MAIN_DTYPE, device=dev)
        self.value_int8 = torch.zeros(num_slots, self.width, dtype=QSA_C8_MAIN_DTYPE, device=dev)
        self.key_scale = torch.zeros(num_slots, num_kv_heads, dtype=self.scale_dtype, device=dev)
        self.value_scale = torch.zeros(num_slots, num_kv_heads, dtype=self.scale_dtype, device=dev)

    def write(
        self,
        slot_mapping: torch.Tensor,
        key_rows: torch.Tensor,
        value_rows: torch.Tensor,
    ) -> None:
        """Fused C8 write: quantize on the T6.2 hook, scatter payload + scales.

        PAD (``< 0``) slots are skipped, matching the paged store kernels.
        """
        recorder: C8ScaleRecorder = {}
        hook = make_c8_kv_quant_hook(self.num_kv_heads, recorder)
        # Payload: INT8 rows scattered through the reserved quant seam.
        qsa_write_kv_to_cache(
            self.key_int8,
            self.value_int8,
            slot_mapping,
            key_rows.reshape(key_rows.shape[0], self.width),
            value_rows.reshape(value_rows.shape[0], self.width),
            quant_hook=hook,
        )
        # Companion scales scattered to the same slots (fp16), in row order.
        qsa_scatter_rows(self.key_scale, slot_mapping, recorder["key_scale"])
        qsa_scatter_rows(self.value_scale, slot_mapping, recorder["value_scale"])

    def _dequantized(self, payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        deq = dequantize_kv_int8(payload, scale.float(), self.num_kv_heads)
        return deq.reshape(self.num_slots, self.num_kv_heads, self.head_dim)

    def dequantized_key_cache(self) -> torch.Tensor:
        """Full dequantized key cache as ``[num_slots, Hkv, D]`` float."""
        return self._dequantized(self.key_int8, self.key_scale)

    def dequantized_value_cache(self) -> torch.Tensor:
        """Full dequantized value cache as ``[num_slots, Hkv, D]`` float."""
        return self._dequantized(self.value_int8, self.value_scale)

    def gather_key_rows(self, slots: torch.Tensor) -> torch.Tensor:
        """Fused C8 read: gather + dequant key rows for ``slots`` (``[n, Hkv, D]``)."""
        payload = qsa_gather_rows(self.key_int8.float(), slots)
        scale = qsa_gather_rows(self.key_scale.float(), slots)
        return self._rows_to_heads(payload, scale)

    def gather_value_rows(self, slots: torch.Tensor) -> torch.Tensor:
        """Fused C8 read: gather + dequant value rows for ``slots`` (``[n, Hkv, D]``)."""
        payload = qsa_gather_rows(self.value_int8.float(), slots)
        scale = qsa_gather_rows(self.value_scale.float(), slots)
        return self._rows_to_heads(payload, scale)

    def _rows_to_heads(self, payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        deq = dequantize_kv_int8(payload, scale, self.num_kv_heads)
        return deq.reshape(payload.shape[0], self.num_kv_heads, self.head_dim)


# =========================================================================
# Accuracy harness: C8-vs-BF16 selection sets and short-context logits
# =========================================================================
def roundtrip_kv_int8(
    rows: torch.Tensor,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dequant(quant(rows)), per_head_scale)`` for round-trip checks."""
    q, scale = quantize_kv_int8(rows, num_kv_heads)
    return dequantize_kv_int8(q, scale, num_kv_heads), scale


def selection_set_agreement(
    bf16_selected: list[set[int]],
    c8_selected: list[set[int]],
) -> dict[str, float]:
    """Agreement metrics between BF16 and C8 per-query selection sets.

    Returns ``mean_retained`` (mean over query tokens of
    ``|bf16 ∩ c8| / |bf16|``, i.e. the fraction of BF16-selected tokens the C8
    keys still select -- the acceptance metric), plus ``mean_jaccard`` and the
    worst per-token ``min_retained`` for diagnostics.
    """
    if len(bf16_selected) != len(c8_selected):
        raise ValueError("selection lists must be aligned per query token")
    retained: list[float] = []
    jaccard: list[float] = []
    for a, b in zip(bf16_selected, c8_selected):
        if not a and not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        retained.append(inter / len(a) if a else 1.0)
        jaccard.append(inter / union if union else 1.0)
    if not retained:
        return {"mean_retained": 1.0, "mean_jaccard": 1.0, "min_retained": 1.0}
    return {
        "mean_retained": sum(retained) / len(retained),
        "mean_jaccard": sum(jaccard) / len(jaccard),
        "min_retained": min(retained),
    }


def logits_relative_error(
    bf16_out: torch.Tensor,
    c8_out: torch.Tensor,
) -> float:
    """Frobenius relative error ``||c8 - bf16|| / ||bf16||`` of attention logits."""
    denom = bf16_out.double().norm()
    if denom == 0:
        return float((c8_out.double().norm()).item())
    return float(((c8_out.double() - bf16_out.double()).norm() / denom).item())


__all__ = [
    "C8_LOGITS_REL",
    "C8_ROUNDTRIP_EPS",
    "C8_SELECTION_AGREEMENT_BAR",
    "C8QSAMainKVCache",
    "INT8_MAX",
    "INT8_MIN",
    "PAD_SLOT_ID",
    "dequantize_kv_int8",
    "logits_relative_error",
    "make_c8_kv_quant_hook",
    "quantize_kv_int8",
    "roundtrip_kv_int8",
    "selection_set_agreement",
]
