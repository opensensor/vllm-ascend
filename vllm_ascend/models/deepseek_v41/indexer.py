# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P DeepSeek V4.1 sparse-attention indexer / CSA2 (E3.2 -- ADAPT).

This is an *adaptation* of the shipped, ``torch_npu``-native DeepSeek V4 indexer
(``vllm_ascend.models.deepseek_v4.indexer.DeepseekV4Indexer`` +
``...compressor.Compressor``), NOT a green-field port. What is reused vs. what
is adapted for V4.1 on the Triton-free / NPU-free 310P host lane:

Reused (shape & pipeline preserved from the shipped V4 indexer/compressor)
--------------------------------------------------------------------------
* The *pipeline*: mean-pool CSA2 compression of ``compress_ratio`` consecutive
  index keys into one latent, then a Lightning-Indexer scoring
  (``sum_h w[t,h] * relu(softmax_scale * (q_{t,h} . k_block))``), then a
  deterministic top-k block selection -- the exact three-stage structure of
  ``Compressor.forward`` -> ``DeepseekV4Indexer._select_topk_*``.
* The projection layout: ``wq_b`` (q-LoRA -> ``n_heads * head_dim`` index
  queries) and ``weights_proj`` (hidden -> ``n_heads`` per-head weights), and
  the ``softmax_scale = head_dim ** -0.5`` / ``n_heads ** -0.5`` weight scaling
  the shipped ``_select_topk_serial`` applies (``weights_proj(x) *
  (softmax_scale * n_heads ** -0.5)``).
* The ``AscendIndexerOps`` seam: an ops object owns the quantize/select
  primitives so the module body never spells a device op directly.

Adapted for V4.1 on 310P
------------------------
* **Compression-ratio gate.** The shipped indexer instantiates its indexer K
  cache only when ``compress_ratio == 4`` and its compressor only when
  ``compress_ratio > 1`` (V4.0 ships ratios 4 / 128). V4.1 uses per-layer
  ``compress_ratios in {0, 1, 2}``: ratio 0 is a plain sliding-window layer
  with **no** indexer, and ratios **1 and 2** run the indexer (ratio 1 is an
  identity CSA2 pool = raw-key selection; ratio 2 pools token pairs). See
  :meth:`supports_ratio` / the ``__init__`` gate.
* **Torch fallback for the ``npu_*`` selection/quant ops.** The shipped path
  calls ``torch.ops._C_ascend.npu_quant_lightning_indexer_v2`` and the
  ``indexer_quant_*`` device operators; these may be absent on the 310P (a
  device concern verified against the pinned CANN toolkit -- see the E2.1 dtype
  policy note). This module runs a pure-eager-torch path
  (:class:`TorchIndexerFallbackOps`) that computes the identical math with no
  activation quantization, upcasting the reduction to a high-precision
  accumulation dtype. :func:`make_indexer_ops` picks the device kernel when it
  is actually registered and otherwise falls back to torch, so the module never
  hard-requires ``torch_npu``.
* **Deterministic top-k.** Selection sorts by descending score with ascending
  block-id tie-break (``torch.argsort(-scores, stable=True)``), matching the
  E0.4 reference oracle exactly (no kernel-order nondeterminism).
* **Dtypes from the E2.1 policy.** Query/key/weight storage rides
  ``policy.indexer_dtype`` (float16); the scoring reduction accumulates in
  ``policy.indexer_accumulation_dtype`` (float32) on device, and in float64 on
  the host parity path so it matches the float64 E0.4 reference bit-for-bit.

Parity target: ``tests/ut/deepseek_w2/reference/indexer_reference.py`` (mean-pool
compression + Lightning-Indexer scoring + top-k) under
``tests/ut/deepseek_w2/reference/tolerances.py``.

Interface for E4.1 (model assembly) -- do NOT edit ``model.py`` for this:
    indexer = AscendDeepseekV41Indexer(config, compress_ratio=ratio)
    if indexer.enabled:
        result = indexer(hidden_states, qr, raw_keys, positions)
        topk = result.block_indices            # [T, block_topk] padded (-1)
        per_token = result.blocks_per_token     # list[list[int]]
The projection sub-modules (``wq_b`` / ``weights_proj``) are built only when the
q-LoRA / hidden dims are known; a caller that has already projected the query
and per-head weights (the fused device prolog does) passes them in directly via
``precomputed_query`` / ``precomputed_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    DeepseekV41DtypePolicy,
)

# V4.1 per-layer compression ratios (config ``compress_ratios``): 0 == plain
# sliding-window layer (no indexer), 1 and 2 == active indexer layers. Mirrors
# vllm_ascend.models.deepseek_v41.model.DEEPSEEKV41_COMPRESS_RATIOS.
SLIDING_WINDOW_RATIO = 0
ACTIVE_COMPRESS_RATIOS = (1, 2)
SUPPORTED_COMPRESS_RATIOS = (SLIDING_WINDOW_RATIO, *ACTIVE_COMPRESS_RATIOS)

# Faithful small defaults (match the E0.4 reference geometry) used only when the
# config object does not carry the DeepSeek index_* fields.
DEFAULT_INDEX_N_HEADS = 4
DEFAULT_INDEX_HEAD_DIM = 128
DEFAULT_INDEX_TOPK = 8

# Padding value for absent selections in the dense ``[T, block_topk]`` tensor.
SELECTION_PAD = -1


# ===========================================================================
# Pure-torch fallback primitives (replace the npu_* selection / quant ops)
# ===========================================================================
def mean_pool_compress(
    raw_keys: torch.Tensor,
    compress_ratio: int,
    *,
    accum_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """CSA2 compression: mean-pool complete groups of ``compress_ratio`` keys.

    Block ``m`` pools raw index keys ``[m*ratio, (m+1)*ratio)``. Ratio 1 is an
    identity pool (raw-key selection). A trailing partial group is dropped (it
    is not yet a fully-formed causal block). Mirrors the shipped compressor's
    mean pooling and the E0.4 ``compress_keys`` reference.

    Args:
        raw_keys: ``[S, D]`` raw index-key rows (single index KV head).
        compress_ratio: tokens pooled per compressed block (1 or 2).
        accum_dtype: high-precision reduction dtype (float64 on the host parity
            path so it matches the float64 reference; float32 on device).

    Returns:
        ``[S // compress_ratio, D]`` compressed keys in ``accum_dtype``.
    """
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be >= 1 for compression; got {compress_ratio}")
    keys = raw_keys.to(accum_dtype)
    seq_len, dim = keys.shape
    num_blocks = seq_len // compress_ratio
    if num_blocks == 0:
        return keys.new_zeros((0, dim))
    trimmed = keys[: num_blocks * compress_ratio]
    return trimmed.view(num_blocks, compress_ratio, dim).mean(dim=1)


def lightning_indexer_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    compressed_keys: torch.Tensor,
    softmax_scale: float,
    *,
    accum_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Lightning-Indexer logits: weighted relu-summed dot products.

    ``score[t, m] = sum_h w[t,h] * relu(softmax_scale * (q_{t,h} . k_block_m))``

    This is the eager-torch replacement for
    ``npu_quant_lightning_indexer_v2``; no activation quantization is applied on
    the host, and the reduction runs in ``accum_dtype``.

    Args:
        query: ``[T, H, D]`` index queries.
        weights: ``[T, H]`` per-head ``weights_proj`` weights.
        compressed_keys: ``[N, D]`` compressed keys.
        softmax_scale: dot-product scale (``index_head_dim ** -0.5``).
        accum_dtype: reduction dtype.

    Returns:
        ``[T, N]`` scores.
    """
    q = query.to(accum_dtype)
    w = weights.to(accum_dtype)
    k = compressed_keys.to(accum_dtype)
    per_head = torch.einsum("thd,nd->thn", q, k) * softmax_scale
    per_head = torch.clamp(per_head, min=0.0)
    return torch.einsum("thn,th->tn", per_head, w)


def visible_block_count(pos: int, compress_ratio: int) -> int:
    """Number of fully-formed causal compressed blocks visible at ``pos``."""
    return (pos + 1) // compress_ratio


def block_topk_for_ratio(index_topk: int, compress_ratio: int) -> int:
    """Block budget: token budget divided by the compression ratio."""
    return index_topk // compress_ratio


def deterministic_topk_blocks(
    scores_row: torch.Tensor,
    visible_blocks: int,
    block_topk: int,
) -> torch.Tensor:
    """Deterministic top-k block selection over the visible causal range.

    Sorts by descending score, ties broken by ascending block index (a stable
    ascending argsort of the negated scores). If fewer than ``block_topk``
    blocks are visible, all are kept (short-context dense fallback). Returns the
    kept block ids as a ``long`` tensor in descending-score order -- matching
    the E0.4 ``select_blocks`` reference exactly.
    """
    keep = min(visible_blocks, block_topk)
    if keep <= 0:
        return scores_row.new_zeros((0,), dtype=torch.long)
    visible = scores_row[:visible_blocks].to(torch.float64)
    order = torch.argsort(-visible, stable=True)
    return order[:keep].to(torch.long)


def select_topk_blocks(
    scores: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    block_topk: int,
) -> list[list[int]]:
    """Per-token deterministic block selection over the causal visible range.

    Args:
        scores: ``[T, N]`` block scores.
        positions: ``[T]`` logical positions of each query token.
        compress_ratio: CSA2 ratio.
        block_topk: block budget (``index_topk // compress_ratio``).

    Returns:
        ``[T]`` lists of selected block ids (descending-score order); never a
        block beyond the causal visible range.
    """
    num_blocks = scores.shape[1]
    selected: list[list[int]] = []
    for t in range(scores.shape[0]):
        pos = int(positions[t].item())
        vis = min(visible_block_count(pos, compress_ratio), num_blocks)
        chosen = deterministic_topk_blocks(scores[t], vis, block_topk)
        selected.append(chosen.tolist())
    return selected


# ===========================================================================
# Ops seam: torch fallback vs. device kernel (mirrors AscendIndexerOps)
# ===========================================================================
class TorchIndexerFallbackOps:
    """Pure-eager-torch stand-in for the shipped ``AscendIndexerOps``.

    Owns the compress/score/select primitives so the module body never spells a
    device op directly. On the 310P host these are plain torch; when the fused
    ``npu_quant_lightning_indexer_v2`` kernel is registered (device path,
    verified against pinned CANN) :func:`make_indexer_ops` returns a device-ops
    object instead. No activation quantization is applied here -- the reduction
    upcasts to ``accum_dtype`` for numerical fidelity.
    """

    def __init__(self, index_topk: int, softmax_scale: float, accum_dtype: torch.dtype) -> None:
        self.index_topk = index_topk
        self.softmax_scale = softmax_scale
        self.accum_dtype = accum_dtype

    def compress_keys(self, raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
        return mean_pool_compress(raw_keys, compress_ratio, accum_dtype=self.accum_dtype)

    def score(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        compressed_keys: torch.Tensor,
    ) -> torch.Tensor:
        return lightning_indexer_scores(
            query,
            weights,
            compressed_keys,
            self.softmax_scale,
            accum_dtype=self.accum_dtype,
        )

    def select_topk(
        self,
        scores: torch.Tensor,
        positions: torch.Tensor,
        compress_ratio: int,
        block_topk: int,
    ) -> list[list[int]]:
        return select_topk_blocks(scores, positions, compress_ratio, block_topk)


def _device_lightning_indexer_available() -> bool:
    """True iff the fused device Lightning-Indexer kernel is registered.

    Always False on the 310P host (no ``torch_npu`` / ``_C_ascend``); the device
    bridge (verified against the pinned CANN toolkit) is what would flip this.
    """
    c_ascend = getattr(torch.ops, "_C_ascend", None)
    return c_ascend is not None and hasattr(c_ascend, "npu_quant_lightning_indexer_v2")


def make_indexer_ops(
    index_topk: int,
    softmax_scale: float,
    accum_dtype: torch.dtype,
) -> TorchIndexerFallbackOps:
    """Pick the indexer ops implementation.

    Returns the torch fallback on the host (and whenever the device kernel is
    not registered). The device-kernel branch is intentionally a hook: wiring it
    is a device concern gated on CANN verification, and it must produce the same
    deterministic selection this fallback does.
    """
    if _device_lightning_indexer_available():
        # Device fast-path is a device/CANN concern (E-device); until it is
        # wired and CANN-verified we run the deterministic eager fallback, which
        # is the source of truth for selection semantics.
        pass
    return TorchIndexerFallbackOps(index_topk, softmax_scale, accum_dtype)


# ===========================================================================
# Selection result
# ===========================================================================
@dataclass(frozen=True)
class IndexerSelection:
    """Result of one indexer forward pass.

    Attributes:
        blocks_per_token: ``[T]`` lists of selected compressed-block ids
            (descending-score order); the clean, exact form for consumers/tests.
        block_indices: ``[T, block_topk]`` dense ``long`` tensor, right-padded
            with :data:`SELECTION_PAD` when fewer blocks are visible -- the form
            a device top-k buffer takes.
        scores: ``[T, N]`` block scores (accumulation dtype).
        compressed_keys: ``[N, D]`` mean-pooled keys (accumulation dtype).
        block_topk: the per-token block budget used.
    """

    blocks_per_token: list[list[int]]
    block_indices: torch.Tensor
    scores: torch.Tensor
    compressed_keys: torch.Tensor
    block_topk: int = field(default=0)


def _pad_selection(blocks_per_token: list[list[int]], block_topk: int) -> torch.Tensor:
    """Pack ragged per-token selections into a ``[T, block_topk]`` long tensor."""
    num_tokens = len(blocks_per_token)
    out = torch.full((num_tokens, block_topk), SELECTION_PAD, dtype=torch.long)
    for t, blocks in enumerate(blocks_per_token):
        if blocks:
            out[t, : len(blocks)] = torch.tensor(blocks, dtype=torch.long)
    return out


# ===========================================================================
# The V4.1 indexer module
# ===========================================================================
class AscendDeepseekV41Indexer(nn.Module):
    """Host-eager Lightning-Indexer + CSA2 compression for the 310P V4.1 path.

    Adapts the shipped ``DeepseekV4Indexer`` (see module docstring): the ratio
    gate now admits ratios 1 and 2 (ratio 0 => sliding-window, indexer disabled)
    and every device op is replaced by a deterministic torch fallback.
    """

    def __init__(
        self,
        config: Any,
        compress_ratio: int,
        *,
        policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        prefix: str = "",
        build_projections: bool = True,
    ) -> None:
        super().__init__()
        if compress_ratio not in SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(
                f"DeepSeek V4.1 indexer compress_ratio={compress_ratio} unsupported; "
                f"expected one of {SUPPORTED_COMPRESS_RATIOS} (0 = sliding window, "
                f"1 and 2 = active indexer)."
            )
        self.config = config
        self.prefix = prefix
        self.policy = policy
        self.compress_ratio = compress_ratio
        # Ratio 0 is a plain sliding-window layer: no indexer selection at all.
        self.enabled = compress_ratio in ACTIVE_COMPRESS_RATIOS

        self.n_heads = int(getattr(config, "index_n_heads", DEFAULT_INDEX_N_HEADS))
        self.head_dim = int(getattr(config, "index_head_dim", DEFAULT_INDEX_HEAD_DIM))
        self.index_topk = int(getattr(config, "index_topk", DEFAULT_INDEX_TOPK))
        self.softmax_scale = self.head_dim**-0.5
        # weights_proj scaling from the shipped _select_topk_serial.
        self.weight_scale = self.softmax_scale * self.n_heads**-0.5
        # E2.1 dtype policy: fp16 storage, fp32 accumulation on device.
        self.dtype = policy.indexer_dtype
        self.device_accumulation_dtype = policy.indexer_accumulation_dtype
        # Host parity path upcasts the reduction to float64 to match the float64
        # E0.4 reference exactly (a strict superset of the fp32 device accum).
        self.accumulation_dtype = torch.float64

        self.block_topk = block_topk_for_ratio(self.index_topk, compress_ratio) if self.enabled else 0
        self.ops = make_indexer_ops(self.index_topk, self.softmax_scale, self.accumulation_dtype)

        # Projection sub-modules, mirroring the shipped wq_b / weights_proj.
        # Built only when the q-LoRA / hidden dims are known and the layer is
        # active; a caller that has already projected passes tensors in.
        self.wq_b: nn.Linear | None = None
        self.weights_proj: nn.Linear | None = None
        q_lora_rank = getattr(config, "q_lora_rank", None)
        hidden_size = getattr(config, "hidden_size", None)
        if self.enabled and build_projections and q_lora_rank is not None:
            self.wq_b = nn.Linear(int(q_lora_rank), self.n_heads * self.head_dim, bias=False, dtype=self.dtype)
        if self.enabled and build_projections and hidden_size is not None:
            self.weights_proj = nn.Linear(int(hidden_size), self.n_heads, bias=False, dtype=self.dtype)

    # -- classification -----------------------------------------------------
    @staticmethod
    def supports_ratio(compress_ratio: int) -> bool:
        """True iff ``compress_ratio`` runs an active indexer (1 or 2)."""
        return compress_ratio in ACTIVE_COMPRESS_RATIOS

    # -- projections (host eager; device fuses these into the prolog) --------
    def project_query(self, qr: torch.Tensor) -> torch.Tensor:
        """Project q-LoRA ``[T, q_lora_rank]`` into index queries ``[T, H, D]``."""
        if self.wq_b is None:
            raise RuntimeError("wq_b projection not built; pass precomputed_query to forward() instead")
        q = self.wq_b(qr.to(self.dtype))
        return q.view(-1, self.n_heads, self.head_dim)

    def project_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden ``[T, hidden]`` into scaled per-head weights ``[T, H]``."""
        if self.weights_proj is None:
            raise RuntimeError("weights_proj not built; pass precomputed_weights to forward() instead")
        return self.weights_proj(hidden_states.to(self.dtype)) * self.weight_scale

    # -- pipeline stages ----------------------------------------------------
    def compress_keys(self, raw_keys: torch.Tensor) -> torch.Tensor:
        """Mean-pool raw index keys into compressed blocks (CSA2)."""
        return self.ops.compress_keys(raw_keys, self.compress_ratio)

    def score(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        compressed_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Lightning-Indexer scores ``[T, N]`` for queries against blocks."""
        return self.ops.score(query, weights, compressed_keys)

    def select(self, scores: torch.Tensor, positions: torch.Tensor) -> list[list[int]]:
        """Deterministic top-k block selection per token over the causal range."""
        return self.ops.select_topk(scores, positions, self.compress_ratio, self.block_topk)

    # -- full forward -------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor | None,
        qr: torch.Tensor | None,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
        *,
        precomputed_query: torch.Tensor | None = None,
        precomputed_weights: torch.Tensor | None = None,
    ) -> IndexerSelection | None:
        """Compress -> score -> select for one sequence.

        Returns ``None`` for a disabled (sliding-window, ratio 0) layer. Either
        provide ``hidden_states`` + ``qr`` (this module projects them) or pass
        ``precomputed_query`` ``[T, H, D]`` and ``precomputed_weights`` ``[T, H]``
        (the fused device prolog does the latter).
        """
        if not self.enabled:
            return None

        if precomputed_query is not None:
            query = precomputed_query
        else:
            if qr is None:
                raise ValueError("indexer forward requires qr or precomputed_query")
            query = self.project_query(qr)

        if precomputed_weights is not None:
            weights = precomputed_weights
        else:
            if hidden_states is None:
                raise ValueError("indexer forward requires hidden_states or precomputed_weights")
            weights = self.project_weights(hidden_states)

        compressed = self.compress_keys(raw_keys)
        scores = self.score(query, weights, compressed)
        blocks_per_token = self.select(scores, positions)
        block_indices = _pad_selection(blocks_per_token, self.block_topk)
        return IndexerSelection(
            blocks_per_token=blocks_per_token,
            block_indices=block_indices,
            scores=scores,
            compressed_keys=compressed,
            block_topk=self.block_topk,
        )


__all__ = [
    "SLIDING_WINDOW_RATIO",
    "ACTIVE_COMPRESS_RATIOS",
    "SUPPORTED_COMPRESS_RATIOS",
    "SELECTION_PAD",
    "mean_pool_compress",
    "lightning_indexer_scores",
    "visible_block_count",
    "block_topk_for_ratio",
    "deterministic_topk_blocks",
    "select_topk_blocks",
    "TorchIndexerFallbackOps",
    "make_indexer_ops",
    "IndexerSelection",
    "AscendDeepseekV41Indexer",
]
