# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-op numerical tolerances for the E0.4 DeepSeek V4.1 eager reference harness.

Every tolerance is a named constant declared here, before any comparison
assertion consumes it, with a documented rationale. Each explains why the bound
is both *safe* (a real regression trips it) and *achievable* (two correct
formulations of the same math agree within it). References compute in float64 /
float32; the bounds apply to comparisons between two results of the *same* op
computed two different ways (packed vs. dense, grouped vs. per-token,
sparse-gather vs. dense-mask).

Naming: ``<COMPONENT>_<KIND>_{RTOL,ATOL}`` (or ``_EPS`` / ``_EXACT`` where the
comparison is absolute / exact).
"""

# ---------------------------------------------------------------------------
# W2 (2-bit) expert weights
# ---------------------------------------------------------------------------
# pack -> unpack must be a bijection on the code field: comparisons are exact
# integer equality (``torch.equal``), exposed here as an intent constant.
W2_PACK_EXACT = 0

# Weight round-trip: the signed int2 grid {-2,-1,0,1} with a per-block scale
# chosen from both tails guarantees every element lands in the nearest-rounding
# interval, so |w - dequant(quant(w))| <= scale/2 (half a grid step) per
# element. Tests assert that hard bound directly plus a float slack for the
# round + float64 arithmetic.
W2_ROUNDTRIP_EPS = 1e-9

# The W2 QDQ linear equals its definition (dequant(quant(x)) @ dequant_w2(w).T)
# up to float64 matmul rounding only, since both sides share the exact same
# dequantized operands.
W2_LINEAR_RTOL = 1e-9
W2_LINEAR_ATOL = 1e-9

# Grouped MoE forward vs. the per-token oracle: the two differ only in how many
# rows are quantized together, and the activation quantizer is strictly
# per-row, so the two agree at float64 matmul-reassociation level.
W2_MOE_RTOL = 1e-9
W2_MOE_ATOL = 1e-9

# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------
# n-gram hash ids are exact integer arithmetic (rolling XOR of int64 products,
# then modulo a prime plus an offset): the vectorized reference and the
# brute-force per-token re-derivation must agree bit for bit.
ENGRAM_HASH_EXACT = 0

# Row gather (int8 codes * ue8m0 power-of-two block scale) is exact in float64
# because the scale is a power of two; the gather and its brute-force
# re-derivation match at rounding level.
ENGRAM_GATHER_RTOL = 1e-12
ENGRAM_GATHER_ATOL = 1e-12

# Engram gate/projection: normalized signed-sqrt sigmoid gate plus a residual
# add. Elementwise ops and small reductions over ``dim`` in float64 agree with
# an independent eager re-implementation at rounding level.
ENGRAM_PROJ_RTOL = 1e-10
ENGRAM_PROJ_ATOL = 1e-12

# ---------------------------------------------------------------------------
# MLA (multi-head latent attention)
# ---------------------------------------------------------------------------
# Absorbed-latent MLA vs. the dense (materialized k/v) oracle: algebraically
# identical (the up-projection is folded into the query/output projections), so
# only float64 softmax + matmul reassociation separates them.
MLA_RTOL = 1e-9
MLA_ATOL = 1e-9

# The decoupled-RoPE dot product computed two equivalent ways (rotate q then
# dot, vs. dot then rotate) matches at float64 rounding level.
MLA_ROPE_RTOL = 1e-10
MLA_ROPE_ATOL = 1e-12

# ---------------------------------------------------------------------------
# Lightning indexer / CSA2 compression
# ---------------------------------------------------------------------------
# Compressed-block scores (weighted relu-summed dot products, float64) vs. a
# brute-force einsum re-derivation, at rounding level.
INDEXER_SCORE_RTOL = 1e-10
INDEXER_SCORE_ATOL = 1e-12
# Selection sets are compared as sets of integer block/token ids -> exact.
INDEXER_SELECTION_EXACT = 0
# Mean-pool compression vs. an explicit per-block average -> rounding level.
INDEXER_COMPRESS_RTOL = 1e-12
INDEXER_COMPRESS_ATOL = 1e-12

__all__ = [
    "W2_PACK_EXACT",
    "W2_ROUNDTRIP_EPS",
    "W2_LINEAR_RTOL",
    "W2_LINEAR_ATOL",
    "W2_MOE_RTOL",
    "W2_MOE_ATOL",
    "ENGRAM_HASH_EXACT",
    "ENGRAM_GATHER_RTOL",
    "ENGRAM_GATHER_ATOL",
    "ENGRAM_PROJ_RTOL",
    "ENGRAM_PROJ_ATOL",
    "MLA_RTOL",
    "MLA_ATOL",
    "MLA_ROPE_RTOL",
    "MLA_ROPE_ATOL",
    "INDEXER_SCORE_RTOL",
    "INDEXER_SCORE_ATOL",
    "INDEXER_SELECTION_EXACT",
    "INDEXER_COMPRESS_RTOL",
    "INDEXER_COMPRESS_ATOL",
]
