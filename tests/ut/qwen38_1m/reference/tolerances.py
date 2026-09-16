# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-op numerical tolerances for the T0.6 eager reference harness.

PRD Sec 8.1 requires every tolerance to be a *named constant declared in source
before any comparison assertion*, with a documented rationale. Each constant
below is consumed by exactly one self-consistency test; the rationale explains
why the bound is both safe (a real regression fails) and achievable (two correct
formulations of the same math agree within it).

Naming: ``<COMPONENT>_<KIND>_{RTOL,ATOL}``. All references compute in float64 or
float32; these bounds apply to comparisons between two float32/float64 results of
the *same* op computed two different ways (e.g. chunked vs. recurrent).
"""

# ---------------------------------------------------------------------------
# GDN (Gated DeltaNet)
# ---------------------------------------------------------------------------
# Chunked vs. unchunked delta rule are algebraically identical but reassociate a
# long float64 recurrence into per-chunk matrix solves. Over sequences of a few
# hundred tokens with |g| decay, accumulated float64 reassociation error stays
# well under 1e-9 relative; 1e-8 leaves head-room without masking a real bug
# (a wrong sign / off-by-one in the WY transform shifts outputs by O(1)).
GDN_CHUNK_RTOL = 1e-8
GDN_CHUNK_ATOL = 1e-9

# Causal depthwise conv reference vs. torch F.conv1d (both float64). Pure linear
# combination of identical inputs -> agreement is at rounding level.
GDN_CONV_RTOL = 1e-10
GDN_CONV_ATOL = 1e-12

# ---------------------------------------------------------------------------
# n-gram hashing
# ---------------------------------------------------------------------------
# Hash identities are exact integer arithmetic. The vectorized reference and the
# brute-force per-token re-derivation must agree *bit for bit*; tolerance is zero
# and comparisons use torch.equal, but we expose the intent as a constant.
NGRAM_HASH_EXACT = 0  # exact integer equality required

# ---------------------------------------------------------------------------
# PLE (position-learning enhancement)
# ---------------------------------------------------------------------------
# PLE gate/conv reference vs. an independent eager (float64) re-implementation of
# the same RMSNorm + sigmoid-gate + dilated-conv math. Only elementwise ops and
# small reductions over the hidden group; float64 agreement is at rounding level.
PLE_GATE_RTOL = 1e-10
PLE_GATE_ATOL = 1e-12
PLE_CONV_RTOL = 1e-10
PLE_CONV_ATOL = 1e-12

# ---------------------------------------------------------------------------
# QSA indexer
# ---------------------------------------------------------------------------
# Selection sets are compared as *sets of integer token ids* -> exact. Logit
# scoring (relu-summed dot products, float64) is compared against a brute-force
# einsum re-derivation at rounding level.
QSA_INDEXER_SCORE_RTOL = 1e-10
QSA_INDEXER_SCORE_ATOL = 1e-12
QSA_INDEXER_SELECTION_EXACT = 0  # exact set/index equality required

# ---------------------------------------------------------------------------
# QSA sparse attention
# ---------------------------------------------------------------------------
# Sparse (selected-token) attention vs. dense causal attention restricted to the
# same selected set. Softmax + weighted sum in float64; log2/exp2 vs. ln/exp
# reformulations agree to ~1e-9. 1e-8 is safe.
QSA_ATTN_RTOL = 1e-8
QSA_ATTN_ATOL = 1e-9

# ---------------------------------------------------------------------------
# W8A8 dynamic INT8 QDQ
# ---------------------------------------------------------------------------
# Round-trip dequant(quant(x)) recovers x only up to one quantization step. The
# correct, tight bound is *absolute per element*: |err| <= scale/2 (half a grid
# step); the reconstruction of a near-zero element has O(1) relative error, so a
# relative bound would be meaningless here. Tests assert the hard scale/2 bound
# directly, plus a small float slack for the round + FP32 arithmetic.
W8A8_ROUNDTRIP_EPS = 1e-4
# The QDQ linear (dequant(quant(x)) @ dequant(w).T) compared against the same
# dequantized weights times the *dequantized* activations is definitional, so
# only FP32 matmul rounding separates them.
W8A8_GEMM_RTOL = 1e-5
W8A8_GEMM_ATOL = 1e-4
# Aggregate closeness of the full QDQ GEMM to the true (un-quantized) FP GEMM:
# per-element quant noise (~scale/2) averages down over the contraction, so the
# Frobenius relative error of the whole output stays well under 5%.
W8A8_LINEAR_REL = 0.05
