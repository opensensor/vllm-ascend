# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch (FP64/FP32) eager reference implementations for Qwen3.8-Flash-Next
/ Qwen4Exp components on Ascend 310P (task T0.6).

These references port the *formulas* (never the Triton code) from the vLLM CUDA
fork so that later 310P parity tasks can compare device kernels against a small,
readable, CPU-only golden implementation. Each module is independently importable
and free of any Triton import.

Modules:
    tolerances              -- per-op numerical tolerances (declared once, here).
    gdn_reference           -- Gated DeltaNet: gating, conv, chunked + unchunked.
    ngram_hash_reference    -- Qwen n-gram hashing with EOS boundary padding.
    ple_reference           -- PLE gate + dilated short convolution.
    qsa_indexer_reference   -- QSA indexer scoring, block selection, budget/expand.
    qsa_attention_reference -- QSA sparse attention (partial rotary, QK-norm, gate).
    w8a8_reference          -- W8A8 dynamic INT8 quantize/dequantize round-trip.
"""
