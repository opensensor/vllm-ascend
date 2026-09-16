# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch (FP64/FP32) eager reference implementations for DeepSeek V4.1
552B W2 (2-bit experts) on Ascend 310P (task E0.4).

These references port the *formulas* (never the Triton code) from the vLLM CUDA
fork (``vllm/models/deepseek_v41/*``) so that later 310P parity tasks can
compare device kernels against a small, readable, CPU-only golden
implementation. Each module is independently importable and free of any Triton
import.

Modules:
    tolerances          -- per-op numerical tolerances (declared once, here).
    w2_moe_reference    -- W2 pack/unpack + INT8 grouped-QDQ SwiGLU MoE.
    engram_reference    -- Engram n-gram hash -> row gather -> gated projection.
    mla_reference       -- Multi-head latent attention (q/kv down-up, decoupled
                           RoPE, latent-KV) vs. a dense oracle.
    indexer_reference   -- Lightning indexer scoring + CSA2 block compression +
                           top-k selection.
"""
