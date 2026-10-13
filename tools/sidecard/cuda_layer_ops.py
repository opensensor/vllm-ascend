# SPDX-License-Identifier: Apache-2.0
"""Owner-local CUDA KDA recurrence; imported only by the CUDA actor."""

import torch
import triton
import triton.language as tl

VALUE_TILE = 8
MHC_STREAMS = 4
MHC_WIDTH = 4096
POST_TILE = 256
INDEXER_HEAD_DIM = 128


@triton.jit
def hadamard_kernel(Input, Output, SCALE: tl.constexpr):
    vector = tl.program_id(0)
    offsets = tl.arange(0, 128)
    value = tl.load(Input + vector * 128 + offsets).to(tl.float32)
    for stride in tl.static_range(7):
        distance = 1 << stride
        other = tl.gather(value, offsets ^ distance, axis=0)
        value = tl.where((offsets & distance) == 0, value + other, other - value)
    # Keep the reference's seven FP32 butterflies, normalization, and two
    # explicit rounding boundaries. Matrix multiplication changes their order.
    value = (value * SCALE).to(tl.bfloat16).to(tl.float16)
    tl.store(Output + vector * 128 + offsets, value)


def hadamard(x):
    if x.device.type != "cuda" or x.ndim < 1 or x.shape[-1] != INDEXER_HEAD_DIM:
        raise ValueError("CUDA indexer rotation requires 128-wide device vectors")
    if x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("CUDA indexer rotation requires contiguous FP32 input")
    output = torch.empty_like(x, dtype=torch.float16)
    if x.numel():
        hadamard_kernel[(x.numel() // INDEXER_HEAD_DIM,)](
            x, output, INDEXER_HEAD_DIM**-0.5, num_warps=4, enable_fp_fusion=False
        )
    return output


@triton.jit
def recurrent_kernel(
    Q, K, V, Gate, Beta, State, Out, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    head = tl.program_id(0)
    block = tl.program_id(1)
    keys = tl.arange(0, D)
    values = block * B + tl.arange(0, B)
    offsets = head * D * D + values[:, None] * D + keys[None, :]
    state = tl.load(State + offsets).to(tl.float32)
    for row in range(T):
        q = tl.load(Q + row * H * D + head * D + keys).to(tl.float32)
        k = tl.load(K + row * H * D + head * D + keys).to(tl.float32)
        v = tl.load(V + row * H * D + head * D + values).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (D**-0.5)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        decay = tl.exp(tl.load(Gate + row * H * D + head * D + keys).to(tl.float32))
        beta = tl.load(Beta + row * H + head).to(tl.float32)
        state *= decay[None, :]
        innovation = (v - tl.sum(state * k[None, :], 1)) * beta
        state += innovation[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(Out + row * H * D + head * D + values, output)
    tl.store(State + offsets, state)


def recurrent(q, k, v, gate, beta, state):
    if q.shape != k.shape or q.shape != v.shape or gate.shape != q.shape or beta.shape != q.shape[:2]:
        raise ValueError("CUDA recurrent KDA input geometry differs")
    rows, heads, width = q.shape
    if width != 128 or state.shape != (heads, width, width) or state.dtype != torch.float32:
        raise ValueError("CUDA recurrent KDA requires 128-wide FP32 VK state")
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    recurrent_kernel[(heads, width // VALUE_TILE)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        state,
        output,
        rows,
        heads,
        width,
        VALUE_TILE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def post_kernel(X, Residual, Post, Comb, Out, D: tl.constexpr, B: tl.constexpr):
    token, block = tl.program_id(0), tl.program_id(1)
    offsets = block * B + tl.arange(0, B)
    streams = tl.arange(0, 4)
    output = tl.full((4, B), 0, tl.float32)
    for source in tl.static_range(4):
        residual = tl.load(Residual + token * 4 * D + source * D + offsets).to(tl.float32)
        coefficients = tl.load(Comb + token * 16 + source * 4 + streams).to(tl.float32)
        output += coefficients[:, None] * residual[None, :]
    x = tl.load(X + token * D + offsets).to(tl.float32)
    post = tl.load(Post + token * 4 + streams).to(tl.float32)
    output += post[:, None] * x[None, :]
    tl.store(Out + token * 4 * D + streams[:, None] * D + offsets[None, :], output.to(tl.bfloat16).to(tl.float32))


def mhc_post(x, residual, post, comb):
    rows, streams, width = residual.shape
    if streams != MHC_STREAMS or width != MHC_WIDTH:
        raise ValueError("CUDA mHC post requires the admitted four-stream geometry")
    output = torch.empty_like(residual)
    post_kernel[(rows, width // POST_TILE)](
        x.contiguous(),
        residual.contiguous(),
        post.contiguous(),
        comb.contiguous(),
        output,
        width,
        POST_TILE,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def pre_mix_kernel(
    Residual, Mixes, Scale, Base, Pre, Post, Comb, RMS_EPS: tl.constexpr, HC_EPS: tl.constexpr, ITERATIONS: tl.constexpr
):
    token = tl.program_id(0)
    flat_offsets = tl.arange(0, 16384)
    residual = tl.load(Residual + token * 16384 + flat_offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(residual * residual, 0) / 16384 + RMS_EPS)
    streams = tl.arange(0, 4)
    scale_pre, scale_post, scale_comb = tl.load(Scale), tl.load(Scale + 1), tl.load(Scale + 2)
    pre_logits = tl.load(Mixes + token * 24 + streams) * inv_rms * scale_pre + tl.load(Base + streams)
    post_logits = tl.load(Mixes + token * 24 + 4 + streams) * inv_rms * scale_post + tl.load(Base + 4 + streams)
    pre = 1 / (1 + tl.exp(-pre_logits)) + HC_EPS
    post = 2 / (1 + tl.exp(-post_logits))
    comb_offsets = streams[:, None] * 4 + streams[None, :]
    logits = tl.load(Mixes + token * 24 + 8 + comb_offsets) * inv_rms * scale_comb + tl.load(Base + 8 + comb_offsets)
    comb = tl.exp(logits - tl.max(logits, 1)[:, None])
    comb = comb / tl.sum(comb, 1)[:, None] + HC_EPS
    comb = comb / (tl.sum(comb, 0)[None, :] + HC_EPS)
    for _ in range(ITERATIONS - 1):
        comb = comb / (tl.sum(comb, 1)[:, None] + HC_EPS)
        comb = comb / (tl.sum(comb, 0)[None, :] + HC_EPS)
    tl.store(Pre + token * 4 + streams, pre)
    tl.store(Post + token * 4 + streams, post.to(tl.bfloat16).to(tl.float32))
    tl.store(Comb + token * 16 + comb_offsets, comb.to(tl.bfloat16).to(tl.float32))


@triton.jit
def pre_input_kernel(Residual, Pre, Norm, Out, NORM_EPS: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, 4096)
    streams = tl.arange(0, 4)
    residual = tl.load(Residual + token * 16384 + streams[:, None] * 4096 + offsets[None, :]).to(tl.float32)
    pre = tl.load(Pre + token * 4 + streams).to(tl.float32)
    mixed = tl.sum(pre[:, None] * residual, 0).to(tl.bfloat16).to(tl.float32)
    value = mixed * tl.rsqrt(tl.sum(mixed * mixed, 0) / 4096 + NORM_EPS)
    value *= tl.load(Norm + offsets).to(tl.float32)
    tl.store(Out + token * 4096 + offsets, value)


def mhc_pre(residual, fn, scale, base, norm, rms_eps, hc_eps, iterations):
    rows = residual.shape[0]
    if residual.shape[1:] != (MHC_STREAMS, MHC_WIDTH):
        raise ValueError("CUDA mHC pre requires the admitted four-stream geometry")
    flat = residual.contiguous().flatten(1)
    mixes = flat @ fn.float().t()
    pre = torch.empty((rows, MHC_STREAMS), device="cuda", dtype=torch.float32)
    post = torch.empty((rows, MHC_STREAMS, 1), device="cuda", dtype=torch.float32)
    comb = torch.empty((rows, MHC_STREAMS, MHC_STREAMS), device="cuda", dtype=torch.float32)
    layer_input = torch.empty((rows, MHC_WIDTH), device="cuda", dtype=torch.float16)
    pre_mix_kernel[(rows,)](
        flat,
        mixes,
        scale.float(),
        base.float(),
        pre,
        post,
        comb,
        rms_eps,
        hc_eps,
        iterations,
        num_warps=8,
        enable_fp_fusion=False,
    )
    pre_input_kernel[(rows,)](flat, pre, norm, layer_input, rms_eps, num_warps=8, enable_fp_fusion=False)
    return post, comb, layer_input
