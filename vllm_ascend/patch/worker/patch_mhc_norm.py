# SPDX-License-Identifier: Apache-2.0
# mHC per-sublayer input RMSNorm for the 310P (Triton-independent).
#
# The tilelang mHC pre kernel fuses the layer's input RMSNorm into layer_input,
# but mhc_pre_torch (which forward_native/forward_oot use on NPU) ignores
# norm_weight/norm_eps entirely, so every mHC layer would run WITHOUT its input
# norm -> the layer input tracks the (growing) fp32 residual accumulator instead
# of being normalized to unit-RMS*gamma, blowing up activations across depth.
#
# This patch reapplies the norm to match CUDA/tilelang semantics:
#   layer_input = layer_input * rsqrt(mean(layer_input^2) + norm_eps) * norm_weight
#
# NOTE: this lives in its OWN module (imported UNCONDITIONALLY) rather than in
# patch_triton.py, because patch_triton.py is only imported `if HAS_TRITON:` and
# the 310P has no Triton -- so the mHC norm never loaded there. This patch has
# no Triton dependency and must apply on every worker regardless of Triton.
try:
    import torch as _t
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch as _mhc_post_torch,
    )
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_pre_torch as _mhc_pre_torch,
    )
    from vllm.model_executor.layers import mhc as _mhc_mod

    # Four-token decode has four mHC streams of 4096 values per token.
    # Keep larger prefill buffers on the native path until separately gated.
    MHC_AI_CORE_ROUND_MAX_ELEMENTS = 65_536

    def _mhc_rms_norm(x, weight, eps):
        # `x` (the mHC layer_input) may arrive in fp32 because the residual
        # accumulator rides fp32 on the 310P. RMSNorm normalizes away the
        # magnitude, so downcast the result to the norm weight's compute dtype
        # (fp16) for the attn/MLP submodules; the fp32 accumulator is carried
        # separately by mhc_post and is never touched here.
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * _t.rsqrt(var + eps) * weight.float()).to(weight.dtype)

    def _round_mhc_state(x, use_fp16: bool = False, use_ai_core: bool = False):
        """Round mHC state to the reference BF16 or experimental FP16 precision.

        The golden runs the mHC residual + hyper-connection mixes in bf16
        (7 stored fraction bits, 8 exponent bits); the 310P carries them in
        fp32 (23 stored fraction bits).
        fp32 is *more* precise but a *different* rounding, and that drift seeds
        the residual and compounds across depth. The default rounds the fp32
        intermediates to bf16 precision while keeping their exponent range.
        The opt-in fp16 round trip tests the faster native 310P conversion; it
        changes the rounding and narrows the exponent range.
        """
        if use_fp16:
            return x.to(_t.float16).to(_t.float32)
        if (
            use_ai_core
            and x.dtype == _t.float32
            and x.is_contiguous()
            and 0 < x.numel() <= MHC_AI_CORE_ROUND_MAX_ELEMENTS
        ):
            return _t.ops._C_ascend.mhc_bf16_round_310(x)
        # Native BF16 conversion is bit-exact for finite FP32 inputs on 310P.
        # The integer emulation launched BitwiseAndScalar on AI CPU for every
        # mHC state tensor, dominating prefill and slowing graph replay.
        return x.to(_t.bfloat16).to(_t.float32)

    def _round_mhc_outputs_batch(post_mix, comb_mix, layer_input, use_ai_core: bool = False):
        """Round the three independent pre outputs with one BF16 cast pair.

        A fused post/pre must still round its residual before computing these
        outputs; only the already-computed outputs can share a conversion.
        """
        outputs = (post_mix, comb_mix, layer_input)
        sizes = tuple(output.numel() for output in outputs)
        flattened = _t.cat([output.reshape(-1) for output in outputs])
        rounded = _round_mhc_state(flattened, use_ai_core=use_ai_core)
        return tuple(piece.view_as(output) for piece, output in zip(rounded.split(sizes), outputs, strict=True))

    def _mhc_pre_torch_sinkhorn_310(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    ):
        """Upstream mHC pre math with its 4x4 Sinkhorn loop in one 310P op."""
        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        if hc_mult != 4:
            raise ValueError("310P fused mHC Sinkhorn requires four residual streams")
        outer_shape = residual.shape[:-2]
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        num_tokens = residual_flat.shape[0]
        x = residual_flat.view(num_tokens, hc_mult * hidden_size).float()
        mixes = _t.matmul(x, fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * _t.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

        pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
        pre_mix = _t.sigmoid(pre_logits) + hc_pre_eps
        post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
        post_mix = _t.sigmoid(post_logits) * hc_post_mult_value
        comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[2] + hc_base[
            2 * hc_mult :
        ].view(1, hc_mult, hc_mult)
        comb_mix = _t.ops._C_ascend.mhc_sinkhorn_310(comb_logits.contiguous(), sinkhorn_repeat, hc_sinkhorn_eps)
        layer_input = _t.sum(pre_mix.unsqueeze(-1) * residual_flat.float(), dim=1).to(residual.dtype)
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    def _mhc_pre_npu(
        self,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        pre_impl = _mhc_pre_torch_sinkhorn_310 if getattr(self, "use_310p_sinkhorn", False) else _mhc_pre_torch
        post_mix, comb_mix, layer_input = pre_impl(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        use_fp16 = getattr(self, "use_310p_fp16_mhc_state", False)
        use_ai_core = getattr(self, "use_310p_ai_core_bf16_round", False)
        if getattr(self, "use_310p_batched_bf16_round", False) and not use_fp16:
            post_mix, comb_mix, layer_input = _round_mhc_outputs_batch(
                post_mix, comb_mix, layer_input, use_ai_core=use_ai_core
            )
        else:
            post_mix = _round_mhc_state(post_mix, use_fp16, use_ai_core)
            comb_mix = _round_mhc_state(comb_mix, use_fp16, use_ai_core)
            layer_input = _round_mhc_state(layer_input, use_fp16, use_ai_core)
        if norm_weight is not None:
            layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
        return post_mix, comb_mix, layer_input

    def _mhc_fused_post_pre_npu(
        self,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        residual_cur = _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        use_fp16 = getattr(self, "use_310p_fp16_mhc_state", False)
        use_ai_core = getattr(self, "use_310p_ai_core_bf16_round", False)
        residual_cur = _round_mhc_state(residual_cur, use_fp16, use_ai_core)
        pre_impl = _mhc_pre_torch_sinkhorn_310 if getattr(self, "use_310p_sinkhorn", False) else _mhc_pre_torch
        post_mix_cur, comb_mix_cur, layer_input_cur = pre_impl(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if getattr(self, "use_310p_batched_bf16_round", False) and not use_fp16:
            post_mix_cur, comb_mix_cur, layer_input_cur = _round_mhc_outputs_batch(
                post_mix_cur, comb_mix_cur, layer_input_cur, use_ai_core=use_ai_core
            )
        else:
            post_mix_cur = _round_mhc_state(post_mix_cur, use_fp16, use_ai_core)
            comb_mix_cur = _round_mhc_state(comb_mix_cur, use_fp16, use_ai_core)
            layer_input_cur = _round_mhc_state(layer_input_cur, use_fp16, use_ai_core)
        if norm_weight is not None:
            layer_input_cur = _mhc_rms_norm(layer_input_cur, norm_weight, norm_eps)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
except ImportError:
    pass
