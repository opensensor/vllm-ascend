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

    def _mhc_rms_norm(x, weight, eps):
        # `x` (the mHC layer_input) may arrive in fp32 because the residual
        # accumulator rides fp32 on the 310P. RMSNorm normalizes away the
        # magnitude, so downcast the result to the norm weight's compute dtype
        # (fp16) for the attn/MLP submodules; the fp32 accumulator is carried
        # separately by mhc_post and is never touched here.
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * _t.rsqrt(var + eps) * weight.float()).to(weight.dtype)

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
        post_mix, comb_mix, layer_input = _mhc_pre_torch(
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
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_torch(
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
        if norm_weight is not None:
            layer_input_cur = _mhc_rms_norm(layer_input_cur, norm_weight, norm_eps)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
except ImportError:
    pass
