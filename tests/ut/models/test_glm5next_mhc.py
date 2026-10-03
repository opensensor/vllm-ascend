# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

from vllm_ascend.models.glm5next.model import Glm5NextDecoderLayer


def _rms_norm(x, weight, epsilon):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("post_scale", [2.0, 1.5])
@pytest.mark.parametrize("with_norm", [False, True])
def test_mhc_native_ops_preserve_deferred_mixing_and_input_norm(post_scale, with_norm):
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.n = 4
    layer.mhc_sinkhorn_iterations = 20
    layer.rms_norm_eps = 1e-6
    layer.hc_eps = 1e-6
    layer.mhc_post_mult_value = post_scale
    torch.manual_seed(0)
    residual = torch.randn(3, 4, 8).bfloat16()
    original = residual.clone()
    fn = torch.randn(24, 32) * 0.1
    scale = torch.tensor([0.5, 0.6, 0.7])
    base = torch.randn(24) * 0.1
    weight = torch.linspace(0.5, 1.5, 8).bfloat16() if with_norm else None
    norm_eps = 1e-5

    def pre_op(**kwargs):
        post, comb, y = mhc_pre_torch(
            kwargs["residual"],
            kwargs["fn"],
            kwargs["hc_scale"],
            kwargs["hc_base"],
            kwargs["rms_eps"],
            kwargs["hc_pre_eps"],
            kwargs["hc_sinkhorn_eps"],
            kwargs["hc_post_mult_value"],
            kwargs["sinkhorn_repeat"],
        )
        if kwargs["norm_weight"] is not None:
            y = _rms_norm(y, kwargs["norm_weight"], kwargs["norm_eps"])
        return post, comb, y

    def fused_post_pre_op(**kwargs):
        mixed = mhc_post_torch(kwargs["x"], kwargs["residual"], kwargs["post_layer_mix"], kwargs["comb_res_mix"])
        post, comb, y = pre_op(**{**kwargs, "residual": mixed})
        return mixed, post, comb, y

    layer.mhc_pre_op = pre_op
    layer.mhc_post_op = mhc_post_torch
    layer.mhc_fused_post_pre_op = fused_post_pre_op

    expected_post, expected_comb, expected_y = mhc_pre_torch(
        residual, fn, scale, base, layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, post_scale, 20
    )
    if with_norm:
        expected_y = _rms_norm(expected_y, weight, norm_eps)
    post, comb, y = layer.hc_pre(residual, fn, scale, base, norm_weight=weight, norm_eps=norm_eps)
    for actual, expected in zip((post, comb, y), (expected_post, expected_comb, expected_y)):
        torch.testing.assert_close(actual, expected)

    mixed = mhc_post_torch(expected_y, residual, expected_post, expected_comb)
    next_post, next_comb, next_y = mhc_pre_torch(
        mixed, fn, scale, base, layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, post_scale, 20
    )
    if with_norm:
        next_y = _rms_norm(next_y, weight, norm_eps)
    torch.testing.assert_close(layer.hc_post(y, residual, post, comb), mixed)
    actual = layer.hc_fused_post_pre(y, residual, post, comb, fn, scale, base, norm_weight=weight, norm_eps=norm_eps)
    for value, expected in zip(actual, (mixed, next_post, next_comb, next_y)):
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(residual, original)
