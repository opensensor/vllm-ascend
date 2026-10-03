# SPDX-License-Identifier: Apache-2.0
"""Default-off 310P mHC output-batching candidate regression checks."""

from types import SimpleNamespace

import torch

from vllm_ascend.patch.worker import patch_mhc_norm as mhc_patch
from vllm_ascend.patch.worker.patch_mhc_norm import _round_mhc_outputs_batch, _round_mhc_state


def test_mhc_batched_outputs_match_separate_rounding_bits():
    generator = torch.Generator().manual_seed(20261003)
    for rows in (1, 4, 512):
        outputs = (
            torch.randn((rows, 4, 1), generator=generator),
            torch.randn((rows, 4, 4), generator=generator),
            torch.randn((rows, 6144), generator=generator),
        )
        expected = tuple(_round_mhc_state(output) for output in outputs)
        actual = _round_mhc_outputs_batch(*outputs)
        for original, single, batched in zip(outputs, expected, actual, strict=True):
            assert batched.shape == original.shape
            torch.testing.assert_close(batched.view(torch.int32), single.view(torch.int32), rtol=0, atol=0)


def test_mhc_batched_outputs_support_noncontiguous_inputs():
    base = torch.arange(96, dtype=torch.float32).reshape(4, 4, 6)
    outputs = (base[:, :, :1], base[:, :, ::2], base[:, :, 1::2])
    actual = _round_mhc_outputs_batch(*outputs)
    for original, batched in zip(outputs, actual, strict=True):
        torch.testing.assert_close(batched, _round_mhc_state(original), rtol=0, atol=0)


def test_mhc_batched_fused_post_pre_rounds_residual_before_pre(monkeypatch):
    residual_after_post = torch.tensor([1.006, -1.006], dtype=torch.float32)
    expected_residual = _round_mhc_state(residual_after_post)
    outputs = (
        torch.tensor([1.006], dtype=torch.float32),
        torch.tensor([-1.006], dtype=torch.float32),
        torch.tensor([1.02], dtype=torch.float32),
    )

    def pre_impl(residual, *_args):
        torch.testing.assert_close(residual.view(torch.int32), expected_residual.view(torch.int32), rtol=0, atol=0)
        return outputs

    monkeypatch.setattr(mhc_patch, "_mhc_post_torch", lambda *_args: residual_after_post)
    monkeypatch.setattr(mhc_patch, "_mhc_pre_torch", pre_impl)
    op = SimpleNamespace(use_310p_batched_bf16_round=True, use_310p_sinkhorn=False)
    actual = mhc_patch._mhc_fused_post_pre_npu(
        op,
        x=None,
        residual=None,
        post_layer_mix=None,
        comb_res_mix=None,
        fn=None,
        hc_scale=None,
        hc_base=None,
        rms_eps=None,
        hc_pre_eps=None,
        hc_sinkhorn_eps=None,
        hc_post_mult_value=None,
        sinkhorn_repeat=None,
    )
    expected = (expected_residual, *(_round_mhc_state(output) for output in outputs))
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.view(torch.int32), reference.view(torch.int32), rtol=0, atol=0)
