# SPDX-License-Identifier: Apache-2.0
"""310P regression test for preformatted Qwen4Exp eager linear weights."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import _format_eager_linear_weights_npu, _GatedResidual, _grouped_rms_norm
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def test_eager_linear_nz_preserves_output_and_skips_small_weight() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")

    dtype_policy = ASCEND_QWEN4EXP_DTYPE_POLICY
    layer = _GatedResidual(
        hc_count=2,
        hidden_size=64,
        lowrank=16,
        eps=1e-6,
        params_dtype=dtype_policy.main_dtype,
        compute_dtype=dtype_policy.accumulation_dtype,
    ).to("npu:0")
    model = nn.Sequential(layer)
    x = torch.randn(1, layer.hyper_hidden, dtype=dtype_policy.main_dtype, device="npu:0")
    before = F.linear(x, layer.input_mix_weight_down)

    _format_eager_linear_weights_npu(model)

    assert torch_npu.get_npu_format(layer.input_mix_weight_down) == ACL_FORMAT_FRACTAL_NZ
    assert torch_npu.get_npu_format(layer.input_mix_weight_up) == ACL_FORMAT_FRACTAL_NZ
    assert torch_npu.get_npu_format(layer.block_inject_weight) != ACL_FORMAT_FRACTAL_NZ
    assert torch.equal(before, F.linear(x, layer.input_mix_weight_down))


@pytest.mark.parametrize("tokens", [1, 256, 2048])
def test_grouped_rms_norm_native_matches_fp32_eager(tokens: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")

    torch.manual_seed(310)
    streams, hidden = 4, 2560
    x = torch.randn(tokens, streams * hidden, dtype=torch.float16, device="npu:0")
    weight = torch.randn(streams * hidden, dtype=torch.float16, device="npu:0") * 0.1
    unit_weight = torch.ones(hidden, dtype=torch.float32, device="npu:0")
    expected = _grouped_rms_norm(x, weight, 1e-6, hidden, torch.float32)
    actual = _grouped_rms_norm(x, weight, 1e-6, hidden, torch.float32, unit_weight)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
