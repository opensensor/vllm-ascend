import pytest
import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

bootstrap_custom_op_env(include_vendor_lib=True)
enable_custom_op()

torch_npu.npu.set_compile_mode(jit_compile=False)

EPS = 1e-6
# Dequantised output can only be as good as one int8 step, and the kernel rounds
# to nearest, so half a step is the floor any correct implementation sits at.
HALF_LSB = 1.0 / 254.0

# 512 and 1536 fund an 8-row loop step and pick the Normal kernel; from 2048 up
# the UB budget forces SingleRow. Both paths need covering -- SingleRow read
# gamma out of bounds until it was fixed, and Normal never ran at all until the
# DataCopyPad calls were replaced (310P silently discards them).
NORMAL_SHAPES = [(1, 512), (13, 512), (255, 512), (1024, 512), (1, 1536)]
SINGLE_ROW_SHAPES = [(1, 2048), (33, 2048), (7, 4096), (1024, 4096), (1, 8192)]
ALL_SHAPES = NORMAL_SHAPES + SINGLE_ROW_SHAPES


def _rms_norm(value, gamma):
    return value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + EPS) * gamma.float()


def _assert_matches(y, scale, expected):
    dequant = y.cpu().float() * scale.cpu().unsqueeze(-1)
    rel = ((dequant - expected).abs().max() / expected.abs().max()).item()
    assert y.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert rel < HALF_LSB * 1.05, f"relative error {rel:.3e} exceeds one half int8 step"


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("num_tokens, hidden", ALL_SHAPES)
def test_npu_rms_norm_dynamic_quant_310(num_tokens, hidden):
    torch.random.manual_seed(0)
    x = (torch.randn(num_tokens, hidden, dtype=torch.float16) * 0.5).npu()
    # A non-unit gamma is what catches the SingleRow out-of-bounds read; with
    # gamma == 1 the kernel looks right even when it never loads gamma at all.
    gamma = torch.randn(hidden, dtype=torch.float16).npu()
    x_before = x.cpu().clone()

    y, scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, gamma, None, None, EPS)

    _assert_matches(y, scale, _rms_norm(x.cpu().float(), gamma.cpu()))
    assert torch.equal(x.cpu(), x_before), "x must not be written through"


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("num_tokens, hidden", ALL_SHAPES)
def test_npu_add_rms_norm_dynamic_quant_310(num_tokens, hidden):
    torch.random.manual_seed(0)
    x = (torch.randn(num_tokens, hidden, dtype=torch.float16) * 0.5).npu()
    residual = (torch.randn(num_tokens, hidden, dtype=torch.float16) * 0.5).npu()
    gamma = torch.randn(hidden, dtype=torch.float16).npu()
    x_before = x.cpu().clone()

    y, scale, residual_out = torch.ops._C_ascend.npu_add_rms_norm_dynamic_quant(x, residual, gamma, None, None, EPS)

    total = x.cpu().float() + residual.cpu().float()
    _assert_matches(y, scale, _rms_norm(total, gamma.cpu()))
    assert torch.equal(residual_out.cpu().float(), total.to(torch.float16).float())
    # The aclnn signature takes x by reference and writes the sum back into it,
    # so the binding has to hand it a copy.
    assert torch.equal(x.cpu(), x_before), "x must not be written through"


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_rms_norm_dynamic_quant_310_rejects_bfloat16():
    x = torch.randn(4, 512, dtype=torch.bfloat16).npu()
    gamma = torch.randn(512, dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError, match="FLOAT16"):
        torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, gamma, None, None, EPS)


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_rms_norm_dynamic_quant_310_rejects_unaligned_hidden():
    # 310P copies GM in whole 32B blocks, so the tiling refuses a hidden size it
    # cannot cover exactly rather than returning a partly written tensor.
    x = torch.randn(4, 500, dtype=torch.float16).npu()
    gamma = torch.randn(500, dtype=torch.float16).npu()
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, gamma, None, None, EPS)
