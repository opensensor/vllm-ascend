"""The fused norm+quant seam must match the three-op path it replaces."""

import os
from types import SimpleNamespace

# The fusion is opt-in; turn it on before anything reads the flag.
os.environ["VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT"] = "1"

import pytest
import torch
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

from vllm_ascend._310p.ops.fused_norm_quant import (
    add_rms_norm_quant,
    apply_quantized,
    can_fuse,
    norm_gamma,
    rms_norm_quant,
)
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod310
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op, maybe_trans_nz
from vllm_ascend.utils import is_310p as is_310p_hw

bootstrap_custom_op_env(include_vendor_lib=True)
enable_custom_op()

torch_npu.npu.set_compile_mode(jit_compile=False)

EPS = 1e-6
# Both paths quantize to int8, so they agree to a whole step at worst; they
# differ only in where the row max is computed.
ONE_LSB = 1.0 / 127.0


@pytest.fixture(autouse=True)
def _vllm_config():
    # RMSNorm is a CustomOp and reads the active config when constructed.
    with set_current_vllm_config(VllmConfig()):
        yield


def _make_layer(hidden, out_features):
    """A stand-in for a loaded W8A8_DYNAMIC linear, in post-load layout."""
    # Built the way process_weights_after_loading leaves a real layer: the
    # checkpoint's [out, in] weight cast to NZ and transposed, scale flattened.
    loaded = torch.randint(-127, 127, (out_features, hidden), dtype=torch.int8).npu()
    weight = maybe_trans_nz(loaded).transpose(0, 1)
    weight_scale = (torch.rand(out_features, dtype=torch.float32) * 0.01 + 0.001).npu()
    return SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        skip_bias_add=False,
        bias=None,
        quant_method=AscendW8A8DynamicLinearMethod310(),
    )


def _reference_norm(x, residual, norm):
    """What AscendGemmaRMSNorm310.forward_oot does, kept explicit."""
    gamma = 1.0 + norm.weight if isinstance(norm, GemmaRMSNorm) else norm.weight
    if residual is None:
        out, _ = torch_npu.npu_rms_norm(x, gamma, norm.variance_epsilon)
        return out, x
    total = x + residual
    out, _ = torch_npu.npu_rms_norm(total, gamma, norm.variance_epsilon)
    return out, total


def _relative(fused, baseline):
    return ((fused.float() - baseline.float()).abs().max() / baseline.float().abs().max()).item()


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("norm_cls", [GemmaRMSNorm, RMSNorm])
@pytest.mark.parametrize("num_tokens, hidden", [(1, 5120), (7, 5120), (256, 5120), (13, 2048)])
def test_fused_matches_three_op_path(norm_cls, num_tokens, hidden):
    torch.random.manual_seed(0)
    out_features = 512
    norm = norm_cls(hidden, eps=EPS).npu()
    norm.weight.data = (torch.randn(hidden, dtype=torch.float16) * 0.1).npu()
    layer = _make_layer(hidden, out_features)

    x = (torch.randn(num_tokens, hidden, dtype=torch.float16) * 0.5).npu()
    residual = (torch.randn(num_tokens, hidden, dtype=torch.float16) * 0.5).npu()

    assert can_fuse(layer, x)

    for resid in (None, residual):
        normed, expected_residual = _reference_norm(x, resid, norm)
        baseline = layer.quant_method.apply(layer, normed)

        if resid is None:
            quantized, scale = rms_norm_quant(x, norm)
            fused_residual = x
        else:
            quantized, scale, fused_residual = add_rms_norm_quant(x, resid, norm)
        fused = apply_quantized(layer, quantized, scale, torch.float16)

        assert _relative(fused, baseline) < ONE_LSB
        assert torch.equal(fused_residual, expected_residual)


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_gemma_gamma_is_one_plus_weight():
    # GemmaRMSNorm scales by 1 + weight; handing the kernel the bare weight is
    # wrong in a way nothing else catches, so pin it.
    gemma = GemmaRMSNorm(128, eps=EPS).npu()
    gemma.weight.data = torch.randn(128, dtype=torch.float16).npu()
    assert torch.equal(norm_gamma(gemma), 1.0 + gemma.weight)

    plain = RMSNorm(128, eps=EPS).npu()
    plain.weight.data = torch.randn(128, dtype=torch.float16).npu()
    assert torch.equal(norm_gamma(plain), plain.weight)


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
def test_can_fuse_declines_what_it_cannot_handle():
    x = torch.randn(4, 512, dtype=torch.float16).npu()
    # A sparse MoE block has no gate_up_proj at all.
    assert not can_fuse(None, x)
    # An unquantized linear has no pre-quantized entry point.
    assert not can_fuse(SimpleNamespace(quant_method=object()), x)
    # The kernel is fp16-only on 310P.
    assert not can_fuse(_make_layer(512, 64), x.to(torch.bfloat16))
