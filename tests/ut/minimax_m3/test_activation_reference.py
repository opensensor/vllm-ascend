# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.minimax_m3.activation_reference import (
    mxfp8_roundtrip,
    nvfp4_roundtrip,
    quantized_dense_projection,
    round_e4m3,
    warmup_activation_reference,
)


def test_e4m3_matches_torch_rounding_and_saturation():
    values = torch.cat((torch.linspace(-500, 500, 8193), torch.linspace(-0.02, 0.02, 8193)))
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(round_e4m3(values), expected, rtol=0, atol=0)


def test_nvfp4_ties_zero_and_sign():
    values = torch.tensor(
        [[6.0, -6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0]]
    ).half()
    expected = torch.tensor(
        [[6.0, -6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0]]
    ).half()
    torch.testing.assert_close(nvfp4_roundtrip(values), expected, rtol=0, atol=0)
    for function in (nvfp4_roundtrip, mxfp8_roundtrip):
        torch.testing.assert_close(function(torch.zeros(3, 64).half()), torch.zeros(3, 64).half())


def test_only_layer_projections_are_selected():
    for name in ("qkv_proj", "index_k_proj", "o_proj"):
        assert quantized_dense_projection("language_model.model.layers.3.self_attn." + name)
    assert quantized_dense_projection("language_model.model.layers.3.block_sparse_moe.shared_experts.gate_up_proj")
    assert not quantized_dense_projection("language_model.model.layers.3.block_sparse_moe.gate")
    assert not quantized_dense_projection("language_model.lm_head")


def test_warmup_checks_both_formats_and_casts():
    warmup_activation_reference("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native NVIDIA quantizer comparison requires CUDA")
def test_against_native_nvidia_quantizers():
    flashinfer = pytest.importorskip("flashinfer")
    from flashinfer.tllm_enums import SfLayout

    generator = torch.Generator(device="cuda").manual_seed(73)
    value = torch.randn(32, 256, generator=generator, device="cuda").half()
    codes, scales = flashinfer.nvfp4_quantize(value, torch.ones(1, device="cuda"), sfLayout=SfLayout.layout_linear)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    codes = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    expected = (
        (lut[codes.long()].unflatten(-1, (-1, 16)) * scales.view(torch.float8_e4m3fn).float()[..., None])
        .flatten(-2)
        .half()
    )
    torch.testing.assert_close(nvfp4_roundtrip(value), expected, rtol=0, atol=0)
    codes, scales = flashinfer.mxfp8_quantize(value, is_sf_swizzled_layout=False)
    expected = (
        (codes.float().unflatten(-1, (-1, 32)) * torch.exp2(scales.float().reshape(32, -1) - 127)[..., None])
        .flatten(-2)
        .half()
    )
    torch.testing.assert_close(mxfp8_roundtrip(value), expected, rtol=0, atol=0)
