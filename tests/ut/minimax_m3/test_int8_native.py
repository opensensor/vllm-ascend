# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.minimax_m3.build_resident_native import build
from tools.minimax_m3.int8_native import prepare_scales


def test_w4_requantization_preserves_block_scale_variation_and_bounds_codes():
    scales = torch.tensor([[[0.0, 0.125, 0.5, 2.0], [0.0, 0.0, 0.0, 0.0]]]).half()
    ratios, output_scale = prepare_scales(scales)
    values = torch.arange(-8, 8).float()
    converted = (ratios[..., None].half() * values.half()).round()
    assert converted.min() >= -127 and converted.max() <= 127
    assert torch.isfinite(ratios).all() and (output_scale > 0).all()
    reconstructed = converted.float() * output_scale[..., None, None]
    expected = scales.float()[..., None] * values
    assert (reconstructed - expected).abs().max() <= output_scale.max()
    assert (ratios[0, 0, 1:] > 0).all()
    assert ratios[0, 1].count_nonzero() == 0


def test_int8_b_tile_gather_maps_k32_n16_without_dropping_values():
    # The stored decoder emits [K256,N16]; INT8 B in nZ needs [K8,N16,K32].
    source = torch.arange(4096).reshape(256, 16)
    ids = torch.arange(4096)
    k = (ids // 512) * 32 + ids % 32
    n = (ids % 512) // 32
    actual = source.flatten()[k * 16 + n].reshape(8, 16, 32)
    assert torch.equal(actual, source.reshape(8, 32, 16).transpose(1, 2))


@pytest.mark.parametrize("options", [{}, {"wide_l1": True}, {"wide_l1": True, "rint_unpack": True, "expert_lanes": 1}])
def test_int8_build_rejects_incompatible_options(tmp_path, options):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="INT8 prototype"):
        build(output, tmp_path, "minimax_w4_test", tmp_path, int8_compute=True, **options)
    assert not output.exists()


def test_byte_pair_sign_extension_covers_every_packed_pair():
    words = torch.arange(65536, dtype=torch.int64)
    for field in (0, 4):
        source = words >> field
        actual = (source & 0x0F0F) | (((source & 0x0808) * 30) & 0xFFFF)
        for byte in (0, 8):
            nibble = (words >> (field + byte)) & 15
            expected = torch.where(nibble >= 8, nibble - 16, nibble) & 255
            assert torch.equal((actual >> byte) & 255, expected)


def test_int8_layout_reorder_keeps_every_code_and_its_scale_block():
    from tools.minimax_m3.int8_native import pack_int8_layout
    from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256

    canonical = torch.randint(0, 256, (32, 512), dtype=torch.uint8, generator=torch.Generator().manual_seed(7))
    stored = pack_int8_layout(pack_w4_nz256(canonical)[None])
    low = stored.view(torch.uint8) & 15
    high = stored.view(torch.uint8) >> 4
    decoded = torch.stack((low.reshape(2, 4, -1), high.reshape(2, 4, -1)), 2)
    decoded = decoded.reshape(2, 4, 8, 16, 32).permute(0, 3, 1, 2, 4).reshape(32, 1024)
    expected = torch.stack((canonical & 15, canonical >> 4), -1).flatten(-2)
    assert torch.equal(decoded, expected)
    assert stored.numel() == canonical.numel()
