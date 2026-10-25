# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.minimax_m3.build_resident_native import build
from tools.minimax_m3.mixed_checkpoint import dequant_nvfp4
from tools.minimax_m3.nvfp4 import pack_source, storage_bytes, unpack_source, weight_lookup
from tools.minimax_m3.nvfp4_native import geometry


@pytest.mark.parametrize("shape", [(16, 256), (32, 1024), (128, 3072), (128, 6144)])
def test_source_bytes_decode_exactly(shape):
    gen = torch.Generator().manual_seed(419)
    rows, cols = shape
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=gen)
    scales = torch.randint(0, 127, (rows, cols // 16), dtype=torch.uint8, generator=gen).view(torch.float8_e4m3fn)
    global_scale = torch.tensor(10931.173)
    original = dequant_nvfp4(codes, scales, global_scale).half()
    bank = pack_source(codes, scales, global_scale)
    assert torch.equal(unpack_source(*bank), original)
    assert sum(t.numel() * t.element_size() for t in bank) == storage_bytes(rows, cols)


def test_every_valid_scale_and_nibble():
    codes = torch.arange(256, dtype=torch.uint8).repeat(128, 1)
    scales = torch.arange(128, dtype=torch.uint8)[:, None].expand(128, 32).clone()
    scales[-1] = 128  # Negative zero is valid; 127 is NaN.
    scales = scales.view(torch.float8_e4m3fn)
    global_scale = torch.tensor(719.123)
    assert torch.equal(
        unpack_source(*pack_source(codes, scales, global_scale)), dequant_nvfp4(codes, scales, global_scale).half()
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_global_scale(value):
    with pytest.raises(ValueError, match="global scale"):
        weight_lookup(torch.tensor(value))


@pytest.mark.parametrize("byte", [127, 129, 255])
def test_invalid_block_scale(byte):
    codes = torch.zeros(16, 128, dtype=torch.uint8)
    scales = torch.full((16, 16), byte, dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="block scales"):
        pack_source(codes, scales, torch.tensor(1.0))


def test_fp16_overflow_rejected():
    with pytest.raises(ValueError, match="overflow"):
        pack_source(
            torch.zeros(16, 128, dtype=torch.uint8),
            torch.full((16, 16), 448.0).to(torch.float8_e4m3fn),
            torch.tensor(1e-10),
        )


def test_native_geometry_rejects_old_w4_scales():
    x = torch.zeros(4, 1024).half()
    codes = torch.zeros(2, 128, 512, dtype=torch.int8)
    scales = torch.zeros(2, 128, 64, dtype=torch.uint8)
    lookup = torch.zeros(2, 4096).half()
    routes = torch.tensor([1, 0, 2, 1], dtype=torch.int32)
    assert geometry(x, codes, scales, routes, lookup)[-1] is True
    with pytest.raises(ValueError, match="raw K16"):
        geometry(x, codes, torch.zeros(2, 4, 32).half(), routes, lookup)
    with pytest.raises(ValueError, match="raw K16"):
        geometry(x, codes, scales, routes, lookup[:, :16])


@pytest.mark.parametrize(
    "options", [{}, {"wide_l1": True, "scale_cache": True}, {"wide_l1": True, "int8_compute": True}]
)
def test_source_build_rejects_w4_specific_modes(tmp_path, options):
    with pytest.raises(ValueError, match="source NVFP4"):
        build(tmp_path / "build", tmp_path, "minimax_w4_source_test", tmp_path, source_nvfp4=True, **options)
    assert not (tmp_path / "build").exists()
