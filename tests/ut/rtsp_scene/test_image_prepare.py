import hashlib
import json

import pytest
import torch
import torch.nn.functional as functional

from tools.rtsp_scene.build_norm_quant import checked_build as checked_norm_build
from tools.rtsp_scene.image_prepare import COEFFICIENT_BITS, checked_build, resize_coefficients


@pytest.mark.parametrize("height,target", [(180, 192), (240, 256)])
def test_fixed_coefficients_match_cpu_byte_resize_at_edges_and_interior(height, target):
    weights, rows = resize_coefficients(height, target)
    generator = torch.Generator().manual_seed(height)
    x = torch.randint(0, 256, (height, 320, 3), dtype=torch.uint8, generator=generator)
    expected = functional.interpolate(
        x.permute(2, 0, 1).unsqueeze(0), size=(target, 320), mode="bicubic", align_corners=False, antialias=True
    )[0].permute(1, 2, 0)
    total = (x[rows].long() * weights.long()[:, :, None, None]).sum(1)
    actual = ((total + (1 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS).clamp(0, 255).byte()
    assert torch.equal(actual, expected)
    assert int(rows.min()) >= 0 and int(rows.max()) < height


def test_unvalidated_resize_geometry_is_rejected():
    with pytest.raises(ValueError, match="validated"):
        resize_coefficients(240, 224)


def test_minimum_pixel_resize_rounds_between_the_two_axes():
    x = torch.randint(0, 256, (180, 320, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(7))
    weights, indices = resize_coefficients(320, 352)
    horizontal = (x[:, indices].long() * weights.long()[None, :, :, None]).sum(2)
    horizontal = ((horizontal + (1 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS).clamp(0, 255).byte()
    weights, indices = resize_coefficients(180, 192)
    vertical = (horizontal[indices].long() * weights.long()[:, :, None, None]).sum(1)
    actual = ((vertical + (1 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS).clamp(0, 255).byte()
    expected = functional.interpolate(
        x.permute(2, 0, 1).unsqueeze(0), size=(192, 352), mode="bicubic", align_corners=False, antialias=True
    )[0].permute(1, 2, 0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("height,target", [(360, 352), (480, 416), (640, 576)])
def test_full_resolution_downsample_uses_fifteen_bit_weights(height, target):
    weights, indices = resize_coefficients(height, target)
    x = torch.randint(0, 256, (height, 64, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(height))
    total = (x[indices].long() * weights.long()[:, :, None, None]).sum(1)
    actual = ((total + (1 << 14)) >> 15).clamp(0, 255).byte()
    expected = functional.interpolate(
        x.permute(2, 0, 1).unsqueeze(0), size=(target, 64), mode="bicubic", align_corners=False, antialias=True
    )[0].permute(1, 2, 0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind", ["image", "norm"])
def test_frozen_image_and_norm_binary_contracts(kind, tmp_path):
    if kind == "image":
        name, check = "image-w320.bin", checked_build
        manifest = {
            "width": 320,
            "coefficient_bits": 14,
            "arguments": 7,
            "output_widths": [320, 352],
            "horizontal_arguments": 5,
        }
        for other in ("image-w352.bin", "image-horizontal.bin"):
            (tmp_path / other).write_bytes(b"frozen")
    else:
        name, check = "norm-quant.bin", checked_norm_build
        manifest = {"rows": 8, "inner": 4096, "arguments": 8}
    binary = tmp_path / name
    binary.write_bytes(b"frozen")
    manifest["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.bin")}
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    assert check(tmp_path) == manifest
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        check(tmp_path)
    manifest["arguments"] += 1
    (tmp_path / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract"):
        check(tmp_path)
