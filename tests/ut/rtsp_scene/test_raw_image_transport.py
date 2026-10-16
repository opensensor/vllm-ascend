"""Verify raw frame order, token grids, fallback and instance isolation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.rtsp_scene.raw_image_transport import pack_images, validate_metadata, with_raw_images


def test_mixed_frame_geometry_and_rgb_order():
    images = []
    for h, w in [(480, 640), (180, 320), (360, 640), (240, 320)]:
        image = torch.empty(3, h, w, dtype=torch.uint8)
        image[0].fill_(7)
        image[1].fill_(113)
        image[2].fill_(239)
        images.append(image)
    packed = pack_images(images)
    assert packed["image_grid_thw"].tolist() == [[1, 26, 36], [1, 12, 22], [1, 22, 40], [1, 16, 20]]
    shapes = validate_metadata(packed["scene_geometry"], packed["image_grid_thw"])
    offset = 0
    for image, (w, h, _, _) in zip(images, shapes, strict=True):
        count = h * w * 3
        restored = packed["scene_rgb"][offset : offset + count].reshape(h, w, 3).permute(2, 0, 1)
        assert torch.equal(image, restored)
        offset += count
    assert offset == packed["scene_rgb"].numel()


@pytest.mark.parametrize(
    "images", [[], [torch.zeros(3, 17, 32, dtype=torch.uint8)], [torch.zeros(3, 360, 640)], [torch.zeros(5)]]
)
def test_reject_unsupported_inputs(images):
    with pytest.raises(ValueError):
        pack_images(images)


@pytest.mark.parametrize("bad_field", ["shape", "dtype", "grid", "unknown", "empty"])
def test_reject_bad_metadata(bad_field):
    geometry = torch.tensor([[360, 640]])
    grid = torch.tensor([[1, 22, 40]])
    if bad_field == "shape":
        geometry = geometry.flatten()
    elif bad_field == "dtype":
        geometry = geometry.float()
    elif bad_field == "grid":
        grid[0, 1] = 23
    elif bad_field == "unknown":
        geometry[0, 0] = 400
    else:
        geometry, grid = geometry[:0], grid[:0]
    with pytest.raises(ValueError):
        validate_metadata(geometry, grid)


def test_processor_copy_and_unsupported_setting_fallback():
    original = Mock(return_value={"pixel_values": "stock"})
    image_processor = SimpleNamespace(
        _preprocess=original,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        rescale_factor=1 / 255,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        resample=3,
        size={"shortest_edge": 65536, "longest_edge": 262144},
    )
    processor = SimpleNamespace(image_processor=image_processor)
    candidate = with_raw_images(processor)
    assert candidate is not processor and candidate.image_processor is not image_processor
    assert processor.image_processor._preprocess is original
    assert not hasattr(processor.image_processor, "scene_original_preprocess")
    args = (
        [torch.zeros(3, 360, 640, dtype=torch.uint8)],
        False,  # An explicit request override must retain the stock processor.
        SimpleNamespace(shortest_edge=65536, longest_edge=262144),
        3,
        True,
        1 / 255,
        True,
        [0.5] * 3,
        [0.5] * 3,
        16,
        2,
        2,
        False,
        "pt",
    )
    assert candidate.image_processor._preprocess(*args, diagnostic=True) == {"pixel_values": "stock"}
    original.assert_called_once_with(*args, diagnostic=True)


def test_same_sized_camera_with_different_pixel_budget_uses_stock_grid():
    original = Mock(return_value={"pixel_values": "stock", "image_grid_thw": "stock-grid"})
    image_processor = SimpleNamespace(
        _preprocess=original,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        rescale_factor=1 / 255,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        resample=3,
        size={"shortest_edge": 65536, "longest_edge": 1048576},
    )
    candidate = with_raw_images(SimpleNamespace(image_processor=image_processor))
    args = (
        [torch.zeros(3, 480, 640, dtype=torch.uint8)],
        True,
        SimpleNamespace(shortest_edge=65536, longest_edge=1048576),
        3,
        True,
        1 / 255,
        True,
        [0.5] * 3,
        [0.5] * 3,
        16,
        2,
        2,
        False,
        "pt",
    )
    result = candidate.image_processor._preprocess(*args)
    assert result["image_grid_thw"] == "stock-grid"
    assert "scene_rgb" not in result
    original.assert_called_once_with(*args)
