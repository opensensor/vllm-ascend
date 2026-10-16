import json
from types import SimpleNamespace

import pytest
import torch

from tools.rtsp_scene.dvpp_tensor_probe import select_camera
from tools.rtsp_scene.tensor_preprocess import patchify, with_device_patchify


@pytest.mark.parametrize("batch,patch,merge,temporal", [(1, 2, 2, 2), (2, 2, 1, 3), (2, 4, 2, 1)])
def test_patch_order_matches_the_reference_nine_dimension_expansion(batch, patch, merge, temporal):
    height, width = patch * merge * 2, patch * merge * 3
    x = torch.arange(batch * 3 * height * width).reshape(batch, 3, height, width)
    gh, gw = height // patch, width // patch
    original = x.reshape(batch, 3, gh // merge, merge, patch, gw // merge, merge, patch).permute(0, 2, 5, 3, 6, 1, 4, 7)
    expected = (
        original.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
        .reshape(batch, gh * gw, 3 * temporal * patch * patch)
    )
    actual, grid_h, grid_w = patchify(x, patch, merge, temporal)
    assert torch.equal(actual, expected)
    assert (grid_h, grid_w) == (gh, gw)


def test_patchify_rejects_unaligned_geometry():
    for x, patch, merge, temporal in [
        (torch.zeros(3, 8, 8), 2, 2, 2),
        (torch.zeros(1, 3, 7, 8), 2, 2, 2),
        (torch.zeros(1, 3, 8, 8), 0, 2, 2),
    ]:
        with pytest.raises(ValueError):
            patchify(x, patch, merge, temporal)


def test_processor_override_is_instance_local():
    original = SimpleNamespace(patchify=lambda: "original")
    candidate = with_device_patchify(original)
    assert original.patchify() == "original"
    actual, _, _ = candidate.patchify(torch.zeros(1, 3, 8, 8), 2, 2, 2)
    assert actual.shape == (1, 16, 24)
    with pytest.raises(ValueError):
        with_device_patchify(SimpleNamespace())


def test_tensor_camera_selection_honors_privacy_and_permissions(tmp_path):
    config = tmp_path / "cameras.json"
    config.write_text(
        json.dumps(
            [{"name": "available", "url": "private"}, {"name": "private", "url": "private", "privacy_mode": True}]
        )
    )
    config.chmod(0o600)
    assert select_camera(config, "available")["name"] == "available"
    with pytest.raises(ValueError, match="enabled"):
        select_camera(config, "private")
    config.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        select_camera(config, "available")
