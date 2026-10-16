"""Qwen image patch ordering without nine-dimensional intermediate copies."""

import copy
from types import MethodType


def patchify(images, patch_size: int, merge_size: int, temporal_patch_size: int):
    if images.ndim != 4 or min(patch_size, merge_size, temporal_patch_size) <= 0:
        raise ValueError("Use BCHW images and positive patch/merge/temporal sizes")
    batch, channels, height, width = images.shape
    factor = patch_size * merge_size
    if height % factor or width % factor:
        raise ValueError("Image dimensions must divide evenly into merged patches")
    grid_h, grid_w = height // patch_size, width // patch_size
    # Preserve the processor's spatial/channel/temporal feature order. Expand
    # time after copying the eight-dimensional spatial permutation, so NPU
    # InplaceCopy never receives the reference's unsupported nine-dimensional view.
    spatial = (
        images.reshape(
            batch,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        .permute(0, 2, 5, 3, 6, 1, 4, 7)
        .contiguous()
    )
    spatial = spatial.reshape(batch, grid_h * grid_w, channels, 1, patch_size * patch_size)
    patches = spatial.expand(-1, -1, -1, temporal_patch_size, -1).reshape(
        batch, grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size
    )
    return patches, grid_h, grid_w


def with_device_patchify(processor):
    if not callable(getattr(processor, "patchify", None)):
        raise ValueError("The image processor must expose Qwen patchify")
    candidate = copy.copy(processor)

    def process(self, images, patch_size, merge_size, temporal_patch_size):
        return patchify(images, patch_size, merge_size, temporal_patch_size)

    candidate.patchify = MethodType(process, candidate)
    return candidate
