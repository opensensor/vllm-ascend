"""Byte RGB transport for an explicitly registered scene processor variant."""

import copy
from types import MethodType

import torch

from tools.rtsp_scene.image_prepare import IMAGE_GEOMETRIES

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MIN_PIXELS = 65536
MAX_PIXELS = 262144


def geometry_for(height, width):
    return next((g for g in IMAGE_GEOMETRIES if g[:2] == (width, height)), None)


def pack_images(images):
    """Pack bytes in image order; metadata remains on CPU across worker IPC."""
    if any(
        image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8 or not image.is_cpu for image in images
    ):
        raise ValueError("Use CPU RGB byte tensors in CHW order")
    geometries = [geometry_for(image.shape[-2], image.shape[-1]) for image in images]
    if not images or any(g is None for g in geometries):
        raise ValueError("Use the validated scene image geometries")
    return {
        "scene_rgb": torch.cat([image.permute(1, 2, 0).contiguous().flatten() for image in images]),
        "scene_geometry": torch.tensor([[g[1], g[0]] for g in geometries], dtype=torch.int64),
        "image_grid_thw": torch.tensor(
            [[1, g[3] // PATCH_SIZE, g[2] // PATCH_SIZE] for g in geometries], dtype=torch.int64
        ),
    }


def raw_preprocess(
    self,
    images,
    do_resize,
    size,
    resample,
    do_rescale,
    rescale_factor,
    do_normalize,
    image_mean,
    image_std,
    patch_size,
    temporal_patch_size,
    merge_size,
    disable_grouping,
    return_tensors,
    **kwargs,
):
    # Requests with different settings, dimensions or tensor types retain the
    # stock processor. Never reinterpret an ordinary float pixel tensor as RGB.
    supported = (
        images
        and do_resize == self.do_resize
        and do_rescale == self.do_rescale
        and do_normalize == self.do_normalize
        and rescale_factor == self.rescale_factor
        and image_mean == self.image_mean
        and image_std == self.image_std
        and resample == self.resample
        and size.shortest_edge == self.size["shortest_edge"]
        and size.longest_edge == self.size["longest_edge"]
        and (size.shortest_edge, size.longest_edge) == (MIN_PIXELS, MAX_PIXELS)
        and (patch_size, temporal_patch_size, merge_size) == (PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)
        and all(
            image.ndim == 3
            and image.shape[0] == 3
            and image.is_cpu
            and image.dtype == torch.uint8
            and geometry_for(image.shape[-2], image.shape[-1]) is not None
            for image in images
        )
    )
    if supported:
        # Import lazily so host layout tests do not require transformers/vLLM.
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(pack_images(images), tensor_type=return_tensors)
    return self.scene_original_preprocess(
        images,
        do_resize,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        disable_grouping,
        return_tensors,
        **kwargs,
    )


def with_raw_images(processor):
    """Copy the cached HF processor; no global monkeypatch or cache mutation."""
    result = copy.copy(processor)
    result.image_processor = copy.copy(processor.image_processor)
    result.image_processor.scene_original_preprocess = processor.image_processor._preprocess
    result.image_processor._preprocess = MethodType(raw_preprocess, result.image_processor)
    return result


def validate_metadata(geometry, grid):
    if geometry.device.type != "cpu" or grid.device.type != "cpu":
        raise ValueError("Scene geometry and image grids must remain on CPU")
    if geometry.dtype != torch.int64 or grid.dtype != torch.int64:
        raise ValueError("Scene dimensions must be integer metadata")
    if geometry.ndim != 2 or geometry.shape[1] != 2 or grid.shape != (geometry.shape[0], 3):
        raise ValueError("Malformed scene geometry or image grid")
    geometries = []
    for (height, width), actual_grid in zip(geometry.tolist(), grid.tolist(), strict=True):
        shape = geometry_for(height, width)
        if shape is None or actual_grid != [1, shape[3] // PATCH_SIZE, shape[2] // PATCH_SIZE]:
            raise ValueError("Scene dimensions and image token grid disagree")
        geometries.append(shape)
    if not geometries:
        raise ValueError("Scene frames cannot be empty")
    return geometries
