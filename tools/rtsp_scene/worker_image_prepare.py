"""Per-worker, bounded, device-only resize/normalization/patch preparation."""

import torch

from tools.rtsp_scene.image_prepare import IMAGE_GEOMETRIES, MAX_FRAMES, DeviceImagePrepare
from tools.rtsp_scene.raw_image_transport import MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE, validate_metadata
from tools.rtsp_scene.tensor_preprocess import patchify


class WorkerImagePreparer:
    def __init__(self, directory, backend, processor, device, verify=None):
        self.verify = verify
        self.preparers = {
            shape: DeviceImagePrepare(directory, backend, processor, shape[1], shape[3], device, input_width=shape[0])
            for shape in IMAGE_GEOMETRIES
        }

    def __call__(self, rgb, geometry, grid):
        shapes = validate_metadata(geometry, grid)
        if rgb.device.type != "npu" or rgb.dtype != torch.uint8 or rgb.ndim != 1 or not rgb.is_contiguous():
            raise ValueError("Use contiguous NPU RGB bytes")
        sizes = [width * height * 3 for width, height, _, _ in shapes]
        if sum(sizes) != rgb.numel():
            raise ValueError("Scene byte length does not match the metadata")
        frames = rgb.split(sizes)
        output = [None] * len(frames)
        for shape, prepare in self.preparers.items():
            indices = [index for index, candidate in enumerate(shapes) if candidate == shape]
            for start in range(0, len(indices), MAX_FRAMES):
                batch = indices[start : start + MAX_FRAMES]
                pixels = torch.stack([frames[i].reshape(shape[1], shape[0], 3) for i in batch])
                patches, _, _ = patchify(prepare(pixels), PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE)
                for index, values in zip(batch, patches, strict=True):
                    output[index] = values
        result = torch.cat(output)
        if self.verify is not None:
            # Load-time kernel gates do not exercise request serialization. Check
            # the first real byte batch once; no CPU pixel work after this gate.
            self.verify(rgb, shapes, result)
            self.verify = None
        return result
