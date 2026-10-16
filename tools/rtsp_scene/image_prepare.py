"""Frozen 320-wide device image preparation matching the pinned CPU processor."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as functional

from tools.rtsp_scene.dense_shapes import kernel_namespace

IMAGE_WIDTH = 320
OUTPUT_WIDTHS = (320, 352)
COEFFICIENT_BITS = 14
RESIZE_HEIGHTS = ((180, 192), (240, 256))
RESIZE_HEIGHTS_640 = ((360, 352), (480, 416))
RESIZE_AXES = (
    (180, 192, 14, 4),
    (240, 256, 14, 4),
    (320, 352, 14, 4),
    (360, 352, 15, 6),
    (480, 416, 15, 6),
    (640, 576, 15, 6),
)
IMAGE_GEOMETRIES = ((320, 180, 352, 192), (320, 240, 320, 256), (640, 360, 640, 352), (640, 480, 576, 416))
TAPS = 4
MAX_FRAMES = 8
FLOAT_INTEGER_LIMIT = 1 << 24


def resize_coefficients(height: int, target: int):
    match = next((axis for axis in RESIZE_AXES if axis[:2] == (height, target)), None)
    if match is None:
        raise ValueError("Use a validated resize geometry")
    bits, taps = match[2:]
    basis = torch.eye(height, dtype=torch.float64).reshape(1, 1, height, height)
    weights = functional.interpolate(basis, size=(target, height), mode="bicubic", align_corners=False, antialias=True)[
        0, 0
    ]
    quantized = torch.round(weights * (1 << bits))
    if quantized.abs().sum(1).max() * 255 >= FLOAT_INTEGER_LIMIT:
        raise ValueError("Coefficients exceed exact FP32 integer arithmetic")
    coefficients = torch.zeros(target, taps, dtype=torch.float32)
    indices = torch.zeros(target, taps, dtype=torch.int64)
    for row in range(target):
        columns = torch.nonzero(quantized[row], as_tuple=True)[0]
        if columns.numel() > taps:
            raise ValueError("Resize footprint exceeds the frozen tap contract")
        coefficients[row, : columns.numel()] = quantized[row, columns].float()
        indices[row, : columns.numel()] = columns
    return coefficients, indices


def normalization_lookup(processor):
    levels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(1, 3, -1, -1).contiguous()
    normalized = processor.rescale_and_normalize(
        levels,
        processor.do_rescale,
        processor.rescale_factor,
        processor.do_normalize,
        processor.image_mean,
        processor.image_std,
    )
    return normalized.half().float().reshape(3, 256).contiguous()


def checked_build(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build.json").read_text())
    if (
        manifest.get("width"),
        manifest.get("coefficient_bits"),
        manifest.get("arguments"),
        manifest.get("output_widths"),
        manifest.get("horizontal_arguments"),
    ) != (
        IMAGE_WIDTH,
        COEFFICIENT_BITS,
        7,
        list(OUTPUT_WIDTHS),
        5,
    ):
        raise ValueError("Unsupported device image preparation contract")
    version = manifest.get("version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported image preparation version")
    names = [*[f"image-w{w}.bin" for w in OUTPUT_WIDTHS], "image-horizontal.bin"]
    if version == 2:
        if manifest.get("geometries") != [list(g) for g in IMAGE_GEOMETRIES]:
            raise ValueError("Unsupported image geometry contract")
        names.extend(["image-w576.bin", "image-w640.bin", "image-horizontal640.bin"])
    for name in names:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"].get(name):
            raise ValueError("Image preparation binary fingerprint mismatch")
    return manifest


class DeviceImagePrepare:
    def __init__(self, directory, backend, processor, height, target, device, input_width=IMAGE_WIDTH):
        self.manifest = checked_build(directory)
        geometry = next((g for g in IMAGE_GEOMETRIES if (g[0], g[1], g[3]) == (input_width, height, target)), None)
        if geometry is None or processor.patch_size != 16 or processor.merge_size != 2:
            raise ValueError("Unsupported image processor geometry")
        if input_width != IMAGE_WIDTH and self.manifest.get("version", 1) != 2:
            raise ValueError("640-pixel images require a version-two build")
        self.input_width, self.width = input_width, geometry[2]
        if processor.get_number_of_image_patches(height, input_width) != target * self.width // processor.patch_size**2:
            raise ValueError("Processor settings do not match the validated resize grid")
        weights, rows = resize_coefficients(height, target)
        self.weights, self.rows = weights.to(device), rows.to(device)
        self.lookup = normalization_lookup(processor).to(device)
        self.indices = torch.tensor(
            [[4 * (3 * x + c) for x in range(self.width)] for c in range(3)], dtype=torch.int32, device=device
        )
        self.configs = {
            n: torch.tensor([n, height, target], dtype=torch.int64, device=device) for n in range(1, MAX_FRAMES + 1)
        }
        self.height, self.target, self.backend = height, target, backend
        kernel_class = getattr(torch.classes, kernel_namespace(backend.manifest)).Kernel
        self.kernel = kernel_class(str(Path(directory) / f"image-w{self.width}.bin"), "rtsp_image_prepare")
        self.horizontal = None
        if self.width != self.input_width:
            weights, indices = resize_coefficients(self.input_width, self.width)
            self.horizontal_weights = weights.t().repeat_interleave(3, dim=1).contiguous().to(device)
            offsets = 4 * (3 * indices.t()[:, :, None] + torch.arange(3))
            self.horizontal_offsets = offsets.reshape(indices.shape[1], -1).int().contiguous().to(device)
            self.horizontal_configs = {
                n: torch.tensor([n * height], dtype=torch.int64, device=device) for n in range(1, MAX_FRAMES + 1)
            }
            filename = "image-horizontal.bin" if input_width == IMAGE_WIDTH else "image-horizontal640.bin"
            self.horizontal = kernel_class(str(Path(directory) / filename), "rtsp_image_horizontal")

    def __call__(self, images):
        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != (self.height, self.input_width, 3)
            or images.dtype != torch.uint8
            or images.device.type != "npu"
            or not images.is_contiguous()
            or not 0 < images.shape[0] <= MAX_FRAMES
            or self.backend.get_format(images) not in (0, 2)
        ):
            raise ValueError("Use contiguous base-format NPU byte frames with the frozen NHWC geometry")
        frames = images.shape[0]
        if self.horizontal is not None:
            resized = torch.ops.npu.empty_with_format(
                [frames, self.height, self.width, 3], dtype=torch.uint8, device=images.device, acl_format=2
            )
            self.backend.launch(
                self.horizontal,
                [images, self.horizontal_weights, self.horizontal_offsets, resized, self.horizontal_configs[frames]],
            )
            images = resized
        output = torch.ops.npu.empty_with_format(
            [frames, 3, self.target, self.width], dtype=torch.float16, device=images.device, acl_format=2
        )
        self.backend.launch(
            self.kernel,
            [images, self.weights, self.rows, self.indices, self.lookup, output, self.configs[images.shape[0]]],
        )
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    cann = args.cann.resolve()
    target = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    hashes = {}
    jobs = [("image_prepare.cpp", f"image-w{width}.bin", [f"-DRTSP_IMAGE_WIDTH={width}"]) for width in OUTPUT_WIDTHS]
    jobs.append(("image_horizontal.cpp", "image-horizontal.bin", []))
    precision = ["-DRTSP_IMAGE_TAPS=6", "-DRTSP_IMAGE_COEFFICIENT_BITS=15"]
    jobs.extend(
        ("image_prepare.cpp", f"image-w{width}.bin", [f"-DRTSP_IMAGE_WIDTH={width}", *precision])
        for width in (576, 640)
    )
    jobs.append(
        (
            "image_horizontal.cpp",
            "image-horizontal640.bin",
            ["-DRTSP_IMAGE_SOURCE_WIDTH=640", "-DRTSP_IMAGE_TARGET_WIDTH=576", *precision],
        )
    )
    for filename, name, flags in jobs:
        binary = args.output / name
        subprocess.run(
            [
                str(args.compiler),
                str(source / filename),
                str(binary),
                "--npu-arch=dav-2002",
                f"--sysroot={cann / 'tools/hcc/sysroot'}",
                f"-isystem{target}",
                f"-isystem{target / 'aarch64-target-linux-gnu'}",
                *flags,
            ],
            check=True,
        )
        hashes[name] = hashlib.sha256(binary.read_bytes()).hexdigest()
        hashes[filename] = hashlib.sha256((source / filename).read_bytes()).hexdigest()
    manifest = {
        "width": IMAGE_WIDTH,
        "coefficient_bits": COEFFICIENT_BITS,
        "arguments": 7,
        "horizontal_arguments": 5,
        "output_widths": list(OUTPUT_WIDTHS),
        "version": 2,
        "geometries": [list(g) for g in IMAGE_GEOMETRIES],
        "sha256": hashes,
    }
    (args.output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
