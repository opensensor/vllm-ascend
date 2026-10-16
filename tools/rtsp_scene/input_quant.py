"""Frozen decode input quantization; shares the existing generic RTC bridge."""

import hashlib
import json
from pathlib import Path

import torch

from tools.rtsp_scene.dense_shapes import kernel_namespace

INPUT_WIDTHS = (2048, 4096, 6144)
MAX_ROWS = 8
SCALE_STRIDE = 8
BASE_FORMATS = (0, 2)


def checked_input_build(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "build.json").read_text())
    if manifest.get("max_rows") != MAX_ROWS or manifest.get("arguments") != 4:
        raise ValueError("Input quantization requires the frozen eight-row, four-argument contract")
    for inner in INPUT_WIDTHS:
        name = f"input-k{inner}.bin"
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"].get(name):
            raise ValueError(f"Input quantization build fingerprint mismatch: {name}")
    return manifest


class InputQuantizer:
    def __init__(self, directory, backend, device):
        directory = Path(directory).resolve()
        self.manifest = checked_input_build(directory)
        self.backend = backend
        self.get_format = backend.get_format
        kernel_class = getattr(torch.classes, kernel_namespace(backend.manifest)).Kernel
        self.kernels = {
            inner: kernel_class(str(directory / f"input-k{inner}.bin"), "rtsp_input_quant") for inner in INPUT_WIDTHS
        }
        self.configs = {rows: torch.tensor([rows], dtype=torch.int64, device=device) for rows in range(1, MAX_ROWS + 1)}

    def __call__(self, x):
        rows, inner = x.shape
        if x.dtype != torch.float16 or not x.is_contiguous() or inner not in INPUT_WIDTHS or not 0 < rows <= MAX_ROWS:
            raise ValueError("Input quantization requires a contiguous supported FP16 decode matrix")
        # Logical contiguity alone does not prove ND physical storage on NPU.
        # Retain the reference quantizer for incoming NZ or other device formats.
        if self.get_format(x) not in BASE_FORMATS:
            codes, scales = self.backend.dynamic_quant(self.backend.format_cast(x, 2))
            return self.backend.format_cast(codes, 29), scales.reshape(rows, 1).expand(
                rows, SCALE_STRIDE
            ).contiguous().reshape(-1)
        output = self.backend.empty(rows, inner, torch.int8, x.device, True)
        scales = torch.empty(rows, SCALE_STRIDE, dtype=torch.float32, device=x.device)
        self.backend.launch(self.kernels[inner], [x, output, scales, self.configs[rows]])
        return output, scales.reshape(-1)
