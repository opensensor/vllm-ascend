"""Create a private model overlay for an explicit, reversible serving experiment."""

import argparse
import json
import os
from pathlib import Path

from tools.rtsp_scene.dense_bench import checked_build
from tools.rtsp_scene.dense_shapes import decode_rows, kernel_namespace
from tools.rtsp_scene.image_prepare import checked_build as checked_image_build
from tools.rtsp_scene.input_quant import MAX_ROWS as INPUT_QUANT_MAX_ROWS
from tools.rtsp_scene.input_quant import checked_input_build

RUNTIME_CONFIG = "rtsp_dense_runtime.json"
DENSE_MODES = ("ffn", "all")
GATE_REPORT_PREFIX = "dense-gate-rank"
VISION_GATE_REPORT_PREFIX = "vision-gate-rank"


def checked_qkv_build(build: Path, control_manifest: dict):
    manifest = checked_build(build)
    if decode_rows(manifest) != decode_rows(control_manifest):
        raise ValueError("QKV and primary kernels must share the decode row limit")
    if kernel_namespace(manifest) == kernel_namespace(control_manifest):
        raise ValueError("QKV and primary builds require distinct native namespaces")
    return {"build": str(build.resolve()), "manifest": manifest}


def prepare_overlay(
    model: Path,
    build: Path,
    output: Path,
    mode: str,
    vision_zero_add_norm: bool = False,
    input_quant_build: Path | None = None,
    device_image_build: Path | None = None,
    qkv_build: Path | None = None,
) -> dict:
    model, build, output = model.resolve(), build.resolve(), output.resolve()
    if mode not in DENSE_MODES or output == model or output.is_relative_to(model):
        raise ValueError("Use a separate overlay directory and a supported dense mode")
    if not (model / "config.json").is_file() or not (model / "model.safetensors.index.json").is_file():
        raise ValueError("Provide the local quantized checkpoint")
    manifest = checked_build(build)
    qkv_config = checked_qkv_build(qkv_build, manifest) if qkv_build is not None else None
    if qkv_config is not None and mode != "all":
        raise ValueError("A QKV-specific build requires dense mode all")
    input_config = None
    image_config = None
    image_model_config = None
    if device_image_build is not None:
        image_config = {"build": str(device_image_build.resolve()), "manifest": checked_image_build(device_image_build)}
        if image_config["manifest"].get("version") != 2:
            raise ValueError("Worker image preparation requires the full-resolution version-two build")
        image_model_config = json.loads((model / "config.json").read_text())
        if image_model_config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
            raise ValueError("Worker image preparation requires the Qwen3-VL architecture")
        image_model_config["architectures"] = ["SceneDeviceQwen3VLForConditionalGeneration"]
    if input_quant_build is not None:
        if decode_rows(manifest) != INPUT_QUANT_MAX_ROWS:
            raise ValueError("NZ input quantization currently requires an eight-row matrix build")
        input_config = {"build": str(input_quant_build.resolve()), "manifest": checked_input_build(input_quant_build)}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for source in model.iterdir():
        if (
            source.is_file()
            and source.name != RUNTIME_CONFIG
            and not source.name.startswith((GATE_REPORT_PREFIX, VISION_GATE_REPORT_PREFIX))
        ):
            (output / source.name).symlink_to(source)
    config = {"build": str(build), "mode": mode, "manifest": manifest}
    if vision_zero_add_norm:
        config["vision_zero_add_norm"] = True
    if input_config is not None:
        config["input_quant"] = input_config
    if qkv_config is not None:
        config["qkv_projection_build"] = qkv_config
    if image_config is not None:
        (output / "config.json").unlink()
        (output / "config.json").write_text(json.dumps(image_model_config, indent=2) + "\n")
        config["device_image_prepare"] = image_config
    (output / RUNTIME_CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=DENSE_MODES, default="ffn")
    parser.add_argument("--vision-zero-add-norm", action="store_true", help="Gate an opt-in 310P vision norm path")
    parser.add_argument("--input-quant-build", type=Path, help="Frozen opt-in NZ input quantization build")
    parser.add_argument("--device-image-build", type=Path, help="Frozen opt-in worker image preparation build")
    parser.add_argument("--qkv-build", type=Path, help="Frozen opt-in QKV-specific tile build")
    args = parser.parse_args()
    os.umask(0o077)
    prepare_overlay(
        args.model,
        args.build,
        args.output,
        args.mode,
        args.vision_zero_add_norm,
        args.input_quant_build,
        args.device_image_build,
        args.qkv_build,
    )


if __name__ == "__main__":
    main()
