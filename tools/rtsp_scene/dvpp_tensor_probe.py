"""Gate same-process DVPP-to-Torch tensors and pinned image preprocessing."""

import argparse
import json
import time
from pathlib import Path


def select_camera(config: Path, name: str) -> dict:
    if config.stat().st_mode & 0o077:
        raise ValueError("Camera configuration must be private")
    matches = [
        c
        for c in json.loads(config.read_text())
        if c["name"] == name and c.get("enabled", True) and not c.get("privacy_mode", False)
    ]
    if len(matches) != 1:
        raise ValueError("Select one enabled camera outside privacy mode")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--camera", default="back-door")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, choices=(320, 640), default=320)
    parser.add_argument("--image-build", type=Path, help="Frozen exact device image preparation candidate")
    parser.add_argument("--matrix-build", type=Path, help="Frozen generic RTC bridge")
    args = parser.parse_args()
    args.build.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Compiler/probe process only; do not import these in model/API workers.
    import torch
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent
    cann = args.cann.resolve()
    module = load(
        name="rtsp_dvpp_tensor",
        sources=[str(source / "dvpp_tensor.cpp")],
        build_directory=str(args.build),
        extra_cflags=["-O2", f"-I{cann / 'include'}"],
        extra_ldflags=[
            f"-L{cann / 'lib64'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-lascendcl",
            "-lacl_dvpp_mpi",
            "-lavformat",
            "-lavcodec",
            "-lavutil",
        ],
    )
    if args.compile:
        return
    if not all((args.config, args.model, args.output)) or args.output.exists() or args.frames <= 0:
        parser.error("Supply camera config, model and fresh private output")
    import torch_npu  # noqa: F401
    from transformers import AutoImageProcessor

    from tools.rtsp_scene.dense_bench import measure_pair
    from tools.rtsp_scene.tensor_preprocess import with_device_patchify

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    camera = select_camera(args.config, args.camera)
    images = []

    def consume(borrowed, sample):
        if borrowed.device.type != "npu" or borrowed.dtype != torch.uint8:
            raise ValueError("Expected an NPU RGB byte tensor")
        # Retain owned storage; the extension synchronizes before frame release.
        images.append(borrowed.clone().permute(2, 0, 1).contiguous())

    start = time.monotonic()
    module.capture(camera["url"], args.device, args.max_side, args.frames, 1.0, consume)
    capture_seconds = time.monotonic() - start
    report = {
        "camera": args.camera,
        "device": args.device,
        "frames": len(images),
        "capture_seconds": capture_seconds,
        "npu_owned_tensors": all(x.device.type == "npu" for x in images),
        "shapes": [list(x.shape) for x in images],
        "image_files_written": 0,
        "decode_transport": "borrowed DVPP device view to owned Torch NPU tensor by device copy",
        "limitations": "Standalone prototype; not connected to serving workers. Host demux remains.",
    }
    processor = AutoImageProcessor.from_pretrained(args.model, max_pixels=262144)
    host_images = [x.cpu() for x in images]  # Validation only, after capture.
    expected = processor(images=host_images, return_tensors="pt")
    device_processor = with_device_patchify(processor)
    try:
        actual = device_processor(images=images, return_tensors="pt", device=f"npu:{args.device}")
        torch.npu.synchronize()
        actual_pixels = actual["pixel_values"]
        report["full_preprocess"] = {
            "supported": True,
            "output_device": str(actual_pixels.device),
            "bitwise_equal": torch.equal(expected["pixel_values"], actual_pixels.cpu()),
            "model_fp16_equal": torch.equal(expected["pixel_values"].half(), actual_pixels.cpu().half()),
            "max_abs_error": float((expected["pixel_values"] - actual_pixels.cpu()).abs().max()),
            "grid_equal": torch.equal(expected["image_grid_thw"], actual["image_grid_thw"].cpu()),
        }
        left, right = measure_pair(
            lambda: processor(images=host_images, return_tensors="pt"),
            lambda: device_processor(images=images, return_tensors="pt", device=f"npu:{args.device}"),
            torch.npu.synchronize,
            10,
            5,
        )
        report["full_preprocess"].update({"cpu": left, "npu": right})
    except Exception as error:
        report["full_preprocess"] = {"supported": False, "error": str(error)[:2200]}
    if args.image_build:
        if args.matrix_build is None:
            parser.error("Device image preparation requires the matrix bridge")
        from tools.rtsp_scene.dense_runtime import KernelBackend
        from tools.rtsp_scene.image_prepare import DeviceImagePrepare
        from tools.rtsp_scene.tensor_preprocess import patchify

        target = int(expected["image_grid_thw"][0, 1]) * processor.patch_size
        backend = KernelBackend(args.matrix_build)
        prepare = DeviceImagePrepare(
            args.image_build,
            backend,
            processor,
            images[0].shape[1],
            target,
            f"npu:{args.device}",
            input_width=images[0].shape[2],
        )
        rgb = torch.stack([image.permute(1, 2, 0) for image in images]).contiguous()

        def resident():
            pixels, _, _ = patchify(
                prepare(rgb), processor.patch_size, processor.merge_size, processor.temporal_patch_size
            )
            return pixels.flatten(0, 1)

        actual = resident().cpu()
        equal = torch.equal(actual, expected["pixel_values"].half())
        report["exact_device_prepare"] = {
            "passed": equal,
            "model_fp16_equal": equal,
            "max_abs_error": float((actual - expected["pixel_values"].half()).abs().max()),
            "output_shape": list(actual.shape),
        }
        if equal:
            left, right = measure_pair(
                lambda: processor(images=host_images, return_tensors="pt")["pixel_values"].half(),
                resident,
                torch.npu.synchronize,
                20,
                8,
            )
            report["exact_device_prepare"].update(
                {"cpu": left, "npu": right, "reduction_percent": 100 * (1 - right["median_us"] / left["median_us"])}
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
