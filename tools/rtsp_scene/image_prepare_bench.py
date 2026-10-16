"""Strict synthetic gate and paired timing for device image preparation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--matrix-build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--input-width", type=int, choices=(320, 640), default=320)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh private output")
    # Only this hardware process initializes Torch-NPU and the local processor.
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoImageProcessor

    from tools.rtsp_scene.dense_bench import measure_pair
    from tools.rtsp_scene.dense_runtime import KernelBackend
    from tools.rtsp_scene.image_prepare import RESIZE_HEIGHTS, RESIZE_HEIGHTS_640, DeviceImagePrepare
    from tools.rtsp_scene.tensor_preprocess import patchify

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    processor = AutoImageProcessor.from_pretrained(args.model, max_pixels=262144)
    backend = KernelBackend(args.matrix_build)
    records = []
    with torch.inference_mode():
        for height, target in RESIZE_HEIGHTS if args.input_width == 320 else RESIZE_HEIGHTS_640:
            prepare = DeviceImagePrepare(
                args.build, backend, processor, height, target, f"npu:{args.device}", input_width=args.input_width
            )
            for frames in (1, 8):
                images = torch.randint(0, 256, (frames, height, args.input_width, 3), dtype=torch.uint8)
                device_images = images.to(f"npu:{args.device}")

                def reference(images=images):
                    return processor(images=list(images.permute(0, 3, 1, 2)), return_tensors="pt")[
                        "pixel_values"
                    ].half()

                def candidate(prepare=prepare, images=device_images):
                    pixels, _, _ = patchify(
                        prepare(images), processor.patch_size, processor.merge_size, processor.temporal_patch_size
                    )
                    return pixels.flatten(0, 1)

                checks = []
                for pattern in ("random", "zero", "maximum", "ramp"):
                    if pattern == "zero":
                        images.zero_()
                    if pattern == "maximum":
                        images.fill_(255)
                    if pattern == "ramp":
                        images.copy_(torch.arange(images.numel()).remainder(256).byte().reshape(images.shape))
                    device_images.copy_(images)
                    expected, actual = reference(), candidate().cpu()
                    checks.append(
                        {
                            "pattern": pattern,
                            "bitwise_equal": torch.equal(expected, actual),
                            "max_abs_error": float((expected - actual).abs().max()),
                        }
                    )
                record = {
                    "height": height,
                    "input_width": args.input_width,
                    "target_height": target,
                    "frames": frames,
                    "device": args.device,
                    "checks": checks,
                    "passed": all(c["bitwise_equal"] for c in checks),
                }
                if record["passed"]:
                    left, right = measure_pair(reference, candidate, torch.npu.synchronize, 20, 8)
                    record.update(
                        {
                            "cpu": left,
                            "npu": right,
                            "reduction_percent": 100 * (1 - right["median_us"] / left["median_us"]),
                        }
                    )
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k not in ("cpu", "npu")}), flush=True)


if __name__ == "__main__":
    main()
