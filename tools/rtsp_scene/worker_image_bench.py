"""Gate worker byte inputs against real processor pixels, including batch order."""

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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh private output")
    # Hardware-only imports; the parent test does not initialize an NPU.
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoImageProcessor

    from tools.rtsp_scene.dense_bench import measure_pair
    from tools.rtsp_scene.dense_runtime import KernelBackend
    from tools.rtsp_scene.raw_image_transport import pack_images
    from tools.rtsp_scene.worker_image_prepare import WorkerImagePreparer

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    processor = AutoImageProcessor.from_pretrained(args.model, max_pixels=262144)
    prepare = WorkerImagePreparer(args.build, KernelBackend(args.matrix_build), processor, f"npu:{args.device}")
    cohorts = [
        ("mixed_order", [(480, 640), (180, 320), (360, 640), (240, 320)] * 2),
        ("chunk_boundary", [(360, 640)] * 9),
        ("live_4_3", [(480, 640)] * 8),
        ("live_16_9", [(360, 640)] * 8),
    ]
    records = []
    with torch.inference_mode():
        for name, sizes in cohorts:
            generator = torch.Generator().manual_seed(20261013)
            images = [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=generator) for h, w in sizes]
            packed = pack_images(images)
            rgb = packed["scene_rgb"].to(f"npu:{args.device}")

            def reference(images=images):
                return processor(images=images, return_tensors="pt")["pixel_values"].half()

            def candidate(rgb=rgb, packed=packed):
                return prepare(rgb, packed["scene_geometry"], packed["image_grid_thw"])

            expected, actual = reference(), candidate().cpu()
            row = {
                "cohort": name,
                "device": args.device,
                "frames": len(images),
                "shape": list(actual.shape),
                "transport_bytes": rgb.numel(),
                "baseline_float_bytes": expected.numel() * 4,
                "bitwise_equal": torch.equal(expected, actual),
                "max_abs_error": float((expected - actual).abs().max()),
            }
            if row["bitwise_equal"]:
                left, right = measure_pair(reference, candidate, torch.npu.synchronize, 20, 8)
                row.update(cpu=left, worker=right)
            records.append(row)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("cpu", "worker")}), flush=True)
    if not all(r["bitwise_equal"] for r in records):
        raise RuntimeError("Worker model input gate failed")


if __name__ == "__main__":
    main()
