"""Measure the installed vLLM image loader, excluding transport and model work."""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="JPEG/PPM fixtures from transport_bench.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) <= 0:
        parser.error("Counts must be positive")
    # Runtime-owned dependencies load only in the benchmark worker.
    from PIL import JpegImagePlugin
    from vllm.multimodal.media.image import ImageMediaIO

    loader = ImageMediaIO()
    result = {
        "loader_module": ImageMediaIO.__module__,
        "formats": {},
        "note": "Cached host bytes; no HTTP or inference.",
    }
    for format_name in ("jpeg", "ppm"):
        root = args.fixtures / format_name
        manifest = json.loads((root / "capture.json").read_text())
        inputs = [(root / frame["file"]).read_bytes() for frame in manifest["frames"]]
        for data in inputs:
            loader.load_bytes(data)
        samples = []
        for _ in range(args.repeats):
            wall, cpu = time.perf_counter(), time.process_time()
            for _ in range(args.iterations):
                for data in inputs:
                    loader.load_bytes(data)
            samples.append(
                {
                    "wall_seconds_per_window": (time.perf_counter() - wall) / args.iterations,
                    "cpu_seconds_per_window": (time.process_time() - cpu) / args.iterations,
                }
            )
        if format_name == "ppm":
            with patch.object(JpegImagePlugin.JpegImageFile, "load", side_effect=AssertionError("JPEG decoder called")):
                for data in inputs:
                    loader.load_bytes(data)
        result["formats"][format_name] = {
            "frames": len(inputs),
            "input_bytes": sum(map(len, inputs)),
            "samples": samples,
            "median_wall_seconds": statistics.median(item["wall_seconds_per_window"] for item in samples),
            "median_cpu_seconds": statistics.median(item["cpu_seconds_per_window"] for item in samples),
            "jpeg_decoder_forbidden_check": format_name == "ppm",
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
