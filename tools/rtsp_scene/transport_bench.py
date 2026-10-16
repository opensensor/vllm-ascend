"""Compare JPEG and raw RGB transport of identical pixels on one model endpoint."""

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from tools.rtsp_scene.runner import SCENE_PROMPT, describe


def process_cpu_seconds(pid: int) -> float:
    # Linux stat fields 14/15; command names can include spaces and parentheses.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="Existing JPEG capture for offline pixel matching")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pids", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.rounds <= 0 or any(pid <= 0 for pid in args.pids):
        parser.error("Rounds and process IDs must be positive")
    # Offline fixture preparation only. Live capture obtains RGB directly from DVPP.
    from PIL import Image

    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    source = json.loads((args.capture / "capture.json").read_text())
    for format_name in ("jpeg", "ppm"):
        directory = args.output / format_name
        directory.mkdir()
        manifest = dict(source, frames=[])
        manifest["frame_format"] = format_name
        for frame in source["frames"]:
            path = args.capture / frame["file"]
            if path.suffix not in {".jpg", ".jpeg"}:
                raise ValueError("Use a JPEG source capture for pixel-matched comparison")
            copied = dict(frame)
            if format_name == "jpeg":
                shutil.copyfile(path, directory / path.name)
            else:
                copied["file"] = path.with_suffix(".ppm").name
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    (directory / copied["file"]).write_bytes(f"P6\n{width} {height}\n255\n".encode() + image.tobytes())
            manifest["frames"].append(copied)
        manifests[format_name] = manifest
        (directory / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for iteration in range(args.rounds):
        prompt = args.output / f"prompt-{iteration}.txt"
        prompt.write_text("Request identifier: " + uuid.uuid4().hex + "\n" + SCENE_PROMPT)
        formats = ("jpeg", "ppm") if iteration % 2 == 0 else ("ppm", "jpeg")
        for format_name in formats:
            directory = args.output / f"round-{iteration}-{format_name}"
            directory.mkdir()
            request = argparse.Namespace(model="qwen3-vl-scene", endpoint=args.endpoint, prompt_file=prompt)
            before_cpu = {pid: process_cpu_seconds(pid) for pid in args.pids}
            before_wall = time.monotonic()
            response = describe(args.output / format_name, manifests[format_name], request, directory)
            result = {
                "round": iteration,
                "format": format_name,
                "latency_seconds": response["latency_seconds"],
                "total_wall_seconds": time.monotonic() - before_wall,
                "preparation_seconds": response["preparation_seconds"],
                "client_cpu_seconds": response["client_cpu_seconds"],
                "request_bytes": response["request_bytes"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "prompt_sha256": response["prompt_sha256"],
                "description": response["description"],
                "process_cpu_seconds": {
                    str(pid): process_cpu_seconds(pid) - value for pid, value in before_cpu.items()
                },
            }
            results.append(result)
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
