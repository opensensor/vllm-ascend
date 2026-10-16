"""Bounded multi-camera capture with a shared inference budget and explicit drops."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tools.rtsp_scene.runner import capture_command, describe, frame_extension, safe_error
from tools.rtsp_scene.stream import collect_timestamps, save_window


@dataclass
class CameraCapture:
    camera: dict
    root: Path
    process: subprocess.Popen
    reader: threading.Thread | None = None
    timestamps: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    next_window: int = 0
    error: str | None = None


def choose_endpoint(camera_index: int, endpoints: list[str]) -> str:
    if not endpoints:
        raise ValueError("At least one model endpoint is required")
    return endpoints[camera_index % len(endpoints)]


def has_budget(jobs: list, maximum: int) -> bool:
    return sum(not job.done() for job in jobs) < maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--decoder", choices=["ffmpeg_cpu", "ascend_dvpp"], default="ascend_dvpp")
    parser.add_argument("--frame-format", choices=["ppm", "jpeg"], default="ppm")
    parser.add_argument("--dvpp-binary", type=Path)
    parser.add_argument("--decode-device", type=int, default=0)
    args = parser.parse_args()
    if min(args.windows, args.frames_per_window, args.max_in_flight) <= 0 or args.config.stat().st_mode & 0o077:
        parser.error("Use positive counts and a private camera configuration")
    configured = json.loads(args.config.read_text())
    cameras = []
    for name in dict.fromkeys(args.camera):
        camera = next((item for item in configured if item["name"] == name), None)
        if camera is None or not camera.get("enabled", True) or camera.get("privacy_mode", False):
            parser.error("Selected camera is missing, disabled or in privacy mode")
        cameras.append(camera)
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    args.sample_fps, args.max_frames, args.clip = 1, args.windows * args.frames_per_window, None
    extension = frame_extension(args)
    captures, jobs, records = [], [], []
    utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        for index, camera in enumerate(cameras):
            root = args.output / f"camera-{index}"
            (root / "frames").mkdir(parents=True)
            command, input_text = capture_command(camera["url"], root / ("frames/frame-%06d" + extension), args)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_text else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            capture = CameraCapture(camera, root, process)
            captures.append(capture)
            capture.reader = threading.Thread(
                target=collect_timestamps,
                args=(process, capture.timestamps, capture.logs, capture.details),
                daemon=True,
            )
            capture.reader.start()
            if input_text:
                process.stdin.write(input_text)
                process.stdin.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_in_flight) as pool:
            while any(capture.next_window < args.windows and capture.error is None for capture in captures):
                for index, capture in enumerate(captures):
                    if capture.error is not None or capture.next_window >= args.windows:
                        continue
                    last = (capture.next_window + 1) * args.frames_per_window - 1
                    final = capture.root / "frames" / f"frame-{last + 1:06d}{extension}"
                    if final.exists() and last in capture.timestamps:
                        directory, manifest = save_window(
                            capture.root,
                            capture.next_window * args.frames_per_window,
                            args.frames_per_window,
                            capture.timestamps,
                            capture.camera["name"],
                            utc,
                            args.decoder,
                            extension,
                        )
                        record = {
                            "camera": capture.camera["name"],
                            "window": capture.next_window,
                            "arrival_seconds": time.monotonic() - started,
                            "earlier_pending": sum(not job.done() for job in jobs),
                            "directory": str(directory.relative_to(args.output)),
                        }
                        if has_budget(jobs, args.max_in_flight):
                            request = argparse.Namespace(
                                model="qwen3-vl-scene", endpoint=choose_endpoint(index, args.endpoint), prompt_file=None
                            )
                            record["endpoint"] = request.endpoint
                            jobs.append(pool.submit(describe, directory, manifest, request))
                            record["job_index"] = len(jobs) - 1
                        else:
                            record["dropped"] = "Inference budget exhausted; capture continued"
                        records.append(record)
                        capture.next_window += 1
                        print(json.dumps(record), flush=True)
                    elif (
                        capture.process.poll() is not None and not capture.reader.is_alive()
                    ) or time.monotonic() - started > args.max_frames + 60:
                        capture.error = "Capture stopped or timed out: " + "".join(capture.logs)[-1500:]
                time.sleep(0.05)
            for record in records:
                if "job_index" in record:
                    try:
                        result = jobs[record["job_index"]].result()
                        record["inference_seconds"] = result["latency_seconds"]
                        record["completion_tokens"] = result["usage"]["completion_tokens"]
                    except Exception as error:
                        record["error"] = safe_error(str(error))
    finally:
        for capture in captures:
            try:
                # Retain normal decoder teardown and its final counters after the last JPEG.
                if capture.next_window < args.windows and capture.process.poll() is None:
                    capture.process.terminate()
                capture.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                capture.process.kill()
                capture.process.wait(timeout=3)
                capture.error = capture.error or "Decoder did not finish after the final sampled frame"
            if capture.reader:
                capture.reader.join(timeout=3)
            capture.process.stderr.close()
            if capture.process.stdin and not capture.process.stdin.closed:
                capture.process.stdin.close()
            if capture.process.returncode:
                capture.error = capture.error or f"Decoder exited with status {capture.process.returncode}"
    summary = {
        "decoder": args.decoder,
        "frame_format": args.frame_format,
        "camera_count": len(cameras),
        "windows_per_camera": args.windows,
        "inference_budget": args.max_in_flight,
        "elapsed_seconds": time.monotonic() - started,
        "completed": sum("inference_seconds" in record for record in records),
        "dropped": sum("dropped" in record for record in records),
        "failed": sum("error" in record for record in records),
        "capture_failed": sum(capture.error is not None for capture in captures),
        "captures": [
            {
                "camera": capture.camera["name"],
                "details": capture.details,
                "exit_status": capture.process.returncode,
                "error": capture.error,
            }
            for capture in captures
        ],
        "windows": records,
        "note": "Finite live test at target 1 FPS. Sampling gaps are retained in decoded PTS; drops are explicit.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"captures", "windows"}}), flush=True)
    if summary["failed"] or any(capture.error for capture in captures):
        raise RuntimeError("One or more capture/inference failures; inspect summary.json")


if __name__ == "__main__":
    main()
