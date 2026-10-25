"""Run a bounded continuous RTSP capture while scene inference proceeds asynchronously."""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.rtsp_scene.runner import capture_command, decoder_details, describe, frame_extension, safe_error


def collect_timestamps(process: subprocess.Popen, timestamps: dict, logs: list, details: dict | None = None) -> None:
    for line in process.stderr:
        logs.append(safe_error(line))
        if len(logs) > 50:
            del logs[0]
        if details is not None:
            details.update(decoder_details(line))
        if line.startswith("DVPP_FRAME "):
            value = json.loads(line.removeprefix("DVPP_FRAME "))
            timestamps[value["id"]] = value["seconds"]
        if "Parsed_showinfo" in line and " n:" in line and "pts_time:" in line:
            index = int(line.split(" n:", 1)[1].split()[0])
            timestamps[index] = float(line.split("pts_time:", 1)[1].split()[0])


def save_window(
    root: Path,
    start: int,
    size: int,
    timestamps: dict,
    camera: str,
    utc: str,
    decoder: str = "ffmpeg_cpu",
    extension: str = ".jpg",
) -> tuple[Path, dict]:
    directory = root / f"window-{start // size:03d}"
    directory.mkdir()
    frames = []
    for index in range(start, start + size):
        source = root / "frames" / f"frame-{index + 1:06d}{extension}"
        shutil.copyfile(source, directory / source.name)
        frames.append({"id": index - start, "file": source.name, "seconds": timestamps[index] - timestamps[0]})
    manifest = {
        "camera": camera,
        "capture_started_utc": utc,
        "decoder": decoder,
        "frame_format": "ppm" if extension == ".ppm" else "jpeg",
        "timestamp_basis": "continuous decoded PTS; capture_started_utc is an approximate wall-clock anchor",
        "frames": frames,
    }
    (directory / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--decoder", choices=["ffmpeg_cpu", "ascend_dvpp"], default="ascend_dvpp")
    parser.add_argument("--frame-format", choices=["ppm", "jpeg"], default="ppm")
    parser.add_argument("--dvpp-binary", type=Path)
    parser.add_argument("--decode-device", type=int, default=0)
    args = parser.parse_args()
    if args.windows <= 0 or args.frames_per_window <= 0 or args.config.stat().st_mode & 0o077:
        parser.error("Use positive window/frame counts and a private camera configuration")
    cameras = json.loads(args.config.read_text())
    camera = next((item for item in cameras if item["name"] == args.camera), None)
    if camera is None or not camera.get("enabled", True) or camera.get("privacy_mode", False):
        parser.error("Camera is missing, disabled, or in privacy mode")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    frame_count = args.windows * args.frames_per_window
    args.sample_fps, args.max_frames, args.clip = 1, frame_count, None
    extension = frame_extension(args)
    command, input_text = capture_command(camera["url"], args.output / "frames" / ("frame-%06d" + extension), args)
    timestamps, logs, jobs, arrivals, pending, capture_details = {}, [], [], [], [], {}
    utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        if input_text:
            process.stdin.write(input_text)
            process.stdin.close()
        reader = threading.Thread(
            target=collect_timestamps, args=(process, timestamps, logs, capture_details), daemon=True
        )
        reader.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2 * len(args.endpoint)) as pool:
                for window in range(args.windows):
                    last = (window + 1) * args.frames_per_window - 1
                    final_frame = args.output / "frames" / f"frame-{last + 1:06d}{extension}"
                    while not final_frame.exists() or last not in timestamps:
                        if process.poll() is not None or time.monotonic() - started > frame_count + 45:
                            raise RuntimeError(
                                "Continuous capture stopped before all frames arrived: " + "".join(logs)[-1000:]
                            )
                        time.sleep(0.05)
                    directory, manifest = save_window(
                        args.output,
                        window * args.frames_per_window,
                        args.frames_per_window,
                        timestamps,
                        camera["name"],
                        utc,
                        args.decoder,
                        extension,
                    )
                    request_args = argparse.Namespace(
                        model="qwen3-vl-scene", endpoint=args.endpoint[window % len(args.endpoint)], prompt_file=None
                    )
                    arrivals.append(time.monotonic() - started)
                    pending.append(sum(not job.done() for job in jobs))
                    jobs.append(pool.submit(describe, directory, manifest, request_args))
                    print(
                        json.dumps({"window": window, "arrived_seconds": arrivals[-1], "earlier_pending": pending[-1]}),
                        flush=True,
                    )
                results = [job.result() for job in jobs]
                capture_code = process.wait(timeout=10)
                if capture_code:
                    raise RuntimeError(f"Continuous capture failed with exit {capture_code}")
        finally:
            if process.poll() is None:
                process.terminate()
            reader.join(timeout=5)
    details = capture_details
    missing_slots = details.get("missed_sample_slots", 0)
    summary = {
        "camera": camera["name"],
        "windows": len(results),
        "frames": frame_count,
        "decoder": args.decoder,
        "frame_format": args.frame_format,
        "decoder_details": details,
        "arrival_seconds": arrivals,
        "earlier_pending_at_arrival": pending,
        "inference_seconds": [result["latency_seconds"] for result in results],
        "elapsed_seconds": time.monotonic() - started,
        "capture_gap": (
            f"Continuous connection; {missing_slots} missing sampling slots retained in PTS"
            if missing_slots
            else "none between sampled windows"
        ),
        "note": "One frame per second; finite test, not full-frame-rate analysis or a sustained capacity guarantee.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
