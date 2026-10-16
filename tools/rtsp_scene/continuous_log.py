"""Continuous DVPP camera sampling and append-only scene JSONL logging."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tools.rtsp_scene.runner import capture_command, decoder_details, describe, safe_error

CAPTURE_FRAME_LIMIT = 2_000_000_000
STALL_SECONDS = 20
HEARTBEAT_SECONDS = 30
RECONNECT_SECONDS = 3
FRAMES_PER_WINDOW = 8
SAMPLE_FPS = 1
FRAME_GAP_THRESHOLD = 1.5


def utc(seconds=None):
    return (
        datetime.fromtimestamp(seconds, timezone.utc).isoformat()
        if seconds is not None
        else datetime.now(timezone.utc).isoformat()
    )


class JSONLog:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = None
        self.day = None

    def write(self, record):
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day != self.day:
            if self.file:
                self.file.close()
            target = self.root / f"scenes-{day}.jsonl"
            self.file = target.open("a", buffering=1)
            target.chmod(0o600)
            link = self.root / ".scenes-current"
            link.unlink(missing_ok=True)
            link.symlink_to(target.name)
            link.replace(self.root / "scenes.jsonl")
            self.day = day
        self.file.write(json.dumps({"logged_at_utc": now.isoformat(), **record}, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


@dataclass
class Capture:
    camera: dict
    root: Path
    session: str = field(default_factory=lambda: uuid.uuid4().hex)
    process: subprocess.Popen | None = None
    reader: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    timestamps: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    next_frame: int = 0
    anchor: float | None = None
    last_frame: float = field(default_factory=time.monotonic)
    connected: bool = False
    previous_window_end: float | None = None

    def read(self):
        for line in self.process.stderr:
            with self.lock:
                self.diagnostics.append(safe_error(line).strip())
                self.diagnostics[:] = self.diagnostics[-20:]
                self.details.update(decoder_details(line))
                if line.startswith("DVPP_FRAME "):
                    frame = json.loads(line.removeprefix("DVPP_FRAME "))
                    self.timestamps[frame["id"]] = frame["seconds"]
                    self.last_frame = time.monotonic()
                    if self.anchor is None:
                        self.anchor = time.time() - frame["seconds"]

    def start(self, args):
        (self.root / "frames").mkdir(parents=True, mode=0o700)
        command, input_text = capture_command(self.camera["url"], self.root / "frames/frame-%06d.ppm", args)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        self.process.stdin.write(input_text)
        self.process.stdin.close()

    def stop(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if self.reader and self.reader.is_alive():
                self.reader.join(timeout=3)
            self.process.stderr.close()
        shutil.rmtree(self.root, ignore_errors=True)


def take_window(capture, size, scratch):
    start = capture.next_frame
    indices = range(start, start + size)
    with capture.lock:
        if any(index not in capture.timestamps for index in indices):
            return None
        if any(not (capture.root / "frames" / f"frame-{index + 1:06d}.ppm").is_file() for index in indices):
            return None
        times = [capture.timestamps[index] for index in indices]
        directory = scratch / (capture.session + f"-w{start // size}")
        directory.mkdir(mode=0o700)
        frames = []
        for index, seconds in zip(indices, times):
            source = capture.root / "frames" / f"frame-{index + 1:06d}.ppm"
            source.rename(directory / source.name)
            del capture.timestamps[index]
            frames.append({"id": index - start, "file": source.name, "seconds": seconds - times[0]})
        capture.next_frame += size
        manifest = {
            "camera": capture.camera["name"],
            "capture_started_utc": utc(capture.anchor + times[0]),
            "decoder": "ascend_dvpp",
            "frame_format": "ppm",
            "timestamp_basis": "decoded PTS anchored to first frame arrival; approximate UTC, not camera NTP",
            "frames": frames,
        }
        record = {
            "camera": capture.camera["name"],
            "session": capture.session,
            "window": start // size,
            "window_start_utc": utc(capture.anchor + times[0]),
            "window_end_utc": utc(capture.anchor + times[-1]),
            "decoded_pts_seconds": times,
            "sample_gaps": sum(right - left > FRAME_GAP_THRESHOLD for left, right in zip(times, times[1:])),
            "gap_before_window_seconds": (
                times[0] - capture.previous_window_end if capture.previous_window_end is not None else None
            ),
            "timestamp_basis": manifest["timestamp_basis"],
        }
        capture.previous_window_end = times[-1]
        return directory, manifest, record


def infer(directory, manifest, request):
    try:
        result = describe(directory, manifest, request)
        return {
            "type": "scene",
            "description": result["description"],
            "inference_seconds": result["latency_seconds"],
            "usage": result["usage"],
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "frame_transport": result.get("frame_transport", "base64"),
            "preparation_seconds": result.get("preparation_seconds"),
            "request_bytes": result.get("request_bytes"),
        }
    except Exception as error:
        return {"type": "inference_error", "error": safe_error(str(error))[:1500]}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def select_cameras(config, names):
    if config.stat().st_mode & 0o077:
        raise ValueError("Camera configuration must be private")
    configured = json.loads(config.read_text())
    selected = [camera for camera in configured if not names or camera["name"] in names]
    if names and {camera["name"] for camera in selected} != set(names):
        raise ValueError("A selected camera is missing")
    if not selected or any(not camera.get("enabled", True) or camera.get("privacy_mode", False) for camera in selected):
        raise ValueError("Select enabled cameras outside privacy mode")
    if len({camera["name"] for camera in selected}) != len(selected):
        raise ValueError("Camera names must be unique")
    return selected


def run(args, stop):
    cameras = select_cameras(args.config, args.camera)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    scratch = (
        args.frame_root / "scene-live"
        if getattr(args, "frame_transport", "base64") == "file"
        else args.output / "scratch"
    )
    # Only this service owns scratch; completed logs are never removed.
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(mode=0o700)
    logger = JSONLog(args.output)
    captures, reconnect, pending = {}, {}, {}
    counters = {c["name"]: {"scenes": 0, "inference_errors": 0, "dropped": 0, "capture_errors": 0} for c in cameras}
    request = argparse.Namespace(
        model=args.model,
        endpoint=args.endpoint,
        prompt_file=None,
        frame_transport=getattr(args, "frame_transport", "base64"),
        frame_root=getattr(args, "frame_root", None),
    )
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_in_flight)
    last_status = 0
    logger.write(
        {
            "type": "service_started",
            "cameras": list(counters),
            "sample_fps": SAMPLE_FPS,
            "frames_per_window": FRAMES_PER_WINDOW,
            "frame_transport": getattr(args, "frame_transport", "base64"),
        }
    )
    try:
        while not stop.is_set():
            for future in list(pending):
                if future.done():
                    record = pending.pop(future)
                    result = future.result()
                    counters[record["camera"]]["scenes" if result["type"] == "scene" else "inference_errors"] += 1
                    logger.write({**record, **result})
            for index, camera in enumerate(cameras):
                name = camera["name"]
                if name not in captures and time.monotonic() >= reconnect.get(name, 0):
                    capture = Capture(camera, scratch / (f"camera-{index}-" + uuid.uuid4().hex))
                    captures[name] = capture
                    try:
                        capture.start(args)
                    except Exception as error:
                        logger.write({"type": "capture_error", "camera": name, "error": safe_error(str(error))})
                        counters[name]["capture_errors"] += 1
                        capture.stop()
                        del captures[name]
                        reconnect[name] = time.monotonic() + RECONNECT_SECONDS
                        continue
                if name not in captures:
                    continue
                capture = captures[name]
                if capture.anchor is not None and not capture.connected:
                    logger.write(
                        {
                            "type": "camera_connected",
                            "camera": name,
                            "session": capture.session,
                            "decoder": dict(capture.details),
                        }
                    )
                    capture.connected = True
                window = take_window(capture, FRAMES_PER_WINDOW, scratch)
                if window:
                    directory, manifest, record = window
                    if len(pending) >= args.max_in_flight:
                        counters[name]["dropped"] += 1
                        logger.write({**record, "type": "inference_dropped", "reason": "Inference budget exhausted"})
                        shutil.rmtree(directory)
                    else:
                        pending[pool.submit(infer, directory, manifest, request)] = record
                dead = capture.process.poll() is not None and not capture.reader.is_alive()
                if dead or time.monotonic() - capture.last_frame > STALL_SECONDS:
                    logger.write(
                        {
                            "type": "capture_error",
                            "camera": name,
                            "session": capture.session,
                            "reason": "Decoder stopped" if dead else "Sampling stalled",
                            "partial_frames_discarded": len(capture.timestamps),
                            "diagnostics": list(capture.diagnostics),
                        }
                    )
                    counters[name]["capture_errors"] += 1
                    capture.stop()
                    del captures[name]
                    reconnect[name] = time.monotonic() + RECONNECT_SECONDS
            if time.monotonic() - last_status >= HEARTBEAT_SECONDS:
                status = {
                    "type": "heartbeat",
                    "pending_requests": len(pending),
                    "cameras": counters,
                    "connected": [name for name, capture in captures.items() if capture.connected],
                }
                logger.write(status)
                temporary = args.output / "status.tmp"
                temporary.write_text(json.dumps({"updated_at_utc": utc(), **status}, indent=2) + "\n")
                temporary.replace(args.output / "status.json")
                last_status = time.monotonic()
            stop.wait(0.05)
    finally:
        for capture in captures.values():
            capture.stop()
        pool.shutdown(wait=True)
        for future, record in pending.items():
            logger.write({**record, **future.result()})
        logger.write({"type": "service_stopped"})
        logger.close()
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dvpp-binary", type=Path, required=True)
    parser.add_argument("--decode-device", type=int, default=0)
    parser.add_argument("--max-in-flight", type=int, default=16)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-vl-scene")
    parser.add_argument("--frame-transport", choices=["base64", "file"], default="base64")
    parser.add_argument(
        "--frame-root", type=Path, help="Private shared directory; this service owns its scene-live child"
    )
    args = parser.parse_args()
    if not 1 <= args.max_in_flight <= 32 or args.decode_device < 0:
        parser.error("Use an inference budget in [1,32] and a nonnegative device")
    if args.frame_transport == "file" and (args.frame_root is None or not args.frame_root.is_dir()):
        parser.error("File frame transport requires an existing private shared frame root")
    os.umask(0o077)
    args.decoder, args.frame_format, args.sample_fps, args.max_frames, args.clip = (
        "ascend_dvpp",
        "ppm",
        SAMPLE_FPS,
        CAPTURE_FRAME_LIMIT,
        None,
    )
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: stop.set())
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.output / "logger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, stop)


if __name__ == "__main__":
    main()
