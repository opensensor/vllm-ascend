import argparse
import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from tools.rtsp_scene.continuous_log import Capture, JSONLog, infer, run, select_cameras, take_window


def test_json_lines_append_and_current_link(tmp_path):
    root = tmp_path / "logs"
    for index in range(2):
        logger = JSONLog(root)
        logger.write({"type": "scene", "camera": "back-door", "index": index})
        logger.close()
    records = [json.loads(line) for line in (root / "scenes.jsonl").read_text().splitlines()]
    assert [record["index"] for record in records] == [0, 1]
    assert all(record["logged_at_utc"] for record in records)
    assert (root / "scenes.jsonl").resolve().stat().st_mode & 0o077 == 0


def test_daily_rotation_retains_previous_log(tmp_path):
    logger = JSONLog(tmp_path)
    with patch("tools.rtsp_scene.continuous_log.datetime") as clock:
        clock.now.side_effect = [datetime(2030, 1, 1, tzinfo=timezone.utc), datetime(2030, 1, 2, tzinfo=timezone.utc)]
        logger.write({"type": "first"})
        logger.write({"type": "second"})
    logger.close()
    assert json.loads((tmp_path / "scenes-2030-01-01.jsonl").read_text())["type"] == "first"
    assert json.loads((tmp_path / "scenes.jsonl").read_text())["type"] == "second"


def test_window_moves_frames_and_preserves_pts_gaps(tmp_path):
    capture = Capture({"name": "back-door"}, tmp_path / "capture", anchor=1000)
    (capture.root / "frames").mkdir(parents=True)
    for index, seconds in enumerate((0, 1, 4, 5, 9, 10, 11, 12)):
        capture.timestamps[index] = seconds
        (capture.root / "frames" / f"frame-{index + 1:06d}.ppm").write_bytes(b"frame")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    directory, manifest, record = take_window(capture, 4, scratch)
    assert record["sample_gaps"] == 1
    assert record["gap_before_window_seconds"] is None
    assert [frame["seconds"] for frame in manifest["frames"]] == [0, 1, 4, 5]
    assert len(list(directory.glob("*.ppm"))) == 4
    _, manifest, record = take_window(capture, 4, scratch)
    assert record["gap_before_window_seconds"] == 4
    assert manifest["frames"][0]["seconds"] == 0
    assert not capture.timestamps and not list((capture.root / "frames").iterdir())
    assert take_window(capture, 4, scratch) is None


def test_inference_failure_is_redacted_and_scratch_removed(tmp_path):
    directory = tmp_path / "window"
    directory.mkdir()
    with patch(
        "tools.rtsp_scene.continuous_log.describe", side_effect=ValueError("rtsp://user:secret@host/live failed")
    ):
        record = infer(directory, {}, argparse.Namespace())
    assert record["type"] == "inference_error"
    assert "secret" not in record["error"]
    assert not directory.exists()


def test_privacy_and_unknown_cameras_fail_closed(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"name": "private", "privacy_mode": True}]))
    config.chmod(0o600)
    with pytest.raises(ValueError, match="privacy"):
        select_cameras(config, None)
    with pytest.raises(ValueError, match="missing"):
        select_cameras(config, ["other"])
    config.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        select_cameras(config, None)


def test_capture_failure_does_not_stop_other_camera(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"name": "bad"}, {"name": "good"}]))
    config.chmod(0o600)
    args = argparse.Namespace(
        config=config, camera=None, output=tmp_path / "output", model="scene", endpoint="test", max_in_flight=1
    )
    stop = threading.Event()
    started = []

    def start(capture, _args):
        started.append(capture.camera["name"])
        if capture.camera["name"] == "bad":
            raise ValueError("rtsp://user:secret@host broken")
        stop.set()
        capture.process = type("Process", (), {"poll": lambda self: None})()
        capture.reader = type("Reader", (), {"is_alive": lambda self: True})()

    with (
        patch.object(Capture, "start", start),
        patch.object(Capture, "stop"),
        patch("tools.rtsp_scene.continuous_log.take_window", return_value=None),
    ):
        run(args, stop)
    assert started == ["bad", "good"]
    records = [json.loads(line) for line in (args.output / "scenes.jsonl").read_text().splitlines()]
    assert any(record["type"] == "capture_error" and record["camera"] == "bad" for record in records)
    assert "secret" not in (args.output / "scenes.jsonl").read_text()
    assert records[-1]["type"] == "service_stopped"


def test_inference_retains_transport_metrics_and_removes_completed_frames(tmp_path):
    directory = tmp_path / "window"
    directory.mkdir()
    response = {
        "description": {"summary": "Static scene", "events": [], "limitations": []},
        "latency_seconds": 4.0,
        "usage": {"completion_tokens": 64},
        "model": "scene",
        "prompt_version": "scene-v3-camera-relative",
        "frame_transport": "file",
        "preparation_seconds": 0.001,
        "request_bytes": 3400,
    }
    with patch("tools.rtsp_scene.continuous_log.describe", return_value=response):
        record = infer(directory, {}, argparse.Namespace())
    assert record["type"] == "scene" and record["frame_transport"] == "file"
    assert record["preparation_seconds"] == 0.001 and record["request_bytes"] == 3400
    assert record["description"] == response["description"]
    assert not directory.exists()


def test_full_inference_budget_logs_drop_and_removes_frames(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"name": "first"}, {"name": "second"}]))
    config.chmod(0o600)
    args = argparse.Namespace(
        config=config, camera=None, output=tmp_path / "output", model="scene", endpoint="test", max_in_flight=1
    )
    stop = threading.Event()

    def start(capture, _args):
        capture.process = type("Process", (), {"poll": lambda self: None})()
        capture.reader = type("Reader", (), {"is_alive": lambda self: True})()
        if capture.camera["name"] == "second":
            stop.set()

    def window(capture, size, scratch):
        directory = scratch / capture.camera["name"]
        directory.mkdir()
        return directory, {}, {"camera": capture.camera["name"], "window": 0}

    def blocked_inference(*_args):
        time.sleep(0.1)
        return {"type": "scene"}

    with (
        patch.object(Capture, "start", start),
        patch.object(Capture, "stop"),
        patch("tools.rtsp_scene.continuous_log.take_window", side_effect=window),
        patch("tools.rtsp_scene.continuous_log.infer", side_effect=blocked_inference),
    ):
        run(args, stop)
    records = [json.loads(line) for line in (args.output / "scenes.jsonl").read_text().splitlines()]
    dropped = [record for record in records if record["type"] == "inference_dropped"]
    assert len(dropped) == 1 and dropped[0]["camera"] == "second"
    assert sum(record["type"] == "scene" for record in records) == 1
    assert not (args.output / "scratch").exists()
