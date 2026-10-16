import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rtsp_scene.capacity import aggregate_metrics
from tools.rtsp_scene.review import render
from tools.rtsp_scene.runner import (
    capture,
    capture_command,
    decoder_details,
    frame_times,
    safe_error,
    validate_description,
)
from tools.rtsp_scene.stream import collect_timestamps, save_window


class SceneDescriptionTest(unittest.TestCase):
    def test_capture_defaults_require_hardware_instead_of_cpu_encoding(self):
        args = argparse.Namespace(clip=None, max_frames=8, sample_fps=1)
        with self.assertRaisesRegex(ValueError, "dvpp-binary"):
            capture_command("rtsp://camera/live", Path("frames/frame-%06d.jpg"), args)
        args.decoder = "ffmpeg_cpu"
        command, input_text = capture_command("rtsp://camera/live", Path("frames/frame-%06d.jpg"), args)
        self.assertEqual(command[0], "ffmpeg")
        self.assertIsNone(input_text)
        args.decoder = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown capture backend"):
            capture_command("rtsp://camera/live", Path("frames/frame-%06d.jpg"), args)

    def test_hardware_backend_keeps_url_off_command_line_and_requires_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "dvpp_decode"
            args = argparse.Namespace(decoder="ascend_dvpp", dvpp_binary=binary, max_frames=8, sample_fps=1)
            with self.assertRaisesRegex(ValueError, "built decoder"):
                capture_command("rtsp://user:secret@camera/live", Path(directory) / "frame-%06d.jpg", args)
            binary.touch()
            command, input_text = capture_command(
                "rtsp://user:secret@camera/live", Path(directory) / "frame-%06d.jpg", args
            )
            self.assertNotIn("secret", " ".join(command))
            self.assertEqual(input_text, "rtsp://user:secret@camera/live\n")
            with self.assertRaisesRegex(ValueError, "line breaks"):
                capture_command("rtsp://camera/live\nsecond input", Path(directory) / "frame-%06d.jpg", args)

    def test_hardware_decoder_errors_are_not_retried_with_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "dvpp_decode"
            binary.touch()
            args = argparse.Namespace(decoder="ascend_dvpp", dvpp_binary=binary, clip=None, max_frames=8, sample_fps=1)
            result = type("Result", (), {"returncode": 1, "stderr": "DVPP_ERROR unsupported stream"})()
            with (
                patch("tools.rtsp_scene.runner.subprocess.run", return_value=result) as run,
                self.assertRaisesRegex(RuntimeError, "unsupported stream"),
            ):
                capture({"name": "test", "url": "rtsp://camera/live"}, root / "capture", args)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((root / "capture/capture.json").exists())

    def test_hardware_timestamps_and_work_counters_are_preserved(self):
        log = (
            'DVPP_INFO {"codec":"hevc","device":0}\n'
            'DVPP_FRAME {"id":0,"seconds":0}\n'
            'DVPP_FRAME {"id":1,"seconds":1.033}\n'
            'DVPP_FRAME {"id":2,"seconds":3.0}\n'
            'DVPP_STATS {"decoded_frames":91,"missed_sample_slots":1}\n'
        )
        self.assertEqual(frame_times(log, 3), [0, 1.033, 3.0])
        self.assertEqual(decoder_details(log)["missed_sample_slots"], 1)
        process = type("Process", (), {"stderr": io.StringIO(log)})()
        timestamps, logs = {}, []
        collect_timestamps(process, timestamps, logs)
        self.assertEqual(timestamps, {0: 0, 1: 1.033, 2: 3.0})
        with self.assertRaisesRegex(RuntimeError, "every sampled frame"):
            frame_times(log, 4)

    def test_decoder_metadata_outlives_the_diagnostic_ring_buffer(self):
        log = 'DVPP_INFO {"codec":"hevc","width":640}\n' + "diagnostic\n" * 60
        process = type("Process", (), {"stderr": io.StringIO(log)})()
        timestamps, logs, details = {}, [], {}
        collect_timestamps(process, timestamps, logs, details)
        self.assertEqual(len(logs), 50)
        self.assertEqual(details, {"codec": "hevc", "width": 640})

    def test_continuous_reader_retains_pts_and_redacts_diagnostics(self):
        process = type(
            "Process",
            (),
            {
                "stderr": io.StringIO(
                    "Opening rtsp://user:password@camera/live\n[Parsed_showinfo_2 @ 0x1] n: 0 pts: 0 pts_time:5.25\n"
                )
            },
        )()
        timestamps, logs = {}, []
        collect_timestamps(process, timestamps, logs)
        self.assertEqual(timestamps, {0: 5.25})
        self.assertNotIn("password", "".join(logs))

    def test_successive_windows_preserve_continuous_media_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            for index in (3, 4):
                (root / "frames" / f"frame-{index:06d}.jpg").write_bytes(b"test frame")
            _, manifest = save_window(root, 2, 2, {0: 9.0, 2: 11.0, 3: 12.0}, "test-camera", "UTC")
            self.assertEqual([frame["seconds"] for frame in manifest["frames"]], [2.0, 3.0])
            self.assertEqual([frame["id"] for frame in manifest["frames"]], [0, 1])

    def test_replica_metrics_preserve_request_weighting(self):
        with patch(
            "tools.rtsp_scene.capacity.metrics",
            side_effect=[
                {"request_queue_time_seconds_sum": 6, "request_queue_time_seconds_count": 2},
                {"request_queue_time_seconds_sum": 1, "request_queue_time_seconds_count": 1},
            ],
        ):
            values = aggregate_metrics(["first", "second"])
        self.assertAlmostEqual(
            values["request_queue_time_seconds_sum"] / values["request_queue_time_seconds_count"], 7 / 3
        )

    def test_review_escapes_model_text_and_handles_missing_clock_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(
                json.dumps(
                    {
                        "capture_started_utc": None,
                        "timestamp_basis": "relative clip PTS",
                        "frames": [{"file": "frame-001.jpg"}],
                    }
                )
            )
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "camera": "back-door",
                        "latency_seconds": 1.0,
                        "description": {"summary": "<script>alert('scene')</script>", "events": [], "limitations": []},
                    }
                )
            )
            page = render(root)
            self.assertNotIn("<script>alert", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertIn("UTC anchor unavailable", page)

    def test_filtered_extra_frame_does_not_invalidate_saved_evidence(self):
        log = "\n".join(f"[Parsed_showinfo_2 @ 0x1] n: {i} pts: {i} pts_time:{i + 4}" for i in range(3))
        self.assertEqual(frame_times(log, 2), [0.0, 1.0])
        with self.assertRaisesRegex(RuntimeError, "every sampled frame"):
            frame_times(log, 4)

    def test_error_redacts_credentials_and_query_tokens(self):
        message = "Input rtsp://user:secret@192.168.52.36/live?token=private failed"
        self.assertEqual(safe_error(message), "Input [camera URL] failed")

    def test_rejects_invented_frame_evidence(self):
        value = {
            "summary": "A person moves.",
            "events": [{"description": "Walks", "start_frame": 0, "end_frame": 8, "visibility": "clear"}],
            "limitations": [],
        }
        with self.assertRaisesRegex(ValueError, "existing frame"):
            validate_description(value, 8)
        value["events"][0]["start_frame"] = 3
        value["events"][0]["end_frame"] = 2
        with self.assertRaisesRegex(ValueError, "ordered"):
            validate_description(value, 8)

    def test_accepts_static_scene_and_explicit_uncertainty(self):
        value = {
            "summary": "An empty counter remains visible.",
            "events": [],
            "limitations": ["The drawer interior is not visible."],
        }
        self.assertEqual(validate_description(value, 8), value)


if __name__ == "__main__":
    unittest.main()
