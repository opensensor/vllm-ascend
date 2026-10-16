import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rtsp_scene.multi_stream import has_budget
from tools.rtsp_scene.paced_capacity import arrival_offset, main


class PacedCapacityTest(unittest.TestCase):
    def test_arrivals_are_fixed_independently_of_inference(self):
        self.assertEqual([arrival_offset(window, 8) for window in range(4)], [0, 8, 16, 24])
        for interval in (0, float("nan"), 61):
            with self.assertRaises(ValueError):
                arrival_offset(0, interval)

    def test_overload_drops_are_explicit_and_do_not_pause_arrivals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "capture.json").write_text(json.dumps({"camera": "fixture", "frames": []}))
            output = root / "output"
            args = [
                "paced_capacity",
                "--captures",
                str(source),
                "--output",
                str(output),
                "--streams",
                "1",
                "--windows",
                "3",
                "--max-in-flight",
                "1",
                "--endpoint",
                "http://127.0.0.1:8003",
            ]
            release = threading.Event()
            checked = 0

            def budget(jobs, maximum):
                nonlocal checked
                checked += 1
                available = has_budget(jobs, maximum)
                if checked == 3:
                    release.set()
                return available

            def describe(*_args):
                if not release.wait(timeout=3):
                    raise RuntimeError("Test did not produce all paced arrivals")
                return {"usage": {"completion_tokens": 1}}

            with (
                patch("sys.argv", args),
                patch("sys.stdout", io.StringIO()),
                patch("tools.rtsp_scene.paced_capacity.time.sleep"),
                patch("tools.rtsp_scene.paced_capacity.describe", side_effect=describe),
                patch("tools.rtsp_scene.paced_capacity.has_budget", side_effect=budget),
            ):
                main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["dropped"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual([item["earlier_pending"] for item in summary["arrivals"]], [0, 1, 1])
