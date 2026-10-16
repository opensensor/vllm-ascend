import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rtsp_scene.multi_stream import main

CAPTURE_SCRIPT = """
import json, sys
from pathlib import Path
pattern = sys.argv[1]
for index in range(2):
    Path(pattern % (index + 1)).write_bytes(b'placeholder-frame')
    print('DVPP_FRAME ' + json.dumps({'id': index, 'seconds': index}), file=sys.stderr, flush=True)
print('DVPP_STATS ' + json.dumps({'sampled_frames': 2, 'cpu_seconds': 0.1}), file=sys.stderr, flush=True)
"""


class MultiStreamTest(unittest.TestCase):
    def test_capture_failure_is_retained_while_other_camera_finishes(self):
        for fail_camera in (False, True):
            with self.subTest(fail_camera=fail_camera), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                configuration = root / "cameras.json"
                configuration.write_text(
                    json.dumps(
                        [
                            {"name": "good", "url": "good"},
                            {"name": "second", "url": "bad" if fail_camera else "good"},
                        ]
                    )
                )
                configuration.chmod(0o600)
                output = root / "results"
                args = [
                    "multi_stream",
                    "--config",
                    str(configuration),
                    "--camera",
                    "good",
                    "--camera",
                    "second",
                    "--output",
                    str(output),
                    "--windows",
                    "1",
                    "--frames-per-window",
                    "2",
                    "--endpoint",
                    "http://127.0.0.1:8004/v1/chat/completions",
                ]

                def capture_command(url, pattern, _args):
                    script = "import sys; print('DVPP_ERROR broken stream', file=sys.stderr); sys.exit(1)"
                    return [sys.executable, "-c", script if url == "bad" else CAPTURE_SCRIPT, str(pattern)], None

                def describe(_directory, manifest, _args):
                    self.assertEqual(len(manifest["frames"]), 2)
                    self.assertEqual(manifest["decoder"], "ascend_dvpp")
                    return {"latency_seconds": 0.01, "usage": {"completion_tokens": 2}}

                with (
                    patch("sys.argv", args),
                    patch("sys.stdout", io.StringIO()),
                    patch("tools.rtsp_scene.multi_stream.capture_command", side_effect=capture_command),
                    patch("tools.rtsp_scene.multi_stream.describe", side_effect=describe),
                ):
                    if fail_camera:
                        with self.assertRaisesRegex(RuntimeError, "capture/inference failures"):
                            main()
                    else:
                        main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["completed"], 1 if fail_camera else 2)
                self.assertEqual(summary["capture_failed"], int(fail_camera))
                self.assertEqual(summary["captures"][0]["exit_status"], 0)
                self.assertEqual(summary["captures"][0]["details"]["sampled_frames"], 2)
                if fail_camera:
                    self.assertIn("broken stream", summary["captures"][1]["error"])
