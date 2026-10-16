import argparse
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, JpegImagePlugin

from tools.rtsp_scene.review import render
from tools.rtsp_scene.runner import describe, frame_extension, image_data_url
from tools.rtsp_scene.stream import save_window


class RawFrameTest(unittest.TestCase):
    def test_rgb_transport_preserves_pixels_without_a_jpeg_decoder(self):
        pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 13, 32])
        data = b"P6\n2 2\n255\n" + pixels
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "frame.ppm"
            frame.write_bytes(data)
            url = image_data_url(frame)
            self.assertTrue(url.startswith("data:image/x-portable-pixmap;base64,"))
            restored = base64.b64decode(url.split(",", 1)[1])
            with (
                patch.object(
                    JpegImagePlugin.JpegImageFile, "load", side_effect=AssertionError("JPEG decoding forbidden")
                ),
                Image.open(io.BytesIO(restored)) as image,
            ):
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.tobytes(), pixels)
            frame.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, "payload mismatch"):
                image_data_url(frame)
            frame.write_bytes(b"P6\n2 2\n65535\n" + pixels)
            with self.assertRaisesRegex(ValueError, "8-bit"):
                image_data_url(frame)

    def test_continuous_window_and_model_request_use_raw_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir()
            pixels = b"P6\n1 1\n255\n\x00\x80\xff"
            (root / "frames/frame-000001.ppm").write_bytes(pixels)
            directory, manifest = save_window(root, 0, 1, {0: 1.0}, "camera", "UTC", "ascend_dvpp", ".ppm")
            self.assertEqual(manifest["frame_format"], "ppm")
            response = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Static test image.",
                                    "events": [],
                                    "limitations": [],
                                }
                            )
                        },
                    }
                ],
                "usage": {"completion_tokens": 10},
            }
            args = argparse.Namespace(model="scene", endpoint="http://127.0.0.1:8004", prompt_file=None)
            with patch(
                "tools.rtsp_scene.runner.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())
            ) as call:
                result = describe(directory, manifest, args)
            body = json.loads(call.call_args.args[0].data)
            url = body["messages"][0]["content"][2]["image_url"]["url"]
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), pixels)
            self.assertEqual(result["image_formats"], [".ppm"])
            self.assertGreater(result["request_bytes"], len(pixels))
            page = render(root)
            self.assertIn("data:image/x-portable-pixmap;base64,", page)
            self.assertIn("putImageData", page)
            self.assertEqual(frame_extension(argparse.Namespace()), ".ppm")
            with self.assertRaisesRegex(ValueError, "Frame format"):
                frame_extension(argparse.Namespace(frame_format="png"))
