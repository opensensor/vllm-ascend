import hashlib
import json
from pathlib import Path

import pytest

from tools.rtsp_scene.fixed_decode import request_body, validate_transport


def test_fixed_workload_preserves_pixels_prompt_and_schema(tmp_path):
    pixel = tmp_path / "frame.ppm"
    pixel.write_bytes(b"P6\n1 1\n255\n" + bytes((12, 34, 56)))
    (tmp_path / "capture.json").write_text(json.dumps({"frames": [{"id": 0, "seconds": 0.5, "file": pixel.name}]}))
    prompt = "Request identifier: identical\nDescribe the scene."
    first = request_body(tmp_path, prompt, 64)
    assert first == request_body(tmp_path, prompt, 64)
    body = json.loads(first)
    assert body["max_tokens"] == 64 and body["ignore_eos"] is True
    assert body["temperature"] == 0
    assert body["messages"][0]["content"][0]["text"] == prompt
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["content"][2]["image_url"]["url"].startswith("data:image/x-portable-pixmap;base64,")
    pixel.write_bytes(b"P6\n1 1\n255\n" + bytes((13, 34, 56)))
    assert hashlib.sha256(first).digest() != hashlib.sha256(request_body(tmp_path, prompt, 64)).digest()


def test_file_workload_uses_the_same_frame_evidence(tmp_path):
    pixel = tmp_path / "frame.ppm"
    pixel.write_bytes(b"P6\n1 1\n255\n" + bytes((12, 34, 56)))
    (tmp_path / "capture.json").write_text(json.dumps({"frames": [{"id": 0, "seconds": 0.5, "file": pixel.name}]}))
    body = json.loads(request_body(tmp_path, "Scene", 64, "file", tmp_path))
    assert body["messages"][0]["content"][2]["image_url"]["url"] == pixel.as_uri()
    assert body["messages"][0]["content"][1]["text"] == "Frame 0, relative time 0.500 seconds:"


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_invalid_stagger_is_rejected(delay):
    with pytest.raises(ValueError, match="finite"):
        validate_transport("base64", None, delay)


def test_file_workload_requires_an_absolute_root(tmp_path):
    for root in (None, Path("relative")):
        with pytest.raises(ValueError, match="absolute"):
            validate_transport("file", root, 0)
    validate_transport("file", tmp_path, 0.5)
