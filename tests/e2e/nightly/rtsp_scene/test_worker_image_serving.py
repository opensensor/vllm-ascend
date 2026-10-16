"""Exercise byte and fallback requests through the real graph-wrapped workers."""

import concurrent.futures
import json
import tempfile
import urllib.request
from pathlib import Path

import pytest
from PIL import Image


def test_byte_and_fallback_requests_with_graph_wrapped_workers(request):
    endpoint = request.config.getoption("--scene-image-serving-endpoint")
    root = request.config.getoption("--scene-image-serving-root")
    model = request.config.getoption("--scene-dense-model")
    if not endpoint or not root or not model:
        pytest.skip("Requires the opt-in scene overlay and its running local TP2 graph server")
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    assert config["architectures"] == ["SceneDeviceQwen3VLForConditionalGeneration"]
    runtime = json.loads((model / "rtsp_dense_runtime.json").read_text())
    assert "device_image_prepare" in runtime
    with tempfile.TemporaryDirectory(prefix="worker-image-test-", dir=root) as directory:
        directory = Path(directory)
        raw, fallback = [], []
        for index in range(8):
            path = directory / f"raw-{index}.ppm"
            Image.new("RGB", (640, 480), (index * 31, 255 - index * 29, index * 23)).save(path)
            path.chmod(0o600)
            raw.append(path)
        path = directory / "fallback.ppm"
        Image.new("RGB", (512, 512), (0, 255, 0)).save(path)
        path.chmod(0o600)
        fallback.append(path)

        def send(paths):
            content = [{"type": "text", "text": "Describe the colors in the images, or say hello if none."}]
            content.extend({"type": "image_url", "image_url": {"url": path.as_uri()}} for path in paths)
            body = json.dumps(
                {
                    "model": "qwen3-vl-scene",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": 16,
                }
            ).encode()
            req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as response:
                assert response.status == 200
                payload = json.load(response)
            assert payload["choices"][0]["message"]["content"].strip()
            assert payload["usage"]["completion_tokens"] > 0
            return payload

        # Separate requests let the runner batch the two distinct field schemas.
        # This also verifies ordinary text requests retain their inherited path.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(send, paths) for paths in (raw, fallback, [])]
            outputs = [job.result() for job in jobs]
        assert outputs[0]["usage"]["prompt_tokens"] >= 8 * 234
        assert outputs[1]["usage"]["prompt_tokens"] >= 256
    # A helper attached to the graph wrapper passes startup and fails the first
    # byte request. These worker-produced receipts require real serialized pixels.
    for rank in (0, 1):
        gate = json.loads((model / f"image-transport-gate-rank{rank}.json").read_text())
        assert gate["rank"] == rank and gate["passed"] and gate["bitwise_equal"]
        assert gate["max_abs_error"] == 0 and gate["source"] == "First real serialized RGB byte request"
        if "qkv_projection_build" in runtime:
            dense = json.loads((model / f"dense-gate-rank{rank}.json").read_text())
            assert dense["qkv_projection_build"] == runtime["qkv_projection_build"]
            qkv = [row for row in dense["records"] if row["module"].endswith(".qkv_proj")]
            assert len(qkv) == 216 and all(row["passed"] and row["bitwise_equal"] for row in qkv)
