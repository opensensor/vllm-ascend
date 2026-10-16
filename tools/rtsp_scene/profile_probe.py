"""Capture a bounded native profile; the truncated text is diagnostic only."""

import argparse
import json
import urllib.request
import uuid
from pathlib import Path

from tools.rtsp_scene.runner import API_TIMEOUT_SECONDS, SCENE_PROMPT, image_data_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "capture.json").read_text())
    content = [{"type": "text", "text": "Request identifier: " + uuid.uuid4().hex + "\n" + SCENE_PROMPT}]
    for frame in manifest["frames"]:
        content.append({"type": "text", "text": f"Frame {frame['id']}, time {frame['seconds']:.3f}:"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(args.capture / frame["file"])}})
    body = {
        "model": "qwen3-vl-scene",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 12,
        "ignore_eos": True,
    }
    server = args.endpoint.removesuffix("/v1/chat/completions")

    def request(path, payload=None):
        message = urllib.request.Request(
            path,
            data=json.dumps(payload).encode() if payload is not None else b"",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(message, timeout=API_TIMEOUT_SECONDS) as response:
            return response.read()

    request(server + "/start_profile")
    try:
        response = json.loads(request(args.endpoint, body))
    finally:
        request(server + "/stop_profile")
    args.output.write_text(
        json.dumps(
            {
                "response": response,
                "diagnostic_only": True,
                "note": ("Output capped at 12 tokens. This is not an accepted scene description or serving benchmark."),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"finish_reason": response["choices"][0]["finish_reason"], "usage": response["usage"]}), flush=True
    )


if __name__ == "__main__":
    main()
