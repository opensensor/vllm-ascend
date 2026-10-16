# SPDX-License-Identifier: Apache-2.0
"""Verify streamed image answers, changed-image isolation and real prefix reuse."""

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for label, color in [("red", (255, 0, 0)), ("blue", (0, 0, 255)), ("red", (255, 0, 0))]:
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 1024), color).save(buffer, format="PNG")
        content = [
            dict(
                type="image_url",
                image_url=dict(url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()),
            ),
            dict(type="text", text="What is the single dominant color of this image? Reply with only the color name."),
        ]
        payload = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=content)],
            max_tokens=32,
            temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),
            stream=True,
            stream_options=dict(include_usage=True),
        )
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        text = ""
        usage = None
        done = False
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                assert "error" not in event, event
                for choice in event.get("choices", []):
                    text += choice.get("delta", {}).get("content") or ""
                if event.get("usage"):
                    usage = event["usage"]
        result = dict(expected=label, text=text, done=done, usage=usage, elapsed_s=time.monotonic() - started)
        result["passed"] = bool(done and usage and text.strip().lower().rstrip(".") == label)
        results.append(result)
        (args.output / "streaming-cache.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["passed"], result
    assert results[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, "Repeated image did not reuse KV prefix"
    # The admitted one-image contract must fail clearly for a two-image request.
    payload["stream"] = False
    payload.pop("stream_options")
    payload["messages"][0]["content"].insert(0, content[0])
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except urllib.error.HTTPError as error:
        receipt = dict(status=error.code, error=json.loads(error.read()))
        (args.output / "over-limit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert error.code == 400, receipt
    else:
        raise AssertionError("Expected HTTP 400 for two images")
    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=10) as response:
        assert response.status == 200
    print("STREAMING_CACHE_GATE_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
