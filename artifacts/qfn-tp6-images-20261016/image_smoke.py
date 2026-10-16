# SPDX-License-Identifier: Apache-2.0
"""Small real-image, changed-image cache isolation, and text smoke gate."""

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def request(content, label, expected):
        payload = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=content)],
            max_tokens=32,
            temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),
        )
        started = time.monotonic()
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as f:
            answer = json.load(f)
        text = answer["choices"][0]["message"]["content"].strip()
        passed = text.lower().strip(".! \n") == expected
        result = dict(
            label=label,
            expected=expected,
            text=text,
            passed=passed,
            elapsed_s=time.monotonic() - started,
            response=answer,
        )
        results.append(result)
        (output / "image-smoke.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert passed, (label, text)

    request("Reply with only the integer: 17 * 23", "text-before", "391")
    for label, color, side in [
        ("red", (255, 0, 0), 256),
        ("blue", (0, 0, 255), 256),
        ("red", (255, 0, 0), 256),
        ("green", (0, 255, 0), 1024),
    ]:
        image = Image.new("RGB", (side, side), color)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        (output / f"{label}-{side}.png").write_bytes(data)
        content = [
            dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(data).decode())),
            dict(type="text", text="What is the single dominant color of this image? Reply with only the color name."),
        ]
        request(content, f"{label}-{side}", label)
    image = Image.new("RGB", (768, 256), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
    draw.text((80, 80), "NPU-4729", font=font, fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()
    (output / "ocr.png").write_bytes(data)
    request(
        [
            dict(type="image_url", image_url=dict(url="data:image/png;base64," + base64.b64encode(data).decode())),
            dict(type="text", text="Read the text in the image. Reply with only that text, exactly as written."),
        ],
        "ocr",
        "npu-4729",
    )
    request("Reply with only the integer: 19 * 17", "text-after", "323")
    print("IMAGE_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    run(args)
