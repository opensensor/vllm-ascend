# SPDX-License-Identifier: Apache-2.0
"""Record HTTP generation and streaming latency from an already running service.

Nonempty output is a transport/runtime gate, not an automatic quality verdict.
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


def stream_request(base_url, endpoint, payload, timeout):
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        base_url + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    chunks = []
    text, reasoning = [], []
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            chunks.append(chunk)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = choice.get("text") or delta.get("content") or ""
                thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if (content or thought) and first is None:
                    first = time.monotonic() - started
                text.append(content)
                reasoning.append(thought)
    elapsed = time.monotonic() - started
    return {
        "status": status,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "first_output_seconds": first,
        "elapsed_seconds": elapsed,
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument(
        "--case", action="append", choices=("completion", "chat", "chunked_prefill", "image", "context_boundary")
    )
    parser.add_argument("--image", type=Path, help="Optional PNG for a color-description chat request")
    parser.add_argument(
        "--long-context-tokens", type=int, default=0, help="Optional repeated-token cache boundary test"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=args.timeout) as response:
        models = json.load(response)
    if args.model not in {model["id"] for model in models["data"]}:
        raise ValueError("Requested model is absent from /v1/models")
    (args.output / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    cases = [
        ("completion", "/v1/completions", {"prompt": "The capital of France is"}),
        (
            "chat",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Say hello in English, in one short sentence."}]},
        ),
        (
            "chunked_prefill",
            "/v1/completions",
            {"prompt": "The quick brown fox jumps over the lazy dog. " * 30 + "\nThe capital of France is"},
        ),
    ]
    if args.image is not None:
        data_url = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
        cases.append(
            (
                "image",
                "/v1/chat/completions",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": "What is the dominant color in this image? Answer briefly."},
                            ],
                        }
                    ]
                },
            )
        )
    if args.long_context_tokens:
        # 19739 is the pinned M3.1 tokenizer's single token for "Hello".
        cases.append(
            (
                "context_boundary",
                "/v1/completions",
                {"prompt": [19739] * args.long_context_tokens, "max_tokens": 2},
            )
        )
    reports = []
    for name, endpoint, fields in cases:
        if args.case and name not in args.case:
            continue
        payload = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens, **fields}
        result = stream_request(args.base_url, endpoint, payload, args.timeout)
        result.update(name=name, request=payload, dummy_weights=args.dummy)
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["status"] != 200 or not (result["text"] or result["reasoning"]):
            raise ValueError(f"{name} did not produce nonempty HTTP 200 output")
        reports.append({key: value for key, value in result.items() if key not in ("chunks", "request")})
        print(json.dumps(reports[-1]), flush=True)
    if not reports:
        raise ValueError("No cases selected; image and context cases require their input options")
    (args.output / "report.json").write_text(
        json.dumps({"runtime_passed": True, "quality_assessed": False, "cases": reports}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
