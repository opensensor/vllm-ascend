"""Fixed-token image inference timing; truncated outputs are NOT scene reports."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from tools.rtsp_scene.capacity import aggregate_metrics
from tools.rtsp_scene.runner import API_TIMEOUT_SECONDS, description_schema, image_reference


def request_body(capture: Path, prompt: str, tokens: int, transport="base64", frame_root=None) -> bytes:
    manifest = json.loads((capture / "capture.json").read_text())
    content = [{"type": "text", "text": prompt}]
    for frame in manifest["frames"]:
        content.extend(
            [
                {"type": "text", "text": f"Frame {frame['id']}, relative time {frame['seconds']:.3f} seconds:"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_reference(capture / frame["file"], transport, frame_root)},
                },
            ]
        )
    return json.dumps(
        {
            "model": "qwen3-vl-scene",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": tokens,
            "ignore_eos": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_description",
                    "strict": True,
                    "schema": description_schema(len(manifest["frames"])),
                },
            },
        }
    ).encode()


def validate_transport(transport: str, frame_root: Path | None, stagger_seconds: float) -> None:
    if transport == "file" and (frame_root is None or not frame_root.is_absolute()):
        raise ValueError("File transport requires an absolute shared frame root")
    if not 0 <= stagger_seconds < float("inf"):
        raise ValueError("Stagger delay must be finite and nonnegative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Natural-description capacity run with prompts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003/v1/chat/completions")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--frame-transport", choices=("base64", "file"), default="base64")
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--stagger-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.tokens <= 0 or args.warmups < 0 or args.output.exists():
        parser.error("Use positive token count and a fresh output directory")
    try:
        validate_transport(args.frame_transport, args.frame_root, args.stagger_seconds)
    except ValueError as error:
        parser.error(str(error))
    os.umask(0o077)
    args.output.mkdir(parents=True)
    sources = json.loads((args.reference / "summary.json").read_text())
    first = sources[0]
    for index in range(args.warmups):
        capture = Path(first["captures"][index % len(first["captures"])])
        name = f"c{first['concurrency']}-r0-{index % first['concurrency']}"
        prompt = (args.reference / name / "request-prompt.txt").read_text()
        body = request_body(capture, prompt, args.tokens, args.frame_transport, args.frame_root)
        request = urllib.request.Request(args.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        if payload["usage"]["completion_tokens"] != args.tokens:
            raise ValueError("Warmup did not complete the fixed token budget")
    summaries = []
    for group in sources:
        concurrency = group["concurrency"]
        before = aggregate_metrics([args.endpoint])
        batches, requests = [], []
        for iteration in range(len(group["batch_seconds"])):
            jobs = []
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for index in range(concurrency):
                    name = f"c{concurrency}-r{iteration}-{index}"
                    capture = Path(group["captures"][index % len(group["captures"])])
                    prompt = (args.reference / name / "request-prompt.txt").read_text()
                    body = request_body(capture, prompt, args.tokens, args.frame_transport, args.frame_root)

                    def send(body=body, name=name, delay=index * args.stagger_seconds):
                        if delay:
                            time.sleep(delay)
                        start = time.monotonic()
                        req = urllib.request.Request(
                            args.endpoint, data=body, headers={"Content-Type": "application/json"}
                        )
                        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as response:
                            payload = json.load(response)
                        record = {
                            "name": name,
                            "body_sha256": hashlib.sha256(body).hexdigest(),
                            "body_bytes": len(body),
                            "latency_seconds": time.monotonic() - start,
                            "usage": payload["usage"],
                            "finish_reason": payload["choices"][0]["finish_reason"],
                            "text": payload["choices"][0]["message"]["content"],
                            "submission_delay_seconds": delay,
                        }
                        if record["usage"]["completion_tokens"] != args.tokens:
                            raise ValueError("The fixed-token server workload did not complete its budget")
                        (args.output / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                        return record

                    jobs.append(pool.submit(send))
                requests.extend(job.result() for job in jobs)
            batches.append(time.monotonic() - started)
        after = aggregate_metrics([args.endpoint])
        means = {}
        for metric in ("request_prefill_time_seconds", "request_decode_time_seconds", "request_queue_time_seconds"):
            count = after.get(metric + "_count", 0) - before.get(metric + "_count", 0)
            total = after.get(metric + "_sum", 0) - before.get(metric + "_sum", 0)
            means[metric] = total / count if count else None
        summaries.append(
            {
                "concurrency": concurrency,
                "requests": len(requests),
                "batch_seconds": batches,
                "fixed_completion_tokens": args.tokens,
                "warmup_requests_excluded": args.warmups,
                "server_mean_seconds": means,
                "diagnostic_only": True,
                "note": "Truncated outputs are not accepted scene descriptions.",
                "frame_transport": args.frame_transport,
                "stagger_seconds": args.stagger_seconds,
            }
        )
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
