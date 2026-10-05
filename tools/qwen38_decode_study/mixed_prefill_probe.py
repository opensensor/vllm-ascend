"""Measure decode pauses when cold long prompts join an active 310P batch.

This is a serving probe, not a model-quality benchmark. It sends exact token-ID
prompts so distinct sessions cannot accidentally reuse one another's prefix.
"""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def prompt(tokenizer, length: int, label: str) -> list[int]:
    head = tokenizer.encode(
        f"<|im_start|>user\nSession {label}. Write a long testing guide.\n",
        add_special_tokens=False,
    ).ids
    filler = tokenizer.encode(
        "Archive note: project source, logs, and test records are ordinary background.\n",
        add_special_tokens=False,
    ).ids
    tail = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
    remaining = length - len(head) - len(tail)
    if remaining < 0:
        raise ValueError("prompt length is shorter than its header and tail")
    return head + (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + tail


def stream(
    base_url: str,
    model: str,
    tokens: list[int],
    output_tokens: int,
    timeout: int,
    first_token: threading.Event | None = None,
) -> dict:
    payload = {
        "model": model,
        "prompt": tokens,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    arrivals = []
    usage = None
    finished = False
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                finished = True
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunk = choice.get("text") or ""
                if chunk:
                    arrivals.append(time.monotonic())
                    digest.update(chunk.encode())
                    if first_token is not None:
                        first_token.set()
    end = time.monotonic()
    if not finished or usage is None or not arrivals:
        raise AssertionError("stream did not finish with usage and output")
    if usage["prompt_tokens"] != len(tokens):
        raise AssertionError("server changed the exact prompt-token count")
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {
        "start": start,
        "first": arrivals[0],
        "end": end,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "completion_tokens": usage["completion_tokens"],
        "time_to_first_token_s": arrivals[0] - start,
        "elapsed_s": end - start,
        "decode_tokens_per_s": (usage["completion_tokens"] - 1) / (arrivals[-1] - arrivals[0]),
        "max_chunk_gap_s": max(gaps, default=0),
        "median_chunk_gap_s": statistics.median(gaps) if gaps else 0,
        "chunk_arrivals": arrivals,
        "output_sha256": digest.hexdigest(),
    }


def profile_control(base_url: str, action: str) -> None:
    request = urllib.request.Request(base_url + "/" + action, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=60):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen38-w4-pipeline-c1-dispatch")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--short-prompt-tokens", type=int, default=256)
    parser.add_argument("--long-prompt-tokens", type=int, default=8192)
    parser.add_argument("--long-requests", type=int, default=3)
    parser.add_argument("--short-output-tokens", type=int, default=512)
    parser.add_argument("--long-output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--run-id", default=None, help="Reuse the same token prompts across restarted servers")
    parser.add_argument("--profile", action="store_true", help="Profile the mixed request phase via the server API")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    run_id = args.run_id or uuid.uuid4().hex
    short_a = prompt(tokenizer, args.short_prompt_tokens, run_id + "-serial")
    short_b = prompt(tokenizer, args.short_prompt_tokens, run_id + "-mixed")
    long_prompts = [
        prompt(tokenizer, args.long_prompt_tokens, f"{run_id}-long-{index}") for index in range(args.long_requests)
    ]
    serial = stream(args.base_url, args.model, short_a, args.short_output_tokens, args.timeout)
    first_token = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.long_requests + 1) as pool:
        short_future = pool.submit(
            stream, args.base_url, args.model, short_b, args.short_output_tokens, args.timeout, first_token
        )
        if not first_token.wait(timeout=args.timeout):
            raise TimeoutError("short request never produced its first token")
        if args.profile:
            profile_control(args.base_url, "start_profile")
        long_launch = time.monotonic()
        try:
            long_futures = [
                pool.submit(stream, args.base_url, args.model, tokens, args.long_output_tokens, args.timeout)
                for tokens in long_prompts
            ]
            mixed = short_future.result()
            long_results = [future.result() for future in long_futures]
        finally:
            if args.profile:
                profile_control(args.base_url, "stop_profile")
    before = [b - a for a, b in zip(mixed["chunk_arrivals"], mixed["chunk_arrivals"][1:]) if b <= long_launch]
    during = [b - a for a, b in zip(mixed["chunk_arrivals"], mixed["chunk_arrivals"][1:]) if b > long_launch]
    result = {
        "run_id": run_id,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "serial": serial,
        "mixed": mixed,
        "long_launch": long_launch,
        "long_requests": long_results,
        "short_gap_before_long_s": max(before, default=0),
        "short_gap_after_long_s": max(during, default=0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "serial_decode_tps": round(serial["decode_tokens_per_s"], 2),
                "mixed_decode_tps": round(mixed["decode_tokens_per_s"], 2),
                "max_gap_after_long_s": round(result["short_gap_after_long_s"], 2),
                "long_ttft_s": [round(item["time_to_first_token_s"], 2) for item in long_results],
                "long_cached_tokens": [item["cached_tokens"] for item in long_results],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
