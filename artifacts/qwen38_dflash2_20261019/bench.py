"""Repeat fixed greedy completions against a llama.cpp /completion endpoint."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = {
    "explanation": (
        "Explain speculative decoding to a software engineer. Include how the "
        "draft and target models interact, and why speedups depend on acceptance.\nAnswer:"
    ),
    "code": (
        "Write a Python function that merges two sorted lists without using "
        "sort(). Explain its time complexity.\nAnswer:"
    ),
    "analysis": (
        "A web service receives 100 requests per second. Each request needs "
        "20 ms of CPU time and waits 80 ms for storage. Explain the difference "
        "between throughput and latency and estimate the concurrency needed.\nAnswer:"
    ),
}


def complete(url: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "return_tokens": True,
    }
    request = Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    if result.get("tokens_predicted") != max_tokens:
        raise RuntimeError(f"expected {max_tokens} tokens: {result!r}")
    if len(result.get("tokens", [])) != max_tokens:
        raise RuntimeError("Server did not return every generated token ID; parity cannot be checked")
    return {
        "elapsed_s": elapsed,
        "content": result["content"],
        "tokens": result["tokens"],
        "tokens_predicted": result["tokens_predicted"],
        "timings": result["timings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    samples = []
    for name, prompt in PROMPTS.items():
        complete(args.url, prompt, args.max_tokens)  # warmup
        for repeat in range(args.repeats):
            result = complete(args.url, prompt, args.max_tokens)
            samples.append({"prompt": name, "repeat": repeat, **result})
            print(
                name,
                repeat,
                result["timings"]["predicted_per_second"],
                flush=True,
            )
    args.out.write_text(json.dumps(samples, indent=2) + "\n")


if __name__ == "__main__":
    main()
