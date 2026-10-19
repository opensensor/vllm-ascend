"""Benchmark complete chat answers with natural EOS and GPU-board energy."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from tools.dflash_research.energy_bench import PowerMonitor, integrate_power

MAX_OUTPUT_TOKENS = 1536

PROMPTS = {
    "code": (
        "Implement a Python LRUCache using OrderedDict with capacity validation, "
        "get and put methods, and a short runnable example. Explain time complexity "
        "in under 100 words."
    ),
    "sql": (
        "In PostgreSQL, write a query computing each customer's rolling revenue "
        "over the previous seven calendar days, including the current date, from "
        "orders(customer_id, order_date date, amount numeric). Explain why ROWS 6 "
        "PRECEDING is not equivalent. Keep the answer under 250 words."
    ),
    "math": (
        "Derive the probability of at least one collision after n independent draws "
        "from m equally likely values. Include the n > m case, the birthday "
        "approximation, and its assumptions. Keep the answer under 250 words."
    ),
    "debug": (
        "A multithreaded work queue loses items during shutdown. Describe the most "
        "likely race and give a concrete close, drain, and join protocol that prevents "
        "loss. Keep the answer under 300 words."
    ),
    "story": (
        "Write a self-contained 250–350 word story about an engineer debugging a "
        "radio telescope during a storm. Include dialogue and a clear ending."
    ),
    "translation": (
        "Translate 'The engineer reviewed the design before the test' into Spanish. "
        "Explain two alternative word choices and their nuances. Reply in under 150 words."
    ),
    "systems": (
        "A service needs 20 ms of CPU time and 80 ms waiting for storage per request. "
        "Assume one CPU core and ample storage concurrency, with no other contention. "
        "Give single-request latency and throughput, maximum throughput, and the "
        "approximate concurrency needed to reach it. Keep the answer under 180 words."
    ),
}


def build_payload(prompt: str, max_tokens: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Answer the user's request directly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_completion(response: dict) -> dict:
    choice = response["choices"][0]
    message = choice["message"]
    completion_tokens = response["usage"]["completion_tokens"]
    if completion_tokens <= 0:
        raise ValueError("chat response has no completion tokens")
    if not isinstance(message.get("content"), str):
        raise ValueError("chat response has no text content")
    return {
        "content": message["content"],
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice["finish_reason"],
        "completion_tokens": completion_tokens,
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "timings": response["timings"],
    }


def complete(url: str, payload: dict) -> tuple[dict, float]:
    request = Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urlopen(request, timeout=300) as stream:
        response = json.load(stream)
    return response, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    complete(args.url, build_payload("Reply with the word ready.", 64))
    records = []
    for name, prompt in PROMPTS.items():
        payload = build_payload(prompt, MAX_OUTPUT_TOKENS)
        with PowerMonitor(args.gpu_index) as monitor:
            response, wall_seconds = complete(args.url, payload)
        parsed = extract_completion(response)
        record = {
            "arm": args.arm,
            "prompt": name,
            "request": payload,
            "response": response,
            "wall_seconds": wall_seconds,
            "gpu_energy_joules": integrate_power(monitor.samples),
            "power_samples": len(monitor.samples),
            **parsed,
        }
        records.append(record)
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        print(
            name,
            parsed["completion_tokens"],
            parsed["finish_reason"],
            round(parsed["timings"]["predicted_per_second"], 2),
            flush=True,
        )


if __name__ == "__main__":
    main()
