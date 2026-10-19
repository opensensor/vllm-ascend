"""Measure matched llama.cpp completions and GPU energy for draft sweeps."""

import argparse
import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = {
    "speculation": (
        "Explain speculative decoding to a compiler engineer, including verification and acceptance.\nAnswer:"
    ),
    "python": "Write a Python LRU cache using OrderedDict, with usage and complexity notes.\nAnswer:",
    "sql": "Write SQL for a rolling 7-day revenue total by customer, then explain the window frame.\nAnswer:",
    "debug": (
        "A multithreaded queue loses work during shutdown. Describe likely races "
        "and a robust shutdown protocol.\nAnswer:"
    ),
    "math": (
        "Derive the probability of a collision after n draws from m equally likely values. "
        "Show the approximation and assumptions.\nAnswer:"
    ),
    "systems": (
        "Compare latency and throughput for a service with 20 ms CPU time and 80 ms storage wait. "
        "Discuss concurrency.\nAnswer:"
    ),
    "translation": (
        "Translate into Spanish, then explain two alternative word choices: "
        "The engineer reviewed the design before the test.\nAnswer:"
    ),
    "story": (
        "Write a short story about an engineer debugging a radio telescope during a storm. "
        "Include dialogue and an ending.\nAnswer:"
    ),
    "summary": (
        "Summarize the tradeoffs among caching, batching, and speculative decoding for an LLM serving system.\nAnswer:"
    ),
    "format": "Give a numbered deployment checklist for a small inference service. Explain each step briefly.\nAnswer:",
}


def integrate_power(samples: list[tuple[float, float]]) -> float:
    """Integrate timestamped watts using trapezoids; timestamps are seconds."""
    if len(samples) < 2:
        raise ValueError("at least two power samples are required")
    energy = 0.0
    for (start, start_w), (end, end_w) in zip(samples, samples[1:]):
        if end <= start or start_w < 0 or end_w < 0:
            raise ValueError("power samples must be ordered and nonnegative")
        energy += (end - start) * (start_w + end_w) / 2
    return energy


class PowerMonitor:
    def __init__(self, gpu_index: int, interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def _sample(self) -> None:
        assert self._nvml is not None and self._handle is not None
        power_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
        self.samples.append((time.perf_counter(), power_w))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        assert self._thread is not None and self._nvml is not None
        self._thread.join()
        self._sample()
        self._nvml.nvmlShutdown()


def complete(url: str, prompt: str, max_tokens: int) -> dict:
    request = Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
                "ignore_eos": True,
                "return_tokens": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    if result.get("tokens_predicted") != max_tokens:
        raise RuntimeError(f"expected {max_tokens} output tokens, got {result.get('tokens_predicted')}")
    if len(result.get("tokens", [])) != max_tokens:
        raise RuntimeError(f"expected {max_tokens} token IDs, got {len(result.get('tokens', []))}")
    return {
        "wall_seconds": elapsed,
        "content": result["content"],
        "tokens": result["tokens"],
        "tokens_predicted": result["tokens_predicted"],
        "timings": result["timings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    complete(args.url, PROMPTS["speculation"], 128)
    records = []
    for name, prompt in PROMPTS.items():
        with PowerMonitor(args.gpu_index) as monitor:
            result = complete(args.url, prompt, args.max_tokens)
        samples = monitor.samples
        record = {
            "arm": args.arm,
            "prompt": name,
            "request": {"prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0, "seed": 42},
            "energy_joules": integrate_power(samples),
            "mean_power_w": sum(power for _, power in samples) / len(samples),
            "power_samples": len(samples),
            **result,
        }
        records.append(record)
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        print(
            name,
            round(result["timings"]["predicted_per_second"], 2),
            round(record["energy_joules"] / args.max_tokens, 3),
            flush=True,
        )


if __name__ == "__main__":
    main()
