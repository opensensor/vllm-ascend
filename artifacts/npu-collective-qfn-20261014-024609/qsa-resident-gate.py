"""Bounded live QSA request suite; never starts or stops a serving process."""

import argparse
import json
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    base = client.request(
        "/tokenize",
        {
            "model": "qwen38-flash-next",
            "messages": [{"role": "user", "content": "Explain why a hash table needs collision resolution."}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )["tokens"]
    filler = client.request(
        "/tokenize",
        {"model": "qwen38-flash-next", "prompt": "A hash table stores keys and values in buckets. ", "add_special_tokens": False},
    )["tokens"]
    result = {"workers_before": client.rpc("resident_status"), "trials": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, length, output_tokens in (
        ("short_before", len(base), 64),
        ("cross_boundary", 2030, 64),
        ("above_boundary", 2100, 32),
        ("short_after", len(base), 64),
    ):
        count = length - len(base)
        padding = (filler * ((count + len(filler) - 1) // len(filler)))[:count]
        # Preserve the user role prefix and complete assistant chat suffix.
        prompt = base[:3] + padding + base[3:]
        assert len(prompt) == length
        request = {
            "model": "qwen38-flash-next",
            "prompt": prompt,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "return_token_ids": True,
            "logprobs": 1,
        }
        started = time.monotonic()
        response = client.request("/v1/completions", request)
        elapsed = time.monotonic() - started
        assert response["usage"]["prompt_tokens"] == length
        assert response["usage"]["completion_tokens"] == output_tokens
        record = {
            "label": label,
            "prompt_tokens": length,
            "output_tokens": output_tokens,
            "seconds": elapsed,
            "tokens_per_second": output_tokens / elapsed,
            "request": request,
            "response": response,
        }
        result["trials"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ("request", "response")}), flush=True)
    result["workers_after"] = client.rpc("resident_status")
    result["graph_inventory"] = client.request("/collective_rpc", {"method": "qfn_graph_inventory", "timeout": 60})
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
