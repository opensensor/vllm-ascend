# SPDX-License-Identifier: Apache-2.0
"""Compare decode after first output, excluding queue and prefill time."""

import argparse
import json
from pathlib import Path

from tools.minimax_m3.tool_probe import chat_request

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
for repeat in range(3):
    payload = {
        "model": "minimax-m31-flash",
        "messages": [
            {
                "role": "user",
                "content": "Write the integers from 1 to 100 in ascending order, separated by commas. No explanation.",
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = chat_request("http://127.0.0.1:8000", payload)
    (args.output / f"request-{repeat}.json").write_text(json.dumps(result, indent=2) + "\n")
    tokens = result["usage"]["completion_tokens"]
    elapsed = result["elapsed_seconds"] - result["first_output_seconds"]
    record = {
        "repeat": repeat,
        "output_tokens": tokens,
        "after_first_seconds": elapsed,
        "tokens_per_second_after_first": (tokens - 1) / elapsed,
        "first_output_seconds": result["first_output_seconds"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    records.append(record)
    print(json.dumps(record), flush=True)
(args.output / "report.json").write_text(json.dumps({"label": args.label, "cases": records}, indent=2) + "\n")
