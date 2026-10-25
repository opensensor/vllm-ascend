# SPDX-License-Identifier: Apache-2.0
"""Run bounded real-model qualification; never exercise full 160K context."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from tools.minimax_m3.tool_probe import chat_request, check_calls, tool


def run(*args):
    print("RUN", *args, flush=True)
    subprocess.run(args, check=True)


parser = argparse.ArgumentParser()
parser.add_argument("--full-verification", action="store_true", help="Optional coherence/tool suite; off by default")
args = parser.parse_args()

for _ in range(120):
    if subprocess.run(["systemctl", "is-active", "--quiet", "minimax-m31"]).returncode == 0:
        break
    time.sleep(5)
else:
    raise RuntimeError("service did not recover")
run(
    "python",
    "-m",
    "tools.minimax_m3.coherence_probe",
    "--case",
    "greeting",
    "--output",
    "/workspace/m31-recovered-greeting",
)
run(
    "python",
    "-m",
    "tools.minimax_m3.resident_benchmark",
    "--manifest",
    "/vllm-workspace/minimax-native-v9/manifest.json",
    "--iterations",
    "10",
    "--output",
    "/workspace/m31-v9-fresh-benchmark.json",
)
run("python", "/workspace/m31-serve-benchmark.py", "--label", "v7", "--output", "/workspace/m31-v7-fresh-decode")
run(
    "python",
    "-m",
    "tools.minimax_m3.activate_equivalent",
    "--manifest",
    "/vllm-workspace/minimax-native-v9/manifest.json",
    "--evidence",
    "/workspace/m31-v9-fresh-benchmark.json",
    "--output",
    "/workspace/m31-v9-activation.json",
)
run("python", "/workspace/m31-serve-benchmark.py", "--label", "v9", "--output", "/workspace/m31-v9-decode")
if args.full_verification:
    run(
        "python",
        "-m",
        "tools.minimax_m3.coherence_probe",
        "--extended",
        "--image",
        "/workspace/red.png",
        "--max-tokens",
        "128",
        "--output",
        "/workspace/m31-v9-coherence",
    )
    result = chat_request(
        "http://127.0.0.1:8000",
        {
            "model": "minimax-m31-flash",
            "messages": [{"role": "user", "content": "Call lookup_code for city Paris. Do not guess the tool result."}],
            "tools": [tool("lookup_code", {"city": {"type": "string"}})],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    Path("/workspace/m31-v9-tool.json").write_text(json.dumps(result, indent=2) + "\n")
    assert check_calls(result, [("lookup_code", {"city": "Paris"})])
print("PERFORMANCE BENCHMARK COMPLETE", flush=True)
