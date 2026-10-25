# SPDX-License-Identifier: Apache-2.0
"""Short real-weight checks only; the user owns the full 160K Kilo test."""

import json
import subprocess
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient
from tools.minimax_m3.tool_probe import chat_request, check_calls, tool


def main():
    root = Path("/workspace/m31-160k-smoke")
    root.mkdir(exist_ok=True)
    client = ResidentClient("http://127.0.0.1:8000", 6)
    client.request("/resident/maintenance")
    try:
        before = client.rpc("resident_status")
        (root / "before.json").write_text(json.dumps(before, indent=2) + "\n")
        models = client.request("/v1/models", method="GET")
        assert models["data"][0]["max_model_len"] == 163840
        (root / "models.json").write_text(json.dumps(models, indent=2) + "\n")
        subprocess.run(
            [
                "python",
                "-m",
                "tools.minimax_m3.coherence_probe",
                "--extended",
                "--image",
                "/workspace/red.png",
                "--case",
                "greeting",
                "--case",
                "image",
                "--case",
                "sparse_retrieval",
                "--max-tokens",
                "128",
                "--output",
                str(root / "coherence"),
            ],
            check=True,
        )
        payload = {
            "model": "minimax-m31-flash",
            "messages": [{"role": "user", "content": "Call lookup_code for city Paris. Do not guess the tool result."}],
            "tools": [tool("lookup_code", {"city": {"type": "string"}})],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = chat_request(client.base_url, payload)
        (root / "tool.json").write_text(json.dumps(result, indent=2) + "\n")
        assert check_calls(result, [("lookup_code", {"city": "Paris"})])
        after = client.rpc("resident_status")
        (root / "after.json").write_text(json.dumps(after, indent=2) + "\n")
        fields = ("rank", "pid", "weight_storage_digest", "generation", "digest")
        assert sorted(tuple(r[k] for k in fields) for r in before) == sorted(tuple(r[k] for k in fields) for r in after)
        (root / "report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "configured_context": 163840,
                    "max_tested_prompt_tokens": 3443,
                    "full_context_test": "Deferred to user in Kilo at explicit request",
                    "checks": ["greeting", "red image", "3443-token retrieval", "streamed tool call"],
                    "resident_identity_unchanged": True,
                },
                indent=2,
            )
            + "\n"
        )
        print("SMOKE PASSED; full 160K test intentionally not run", flush=True)
    finally:
        client.request("/resident/maintenance", method="DELETE")


if __name__ == "__main__":
    main()
