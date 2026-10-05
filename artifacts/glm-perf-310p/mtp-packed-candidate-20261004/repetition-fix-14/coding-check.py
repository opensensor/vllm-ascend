"""Bounded coding completion for manual coherence review."""

import json
import time
import urllib.request
from pathlib import Path

payload = {
    "model": "glm53-flash-selective-w3",
    "messages": [
        {
            "role": "user",
            "content": (
                "Write a Python function stable_unique(items) that removes duplicate strings "
                "while preserving their first occurrence order. Include three assert examples, "
                "including empty input. Return only Python code."
            ),
        }
    ],
    "temperature": 0,
    "seed": 42,
    "max_tokens": 256,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
start = time.monotonic()
request = urllib.request.Request(
    "http://127.0.0.1:8001/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)
result["elapsed_seconds"] = time.monotonic() - start
Path("/home/matteius/experiments/glm-w3-20261004/repetition-fix-14/coding-response.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
