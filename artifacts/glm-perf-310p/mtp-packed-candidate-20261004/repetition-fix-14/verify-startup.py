"""Run the same quality/concurrency/tool checks after ordinary startup."""

import subprocess
import sys
import time
import urllib.request

for attempt in range(600):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise TimeoutError("Server did not become healthy")
print("Server healthy; running qualification", flush=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "tools.glm_perf.suite",
        "--model",
        "glm53-flash-selective-w3",
        "--workload",
        "quality",
        "--workload",
        "fault4",
        "--workload",
        "tool",
        "--output",
        "/home/matteius/experiments/glm-w3-20261004/repetition-fix-14/startup_gate.jsonl",
    ],
    check=True,
)
