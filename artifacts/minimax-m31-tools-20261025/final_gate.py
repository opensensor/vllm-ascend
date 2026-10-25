"""Verify the installed tool-enabled service and a cold 16K retrieval."""

import fcntl
import json
import subprocess
import sys
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.minimax_m3.resident import candidate_source

root = Path("/srv/ai/storage/artifacts/minimax-m31-tools-20261025")
client = ResidentClient("http://127.0.0.1:8000", 6)
with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not client.request("/resident/maintenance", method="GET")["maintenance"]
    client.request("/resident/maintenance", method="POST")
    try:
        subprocess.run(
            [sys.executable, "-m", "tools.minimax_m3.tool_probe", "--output", str(root / "final-tools")], check=True
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.minimax_m3.coherence_probe",
                "--output",
                str(root / "final-short"),
                "--max-tokens",
                "128",
                "--case",
                "greeting",
                "--case",
                "image",
                "--image",
                "/workspace/red.png",
            ],
            check=True,
        )
        # Clear both scheduler prefix state and physical caches before measuring.
        workers = client.switch(
            Control(uuid.uuid4().hex, "direct-target", "native_v6", candidate_source(native_name="minimax_w4_v6"))
        )
        (root / "final-before-16k.json").write_text(json.dumps(workers, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.minimax_m3.coherence_probe",
                "--output",
                str(root / "final-16k"),
                "--max-tokens",
                "128",
                "--extended",
                "--case",
                "sparse_retrieval",
                "--records",
                "880",
            ],
            check=True,
        )
        (root / "final-workers.json").write_text(json.dumps(client.rpc("resident_status"), indent=2) + "\n")
    finally:
        if not client.request("/is_paused", method="GET")["is_paused"]:
            client.request("/resident/maintenance", method="DELETE")
