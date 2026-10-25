"""Compare real serving, retaining the candidate only after coherence gates."""

import fcntl
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.resident import candidate_source

root = Path("/srv/ai/storage/artifacts/minimax-m31-tools-20261025")
client = ResidentClient("http://127.0.0.1:8000", 6)


def save(name, value):
    (root / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def probe(name, extra):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.minimax_m3.coherence_probe",
            "--output",
            str(root / name),
            "--max-tokens",
            "128",
            *extra,
        ],
        check=True,
    )


with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if client.request("/resident/maintenance", method="GET")["maintenance"]:
        raise RuntimeError("Another controller holds maintenance")
    client.request("/resident/maintenance", method="POST")
    passed = False
    try:
        save("baseline-before", client.switch(Control(uuid.uuid4().hex, "direct-target")))
        probe(
            "baseline-coherence",
            [
                "--extended",
                "--case",
                "greeting",
                "--case",
                "france",
                "--case",
                "addition",
                "--case",
                "sparse_retrieval",
            ],
        )
        save(
            "native-v6-load",
            client.load_native(
                NativeManifest(json.loads(Path("/vllm-workspace/minimax-native-v6/manifest.json").read_text()))
            ),
        )
        source = candidate_source(Path("/workspace/minimax-attention-tile64.py").read_text(), "minimax_w4_v6")
        started = time.monotonic()
        receipts = client.switch(Control(uuid.uuid4().hex, "direct-target", "tile64_native_v6", source))
        save("combined-switch", {"elapsed_seconds": time.monotonic() - started, "workers": receipts})
        probe("combined-coherence", ["--extended", "--image", "/workspace/red.png"])
        # 430 inventory records reproduce the existing 7943-token retrieval.
        probe("combined-8k", ["--extended", "--case", "sparse_retrieval", "--records", "430"])
        save("combined-after", client.rpc("resident_status"))
        passed = True
    finally:
        if not passed:
            save("failed-candidate-rollback", client.switch(Control(uuid.uuid4().hex, "direct-target")))
        if not client.request("/is_paused", method="GET")["is_paused"]:
            client.request("/resident/maintenance", method="DELETE")
    save("serving-candidate", {"passed": passed})
