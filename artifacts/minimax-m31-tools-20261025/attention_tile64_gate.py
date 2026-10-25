"""Run isolated kernel benchmarks while the resident service is drained."""

import fcntl
import subprocess
import sys
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient

root = Path("/srv/ai/storage/artifacts/minimax-m31-tools-20261025")
client = ResidentClient("http://127.0.0.1:8000", 6)
with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise RuntimeError("Another operation has already paused the service")
    if client.request("/resident/maintenance", method="GET")["maintenance"]:
        raise RuntimeError("Another operation holds maintenance")
    client.request("/resident/maintenance", method="POST")
    try:
        assert client.request("/pause?mode=wait&clear_cache=false")["status"] == "paused"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.minimax_m3.attention_benchmark",
                "/srv/ai/storage/artifacts/minimax-m31-20261024/layer3",
                "--baseline",
                "/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/minimax_m3.py",
                "--candidate",
                "/workspace/minimax-attention-tile64.py",
                "--device",
                "npu:4",
                "--output",
                str(root / "attention-tile64.json"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.minimax_m3.probe_resident_native",
                "/vllm-workspace/minimax-native-v6/manifest.json",
                "--device",
                "npu:4",
                "--output",
                str(root / "native-v6-standalone.json"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.minimax_m3.native_expert_benchmark",
                "/srv/ai/storage/artifacts/minimax-m31-20261024/expert0",
                "--manifest",
                "/vllm-workspace/minimax-native-v6/manifest.json",
                "--binding",
                "/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so",
                "--device",
                "npu:4",
                "--output",
                str(root / "native-v6-experts.json"),
            ],
            check=True,
        )
    finally:
        client.resume()
        client.request("/resident/maintenance", method="DELETE")
