# SPDX-License-Identifier: Apache-2.0
"""Run isolated real-weight checks on the loaded 48K/512-token profile."""

import json
import subprocess
from pathlib import Path

from tools.glm_perf.resident_harness import ResidentClient


def main():
    root = Path("/workspace/m31-context-perf")
    root.mkdir(exist_ok=True)
    client = ResidentClient("http://127.0.0.1:8000", 6, timeout=1800)
    client.request("/resident/maintenance")
    try:
        before = client.rpc("resident_status")
        (root / "before.json").write_text(json.dumps(before, indent=2) + "\n")
        assert all("minimax_w4_v7" in row["native_loaded"] for row in before)
        (root / "models.json").write_text(json.dumps(client.request("/v1/models", method="GET"), indent=2) + "\n")
        for name, options in (
            ("coherence", ["--extended", "--image", "/workspace/red.png"]),
            ("cold-16k", ["--extended", "--case", "sparse_retrieval", "--records", "880"]),
            ("cold-48k", ["--extended", "--case", "sparse_retrieval", "--records", "2600"]),
        ):
            client.request("/pause?mode=wait&clear_cache=true")
            client.rpc("resident_reset")
            client.request("/resume")
            print("START " + name, flush=True)
            subprocess.run(
                [
                    "python",
                    "-m",
                    "tools.minimax_m3.coherence_probe",
                    "--max-tokens",
                    "128",
                    "--output",
                    str(root / name),
                    *options,
                ],
                check=True,
            )
        print("START tools", flush=True)
        subprocess.run(["python", "-m", "tools.minimax_m3.tool_probe", "--output", str(root / "tools")], check=True)
        after = client.rpc("resident_status")
        (root / "after.json").write_text(json.dumps(after, indent=2) + "\n")
        fields = ("rank", "pid", "weight_storage_digest", "generation", "digest")
        assert sorted(tuple(r[k] for k in fields) for r in before) == sorted(tuple(r[k] for k in fields) for r in after)
        print("ALL PASSED; worker/storage identities unchanged", flush=True)
    finally:
        client.request("/resident/maintenance", method="DELETE")


if __name__ == "__main__":
    main()
