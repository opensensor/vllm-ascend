"""Wait for this API process's six workers, then apply the qualified CPU masks."""

import subprocess
import sys
import time
from pathlib import Path

api = int(sys.argv[1])
root = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")


def children(pid):
    p = Path(f"/proc/{pid}/task/{pid}/children")
    return [int(item) for item in p.read_text().split()] if p.exists() else []


def command(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return b""


for _ in range(600):
    api_command = command(api)
    if b"start-tp6.sh" in api_command:
        time.sleep(1)
        continue
    assert b"qfn_resident_server" in api_command, "API process exited or changed identity"
    engines = [pid for pid in children(api) if command(pid).startswith(b"VLLM::EngineCore")]
    if len(engines) == 1:
        candidates = children(engines[0])
        workers = [
            [pid for pid in candidates if command(pid).startswith(f"VLLM::Worker_TP{i}_EP{i}".encode())]
            for i in range(6)
        ]
        if all(len(found) == 1 for found in workers):
            subprocess.run(
                [
                    "/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python",
                    "/srv/ai/src/qwen-max-context-20261010/affinity-mtp-tp6.py",
                    "--api-pid",
                    str(api),
                    "--engine-pid",
                    str(engines[0]),
                    "--workers",
                    *[str(found[0]) for found in workers],
                ],
                check=True,
            )
            break
    time.sleep(1)
else:
    raise RuntimeError("Timed out waiting for six workers")
