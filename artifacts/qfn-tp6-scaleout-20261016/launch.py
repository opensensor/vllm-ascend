"""Launch a fresh qualified TP6 session; preserve all previous run evidence."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
version = sys.argv[1]
assert version.isalnum()
run_root = ROOT / ("resident-" + version)
assert not run_root.exists()
gate = json.loads((ROOT / "collective-graph-full/result.json").read_text())
assert gate["passed"] and gate["graphs_retired"]
assert all(rank["ownership_released"] for rank in gate["ranks"])
library = ROOT / "ple-host-six-v1/libple_host_gather.so"
digest = hashlib.sha256(library.read_bytes()).hexdigest()
for rank in range(6):
    transport = json.loads((ROOT / f"ple-transport-rank{rank}.json").read_text())
    assert transport["passed"] and transport["rank"] == rank
    assert transport["library_sha256"] == digest and transport["owners_closed"] == 14
command = ["bash", str(ROOT / "start-tp6.sh"), "--run-root", str(run_root)]
for device in range(6):
    command.extend(["--dma-reserve-device", str(device)])
with (ROOT / ("launch-" + version + ".log")).open("w") as log:
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
(ROOT / "api.pid").write_text(str(process.pid) + "\n")
with (ROOT / ("thermal-" + version + ".jsonl")).open("w") as log:
    monitor = subprocess.Popen(
        ["bash", str(ROOT / "run.sh"), str(ROOT / "monitor.py"), str(run_root), str(process.pid)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
with (ROOT / ("affinity-" + version + ".json")).open("w") as log:
    affinity = subprocess.Popen(
        ["python3", str(ROOT / "affinity.py"), str(process.pid)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(json.dumps(dict(api_pid=process.pid, monitor_pid=monitor.pid, affinity_pid=affinity.pid, run_root=str(run_root))))
