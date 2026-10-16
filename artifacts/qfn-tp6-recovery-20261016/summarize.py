# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

root = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
run = root / "resident-v7"
files = [
    "hang-v5/rank0-gdb.txt",
    "hang-v5/rank0-python.txt",
    "hc-producer-v3/provenance.json",
    "restore.py",
    "install-hc-producer.py",
    "resident-v7/hc-exact-gate.json",
    "resident-v7/stability.jsonl",
    "resident-v7/private-state-gate.json",
    "mamba-resident-v2/provenance.json",
    "resident-v7/state-install-v3.json",
    "resident-v7/state-six-decode-gate.json",
    "resident-v7/state-on.json",
    "resident-v7/state-retained-off.json",
    "resident-v7/final-smoke.json",
]
files += [str(p.relative_to(root)) for p in run.glob("state-ab-*.jsonl")]
result = {
    "host": "192.168.53.187",
    "run": "resident-v7",
    "files": {
        name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(), "bytes": (root / name).stat().st_size}
        for name in files
        if (root / name).exists()
    },
}
result["hc"] = json.loads((run / "hc-exact-gate.json").read_text())
result["retained_state"] = json.loads((run / "state-retained-off.json").read_text())["commit"]
result["state_verification"] = json.loads((run / "state-on.json").read_text())["commit"]
result["stability"] = [
    {k: v for k, v in json.loads(line).items() if k != "text"}
    for line in (run / "stability.jsonl").read_text().splitlines()
]
result["ab"] = {}
for label in ("a3", "b3", "a4", "b4"):
    p = run / f"state-ab-{label}.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    result["ab"][label] = [{k: v for k, v in row.items() if k != "text"} for row in rows]
(root / "recovery-summary.json").write_text(json.dumps(result, indent=2) + "\n")
