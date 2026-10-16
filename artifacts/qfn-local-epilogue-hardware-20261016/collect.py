"""Collect small immutable build and runtime receipts without credentials."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


builds = {}
for name in ("build", "build-v2"):
    directory = ROOT / name
    builds[name] = {p.name: digest(p) for p in directory.iterdir() if p.suffix in (".cpp", ".h", ".bin")}
builds["tested_python"] = {name: digest(ROOT / name) for name in ("qfn_route_finalize.py", "local_route_epilogue.py")}
(ROOT / "hardware-builds.json").write_text(json.dumps(builds, indent=2) + "\n")
decode = {}
for action in ("off", "on"):
    decode[action] = {}
    for kind in ("decode-benchmark", "concurrency"):
        path = BASE / "qsa-resident-v1" / f"{kind}-localepilogue-{action}-v16.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for row in data.get("trials", []) + data.get("requests", []):
            row.pop("response", None)
        for phase in ("before", "after"):
            if phase in data:
                data[phase] = [
                    {
                        k: r[k]
                        for k in ("rank", "pid", "weight_storage_digest", "graphs_dirty", "native_failed")
                        if k in r
                    }
                    for r in data[phase]
                ]
        for rank in data.get("graph_inventory", []):
            for graph in rank.get("graphs", []):
                graph.pop("segments", None)
        decode[action][kind] = data
(ROOT / "decode-results.json").write_text(json.dumps(decode, indent=2) + "\n")
for name in ("final-audit",):
    path = ROOT / name / "receipt.json"
    if path.exists():
        data = json.loads(path.read_text())
        compact = {k: data[k] for k in ("applied", "resume")}
        (ROOT / (name + ".json")).write_text(json.dumps(compact, indent=2) + "\n")
