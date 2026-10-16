"""Record authenticated owner status without exposing control credentials."""

import json
import sys
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
run = ROOT / sys.argv[1]
assert run.parent == ROOT and run.name.startswith("resident-")
client = QfnResidentClient("http://127.0.0.1:8001", run / "control.token", expected_workers=6)
workers = client.rpc("resident_status")
assert len(workers) == 6
assert all(not w["native_failed"] and not w["graphs_dirty"] for w in workers)
assert all(w["custom_collective"]["state"] == "active" for w in workers)
assert client.request("/is_paused", method="GET")["is_paused"] is False
engine = client.request("/qfn_engine_control", {"operation": "status"})
assert not engine["failed"] and not engine["paused"]
report = dict(
    ready=True,
    engine=engine,
    workers=workers,
    models=client.request("/v1/models", method="GET"),
    thermal=admit(query(), devices=tuple(range(6)), max_temperature_c=90),
)
(run / "final-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(ready=True, workers=[w["pid"] for w in workers], thermal=report["thermal"])), flush=True)
