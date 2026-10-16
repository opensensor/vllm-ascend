"""Bounded thermal admission and drained private qualification."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.qwen4exp.thermal_guard import admit, query
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
p = argparse.ArgumentParser()
p.add_argument("phase")
a = p.parse_args()
thermal = admit(query())
(ROOT / (a.phase + "-thermal-before.json")).write_text(json.dumps(thermal, indent=2) + "\n")
assert thermal["passed"], thermal
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
source = (
    "PHASE = "
    + repr(a.phase.removesuffix("-v2"))
    + "\nVERSION = "
    + str(2 if a.phase.endswith("-v2") else 1)
    + "\n"
    + (ROOT / "gate.py").read_text()
)
result = apply(client, source, ROOT / (a.phase + "-gate"))
receipt = result["applied"]["applied"]
(ROOT / (a.phase + "-results.json")).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
(ROOT / (a.phase + "-thermal-after.json")).write_text(json.dumps(admit(query()), indent=2) + "\n")
