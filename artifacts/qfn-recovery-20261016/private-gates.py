"""Run short private operator checks after serving restoration."""

import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.qwen4exp.thermal_guard import admit, query
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
for name in ("swiglu-gate", "activation-layout-gate", "activation-epilogue-gate"):
    admission = admit(query(), max_temperature_c=85)
    print("THERMAL", admission, flush=True)
    if not admission["passed"]:
        raise RuntimeError("private experiment admission rejected")
    receipt = apply(client, (ROOT / (name + ".py")).read_text(), ROOT / name)
    print(name, json.dumps(receipt["applied"]), flush=True)
print("PRIVATE GATES COMPLETE", flush=True)
