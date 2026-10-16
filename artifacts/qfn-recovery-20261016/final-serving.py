"""Final bounded decode regressions and retained-owner audit."""

import json
import subprocess
import sys
import time
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
RUN = BASE / "resident-live-v16"
records = []
for name, args in (
    ("decode", ["--trials", "2", "--tokens", "256"]),
    ("concurrency", ["--concurrency", "4", "--tokens", "256"]),
):
    before = admit(query(), max_temperature_c=85)
    if not before["passed"]:
        raise RuntimeError("final serving benchmark admission rejected")
    subprocess.run(
        [
            sys.executable,
            str(BASE / f"qsa-{name}-benchmark.py"),
            "--run-root",
            str(RUN),
            "--label",
            "activation-v16",
            *args,
        ],
        check=True,
    )
    records.append(dict(name=name, before=before, after=admit(query())))
    (ROOT / "final-thermal.json").write_text(json.dumps(records, indent=2) + "\n")
    if name == "decode":
        time.sleep(15)
subprocess.run(
    [sys.executable, str(ROOT / "control.py"), str(ROOT / "audit.py"), str(ROOT / "final-audit")], check=True
)
print("FINAL SERVING GATES COMPLETE", flush=True)
