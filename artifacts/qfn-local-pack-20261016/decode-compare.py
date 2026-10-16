"""Check the short decode paths, where neither packing mode should execute."""

import json
import subprocess
import sys
import time
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
RUN = BASE / "resident-live-v16"


def control(action, label):
    subprocess.run([sys.executable, str(ROOT / "local-control.py"), action, label], check=True)


if __name__ == "__main__":
    records = []
    try:
        for action in ("off", "on"):
            control(action, f"decode-{action}-before")
            for name, args in (
                ("decode", ["--trials", "2", "--tokens", "256"]),
                ("concurrency", ["--concurrency", "4", "--tokens", "256"]),
            ):
                before = admit(query(), max_temperature_c=85)
                assert before["passed"], before
                subprocess.run(
                    [
                        sys.executable,
                        str(BASE / f"qsa-{name}-benchmark.py"),
                        "--run-root",
                        str(RUN),
                        "--label",
                        f"localpack-{action}-v16",
                        *args,
                    ],
                    check=True,
                )
                records.append(dict(action=action, workload=name, before=before, after=admit(query())))
                (ROOT / "decode-compare-thermal.json").write_text(json.dumps(records, indent=2) + "\n")
                time.sleep(20)
            control("status", f"decode-{action}-after")
    finally:
        control("on", "decode-final-on")
