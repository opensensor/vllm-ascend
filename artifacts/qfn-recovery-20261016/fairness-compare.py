"""Compare first-sample prefill fairness while keeping activation fusion enabled."""

import json
import subprocess
import sys
import time
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
records = []
try:
    for index, mode in enumerate(("baseline", "candidate")):
        subprocess.run([sys.executable, str(ROOT / "budget-compare.py"), mode, f"fairness-{index}-{mode}"], check=True)
        before = admit(query(), max_temperature_c=85)
        if not before["passed"]:
            raise RuntimeError("fairness comparison rejected by benchmark guard")
        label = "fairness-" + mode
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "double-prefill.py"),
                "--paragraphs",
                "80",
                "100",
                "--pairs",
                "1",
                "--label",
                label,
            ],
            check=True,
        )
        records.append(
            dict(
                mode=mode,
                before=before,
                after=admit(query()),
                requests=json.loads((ROOT / (label + ".json")).read_text()),
            )
        )
        (ROOT / "fairness-comparison.json").write_text(json.dumps(records, indent=2) + "\n")
        if index == 0:
            time.sleep(20)
finally:
    subprocess.run(
        [sys.executable, str(ROOT / "budget-compare.py"), "candidate", "fairness-final-candidate"], check=True
    )
