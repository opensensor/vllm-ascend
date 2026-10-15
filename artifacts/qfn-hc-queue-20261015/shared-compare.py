"""Qualify narrowed residuals in serving and alternate against retained fusion."""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015")
RUN = "/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v15"
PROFILE = "/srv/ai/storage/artifacts/qfn-mixed-opt-20261015/profile-live.py"


def run(*args):
    if str(args[0]) == PROFILE:
        subprocess.run([sys.executable, "-m", "tools.qwen4exp.thermal_guard"], check=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


if __name__ == "__main__":
    run(ROOT / "shared-control.py", "verify", ROOT / "shared-verify")
    try:
        run(PROFILE, "--run-root", RUN, "--label", "hc-shared-shadow", "--mode", "spans", "--paragraphs", 450)
        run(
            PROFILE,
            "--run-root",
            RUN,
            "--label",
            "hc-shared-large-shadow",
            "--mode",
            "spans",
            "--paragraphs",
            30,
            "--prefill-only",
        )
        for index, mode in enumerate(("baseline", "on", "baseline", "on")):
            label = f"hc-shared-{index}-{mode}"
            run(ROOT / "shared-control.py", mode, ROOT / label)
            run(PROFILE, "--run-root", RUN, "--label", label, "--mode", "spans", "--paragraphs", 450)
    finally:
        run(ROOT / "shared-control.py", "baseline", ROOT / "shared-restore-baseline")
