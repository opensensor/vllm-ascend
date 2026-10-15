"""Shadow the queue candidate, compare two pairs, and restore the retained path."""

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
    run(ROOT / "control.py", "verify", ROOT / "verify")
    try:
        run(PROFILE, "--run-root", RUN, "--label", "hc-queue-shadow", "--mode", "spans", "--paragraphs", 60)
        for index, mode in enumerate(("draining", "queued", "draining", "queued")):
            label = f"hc-queue-{index}-{mode}"
            run(ROOT / "control.py", mode, ROOT / label)
            run(PROFILE, "--run-root", RUN, "--label", label, "--mode", "spans", "--paragraphs", 450)
    finally:
        run(ROOT / "control.py", "draining", ROOT / "restore-draining")
