"""Qualify mixed serving with the retained HC and packing candidates together."""

import subprocess
import sys
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-pack-followup-20261016")


def run(name, *args):
    subprocess.run([sys.executable, str(ROOT / name), *map(str, args)], check=True)


if __name__ == "__main__":
    admission = admit(query(), max_temperature_c=85)
    assert admission["passed"], admission
    run("shared-control.py", "on", ROOT / "shared-retained-on")
    run("pack-control.py", "verify", "pack32-mixed-verify")
    try:
        run("profile-short-mixed.py", "--label", "pack32-mixed-shadow", "--mode", "spans", "--paragraphs", "80")
    finally:
        run("pack-control.py", "on", "pack32-mixed-on")
    run("control.py", ROOT / "audit.py", ROOT / "combined-audit")
