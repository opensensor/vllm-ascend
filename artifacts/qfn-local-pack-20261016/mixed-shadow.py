"""Qualify mixed serving with the retained HC and packing candidates together."""

import subprocess
import sys
from pathlib import Path

from tools.qwen4exp.thermal_guard import admit, query

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")


def run(name, *args):
    subprocess.run([sys.executable, str(ROOT / name), *map(str, args)], check=True)


if __name__ == "__main__":
    admission = admit(query(), max_temperature_c=85)
    assert admission["passed"], admission
    run("local-control.py", "verify", "local-mixed-verify")
    try:
        run("profile-short-mixed.py", "--label", "local-mixed-shadow", "--mode", "spans", "--paragraphs", "80")
    finally:
        run("local-control.py", "on", "local-mixed-on")
    run("control.py", ROOT / "audit.py", ROOT / "combined-audit")
