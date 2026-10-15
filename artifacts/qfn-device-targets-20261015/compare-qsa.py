"""Alternate original QSA and cached block-table addressing on uncached prompts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015")
PROFILE = Path("/srv/ai/storage/artifacts/qfn-mixed-opt-20261015/profile-live.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    for action, label in (("off", "off-a"), ("on", "on-a"), ("off", "off-b"), ("on", "on-b")):
        with (ROOT / f"compare2-{label}-control.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "qsa-control.py"),
                    action,
                    "--run-root",
                    str(args.run_root),
                    "--output",
                    str(ROOT / f"compare2-{label}"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        subprocess.run(
            [
                sys.executable,
                str(PROFILE),
                "--run-root",
                str(args.run_root),
                "--label",
                f"qsa-table2-{label}",
                "--mode",
                "spans",
                "--paragraphs",
                "450",
            ],
            check=True,
        )
        print(json.dumps(dict(completed=label)), flush=True)


if __name__ == "__main__":
    main()
