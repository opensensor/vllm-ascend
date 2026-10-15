"""Alternate original QSA and cached block-table addressing on uncached prompts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    for action, label in (("off", "off"), ("on", "on")):
        with (ROOT / f"decode-check-{label}-control.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "qsa-control.py"),
                    action,
                    "--run-root",
                    str(args.run_root),
                    "--output",
                    str(ROOT / f"decode-check-{label}"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        subprocess.run(
            [
                sys.executable,
                "/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qsa-concurrency-benchmark.py",
                "--run-root",
                str(args.run_root),
                "--label",
                f"device-targets-c4-{label}",
                "--concurrency",
                "4",
                "--tokens",
                "256",
            ],
            check=True,
        )
        print(json.dumps(dict(completed=label)), flush=True)


if __name__ == "__main__":
    main()
