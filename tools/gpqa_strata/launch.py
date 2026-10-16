"""Run IQ2_XS and IQ4_XS GPQA smoke gates, then optionally both full replays."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="Run 198 questions per arm after both smoke gates pass")
    parser.add_argument(
        "--no-tail-skip", action="store_true", help="Opt out of Strata's approximate tail-expert skipping"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    variant = "-no-tail-skip" if args.no_tail_skip else ""
    runner = Path(__file__).with_name("experiment.py")
    for phase, limit in (("smoke", 5), ("full", 198)):
        if phase == "full" and not args.full:
            break
        for arm in ("iq2", "iq4"):
            output = root / "results" / f"{arm}{variant}-{phase}"
            command = [
                sys.executable,
                str(runner),
                "run",
                "--root",
                str(root),
                "--cases",
                str(root / "reference/cases.json"),
                "--baseline",
                str(root / "reference/results.jsonl"),
                "--config",
                str(root / "configs" / f"{arm}{variant}.json"),
                "--output",
                str(output),
                "--limit",
                str(limit),
            ]
            print(f"Starting {arm}{variant} {phase}: {limit} frozen questions", flush=True)
            subprocess.run(command, check=True)
            summary = json.loads((output / "summary.json").read_text())
            if summary["completed"] != limit or summary["extracted"] == 0:
                raise RuntimeError(f"Incomplete or unextractable {arm} {phase}; inspect its ledger")
    if args.full:
        iq2 = root / "results" / f"iq2{variant}-full"
        iq4 = root / "results" / f"iq4{variant}-full"
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "summarize",
                "--root",
                str(root),
                "--cases",
                str(root / "reference/cases.json"),
                "--baseline",
                str(iq4 / "results.jsonl"),
                "--config",
                str(root / "configs" / f"iq2{variant}.json"),
                "--output",
                str(iq2),
                "--summary-name",
                "paired-strata-iq4-summary.json",
            ],
            check=True,
        )
        print("Saved IQ2 versus Strata IQ4 in paired-strata-iq4-summary.json.", flush=True)


if __name__ == "__main__":
    main()
