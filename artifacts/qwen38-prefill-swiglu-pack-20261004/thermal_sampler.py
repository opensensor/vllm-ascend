# SPDX-License-Identifier: Apache-2.0
"""Record per-device 310P telemetry during the isolated prefill test."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3)
    args = parser.parse_args()
    if args.interval <= 0 or args.output.exists() or args.stop.exists():
        parser.error("interval must be positive and output/stop paths must be new")
    with args.output.open("x") as output:
        while not args.stop.exists():
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
            output.write(
                json.dumps(
                    {
                        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                        "returncode": result.returncode,
                        "info": result.stdout,
                    }
                )
                + "\n"
            )
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
