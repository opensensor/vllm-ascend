#!/usr/bin/env python3
"""Read-only five-second thermal record for this dedicated test process."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/matteius/experiments/qwen38-prefix-npu-20261008")


def record(pid, output):
    with (ROOT / output).open("a", buffering=1) as record:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            try:
                raw = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
                temps = [int(line.split("NA")[1].split()[0]) for line in raw.splitlines() if "310P3" in line]
                record.write(json.dumps({"time": time.time(), "temperatures_c": temps, "api_pid": pid}) + "\n")
            except (subprocess.SubprocessError, ValueError) as error:
                record.write(json.dumps({"time": time.time(), "error": str(error)}) + "\n")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", default="thermal.jsonl")
    args = parser.parse_args()
    record(args.pid, args.output)


if __name__ == "__main__":
    main()
