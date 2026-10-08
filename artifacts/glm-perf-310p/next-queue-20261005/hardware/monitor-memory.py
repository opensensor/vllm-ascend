# SPDX-License-Identifier: Apache-2.0
"""Record device memory/temperature while one identified server process lives."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pid_file", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
process = psutil.Process(int(args.pid_file.read_text()))
created = process.create_time()
with args.output.open("w") as output:
    while process.is_running() and process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        output.write(
            json.dumps(
                {
                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "returncode": result.returncode,
                    "output": result.stdout,
                    "error": result.stderr,
                }
            )
            + "\n"
        )
        output.flush()
        time.sleep(15)
