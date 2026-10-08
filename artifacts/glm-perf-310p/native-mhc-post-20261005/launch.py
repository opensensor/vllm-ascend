# SPDX-License-Identifier: Apache-2.0
"""Explicit launch, never called by a test import."""

import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, choices=[640, 1280], default=640)
parser.add_argument("--mixer", choices=["baseline", "native"], default="baseline")
parser.add_argument("--resident", choices=["on", "off"], default="on")
parser.add_argument("--context", default="-1")
parser.add_argument("--label", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
command = [
    "bash",
    str(root / "serve-candidate.sh"),
    "/srv/ai/src/glm-selective-w3-nz-test-20261004",
    "/srv/ai/models/GLM-5.3-Flash-selective-W3-310p",
    args.context,
    "0.70",
    "/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004",
    "4",
    "graph",
    str(args.batch),
    "histogram",
    "",
    "/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/opp-qsa-cube512-candidate",
    "on",
    "off",
    "off",
    "off",
    "",
    "1",
    args.mixer,
    args.resident,
]
with (root / f"serve-{args.label}.log").open("wb") as log:
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
(root / f"{args.label}.pid").write_text(str(process.pid) + "\n")
print(process.pid)
