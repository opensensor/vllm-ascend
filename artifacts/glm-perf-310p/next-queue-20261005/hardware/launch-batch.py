# SPDX-License-Identifier: Apache-2.0
"""Explicit GLM batch trial launch. Never invoked by importing test helpers."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, choices=(640, 1280, 2560), required=True)
parser.add_argument("--kv-fraction", type=float, default=0.75)
parser.add_argument("--kv-bytes", type=int, help="explicit budget after this batch's memory profile")
parser.add_argument("--label", required=True)
args = parser.parse_args()
if not 0 < args.kv_fraction <= 1 or not args.label.replace("_", "").isalnum():
    parser.error("invalid fraction or label")
if args.kv_bytes is not None and args.kv_bytes <= 0:
    parser.error("explicit cache bytes must be positive")
root = Path(__file__).resolve().parent
qualified = Path("/home/matteius/experiments/glm-kpool-live-score-20261005")
source = (qualified / "serve-candidate.sh").read_text()
# Keep the qualified extension/binding directory even though this script is
# recorded in the new experiment directory. Admit 2560 only explicitly.
source = source.replace("max_batched_tokens > 1280", "max_batched_tokens > 2560")
source = source.replace("from 64 through 1280", "from 64 through 2560")
old_audit = 'audit_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)'
assert old_audit in source
source = source.replace(old_audit, "audit_root=" + str(qualified))
if args.kv_bytes is not None:
    token = '--max-model-len "$max_model_len"'
    assert source.count(token) == 1
    source = source.replace(token, token + " --kv-cache-memory-bytes " + str(args.kv_bytes))
launcher = root / ("serve-" + args.label + ".sh")
if launcher.exists():
    raise FileExistsError(launcher)
launcher.write_text(source)
package = root / "opp-batch20480"
command = [
    "bash",
    str(launcher),
    "/srv/ai/src/glm-selective-w3-nz-test-20261004",
    "/srv/ai/models/GLM-5.3-Flash-selective-W3-310p",
    "311040",
    str(args.kv_fraction),
    str(package),
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
    "20480",
    "1",
    "native",
    "public-local-control",
    "native",
]
(root / (args.label + "-launch.json")).write_text(
    json.dumps(
        {
            "command": command,
            "launcher_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "native_route_cap": 20480,
            "context": 311040,
            "kv_cache_memory_bytes": args.kv_bytes,
        },
        indent=2,
    )
)
with (root / ("serve-" + args.label + ".log")).open("wb") as output:
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
    )
(root / (args.label + ".pid")).write_text(str(process.pid) + "\n")
print(process.pid)
