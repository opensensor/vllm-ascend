"""Restore the qualified six-rank owners after a fresh server reaches readiness."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
run = ROOT / sys.argv[1]
assert run.parent == ROOT and run.name.startswith("resident-")
assert (run / "control.token").is_file()


def invoke(*args):
    subprocess.run(["bash", str(ROOT / "run.sh"), *map(str, args)], check=True)


invoke(ROOT / "install-hc-exact.py", run.name)
common = ["-m", "tools.qwen4exp.apply_ple_serving", "--run-root", run, "--expected-workers", "6", "--max-requests", "6"]
invoke(
    *common,
    "--output",
    run / "ple-install",
    "--table",
    ROOT / "ple-table.json",
    "--library",
    ROOT / "ple-host-six-v1/libple_host_gather.so",
    "--selector-build",
    ROOT / "ple-selector-six-v2",
)
invoke(ROOT / "prefix-gate.py", run.name)
invoke(ROOT / "six-decode-gate.py", run.name)
invoke(*common, "--action", "status", "--output", run / "ple-verified")
receipt = json.loads((run / "ple-verified/receipt.json").read_text())
workers = receipt["applied"]["applied"]["workers"]
assert all(w["native_ple"]["native_calls"].get("18", 0) > 0 for w in workers)
invoke(*common, "--action", "on", "--output", run / "ple-on")
print("Six-rank HC and PLE restored; real serving verification passed.", flush=True)
