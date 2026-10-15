"""Restore later qualified owners, then verify HC fusion on real serving calls."""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-hc-epilogues-20261015")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
RUN = BASE / "resident-live-v15"
PLE = Path("/srv/ai/storage/artifacts/ple-host-gather-20261015")
MIXED = Path("/srv/ai/storage/artifacts/qfn-mixed-opt-20261015")
DEVICE = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015")


def run(*args):
    print("RUN", *args, flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


run(ROOT / "control.py", ROOT / "restore-gdn.py", ROOT / "restore-gdn")
run(MIXED / "control.py", "gate", "--run-root", RUN, "--label", "hc-restore-score-gate-v15")
run(MIXED / "control.py", "install", "--run-root", RUN, "--label", "hc-restore-score-v15")
run(MIXED / "budget-control.py", "install", "--limit", 512, "--run-root", RUN, "--output", ROOT / "restore-budget")
run(
    "/srv/ai/storage/artifacts/qfn-next-profile-20261015/merge-control.py",
    "install",
    "--run-root",
    RUN,
    "--output",
    ROOT / "restore-merge",
)
run(ROOT / "control.py", ROOT / "restore-qsa-gate.py", ROOT / "restore-qsa-gate")
run(DEVICE / "qsa-control.py", "install", "--run-root", RUN, "--output", ROOT / "restore-qsa")
run(
    "-m",
    "tools.qwen4exp.apply_ple_serving",
    "--run-root",
    RUN,
    "--output",
    ROOT / "restore-ple",
    "--table",
    PLE / "serving-table-v1.json",
    "--library",
    DEVICE / "shared-mapping-v1/libple_host_gather.so",
    "--selector-build",
    BASE / "ple-request-build-v1",
)
run(
    "-m",
    "tools.qwen4exp.apply_hc_epilogues",
    "install",
    "--run-root",
    RUN,
    "--output",
    ROOT / "hc-install",
    "--module",
    PLE / "serving-source-v2/tools/qwen4exp/hc_epilogues.py",
    "--binary",
    ROOT / "build-v3/native_hc_epilogues.bin",
    "--bridge",
    "/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/qwen_hc_combine_v1.so",
)
run(
    MIXED / "profile-live.py",
    "--run-root",
    RUN,
    "--label",
    "hc-epilogues-shadow-v15",
    "--mode",
    "spans",
    "--paragraphs",
    450,
)
print("RESTORATION AND SHADOW RUN COMPLETE", flush=True)
