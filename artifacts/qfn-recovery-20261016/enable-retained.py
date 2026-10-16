"""Alternate only HC epilogues on the restored, verified serving stack."""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
HC = Path("/srv/ai/storage/artifacts/qfn-hc-epilogues-20261015")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
PLE = Path("/srv/ai/storage/artifacts/ple-host-gather-20261015")


def run(*args):
    print("RUN", *args, flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def hc(action, label):
    run(
        "-m",
        "tools.qwen4exp.apply_hc_epilogues",
        action,
        "--run-root",
        RUN,
        "--output",
        ROOT / label,
        "--module",
        PLE / "serving-source-v2/tools/qwen4exp/hc_epilogues.py",
        "--binary",
        HC / "build-v3/native_hc_epilogues.bin",
        "--bridge",
        "/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/qwen_hc_combine_v1.so",
    )


if __name__ == "__main__":
    # These switches require real serving verification counters from the shadow run.
    run(
        "/srv/ai/storage/artifacts/qfn-next-profile-20261015/merge-control.py",
        "on",
        "--run-root",
        RUN,
        "--output",
        ROOT / "restore-merge-on",
    )
    run(
        "/srv/ai/storage/artifacts/qfn-device-targets-20261015/qsa-control.py",
        "on",
        "--run-root",
        RUN,
        "--output",
        ROOT / "restore-qsa-on",
    )
    run(
        "-m",
        "tools.qwen4exp.apply_ple_serving",
        "--action",
        "on",
        "--run-root",
        RUN,
        "--output",
        ROOT / "restore-ple-on",
    )
    hc("on", "hc-qualified")
    print("RETAINED PATHS ENABLED AFTER SERVING SHADOW", flush=True)
