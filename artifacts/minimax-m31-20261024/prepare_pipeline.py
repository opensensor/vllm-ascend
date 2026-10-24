"""Continue the authorized download and conversion after the interactive probes."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path("/run/media/matteius/20TB-drive/vllm-ascend")
ROOT = REPO / "artifacts/minimax-m31-20261024"
SOURCE = Path("/run/media/matteius/20TB-drive/models/MiniMax-M3.1-Flash")
DESTINATION = SOURCE.with_name("MiniMax-M3.1-Flash-W4")
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
DOWNLOAD_PID = 1537493


def write_status(phase, **details):
    value = {
        "phase": phase,
        "updated_unix": time.time(),
        "pid": psutil.Process().pid,
        "source": str(SOURCE),
        "destination": str(DESTINATION),
        "revision": REVISION,
        "service_started": False,
        **details,
    }
    temporary = ROOT / "pipeline-status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(ROOT / "pipeline-status.json")


def run(command, phase, log):
    write_status(phase, command=command)
    with (ROOT / log).open("a") as handle:
        subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    sources = [
        *sorted((REPO / "tools/minimax_m3").glob("*.py")),
        REPO / "vllm_ascend/_310p/ops/minimax_m3.py",
        REPO / "tools/deepseek_w2/w2_format.py",
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (ROOT / "pipeline-source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    write_status("waiting_for_existing_download", download_pid=DOWNLOAD_PID)
    try:
        download = psutil.Process(DOWNLOAD_PID)
        if "MiniMaxAI/MiniMax-M3.1-Flash" not in download.cmdline() or REVISION not in download.cmdline():
            raise RuntimeError("Download PID no longer identifies the expected pinned download")
        while download.is_running() and download.status() != psutil.STATUS_ZOMBIE:
            time.sleep(30)
    except psutil.NoSuchProcess:
        pass
    run(
        [
            "/home/matteius/.local/bin/hf",
            "download",
            "MiniMaxAI/MiniMax-M3.1-Flash",
            "--revision",
            REVISION,
            "--include",
            "model-*.safetensors",
            "--local-dir",
            str(SOURCE),
        ],
        "finishing_pinned_download",
        "download-recheck.log",
    )
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected for path, expected in hashes.items()):
        raise RuntimeError("Conversion sources changed while downloading; review before resuming")
    run(
        [
            sys.executable,
            "-m",
            "tools.minimax_m3.export_mixed_w4",
            str(SOURCE),
            str(DESTINATION),
            "--threads",
            "16",
            "--mse-backend",
            "numba",
        ],
        "converting",
        "full-conversion.log",
    )
    run(
        [
            sys.executable,
            "-m",
            "tools.minimax_m3.serving_budget",
            str(DESTINATION),
            "--device-gib",
            "45.6",
            "--utilization",
            "0.95",
            "--output",
            str(ROOT / "converted-budget.json"),
        ],
        "checking_complete_export",
        "converted-budget.log",
    )
    write_status("export_complete_hardware_service_not_validated")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", error=repr(error))
        raise
