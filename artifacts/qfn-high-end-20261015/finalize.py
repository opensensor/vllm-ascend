"""Measure retained decode and audit every optimization after the mixed trials."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-high-end-20261015")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    for script, extra in (
        ("qsa-decode-benchmark.py", []),
        ("qsa-concurrency-benchmark.py", ["--concurrency", "4", "--tokens", "256"]),
    ):
        subprocess.run(
            [
                sys.executable,
                str(BASE / script),
                "--run-root",
                str(args.run_root),
                "--label",
                "high-end-gdn-v1",
                *extra,
            ],
            check=True,
        )
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    receipt = apply(client, (ROOT / "final-audit-source.py").read_text(), ROOT / "final-audit")
    assert not client.request("/is_paused", method="GET")["is_paused"]
    assert client.request("/health", method="GET") is not False
    print(json.dumps(dict(final_audit=receipt["applied"]["passed"], serving=True)), flush=True)


if __name__ == "__main__":
    main()
