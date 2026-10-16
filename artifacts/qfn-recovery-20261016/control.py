"""Run a drained resident candidate, preserving existing owner identities."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    c = QfnResidentClient(
        "http://127.0.0.1:8001",
        Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16/control.token"),
        timeout=900,
    )
    print(json.dumps(apply(c, a.source.read_text(), a.output)["applied"], indent=2))
