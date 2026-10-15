"""Layer a bounded GDN experiment onto an explicit qualified payload."""

import argparse
import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient


def build_source(base, mode):
    if mode != "probe":
        raise ValueError("private GDN probe only")
    source = base + (
        "\nbase_replacements = replacements\n"
        "from tools.qwen4exp.resident_candidates import qfn_gdn_projection_probe\n"
        "importlib.reload(qfn_gdn_projection_probe)\n"
        "def replacements():\n"
        "    result = base_replacements()\n"
        "    result.update(qfn_gdn_projection_probe.replacements())\n"
        "    return result\n"
    )
    compile(source, "gdn_payload", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe",))
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError("preserve existing receipts; choose a new path")
    base = json.loads(args.base_receipt.read_text())
    if base["control"]["candidate"] not in ("qsa_hc_fused", "qsa_gather_fused"):
        raise ValueError("GDN must extend an explicit qualified HC or gather payload")
    if len(base.get("after", ())) != 4:
        raise ValueError("base payload must have four worker installation receipts")
    control = Control(
        generation=uuid.uuid4().hex,
        candidate=base["control"]["candidate"] + "_gdn_" + args.mode,
        source=build_source(base["control"]["source"], args.mode),
        recapture=True,
    )
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    result = dict(
        control=dataclasses.asdict(control),
        base_receipt=str(args.base_receipt),
        base_receipt_sha256=hashlib.sha256(args.base_receipt.read_bytes()).hexdigest(),
        before=client.rpc("resident_status"),
    )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    result["after"] = client.switch(control)
    result["seconds"] = time.monotonic() - started
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(candidate=control.candidate, seconds=result["seconds"], pids=[row["pid"] for row in result["after"]])
        )
    )


if __name__ == "__main__":
    main()
