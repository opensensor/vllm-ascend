"""Layer a bounded geometry experiment onto an explicit qualified payload."""

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
    if mode not in ("probe", "fused", "profile"):
        raise ValueError("unknown geometry mode")
    source = base + (
        "\nbase_replacements = replacements\n"
        "from tools.qwen4exp import qsa_geometry\n"
        "importlib.reload(qsa_geometry)\n"
        "from tools.qwen4exp.resident_candidates import qsa_graph_selection, qfn_qsa_geometry\n"
        "importlib.reload(qsa_graph_selection)\n"
        "importlib.reload(qfn_qsa_geometry)\n"
        "def replacements():\n"
        "    result = base_replacements()\n"
        "    result.update(qfn_qsa_geometry.replacements())\n"
    )
    if mode in ("fused", "profile"):
        source += (
            "    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']\n"
            "    if len(workers) != 1:\n"
            "        raise RuntimeError('expected exactly one resident worker')\n"
            "    result.update(qfn_qsa_geometry.make_model_replacements(workers[0]))\n"
        )
    if mode == "profile":
        source += (
            "    from tools.qwen4exp.resident_candidates import qfn_native_profile\n"
            "    result.update(qfn_native_profile.replacements())\n"
        )
    source += "    return result\n"
    compile(source, "geometry_payload", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "fused", "profile"))
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError("preserve existing receipts; choose a new path")
    base = json.loads(args.base_receipt.read_text())
    if base["control"]["candidate"] not in ("qsa_hc_fused", "qsa_gather_fused"):
        raise ValueError("geometry must extend an explicit qualified HC or gather payload")
    if len(base.get("after", ())) != 4:
        raise ValueError("base payload must have four worker installation receipts")
    control = Control(
        generation=uuid.uuid4().hex,
        candidate=base["control"]["candidate"] + "_geometry_" + args.mode,
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
