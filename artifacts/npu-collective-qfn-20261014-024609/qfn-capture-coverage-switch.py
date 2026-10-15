"""Extend a qualified resident payload to C4, or explicitly restore C1/C2 coverage."""

import argparse
import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extended", "extended_prefix", "baseline"))
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("preserve prior receipts")
    base = json.loads(args.base_receipt.read_text())
    if base["control"]["candidate"] != "qsa_gather_fused_geometry_v2_fused" or len(base["after"]) != 4:
        raise ValueError("expected qualified combined base with all-rank receipts")
    sizes = (3, 6, 9, 12) if args.mode != "baseline" else (3, 6)
    prefix_sizes = sizes if args.mode == "extended_prefix" else (3,)
    source = base["control"]["source"] + (
        "\ncapture_coverage_base_replacements = replacements\n"
        "from tools.qwen4exp.resident_candidates import qfn_capture_coverage\n"
        "importlib.reload(qfn_capture_coverage)\n"
        "def replacements():\n"
        "    result = capture_coverage_base_replacements()\n"
        "    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']\n"
        "    if len(workers) != 1:\n"
        "        raise RuntimeError('expected one resident worker')\n"
        "    result.update(qfn_capture_coverage.replacements(\n"
        f"        workers[0], sizes={sizes!r}, prefix_sizes={prefix_sizes!r}))\n"
        "    return result\n"
    )
    compile(source, "capture_coverage_payload", "exec")
    control = Control(
        generation=uuid.uuid4().hex,
        candidate="qsa_combined_coverage_" + args.mode,
        source=source,
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
    result["inventory"] = client.rpc("qfn_graph_inventory")
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    assert len(result["inventory"]) == 4
    for rank in result["inventory"]:
        assert rank["capture_coverage"]["runner_sizes"] == list(sizes)
        assert rank["capture_coverage"]["configured_sizes"] == list(sizes)
        assert rank["capture_coverage"]["maximum"] == max(sizes)
        assert any(item["capture_sizes"] == list(sizes) for item in rank["qsa_cold_variants"])
        assert all(item["prefix_sizes"] == list(prefix_sizes) for item in rank["qsa_cold_variants"])
    print(
        json.dumps(
            dict(
                candidate=control.candidate,
                sizes=sizes,
                seconds=result["seconds"],
                pids=[row["pid"] for row in result["after"]],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
