"""Install a bounded acceptance-reuse experiment, then toggle without recapture."""

import argparse
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "upgrade", "on", "off", "verify"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "iteration-closure-v1" / f"switch-{args.label}.json"
    if output.exists() or Path(args.label).name != args.label:
        raise ValueError("use a new plain label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v4/control.token")
    assert not client.request("/is_paused", method="GET")["is_paused"]
    result = dict(before=client.rpc("resident_status"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.action in ("install", "upgrade"):
        base = json.loads((ROOT / "resident-live-v4/requalification/restore-w4-650-down_proj.json").read_text())[
            "control"
        ]
        assert all(
            row["digest"] == base_digest(base)
            or (args.action == "upgrade" and row["candidate"] == "qsa_combined_acceptance_snapshot")
            for row in result["before"]
        )
        module_path = ROOT / "resident-source-v1/tools/qwen4exp/resident_candidates/qfn_acceptance_snapshot.py"
        module_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        source = (
            base["source"]
            + f"\n# acceptance module sha256: {module_digest}\n"
            + """
acceptance_base_replacements = replacements
from tools.qwen4exp.resident_candidates import qfn_acceptance_snapshot
importlib.reload(qfn_acceptance_snapshot)
def replacements():
    result = acceptance_base_replacements()
    result.update(qfn_acceptance_snapshot.replacements())
    return result
"""
        )
        control = Control(
            generation=uuid.uuid4().hex, candidate="qsa_combined_acceptance_snapshot", source=source, recapture=True
        )
        result["control"] = dataclasses.asdict(control)
        output.write_text(json.dumps(result, indent=2) + "\n")
        result["after"] = client.switch(control)
    else:
        client.request("/pause?mode=wait&clear_cache=false")
        try:
            result["ranks"] = client.rpc("qfn_cpu_profile_stop", args.action != "off", args.action == "verify")
            assert len(result["ranks"]) == 4 and all(row.get("passed") for row in result["ranks"])
            result["after"] = client.rpc("resident_status")
        finally:
            output.write_text(json.dumps(result, indent=2) + "\n")
        client.resume()
    identity = lambda rows: [(row["pid"], row["weight_storage_digest"]) for row in rows]
    assert identity(result["before"]) == identity(result["after"])
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["after"])
    result["inventory"] = client.rpc("qfn_graph_inventory")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(action=args.action, pids=[r["pid"] for r in result["after"]], ranks=result.get("ranks"))))


def base_digest(base):
    return Control(**base).digest


if __name__ == "__main__":
    main()
