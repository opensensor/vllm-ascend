"""Install device PLE history on the qualified v6 transition owner."""

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
    parser.add_argument("action", choices=("install", "refresh", "verify", "on", "off"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v6/requalification" / f"ple-history-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("use a new plain label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    assert not client.request("/is_paused", method="GET")["is_paused"]
    result = dict(before=client.rpc("resident_status"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.action in ("install", "refresh"):
        base = json.loads((ROOT / "resident-live-v6/requalification/restore-device-transition.json").read_text())[
            "control"
        ]
        expected = (
            base
            if args.action == "install"
            else json.loads((ROOT / "resident-live-v6/requalification/ple-history-install-v1.json").read_text())[
                "control"
            ]
        )
        assert all(row["digest"] == Control(**expected).digest for row in result["before"])
        files = ("iteration_transition.py", "ple_history_runtime.py", "resident_candidates/qfn_ple_history.py")
        result["source_hashes"] = {
            name: hashlib.sha256((ROOT / "resident-source-v1/tools/qwen4exp" / name).read_bytes()).hexdigest()
            for name in files
        }
        source = (
            base["source"]
            + "\n# PLE history hashes: "
            + json.dumps(result["source_hashes"], sort_keys=True)
            + "\n"
            + """
history_base_replacements = replacements
from tools.qwen4exp import iteration_transition
importlib.reload(iteration_transition)
from tools.qwen4exp import ple_history_runtime
from tools.qwen4exp.resident_candidates import qfn_ple_history
importlib.reload(ple_history_runtime)
importlib.reload(qfn_ple_history)
def replacements(native_resources):
    result = history_base_replacements(native_resources)
    commit = native_resources[qfn_mamba_commit.NATIVE_NAME]
    history = getattr(commit, 'ple_history_v2', None)
    if history is None:
        history = ple_history_runtime.ResidentPLEHistory.__new__(ple_history_runtime.ResidentPLEHistory)
        commit.ple_history_v2 = history
        history.failed = True
        history.__init__(commit.device_transition_v1)
        history.construction_complete = True
    if not getattr(history, 'construction_complete', False) or history.failed:
        raise RuntimeError('PLE history construction failed; retain owners')
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    result.update(qfn_ple_history.make_replacements(result, history, NPUModelRunner310))
    return result
"""
        )
        control = Control(
            generation=uuid.uuid4().hex, candidate="qsa_combined_device_history", source=source, recapture=True
        )
        result["control"] = dataclasses.asdict(control)
        output.write_text(json.dumps(result, indent=2) + "\n")
        result["installed"] = client.switch(control)
    else:
        client.request("/pause?mode=wait&clear_cache=false")
        result["ranks"] = client.rpc(
            "qfn_cpu_profile_stop",
            json.dumps(dict(ple_history=True, enabled=args.action != "off", verify=args.action == "verify")),
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        assert all(row.get("passed") for row in result["ranks"])
        client.resume()
    result["after"] = client.rpc("resident_status")
    assert all(not row["graphs_dirty"] and not row["native_failed"] for row in result["after"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(action=args.action, ranks=result.get("ranks")), indent=2))


if __name__ == "__main__":
    main()
