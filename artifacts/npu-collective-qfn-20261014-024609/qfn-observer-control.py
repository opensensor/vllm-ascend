"""Control overlapped worker output observation on the queued v6 service."""

import argparse
import dataclasses
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v6/requalification" / f"observer-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("use a fresh plain label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    result = dict(before=client.rpc("resident_status"))

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        if args.action == "install":
            base = json.loads((ROOT / "resident-live-v6/requalification/queued-install-v1.json").read_text())["control"]
            assert all(row["digest"] == Control(**base).digest for row in result["before"])
            source = (
                base["source"]
                + """
observer_base_replacements = replacements
from tools.qwen4exp import output_observer
from tools.qwen4exp.resident_candidates import qfn_output_observer
from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import original_base, RUNNER
importlib.reload(output_observer)
importlib.reload(qfn_output_observer)
def replacements(native_resources):
    result = observer_base_replacements(native_resources)
    commit = native_resources[qfn_mamba_commit.NATIVE_NAME]
    resource = getattr(commit, 'output_observer_v1', None)
    if resource is None:
        resource = output_observer.OutputObserver(
            commit.device_transition_v1, original_base(result[RUNNER + '._bookkeeping_sync']))
        commit.output_observer_v1 = resource
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    result.update(qfn_output_observer.make_replacements(result, resource, NPUModelRunner))
    return result
"""
            )
            control = Control(
                generation=uuid.uuid4().hex, candidate="qsa_combined_queued_observer", source=source, recapture=True
            )
            result["control"] = dataclasses.asdict(control)
            save()
            result["installed"] = client.switch(control)
        else:
            client.request("/pause?mode=wait&clear_cache=false")
            payload = dict(output_observer=True)
            payload.update(dict(status=True) if args.action == "status" else dict(enabled=args.action == "on"))
            result["worker"] = client.rpc("qfn_cpu_profile_stop", json.dumps(payload))
            save()
            assert len(result["worker"]) == 4 and all(row.get("passed") for row in result["worker"])
            client.resume()
        result["after"] = client.rpc("resident_status")
        result["engine"] = client.request("/qfn_engine_control", dict(operation="status"))
        save()
        print(json.dumps({key: result[key] for key in ("worker", "engine") if key in result}, indent=2))
    except BaseException as error:
        result["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
