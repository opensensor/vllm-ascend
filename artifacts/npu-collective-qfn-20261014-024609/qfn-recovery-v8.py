"""Rebuild private native ownership before restoring the qualified combined payload."""

import argparse
import dataclasses
import json
import subprocess
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
OUTPUT = ROOT / "resident-live-v8/requalification"
BUILDS = {
    "w4": "/srv/ai/storage/artifacts/qfn-route-finalize-compile-20261014-v1/build-queue-v3",
    "hc": "/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build",
    "gather": "/srv/ai/storage/artifacts/qfn-qsa-gather-staging-compile-20261014-v1/build-r2",
    "geometry": "/srv/ai/storage/artifacts/qfn-qsa-geometry-compile-20261014-v2/build-r2",
}
PROBE_SOURCE = """
import json
from tools.qwen4exp.resident_candidates import (
    qfn_reset_barrier, qfn_w4_prefill_probe, qfn_hc_combine,
    qfn_qsa_gather_staging, qfn_qsa_geometry_v2)
from vllm_ascend._310p.worker_310p import NPUWorker310
def replacements():
    modules = dict(w4=qfn_w4_prefill_probe, hc=qfn_hc_combine,
                   gather=qfn_qsa_gather_staging, geometry=qfn_qsa_geometry_v2)
    gates = {name: module.make_rpc(NPUWorker310.qfn_cpu_profile_stop)
             for name, module in modules.items()}
    def gate(worker, payload):
        try:
            request = json.loads(payload)
            return gates[request['kind']](worker, json.dumps(request['request']))
        except Exception as error:
            return dict(rank=worker.rank, passed=False, error=str(error))
    return {**qfn_reset_barrier.replacements(),
            'vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop':gate}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "load", "validate", "schedule", "restore"))
    parser.add_argument("--kind", choices=tuple(BUILDS), default="w4")
    parser.add_argument("--rows", type=int, default=650)
    parser.add_argument("--projection", choices=("gate_up_proj", "down_proj"), default="down_proj")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    output = OUTPUT / f"{args.action}-{args.kind}-{args.rows}-{args.projection}.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v8/control.token")
    assert not client.request("/is_paused", method="GET")["is_paused"]
    result = dict(before=client.rpc("resident_status"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.action in ("install", "restore"):
        base = json.loads((ROOT / "qsa-resident-v1/restore-c4-prefix-final-v3.json").read_text())["control"]
        control = Control(
            generation=uuid.uuid4().hex,
            recapture=True,
            candidate="qfn_recovery_probe" if args.action == "install" else base["candidate"],
            source=PROBE_SOURCE if args.action == "install" else base["source"],
        )
        result["control"] = dataclasses.asdict(control)
        output.write_text(json.dumps(result, indent=2) + "\n")
        result["after"] = client.switch(control)
    else:
        request = dict(action="preflight", build=BUILDS[args.kind], repeats=3)
        if args.kind == "geometry":
            request["replays"] = 20
        else:
            request["rows"] = [args.rows]
        if args.kind == "w4":
            request.update(version=3, projection=args.projection)
        client.request("/pause?mode=wait&clear_cache=false")
        try:
            result["preflight"] = client.rpc("qfn_cpu_profile_stop", json.dumps(dict(kind=args.kind, request=request)))
            assert all(row.get("passed") for row in result["preflight"])
            request["action"] = args.action
            result["request"] = request
            result["ranks"] = client.rpc("qfn_cpu_profile_stop", json.dumps(dict(kind=args.kind, request=request)))
            assert all(row.get("passed") for row in result["ranks"])
            result["after"] = client.rpc("resident_status")
            result["resume"] = client.resume()
        finally:
            output.write_text(json.dumps(result, indent=2) + "\n")
    result["sensors"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(action=args.action, kind=args.kind, rows=args.rows, pids=[row["pid"] for row in result["after"]])
        ),
        flush=True,
    )
    print(result["sensors"], flush=True)


if __name__ == "__main__":
    main()
