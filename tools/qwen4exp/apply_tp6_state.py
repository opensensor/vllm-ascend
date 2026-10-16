# SPDX-License-Identifier: Apache-2.0
"""Qualify and attach TP6 grouped state commits to the retained HC/PLE server.

Installation starts in diagnostic mode. Disable verification only after actual
serving has exercised all six request rows and checkpoint boundary copies.
This controller does not enable the engine queue or replace PLE ownership.
"""

import argparse
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient

PREPARE = """
import gc
import torch
from tools.qwen4exp.mamba_commit_runtime import ResidentMambaCommit

def prepare():
    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']
    if len(workers) != 1 or workers[0].model_runner.execute_model_state is not None:
        raise RuntimeError('requires one drained resident worker')
    kernel = getattr(torch.classes, NAMESPACE).Kernel(BINARY, 'qwen_mamba_commit_group_v1')
    return ResidentMambaCommit(workers[0].model_runner, kernel, getattr(torch.ops, NAMESPACE).launch)

def validate(resource):
    if resource.failed or sorted(resource.records) != list(range(1, 7)):
        raise RuntimeError('all six request shapes must capture successfully')
    return dict(passed=True, slots=resource.slots, **resource.status())
"""

COMPOSE = """
from types import ModuleType
tp6_state_commit = ModuleType("tp6_state_commit_v1")
exec(compile(STATE_SOURCE, "<tp6-state-commit>", "exec"), tp6_state_commit.__dict__)
hc_replacements = replacements

def replacements(native_resources):
    result = hc_replacements(native_resources)
    result.update(tp6_state_commit.replacements(native_resources))
    return result
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=("install", "verify", "on", "off"), default="install")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing receipts")
    client = QfnResidentClient(args.url, args.run_root / "control.token", expected_workers=6, timeout=1800)
    result = dict(before=client.rpc("resident_status"))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if args.action == "install":
        if args.build is None:
            parser.error("install requires --build")
        root = Path(__file__).parent
        hc_source = (root / "resident_candidates/exact_hc_residual.py").read_text()
        digest = hashlib.sha256(hc_source.encode()).hexdigest()
        if any(row["digest"] != digest for row in result["before"]):
            raise RuntimeError("install requires the retained exact HC candidate; refusing to discard other patches")
        provenance = json.loads((args.build / "provenance.json").read_text())
        if (
            provenance["namespace"] not in ("qwen_mamba_commit_v1", "qwen_mamba_commit_v2")
            or provenance["flush_host_queue"] is not False
        ):
            raise ValueError("requires the qualified graph-compatible commit bridge")
        assets = [*provenance["binaries"].values()]
        for name in (
            "mamba_commit.py",
            "mamba_commit_runtime.py",
            "runtime_geometry.py",
            "resident_candidates/qfn_mamba_commit.py",
            "resident_candidates/qfn_acceptance_snapshot.py",
        ):
            path = root / name
            assets.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = NativeManifest(
            dict(
                name="qfn_mamba_commit_v4",
                libraries=[provenance["bridge"]],
                assets=assets,
                operators=[provenance["namespace"] + "::launch"],
                validation_source="NAMESPACE = "
                + repr(provenance["namespace"])
                + "\nBINARY = "
                + repr(str(args.build.resolve() / "native_mamba_commit.bin"))
                + "\n"
                + PREPARE,
            )
        )
        result["manifest"] = manifest.value
        save()
        result["loaded"] = client.load_native(manifest)
        save()
        state_source = (root / "resident_candidates/tp6_state_commit.py").read_text()
        source = hc_source + "\nSTATE_SOURCE = " + repr(state_source) + "\n" + COMPOSE
        control = Control(uuid.uuid4().hex, candidate="tp6_hc_state_commit", source=source, recapture=True)
        result["control"] = dataclasses.asdict(control)
        save()
        result["installed"] = client.switch(control)
        save()
    client.request("/pause?mode=wait&clear_cache=false")
    enabled = args.action != "off"
    verify = args.action in ("install", "verify")
    try:
        result["commit"] = client.rpc(
            "qfn_cpu_profile_stop", json.dumps(dict(tp6_commit=True, enabled=enabled, verify=verify))
        )
        if not all(row.get("passed") for row in result["commit"]):
            raise RuntimeError("commit control failed; retain paused workers")
        result["after"] = client.rpc("resident_status")
        before = [(r["pid"], r["weight_storage_digest"]) for r in result["before"]]
        after = [(r["pid"], r["weight_storage_digest"]) for r in result["after"]]
        if before != after or any(r["graphs_dirty"] or r["native_failed"] for r in result["after"]):
            raise RuntimeError("resident identity or native state changed unexpectedly")
        result["resumed"] = client.resume()
    finally:
        save()
    print(json.dumps(dict(passed=True, action=args.action, workers=result["commit"])))


if __name__ == "__main__":
    main()
