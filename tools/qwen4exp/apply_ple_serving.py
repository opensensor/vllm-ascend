# SPDX-License-Identifier: Apache-2.0
"""Attach PLE at the existing eager boundary without replacing resident patches."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

ENGINE_SOURCE = """
def operation(worker):
    import importlib
    import torch
    if worker.model_runner.execute_model_state is not None:
        raise RuntimeError("drain worker before changing PLE")
    state = getattr(worker, "_qfn_native_ple_serving", None)
    if ACTION == "install":
        if state is not None:
            raise RuntimeError("native PLE owner already exists; preserve its lifetime")
        # Keep all owners on the worker through failed admission and recapture.
        state = worker._qfn_native_ple_serving = dict(namespace={}, originals=[])
        exec(compile(ADAPTER, "<native-ple-adapter>", "exec"), state["namespace"])
        ns = state["namespace"]
        state["manager"] = manager = ns["ServingPLE"](CONFIG)
        replacements = ns["make_replacements"](CONFIG, manager=manager)
        torch.npu.synchronize()
        device = worker.model_runner.device
        for tokens in (3, 6, 9, 12):
            key = (tokens, 4, device.index)
            manager.lookups[key] = ns["Lookup"](manager, tokens, 4, device)
        # Leave scheduler, resident PatchSession, RPC controls and graphs live.
        for target, replacement in replacements.items():
            module, attributes = target.split(":")
            owner = importlib.import_module(module)
            names = attributes.split(".")
            for name in names[:-1]:
                owner = getattr(owner, name)
            state["originals"].append((owner, names[-1], getattr(owner, names[-1])))
            setattr(owner, names[-1], replacement)
        state["installed"] = True
    elif state is None or not state.get("installed"):
        raise RuntimeError("native PLE is not installed")
    manager = state["manager"]
    if manager.failed or manager.busy:
        raise RuntimeError("failed or busy PLE retains its owners")
    if ACTION in ("on", "off", "verify"):
        if ACTION == "on" and not manager.checked:
            raise RuntimeError("verify real lookups before disabling verification")
        manager.enabled = ACTION != "off"
        manager.verify = ACTION == "verify"
        manager.elapsed.clear()
        manager.native_elapsed.clear()
    return dict(rank=worker.rank, passed=True, native_ple=manager.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError("drain engine before changing PLE")
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        workers = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(workers) == 4 and all(w.get("passed") for w in workers), workers=workers)

def prepare(engine):
    return Candidate()
"""


def engine_source(action, adapter="", config=None):
    if action not in ("install", "on", "off", "verify", "status"):
        raise ValueError("invalid PLE action")
    return f"ACTION = {action!r}\nADAPTER = {adapter!r}\nCONFIG = {config!r}\n" + ENGINE_SOURCE


def preserved_identity(workers):
    """Guard resident patches as well as model and collective graph ownership."""
    return sorted(
        (
            worker["rank"],
            worker["pid"],
            worker["weight_storage_digest"],
            worker.get("generation"),
            worker.get("digest"),
            worker.get("candidate"),
            json.dumps(worker.get("native_loaded"), sort_keys=True),
            json.dumps(worker.get("custom_collective", {}).get("graph_observer"), sort_keys=True),
        )
        for worker in workers
    )


def apply(client, source, output):
    result = dict(before=client.rpc("resident_status"))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "candidate.py").write_text(source)

    def save():
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")

    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "native_ple_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
    try:
        result["prepared"] = client.request(
            "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
        )
        save()
        if not result["prepared"].get("passed"):
            raise RuntimeError("PLE preparation failed; inspect receipt before resuming")
        result["applied"] = client.request(
            "/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest)
        )
        save()
        if not result["applied"].get("passed"):
            raise RuntimeError("PLE apply failed; keep paused and retain owners")
        result["after"] = client.rpc("resident_status")
        if preserved_identity(result["before"]) != preserved_identity(result["after"]):
            raise RuntimeError("PLE changed resident patches, weights or collective graphs; keep paused")
        if any(row["graphs_dirty"] or row["native_failed"] for row in result["after"]):
            raise RuntimeError("resident state failed during PLE attachment; keep paused")
        result["resume"] = client.resume()
    finally:
        save()
    return result


def main():
    from tools.sidecard.qfn_resident_server import QfnResidentClient

    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run-root", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("table", "library", "selector-build"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--action", choices=("install", "on", "off", "verify", "status"), default="install")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    config, adapter = None, ""
    if args.action == "install":
        if any(value is None for value in (args.table, args.library, args.selector_build)):
            parser.error("install requires --table, --library and --selector-build")
        manifest = json.loads((args.selector_build / "provenance.json").read_text())
        binary = Path(manifest["binary"]["path"]).resolve(strict=True)
        if binary.parent != args.selector_build.resolve() or manifest["entry"] != "qwen_ple_request_v1":
            raise ValueError("selector build differs from expected contract")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["binary"]["sha256"]:
            raise ValueError("selector binary differs from provenance")
        library = args.library.resolve(strict=True)
        config = dict(
            table=str(args.table.resolve(strict=True)),
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            binary=str(binary),
            binary_sha256=manifest["binary"]["sha256"],
        )
        adapter = Path(__file__).with_name("ple_serving.py").read_text()
    source = engine_source(args.action, adapter, config)
    client = QfnResidentClient(args.url, args.run_root / "control.token", timeout=1800)
    result = apply(client, source, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
