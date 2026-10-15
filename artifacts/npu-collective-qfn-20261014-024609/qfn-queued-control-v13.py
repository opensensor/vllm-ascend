"""Control the v6 queued-engine trial without changing public service identity."""

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
    parser.add_argument("action", choices=("inspect", "install", "on", "off", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v13/requalification" / f"queued-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("use a fresh plain label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v13/control.token")
    result = dict(
        before=client.rpc("resident_status"),
        engine_before=client.request("/qfn_engine_control", {"operation": "status"}),
    )

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        if args.action == "install":
            base = json.loads((ROOT / "resident-live-v13/requalification/ple-history-install-v2.json").read_text())[
                "control"
            ]
            assert all(row["digest"] == Control(**base).digest for row in result["before"])
            source = (
                base["source"]
                + """
queued_base_replacements = replacements
from tools.qwen4exp import queued_schedule
from tools.qwen4exp.resident_candidates import qfn_queued_schedule
importlib.reload(queued_schedule)
importlib.reload(qfn_queued_schedule)
def replacements(native_resources):
    result = queued_base_replacements(native_resources)
    commit = native_resources[qfn_mamba_commit.NATIVE_NAME]
    resource = getattr(commit, 'queued_schedule_v1', None)
    if resource is None:
        resource = queued_schedule.QueuedScheduleBridge(commit.runner)
        commit.queued_schedule_v1 = resource
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    result.update(qfn_queued_schedule.make_replacements(result, resource, NPUModelRunner))
    return result
"""
            )
            control = Control(
                generation=uuid.uuid4().hex, candidate="qsa_combined_queued_schedule", source=source, recapture=True
            )
            result["control"] = dataclasses.asdict(control)
            save()
            result["installed"] = client.switch(control)
        elif args.action == "status":
            client.request("/pause?mode=wait&clear_cache=false")
            result["worker"] = client.rpc("qfn_cpu_profile_stop", json.dumps(dict(queued_schedule=True, status=True)))
            client.resume()
        else:
            client.request("/pause?mode=wait&clear_cache=false")
            enabled = args.action != "off"
            source = f"""
import importlib
from tools.qwen4exp import queued_engine
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
importlib.reload(queued_engine)
def prepare(engine):
    return queued_engine.QueuedEngineControl(engine, enabled={enabled!r}, async_scheduler_class=AsyncScheduler)
"""
            revision = "queued_" + uuid.uuid4().hex
            digest = hashlib.sha256(source.encode()).hexdigest()
            result["source"] = source
            result["prepared"] = client.request(
                "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
            )
            save()
            if not result["prepared"].get("passed"):
                # Preparation cannot mutate the engine or worker configuration.
                client.resume()
                raise RuntimeError(result["prepared"])
            if args.action == "inspect":
                result["discarded"] = client.request(
                    "/qfn_engine_control", dict(operation="discard", revision=revision)
                )
            else:
                # Workers are drained and cannot see a request until the engine
                # transaction succeeds. A failed apply stays paused for review.
                result["worker"] = client.rpc(
                    "qfn_cpu_profile_stop", json.dumps(dict(queued_schedule=True, enabled=enabled))
                )
                save()
                assert len(result["worker"]) == 4 and all(row.get("passed") for row in result["worker"])
                result["applied"] = client.request(
                    "/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest)
                )
                save()
                assert result["applied"].get("passed")
            client.resume()
        result["after"] = client.rpc("resident_status")
        result["engine_after"] = client.request("/qfn_engine_control", dict(operation="status"))
        save()
        print(
            json.dumps(
                {key: result[key] for key in ("prepared", "worker", "applied", "engine_after") if key in result},
                indent=2,
            )
        )
    except BaseException as error:
        result["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
