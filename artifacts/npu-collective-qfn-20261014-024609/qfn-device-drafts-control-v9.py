"""Install/qualify guarded device draft consumption in the queued v6 bridge."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def refresh_worker(worker):
    import importlib
    import inspect
    from tools.qwen4exp import queued_schedule
    from tools.qwen4exp.resident_candidates import qfn_queued_schedule
    from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER
    from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC
    from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET
    from tools.qwen4exp.resident_candidates.qfn_ple_history import STAGE_TARGET
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend._310p.worker_310p import NPUWorker310
    importlib.reload(queued_schedule)
    importlib.reload(qfn_queued_schedule)
    commit = worker._resident_session().native_resources['qfn_mamba_commit_v4']
    owner = commit.queued_schedule_v1
    if worker.model_runner.execute_model_state is not None or commit.device_transition_v1.pending is not None:
        raise RuntimeError('drain worker state before changing queued packet consumers')
    control = NPUWorker310.qfn_cpu_profile_stop
    if control.__module__ == qfn_queued_schedule.__name__:
        control = inspect.getclosurevars(control).nonlocals['previous_control']
    inputs = NPUModelRunner310._prepare_input_ids
    stage = NPUModelRunner310._stage_qwen4exp_ple_inputs
    if inputs.__module__ == qfn_queued_schedule.__name__:
        inputs = inputs._qfn_transition_base
    if stage.__module__ == qfn_queued_schedule.__name__:
        stage = stage._qfn_transition_base
    base = {RPC: control, INPUT_TARGET: inputs, STAGE_TARGET: stage}
    replacements = qfn_queued_schedule.make_replacements(base, owner, NPUModelRunner)
    owner.transition, owner.history = commit.device_transition_v1, commit.ple_history_v2
    owner.device_drafts = getattr(owner, 'device_drafts', False)
    owner.device_draft_steps = getattr(owner, 'device_draft_steps', 0)
    owner.expected_device_inputs = owner.expected_device_history = None
    owner.__class__ = queued_schedule.QueuedScheduleBridge
    NPUModelRunner.execute_model = replacements[RUNNER + '.execute_model']
    NPUModelRunner310._prepare_input_ids = replacements[INPUT_TARGET]
    NPUModelRunner310._stage_qwen4exp_ple_inputs = replacements[STAGE_TARGET]
    NPUWorker310.qfn_cpu_profile_stop = replacements[RPC]
    # Keep the already-installed observer discoverable by future cold reloads.
    fn = NPUModelRunner._bookkeeping_sync
    if fn.__module__ == 'tools.qwen4exp.resident_candidates.qfn_output_observer':
        fn._qfn_transition_base = inspect.getclosurevars(fn).nonlocals['fallback']
    return dict(rank=worker.rank, passed=True, **owner.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine before changing device packet consumers')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(refresh_worker)
        return dict(passed=len(rows) == 4 and all(row.get('passed') for row in rows), workers=rows)
def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v9/requalification" / f"device-drafts-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("use a fresh plain label")
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v9/control.token")
    result = {}

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    client.request("/pause?mode=wait&clear_cache=false")
    try:
        if args.action == "install":
            result["source_hashes"] = {
                name: hashlib.sha256((ROOT / "resident-source-v1/tools/qwen4exp" / name).read_bytes()).hexdigest()
                for name in ("queued_schedule.py", "resident_candidates/qfn_queued_schedule.py")
            }
            source = SOURCE + "\n# " + json.dumps(result["source_hashes"], sort_keys=True)
            revision, digest = "drafts_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
            result["source"] = source
            result["prepared"] = client.request(
                "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
            )
            save()
            assert result["prepared"].get("passed")
            result["applied"] = client.request(
                "/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest)
            )
            save()
            assert result["applied"].get("passed")
        else:
            payload = dict(queued_schedule=True)
            payload.update(dict(status=True) if args.action == "status" else dict(device_drafts=args.action == "on"))
            result["worker"] = client.rpc("qfn_cpu_profile_stop", json.dumps(payload))
            save()
            assert len(result["worker"]) == 4 and all(row.get("passed") for row in result["worker"])
        client.resume()
        save()
        print(json.dumps(result.get("worker", result.get("applied")), indent=2))
    except BaseException as error:
        result["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
