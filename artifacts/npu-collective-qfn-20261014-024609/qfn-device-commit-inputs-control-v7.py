"""Cold qualification and controls for GDN metadata graphs."""

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
    import torch
    from tools.qwen4exp import device_commit_inputs
    importlib.reload(device_commit_inputs)
    commit=worker._resident_session().native_resources['qfn_mamba_commit_v4']
    if commit.device_transition_v1.pending is not None or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('drain transition consumers')
    torch.npu.synchronize()
    owner=getattr(commit,'device_commit_inputs_v1',None)
    if ACTION=='install':
        if owner is not None: raise RuntimeError('device input owner already exists')
        if commit.checkpoint_preparation_v1.backend!='fused': raise RuntimeError('qualified fused checkpoint required')
        owner=device_commit_inputs.DeviceCommitInputs.__new__(device_commit_inputs.DeviceCommitInputs)
        commit.device_commit_inputs_v1=owner
        owner.__init__(commit)
        owner.construction_complete=True
        owner.install()
        owner.enabled=owner.verify=True
    elif owner is None or not getattr(owner,'construction_complete',False) or owner.failed:
        raise RuntimeError('failed or missing owner must remain retained')
    elif ACTION!='status':
        if ACTION=='on' and (not owner.checked or not owner.checked_blocks): raise RuntimeError('live metadata and bank checks required')
        owner.enabled=ACTION!='off'
        owner.verify=ACTION=='verify'
    return dict(rank=worker.rank,passed=True,owner=owner.status())
class Candidate:
    def validate(self,engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before changing device commit input controls')
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(refresh_worker)
        return dict(passed=len(rows)==4 and all(row.get('passed') for row in rows),workers=rows)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v7/requalification" / f"device-commit-inputs-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/device_commit_inputs.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v7/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "checkpoint_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
    result["prepared"] = client.request(
        "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
    )
    save()
    assert result["prepared"].get("passed")
    result["applied"] = client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
    save()
    assert result["applied"].get("passed")
    client.resume()
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
