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
    from vllm.v1.worker import mamba_utils
    from tools.qwen4exp import checkpoint_preparation_runtime
    importlib.reload(checkpoint_preparation_runtime)
    commit=worker._resident_session().native_resources['qfn_mamba_commit_v4']
    owner=commit.checkpoint_preparation_v1
    if commit.device_transition_v1.pending is not None or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('drain checkpoint consumers')
    if owner.failed or not owner.checked_blocks: raise RuntimeError('qualified checkpoint graphs required')
    torch.npu.synchronize()
    if ACTION=='refresh':
        owner.materialize_counts()
        owner.__class__=checkpoint_preparation_runtime.CheckpointPreparationGraphs
        owner.install(mamba_utils)
    elif ACTION=='upgrade':
        if hasattr(owner,'lazy_readback'): raise RuntimeError('readback upgrade already installed')
        owner.__class__=checkpoint_preparation_runtime.CheckpointPreparationGraphs
        owner.lazy_readback=False
        owner.deferred_counts=None
        owner.avoided_readbacks=owner.materialized_readbacks=0
        owner.original_retire=owner.transition.retire_commit
        owner.install(mamba_utils)
    elif ACTION!='status':
        owner.materialize_counts()
        if not owner.enabled or owner.verify: raise RuntimeError('normal qualified checkpoint mode required')
        owner.lazy_readback=ACTION=='on'
    return dict(rank=worker.rank,passed=True,owner=owner.status())
class Candidate:
    def validate(self,engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before changing readback controls')
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(refresh_worker)
        return dict(passed=len(rows)==4 and all(row.get('passed') for row in rows),workers=rows)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upgrade", "refresh", "on", "off", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v8/requalification" / f"checkpoint-readback-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/checkpoint_preparation_runtime.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v8/control.token")
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
