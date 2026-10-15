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
    import hashlib
    import importlib
    import inspect
    from vllm.v1.worker import mamba_utils
    from vllm_ascend.patch.worker.patch_mamba_utils import _postprocess_mamba_align_gpu_cpu_fallback
    from tools.qwen4exp import host_commit_metadata
    from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import original_base, QUALIFIED_FALLBACK_SHA256
    importlib.reload(host_commit_metadata)
    commit=worker._resident_session().native_resources['qfn_mamba_commit_v4']
    if worker.model_runner.execute_model_state is not None or commit.device_transition_v1.pending is not None:
        raise RuntimeError('drain commit metadata consumers')
    if original_base(mamba_utils.postprocess_mamba_align_gpu) is not _postprocess_mamba_align_gpu_cpu_fallback:
        raise RuntimeError('legacy consumer is not the qualified host fallback')
    if hashlib.sha256(inspect.getsource(_postprocess_mamba_align_gpu_cpu_fallback).encode()).hexdigest()!=QUALIFIED_FALLBACK_SHA256:
        raise RuntimeError('legacy consumer source differs')
    owner=getattr(commit,'host_commit_metadata_v1',None)
    if owner is None:
        owner=host_commit_metadata.HostCommitMetadata(worker.model_runner._mamba_bufs.postprocess_align,
                                                     mamba_utils.stage_postprocess_inputs_to_gpu)
        commit.host_commit_metadata_v1=owner
        mamba_utils.stage_postprocess_inputs_to_gpu=owner.stage
    if ACTION!='status': owner.enabled=ACTION!='off'
    return dict(rank=worker.rank,passed=True,owner=owner.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine and workers before changing metadata graphs')
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
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v9/requalification" / f"host-commit-metadata-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/host_commit_metadata.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v9/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "metadata_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
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
