"""Cold qualification and controls for GDN metadata graphs."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
SOURCE = """
def inspect_worker(worker):
    import inspect, hashlib
    from vllm.v1.worker import mamba_utils
    runner=worker.model_runner
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    funcs = dict(execute=NPUModelRunner.execute_model._qfn_queued_base)
    return dict(rank=worker.rank, sources={k:inspect.getsource(v) for k,v in funcs.items()},
                files={k:inspect.getfile(v) for k,v in funcs.items()})
class Candidate:
    def validate(self,engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain before inspect')
        return dict(passed=True)
    def apply(self,engine):
        self.validate(engine)
        return dict(passed=True,workers=engine.model_executor.collective_rpc(inspect_worker))
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v7/requalification" / f"precopy-audit-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/host_commit_metadata.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ACTION", repr(args.action)) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v7/control.token")
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
