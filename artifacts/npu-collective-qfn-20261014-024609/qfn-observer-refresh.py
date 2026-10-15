"""Cold replacement of observer methods/buffers; retains graphs and executors."""

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
    from tools.qwen4exp import output_observer
    importlib.reload(output_observer)
    owner = worker._resident_session().native_resources['qfn_mamba_commit_v4'].output_observer_v1
    if owner.future is not None or owner.failed or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('observer must be idle and healthy before method replacement')
    if isinstance(owner.host_samples, torch.Tensor):
        old = owner.host_samples
        owner.host_samples = {old.dtype: old, torch.int32: torch.empty((4, 3), dtype=torch.int32, pin_memory=True)}
    owner.__class__ = output_observer.OutputObserver
    return dict(rank=worker.rank, passed=True, **owner.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine and workers before replacing observer methods')
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(refresh_worker)
        return dict(passed=len(rows) == 4 and all(row.get('passed') for row in rows), workers=rows)
def prepare(engine):
    return Candidate()
"""


def main():
    output = ROOT / "resident-live-v6/requalification/observer-refresh-v2.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v6/control.token")
    result = dict(source=SOURCE)
    output.write_text(json.dumps(result, indent=2))
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "observer_" + uuid.uuid4().hex, hashlib.sha256(SOURCE.encode()).hexdigest()
    result["prepared"] = client.request(
        "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=SOURCE)
    )
    output.write_text(json.dumps(result, indent=2))
    assert result["prepared"].get("passed")
    result["applied"] = client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
    output.write_text(json.dumps(result, indent=2))
    assert result["applied"].get("passed")
    client.resume()
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
