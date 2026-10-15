"""Drain-controlled PLE readback scheduling; preserves all graph owners."""

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
    from tools.qwen4exp import ple_history_runtime
    importlib.reload(ple_history_runtime)
    owner = worker._resident_session().native_resources['qfn_mamba_commit_v4'].ple_history_v2
    if owner.failed or owner.transition.pending is not None or worker.model_runner.execute_model_state is not None:
        raise RuntimeError('history must be idle and healthy before method replacement')
    torch.npu.synchronize()
    if not hasattr(owner, 'packet_done'):
        owner.packet_done = torch.npu.Event()
        owner.early_packets = owner.ready_packets = 0
    owner.__class__ = ple_history_runtime.ResidentPLEHistory
    owner.early_readback = ENABLED
    owner.pending = owner.staged = owner.continuation = owner.packet_requests = None
    return dict(rank=worker.rank, passed=True, **owner.status())

class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError('drain engine and workers before changing readback scheduling')
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
    parser.add_argument("action", choices=("on", "off"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    output = ROOT / "resident-live-v13/requalification" / f"ple-readback-{args.label}.json"
    if Path(args.label).name != args.label or output.exists():
        raise ValueError("fresh plain label required")
    source_hash = hashlib.sha256(
        (ROOT / "resident-source-v1/tools/qwen4exp/ple_history_runtime.py").read_bytes()
    ).hexdigest()
    source = SOURCE.replace("ENABLED", repr(args.action == "on")) + "\n# source sha256: " + source_hash
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v13/control.token")
    result = dict(source=source)

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    client.request("/pause?mode=wait&clear_cache=false")
    revision, digest = "readback_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
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
