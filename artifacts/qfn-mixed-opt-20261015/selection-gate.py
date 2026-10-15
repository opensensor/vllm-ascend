"""Inspect private selection tiling without changing model dispatch."""

import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v14")
SOURCE = """
def operation(worker):
    worker = getattr(worker, 'worker', worker)
    try:
        from tools.qwen4exp.tiled_qsa_selection import private_gate
        return private_gate(worker)
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=str(error))
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row['passed'] for row in rows),workers=rows)
def prepare(engine):return Candidate()
"""
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
result = apply(client, SOURCE, Path("/srv/ai/storage/artifacts/qfn-mixed-opt-20261015/v14-selection-gate"))
print(json.dumps(result["applied"], indent=2))
