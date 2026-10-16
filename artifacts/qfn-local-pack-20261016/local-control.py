"""Toggle local packing while retaining full packing and the qualified finalizer."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
SOURCE = """
def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE
        assert worker.model_runner.execute_model_state is None
        state = getattr(worker, "_qfn_pack_serving", None)
        assert state is not None and state["owner"].local_kernel is not None
        owner = state["owner"]
        assert not owner.failed and not owner.mismatches
        assert W4SparseMoE._forward_grouped is state["replacement"]
        if ACTION == "on": assert owner.checked >= 48
        if not check_only:
            if ACTION == "on": owner.local_routes, owner.enabled, owner.verify = True, True, False
            if ACTION == "off": owner.local_routes, owner.verify = False, False
            if ACTION == "verify": owner.local_routes, owner.enabled, owner.verify = True, True, True
        return dict(rank=worker.rank, passed=True, activation=owner.status(), finalizer_preserved=True)
    except Exception as error:
        import traceback
        return dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())
def admission(worker): return operation(worker, check_only=True)
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.rows) == 4 and all(row["passed"] for row in self.rows), workers=self.rows)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)
def prepare(engine): return Candidate()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("on", "off", "verify", "status"))
    parser.add_argument("label")
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
    receipt = apply(client, f"ACTION={args.action!r}\n" + SOURCE, ROOT / args.label)
    print(json.dumps(receipt["applied"], indent=2))
