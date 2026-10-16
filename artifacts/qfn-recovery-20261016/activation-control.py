"""Attach exact activation fusion while retaining the qualified W4 finalizer."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
SOURCE = """
def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import importlib
        import inspect
        import torch
        from tools.qwen4exp import w4_activation_epilogue
        from tools.qwen4exp.resident_candidates import qfn_route_finalize
        from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE
        assert worker.model_runner.execute_model_state is None
        state = getattr(worker, "_qfn_activation_serving", None)
        if state is None:
            assert ACTION == "install"
            private = worker._qfn_activation_epilogue_test
            assert private["receipt"]["passed"]
            current = W4SparseMoE._forward_grouped
            closure = inspect.getclosurevars(current).nonlocals
            assert current._qfn_route_finalize_base is closure["original"]
            resource = closure["resource"]
            assert inspect.getclosurevars(closure["implementation"]).nonlocals["_finalize"] is resource
            importlib.reload(w4_activation_epilogue)
            importlib.reload(qfn_route_finalize)
            owner = w4_activation_epilogue.W4ActivationEpilogue(
                private["kernel"], torch.ops.qwen_hc_combine_v1.launch)
            state = dict(owner=owner, original=current, finalizer=resource)
            worker._qfn_activation_serving = state
            state["replacement"] = qfn_route_finalize.build_forward(
                current, resource, eager_only=closure["eager_only"], activation=owner)
        owner = state["owner"]
        assert not owner.failed and not owner.mismatches
        if ACTION == "install":
            assert W4SparseMoE._forward_grouped is state["original"]
            if not check_only:
                W4SparseMoE._forward_grouped = state["replacement"]
        else:
            assert W4SparseMoE._forward_grouped is state["replacement"]
            if ACTION == "on": assert owner.checked >= 48
            if not check_only:
                if ACTION == "on": owner.enabled, owner.verify = True, False
                if ACTION == "off": owner.enabled = False
                if ACTION == "verify": owner.enabled, owner.verify = True, True
                if ACTION == "restore": W4SparseMoE._forward_grouped = state["original"]
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
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status", "restore"))
    parser.add_argument("label")
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
    receipt = apply(client, f"ACTION={args.action!r}\n" + SOURCE, ROOT / args.label)
    print(json.dumps(receipt["applied"], indent=2))
