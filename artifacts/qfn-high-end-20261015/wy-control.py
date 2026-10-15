"""Drained ownership transitions for the qualified private WY graph."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper, check_only=False):
    import importlib
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from tools.qwen4exp.wy_graph import WYGraph
        module = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
        assert worker.model_runner.execute_model_state is None
        owner = getattr(worker, "_qfn_wy_graph", None)
        if ACTION == "install":
            assert owner is None
            probe = worker._qfn_wy_graph_probe
            assert module._compute_kernel_inputs_from_torch_wy is probe.original
            assert module._UT_USE_BLOCKED_INVERSE and module._WY_GROUPED_GRAM
            owner = WYGraph(probe)
            if not check_only:
                worker._qfn_wy_graph = owner
                module._compute_kernel_inputs_from_torch_wy = owner
        else:
            assert owner is not None
            assert module._compute_kernel_inputs_from_torch_wy is owner
            if ACTION == "on": assert owner.checked >= 36 and not owner.failed
            if not check_only:
                if ACTION == "on": owner.verify, owner.enabled = False, True
                if ACTION == "off": owner.enabled = False
                if ACTION == "verify": owner.verify, owner.enabled = True, True
                if ACTION == "restore":
                    module._compute_kernel_inputs_from_torch_wy = owner.original
                    worker._qfn_wy_graph = None
        return dict(rank=worker.rank, passed=True, wy_graph=owner.status())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=type(error).__name__ + ": " + str(error))
def admission(worker): return operation(worker, check_only=True)
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "admitted"):
            self.admitted = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.admitted)==4 and all(r["passed"] for r in self.admitted),workers=self.admitted)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(r["passed"] for r in rows), workers=rows)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status", "restore"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, f"ACTION={args.action!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
