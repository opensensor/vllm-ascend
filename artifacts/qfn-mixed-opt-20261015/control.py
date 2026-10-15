"""Drained controller for a reversible mixed-QSA candidate on resident server."""

import argparse
import ast
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-mixed-opt-20261015")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v14")
SOURCE = """
def operation(worker, check_only=False):
    worker = getattr(worker, 'worker', worker)
    try:
        import importlib
        from tools.qwen4exp import mixed_qsa
        if worker.model_runner.execute_model_state is not None:
            raise RuntimeError('worker must be drained')
        if getattr(worker, '_qfn_mixed_qsa', None) is None:
            mixed_qsa = importlib.reload(mixed_qsa)
        if ACTION == 'gate': return mixed_qsa.private_gate(worker)
        if ACTION == 'status':
            return mixed_qsa.status(worker)
        if ACTION == 'restore':
            if check_only: return dict(rank=worker.rank,passed=True)
            return mixed_qsa.restore(worker)
        if not getattr(worker, '_qfn_mixed_qsa_gate', {}).get('passed'):
            raise RuntimeError('private mixed-QSA gate required before install')
        return mixed_qsa.install(worker, MODE, FACTORY, check_only=check_only)
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=str(error))
def admission(worker): return operation(worker, check_only=True)
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, 'admitted'):
            self.admitted=engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.admitted)==4 and all(row['passed'] for row in self.admitted),workers=self.admitted)
    def apply(self, engine):
        self.validate(engine)
        workers=self.admitted if ACTION=='gate' else engine.model_executor.collective_rpc(operation)
        return dict(passed=len(workers)==4 and all(row['passed'] for row in workers),workers=workers)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gate", "install", "restore", "status"))
    parser.add_argument("--mode", choices=("score",), default="score")
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", type=Path, default=RUN)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    factory_text = Path("tools/qwen4exp/resident_candidates/qsa_graph_dispatch.py").read_text()
    factory_node = next(
        n
        for n in ast.parse(factory_text).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_qualified_forward"
    )
    factory_source = ast.get_source_segment(factory_text, factory_node)
    source = f"ACTION={args.action!r}\nMODE={args.mode!r}\nFACTORY={factory_source!r}\n" + SOURCE
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, source, ROOT / args.label)
    print(json.dumps(result["applied"], indent=2), flush=True)


if __name__ == "__main__":
    main()
