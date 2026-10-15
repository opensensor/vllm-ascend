"""Drained ownership transitions for the qualified private QSA table cache."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper, check_only=False):
    worker=getattr(wrapper,"worker",wrapper)
    try:
        state=worker._qfn_qsa_table_probe
        owner=state["owner"]
        assert state["receipt"]["passed"] and not owner.failed
        assert worker.model_runner.execute_model_state is None
        if ACTION=="install":
            assert state["scope"]["qsa_sparse_attention_310"] is state["original"]
            if not check_only:
                owner.enabled,owner.verify=True,True
                state["scope"]["qsa_sparse_attention_310"]=owner
        else:
            assert state["scope"]["qsa_sparse_attention_310"] is owner
            if ACTION=="on": assert owner.checked>=52
            if not check_only:
                if ACTION=="on": owner.enabled,owner.verify=True,False
                if ACTION=="off": owner.enabled=False
                if ACTION=="verify": owner.enabled,owner.verify=True,True
                if ACTION=="restore": state["scope"]["qsa_sparse_attention_310"]=state["original"]
        return dict(rank=worker.rank,passed=True,qsa_table=owner.status())
    except Exception as error:
        return dict(rank=worker.rank,passed=False,error=repr(error))
def admission(worker): return operation(worker,check_only=True)
class Candidate:
    def validate(self,engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self,"rows"): self.rows=engine.model_executor.collective_rpc(admission)
        return dict(passed=all(row["passed"] for row in self.rows),workers=self.rows)
    def apply(self,engine):
        self.validate(engine)
        rows=engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in rows),workers=rows)
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
