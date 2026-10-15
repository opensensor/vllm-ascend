"""Compare only the qualified HC epilogue launch bridge in drained serving."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from vllm_ascend.models.qwen4_exp.model import _GatedResidual
        state = worker._qfn_hc_epilogues
        trial = worker._qfn_hc_queue_candidate
        adapter = state["adapter"]
        assert _GatedResidual.mix is adapter.replacement
        assert trial["gate"]["passed"] and not trial["owner"].failed
        assert not adapter.owner.failed and not adapter.mismatches
        if "draining_candidate" not in trial:
            trial["draining_candidate"] = adapter.candidate
            trial["queued_candidate"] = state["final_namespace"]["make_mix"](trial["owner"], state["original"])
        if ACTION == "verify":
            trial["verification_start"] = adapter.checked
            adapter.candidate = trial["queued_candidate"]
            adapter.verify = True
        elif ACTION == "queued":
            assert adapter.checked - trial["verification_start"] >= 48
            adapter.candidate = trial["queued_candidate"]
            adapter.verify = False
        elif ACTION == "draining":
            adapter.candidate = trial["draining_candidate"]
            adapter.verify = False
        adapter.enabled = True
        return dict(rank=worker.rank, passed=True, status=adapter.status(),
                    queue_failed=trial["owner"].failed,
                    queued=adapter.candidate is trial["queued_candidate"])
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))

class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in rows) and len(rows) == 4, workers=rows)

def prepare(engine): return Candidate()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "queued", "draining"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    client = QfnResidentClient(
        "http://127.0.0.1:8001",
        Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v15/control.token"),
        timeout=900,
    )
    result = apply(client, f"ACTION={args.action!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))
