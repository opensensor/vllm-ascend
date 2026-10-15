"""Attach a gated shared-operand adapter while preserving all native owners."""

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
        trial = worker._qfn_shared_hc_operand
        assert trial["gate"]["passed"] and not state["owner"].failed
        if "adapter" not in trial:
            assert ACTION == "verify"
            trial["previous"] = state["adapter"]
            trial["adapter"] = trial["namespace"]["HCEpilogueAdapter"](
                state["owner"], state["original"], share_operand=True)
        active = state["adapter"]
        assert _GatedResidual.mix is active.replacement
        adapter = trial["previous"] if ACTION == "baseline" else trial["adapter"]
        assert not adapter.mismatches and not adapter.owner.failed
        if ACTION == "on": assert adapter.checked >= 48
        adapter.enabled = True
        adapter.verify = ACTION == "verify"
        state["adapter"] = adapter
        _GatedResidual.mix = adapter.replacement
        return dict(rank=worker.rank, passed=True, status=adapter.status())
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
    parser.add_argument("action", choices=("verify", "on", "baseline"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    client = QfnResidentClient(
        "http://127.0.0.1:8001",
        Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v15/control.token"),
        timeout=900,
    )
    result = apply(client, f"ACTION={args.action!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))
