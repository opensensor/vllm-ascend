"""Swap only the drained scheduler budget for a bounded before/after comparison."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
SOURCE = """
from pathlib import Path
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        previous = engine._qfn_mixed_budget
        assert engine.scheduler.schedule is previous.wrapper
        assert previous.max_prefill_tokens == 512
        assert engine.async_scheduling and engine.batch_queue_size == 2
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        from tools.qwen4exp.mixed_prefill_budget import MixedPrefillBudget
        namespace = {"__name__": "qfn_baseline_mixed_prefill_budget"}
        if MODE == "baseline":
            path = Path("/srv/ai/storage/artifacts/qfn-recovery-20261016/baseline_mixed_prefill_budget.py")
            exec(compile(path.read_text(), str(path), "exec"), namespace)
            cls = namespace["MixedPrefillBudget"]
        else:
            cls = MixedPrefillBudget
        previous = engine._qfn_mixed_budget
        previous.restore()
        owner = cls(engine.scheduler, previous.max_prefill_tokens)
        engine._qfn_mixed_budget = owner
        return dict(passed=True, mode=MODE, budget=owner.status())
def prepare(engine): return Candidate()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "candidate"))
    parser.add_argument("label")
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
    result = apply(client, f"MODE={args.mode!r}\n" + SOURCE, ROOT / args.label)
    print(json.dumps(result["applied"], indent=2))
