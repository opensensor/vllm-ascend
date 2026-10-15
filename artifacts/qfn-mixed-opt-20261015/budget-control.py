"""Drained admission for a conditional mixed-prefill chunk budget."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        owner = getattr(engine, '_qfn_mixed_budget', None)
        if ACTION == 'install':
            assert owner is None
            assert type(engine.scheduler).__name__ == 'QueuedBalanceScheduler'
            assert LIMIT >= 128 and LIMIT % 128 == 0
        else:
            assert owner is not None
            assert engine.scheduler.schedule is owner.wrapper
        return dict(passed=True)
    def apply(self, engine):
        self.validate(engine)
        from tools.qwen4exp.mixed_prefill_budget import MixedPrefillBudget
        if ACTION == 'install':
            engine._qfn_mixed_budget = MixedPrefillBudget(engine.scheduler, LIMIT)
        owner = engine._qfn_mixed_budget
        result = owner.status()
        if ACTION == 'restore':
            owner.restore()
            engine._qfn_mixed_budget = None
        return dict(passed=True, budget=result, action=ACTION)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "restore", "status"))
    parser.add_argument("--limit", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = f"ACTION={args.action!r}\nLIMIT={args.limit!r}\n" + SOURCE
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, source, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
