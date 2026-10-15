"""Protect pending first samples using the existing scheduler prefill cap."""

from pathlib import Path


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        owner = engine._qfn_mixed_budget
        assert engine.scheduler.schedule is owner.wrapper
        assert owner.max_prefill_tokens == 512
        assert engine.async_scheduling and engine.batch_queue_size == 2
        return dict(passed=True)

    def apply(self, engine):
        self.validate(engine)
        namespace = {"__name__": "qfn_pending_first_sample_budget"}
        path = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015/mixed_prefill_budget.py")
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        previous = engine._qfn_mixed_budget
        self.previous, self.namespace = previous, namespace
        previous.restore()
        owner = namespace["MixedPrefillBudget"](engine.scheduler, previous.max_prefill_tokens)
        owner.calls, owner.capped_steps = previous.calls, previous.capped_steps
        engine._qfn_mixed_budget = owner
        return dict(passed=True, budget=owner.status())


def prepare(engine):
    return Candidate()
