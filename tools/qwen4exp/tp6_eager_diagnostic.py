# SPDX-License-Identifier: Apache-2.0
"""Temporary target-only eager diagnostic; retains all existing graph owners."""

import inspect
from functools import wraps
from pathlib import Path

from tools.qwen4exp import tp6_finalize_gate


class ForceEager:
    def __init__(self, runner):
        self.runner = runner
        self.name = "_determine_batch_execution_and_padding"
        self.original = getattr(runner, self.name)
        self.owned = self.name in vars(runner)
        self.previous = vars(runner).get(self.name)
        self.signature = inspect.signature(self.original)
        if "force_eager" not in self.signature.parameters:
            raise ValueError("target runner has no explicit eager override")
        self.calls = 0
        self.installed = False

        @wraps(self.original)
        def execute(*args, **kwargs):
            arguments = self.signature.bind(*args, **kwargs)
            arguments.arguments["force_eager"] = True
            self.calls += 1
            return self.original(*arguments.args, **arguments.kwargs)

        self.wrapper = execute

    def install(self):
        if getattr(self.runner, self.name) != self.original or self.installed:
            raise RuntimeError("eager diagnostic ownership changed")
        setattr(self.runner, self.name, self.wrapper)
        self.installed = True

    def restore(self):
        if not self.installed or getattr(self.runner, self.name) is not self.wrapper:
            raise RuntimeError("eager diagnostic ownership changed")
        if self.owned:
            setattr(self.runner, self.name, self.previous)
        else:
            delattr(self.runner, self.name)
        self.installed = False


ENGINE = """
def operation(wrapper, mutate):
    worker = getattr(wrapper, "worker", wrapper)
    owner = getattr(worker, "_qfn_force_eager_diagnostic", None)
    try:
        namespace = {}
        exec(compile(SNAPSHOT, "<eager-snapshot>", "exec"), namespace)
        before = namespace["worker_snapshot"](worker)
        if getattr(worker.model_runner, "execute_model_state", None) is not None:
            raise RuntimeError("worker must be drained")
        if owner is None:
            resources = {}
            exec(compile(RESOURCE, "<eager-diagnostic>", "exec"), resources)
            owner = resources["ForceEager"](worker.model_runner)
            worker._qfn_force_eager_diagnostic = owner
        if mutate:
            if ENABLE and not owner.installed:
                owner.install()
            elif not ENABLE and owner.installed:
                owner.restore()
        after = namespace["worker_snapshot"](worker)
        return dict(passed=before==after, rank=worker.rank, installed=owner.installed, calls=owner.calls,
                    before=before, after=after)
    except Exception as error:
        return dict(passed=False, rank=worker.rank, error=repr(error))
class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError("drain first")
        rows = engine.model_executor.collective_rpc(operation, args=(False,))
        return dict(passed=len(rows)==6 and all(row["passed"] for row in rows), workers=rows)
    def apply(self, engine):
        rows = engine.model_executor.collective_rpc(operation, args=(True,))
        return dict(passed=len(rows)==6 and all(row["passed"] for row in rows), workers=rows)
def prepare(engine):
    return Candidate()
"""


def engine_source(enabled):
    resource = Path(__file__).read_text().replace("from tools.qwen4exp import tp6_finalize_gate", "")
    return (
        f"ENABLE={enabled!r}\nRESOURCE={resource!r}\n"
        f"SNAPSHOT={Path(tp6_finalize_gate.__file__).read_text()!r}\n" + ENGINE
    )
