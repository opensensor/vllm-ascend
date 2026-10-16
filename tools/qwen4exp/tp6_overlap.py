# SPDX-License-Identifier: Apache-2.0
"""Drained, reversible TP6 early-draft and exact-prefix controls."""

import hashlib
import inspect
import textwrap
from types import FunctionType

from tools.qwen4exp.resident_candidates import qfn_early_tp_draft

BASE_QSA_SHA256 = "7e4c67ed1e2854fb8a696258fc3f5e8c128a93b5780a986b1ff0237c1afb0df2"


def build_selection(original, source):
    if hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest() != BASE_QSA_SHA256:
        raise ValueError("requires the profiled QSA selection source")
    namespace = dict(original.__globals__)
    exec(compile(source, "<qfn-exact-complete-prefix>", "exec"), namespace)
    compiled = namespace[original.__name__]
    result = FunctionType(compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__)
    result.__kwdefaults__, result.__annotations__ = compiled.__kwdefaults__, compiled.__annotations__
    return result


class OverlapControl:
    def __init__(self, worker, source):
        # Loaded only by an explicit controller on a paused worker. Keep every
        # original function and pending verification owner for the worker life.
        from vllm_ascend.models.qwen4_exp import model
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        self.worker = worker
        self.runner_class = NPUModelRunner
        self.model = model
        self.sample = NPUModelRunner.sample_tokens
        self.selection = model.qsa_indexer_select_groups_310
        self.early_sample = qfn_early_tp_draft.build_forward(self.sample)
        self.prefix_selection = build_selection(self.selection, source)
        self.early = self.prefix = self.verify = self.failed = False
        self.early_calls = self.prefix_calls = self.checked_calls = 0
        self.checked_capacities = set()

    def install(self):
        owner = self

        def sample_tokens(runner, *args, **kwargs):
            if owner.failed:
                raise RuntimeError("overlap verification failed; retain owners")
            if owner.early:
                state = runner.execute_model_state
                if state is not None and qfn_early_tp_draft.eligible(runner, state[2], state[-1], state[0]):
                    owner.early_calls += 1
                return owner.early_sample(runner, *args, **kwargs)
            return owner.sample(runner, *args, **kwargs)

        def selection(*args, **kwargs):
            if owner.failed:
                raise RuntimeError("prefix verification failed; retain owners")
            if not owner.prefix:
                return owner.selection(*args, **kwargs)
            result = owner.prefix_selection(*args, **kwargs)
            owner.prefix_calls += 1
            if owner.verify:
                import torch

                owner.failed = True
                expected = owner.selection(*args, **kwargs)
                for name in ("group_indices", "group_counts", "tail_starts", "tail_counts"):
                    torch.testing.assert_close(getattr(result, name), getattr(expected, name), rtol=0, atol=0)
                owner.checked_calls += 1
                owner.checked_capacities.add(result.group_indices.shape[1])
                owner.failed = False
            return result

        self.runner_class.sample_tokens = sample_tokens
        self.model.qsa_indexer_select_groups_310 = selection

    def configure(self, *, early, prefix, verify):
        if any(type(value) is not bool for value in (early, prefix, verify)) or (verify and not prefix):
            raise ValueError("expected boolean early/prefix/verify; verification requires prefix")
        if self.failed or self.worker.model_runner.execute_model_state is not None:
            raise RuntimeError("drain healthy execution before changing overlap")
        if prefix and not verify and not self.checked_calls:
            raise RuntimeError("verify actual selection outputs before fast prefix mode")
        self.early, self.prefix, self.verify = early, prefix, verify
        return self.status()

    def status(self):
        return dict(
            rank=self.worker.rank,
            passed=not self.failed,
            early=self.early,
            prefix=self.prefix,
            verify=self.verify,
            failed=self.failed,
            early_calls=self.early_calls,
            prefix_calls=self.prefix_calls,
            checked_calls=self.checked_calls,
            checked_capacities=sorted(self.checked_capacities),
        )
