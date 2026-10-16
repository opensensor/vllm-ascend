# SPDX-License-Identifier: Apache-2.0
"""Drained, reversible TP6 pacing experiments without replacing the scheduler."""

from dataclasses import asdict, replace
from pathlib import Path


def require_idle(engine):
    scheduler = engine.scheduler
    if (
        not engine._pause_complete()
        or engine.batch_queue
        or scheduler.get_num_unfinished_requests()
        or getattr(scheduler, "_prefill_pending_step", None) is not None
        or getattr(scheduler, "_prefill_queued_steps", {})
    ):
        raise RuntimeError("pause and drain all requests and scheduler outputs first")
    if type(scheduler).__name__ != "QwenPrefillPacedScheduler":
        raise RuntimeError("expected the qualified paced scheduler")
    if engine.vllm_config.parallel_config.tensor_parallel_size != 6:
        raise RuntimeError("this experiment requires TP6")
    return scheduler


class PacingCandidate:
    def __init__(self, engine, decode_steps, chunk_tokens):
        scheduler = require_idle(engine)
        self.scheduler = scheduler
        self.previous = scheduler._prefill_pacing
        self.baseline = getattr(engine, "_qfn_tp6_pacing_baseline", self.previous)
        if decode_steps is None:
            if chunk_tokens is not None:
                raise ValueError("restore cannot change chunk size")
            self.policy = self.baseline
        else:
            # The existing policy validates integer bounds and alignment.
            options = dict(decode_only_steps=decode_steps)
            if chunk_tokens is not None:
                options.update(min_tokens=chunk_tokens, initial_tokens=chunk_tokens, max_tokens=chunk_tokens)
            config = replace(self.baseline.config, **options)
            self.policy = type(self.previous)(config)
            self.policy.ms_per_token = self.previous.ms_per_token

    def validate(self, engine):
        scheduler = require_idle(engine)
        if scheduler is not self.scheduler or scheduler._prefill_pacing is not self.previous:
            raise RuntimeError("pacing ownership changed after preparation")
        return dict(passed=True, before=asdict(self.previous.config), after=asdict(self.policy.config))

    def apply(self, engine):
        receipt = self.validate(engine)
        engine._qfn_tp6_pacing_baseline = self.baseline
        self.scheduler._prefill_pacing = self.policy
        self.scheduler._prefill_decode_steps_remaining = 0
        return receipt


def engine_source(decode_steps=None, chunk_tokens=None):
    source = Path(__file__).read_text()
    return source + f"\ndef prepare(engine):\n    return PacingCandidate(engine, {decode_steps!r}, {chunk_tokens!r})\n"
