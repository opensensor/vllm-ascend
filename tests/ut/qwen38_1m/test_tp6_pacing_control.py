# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.qwen4exp.tp6_pacing_control import PacingCandidate
from vllm_ascend.core.qwen_prefill_pacing import PrefillPacingConfig, PrefillPacingPolicy


class QwenPrefillPacedScheduler:
    def __init__(self):
        self._prefill_pacing = PrefillPacingPolicy(
            PrefillPacingConfig(initial_tokens=128, max_tokens=640, decode_only_steps=8)
        )
        self._prefill_pacing.ms_per_token = 5
        self._prefill_pending_step = None
        self._prefill_decode_steps_remaining = 7
        self.unfinished = 0

    def get_num_unfinished_requests(self):
        return self.unfinished


def engine():
    return SimpleNamespace(
        scheduler=QwenPrefillPacedScheduler(),
        _pause_complete=lambda: True,
        batch_queue=[],
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=6)),
    )


def test_changes_only_pacing_and_restores_original_owner():
    e = engine()
    old = e.scheduler._prefill_pacing
    candidate = PacingCandidate(e, 4, 256)
    assert e.scheduler._prefill_pacing is old
    assert candidate.validate(e)["passed"]
    candidate.apply(e)
    assert e.scheduler._prefill_pacing.config.decode_only_steps == 4
    assert e.scheduler._prefill_pacing.budget(2048) == 256
    assert e.scheduler._prefill_decode_steps_remaining == 0
    assert old.config.decode_only_steps == 8
    PacingCandidate(e, 2, None).apply(e)
    assert e.scheduler._prefill_pacing.config.max_tokens == 640
    PacingCandidate(e, None, None).apply(e)
    assert e.scheduler._prefill_pacing is old


@pytest.mark.parametrize("steps,chunk", [(True, None), (-1, None), (65, None), (4, 129), (4, 0), (None, 128)])
def test_rejects_invalid_configuration(steps, chunk):
    with pytest.raises(ValueError):
        PacingCandidate(engine(), steps, chunk)


@pytest.mark.parametrize("state", ["paused", "requests", "queue", "pending", "queued"])
def test_refuses_inflight_work(state):
    e = engine()
    if state == "paused":
        e._pause_complete = lambda: False
    elif state == "requests":
        e.scheduler.unfinished = 1
    elif state == "queue":
        e.batch_queue.append(object())
    elif state == "pending":
        e.scheduler._prefill_pending_step = object()
    else:
        e.scheduler._prefill_queued_steps = {1: object()}
    with pytest.raises(RuntimeError, match="drain"):
        PacingCandidate(e, 4, None)


def test_refuses_stale_owner():
    e = engine()
    candidate = PacingCandidate(e, 4, None)
    PacingCandidate(e, 2, None).apply(e)
    with pytest.raises(RuntimeError, match="ownership"):
        candidate.apply(e)


def test_refuses_wrong_parallel_size_and_scheduler():
    e = engine()
    e.vllm_config.parallel_config.tensor_parallel_size = 4
    with pytest.raises(RuntimeError, match="TP6"):
        PacingCandidate(e, 4, None)
    e = engine()
    e.scheduler = SimpleNamespace(get_num_unfinished_requests=lambda: 0)
    with pytest.raises(RuntimeError, match="qualified paced scheduler"):
        PacingCandidate(e, 4, None)
