# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from tools.qwen4exp.mixed_prefill_budget import MixedPrefillBudget


class Scheduler:
    def __init__(self, outputs=(), limit=0):
        self.running = [SimpleNamespace(num_output_tokens=n) for n in outputs]
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=limit)

    def schedule(self):
        return self.scheduler_config.long_prefill_token_threshold


@pytest.mark.parametrize(
    "outputs,limit,expected",
    [
        ((), 0, 0),
        ((0,), 0, 0),
        ((0, 0), 0, 0),
        ((3, 0), 0, 512),
        ((3, 0), 256, 256),
        ((3, 0), 1024, 512),
        ((0,), 1024, 1024),
    ],
)
def test_cap_only_with_output_and_preserve_stricter_existing_limit(outputs, limit, expected):
    scheduler = Scheduler(outputs, limit)
    owner = MixedPrefillBudget(scheduler, 512)
    assert scheduler.schedule() == expected
    assert scheduler.scheduler_config.long_prefill_token_threshold == limit
    owner.restore()
    assert "schedule" not in vars(scheduler)
    assert scheduler.schedule() == limit


def test_restore_config_even_if_scheduler_fails():
    scheduler = Scheduler((3, 0))

    def fail():
        assert scheduler.scheduler_config.long_prefill_token_threshold == 512
        raise RuntimeError("allocation failed")

    scheduler.schedule = fail
    owner = MixedPrefillBudget(scheduler, 512)
    with pytest.raises(RuntimeError, match="allocation failed"):
        scheduler.schedule()
    assert scheduler.scheduler_config.long_prefill_token_threshold == 0
    owner.restore()
    assert scheduler.schedule is fail


def test_refuse_to_remove_another_wrapper():
    scheduler = Scheduler()
    owner = MixedPrefillBudget(scheduler, 512)
    scheduler.schedule = lambda: None
    with pytest.raises(RuntimeError, match="ownership changed"):
        owner.restore()


@pytest.mark.parametrize("limit", [0, -128, 1, 129])
def test_invalid_checkpoint_geometry(limit):
    with pytest.raises(ValueError, match="128-token"):
        MixedPrefillBudget(Scheduler(), limit)
