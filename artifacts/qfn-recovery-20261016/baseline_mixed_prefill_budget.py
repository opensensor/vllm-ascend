# SPDX-License-Identifier: Apache-2.0
"""Bound prefill chunks while an existing request is producing output."""


class MixedPrefillBudget:
    """Reversible scheduler wrapper; install and restore only while drained.

    Preserve the scheduler's request ordering, cache accounting and async
    placeholder lifecycle. Only its existing per-prefill cap is changed for
    the duration of schedule(). Prefill-only and decode-only token budgets
    otherwise remain under the original scheduler's control.
    """

    def __init__(self, scheduler, max_prefill_tokens):
        if max_prefill_tokens < 128 or max_prefill_tokens % 128:
            raise ValueError("prefill cap must be a positive multiple of the 128-token checkpoint block")
        self.scheduler = scheduler
        self.max_prefill_tokens = max_prefill_tokens
        self.original = scheduler.schedule
        self.had_instance_schedule = "schedule" in vars(scheduler)
        self.capped_steps = 0
        self.calls = 0
        self.wrapper = self.schedule
        scheduler.schedule = self.wrapper

    def schedule(self, *args, **kwargs):
        config = self.scheduler.scheduler_config
        original_limit = config.long_prefill_token_threshold
        has_decoder = any(request.num_output_tokens > 0 for request in self.scheduler.running)
        if has_decoder:
            config.long_prefill_token_threshold = (
                min(original_limit, self.max_prefill_tokens) if original_limit > 0 else self.max_prefill_tokens
            )
            self.capped_steps += 1
        self.calls += 1
        try:
            return self.original(*args, **kwargs)
        finally:
            config.long_prefill_token_threshold = original_limit

    def restore(self):
        if self.scheduler.schedule is not self.wrapper:
            raise RuntimeError("scheduler schedule ownership changed")
        if self.had_instance_schedule:
            self.scheduler.schedule = self.original
        else:
            del self.scheduler.schedule

    def status(self):
        return dict(max_prefill_tokens=self.max_prefill_tokens, calls=self.calls, capped_steps=self.capped_steps)
