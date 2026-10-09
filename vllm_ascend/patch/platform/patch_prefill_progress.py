# SPDX-License-Identifier: Apache-2.0
"""Expose CPU scheduler prompt totals beside the existing chunk iteration logs.

No worker tensors, request text, token IDs or KV state are read. Progress is
explicitly scheduled work: async scheduling can run ahead of device completion.
Remove this wrapper when upstream iteration logging includes request totals.
"""

import functools

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler


def _enabled(scheduler):
    return scheduler.log_stats and getattr(scheduler.observability_config, "enable_logging_iteration_details", False)


def _progress(request, scheduled):
    total = request.num_prompt_tokens
    before = min(total, max(0, request.num_computed_tokens))
    chunk = min(max(0, scheduled), total - before)
    if chunk == 0:
        return None
    prefill = getattr(request, "prefill_stats", None)
    cached = getattr(prefill, "num_cached_tokens", getattr(request, "num_cached_tokens", None))
    # Older revisions use -1 before the first cache lookup. Unknown reuse must
    # not be presented as zero or inferred from an optimistic computed cursor.
    if cached is None or cached < 0:
        cached = "unknown"
    else:
        cached = min(total, cached)
    through = before + chunk
    return request.request_id, total, cached, chunk, through, total - through


def install_prefill_progress(scheduler_class):
    """Install once; inherit the existing iteration-log setting and scheduler."""
    if getattr(scheduler_class._update_after_schedule, "__ascend_prefill_progress__", False):
        return False
    original_add = scheduler_class.add_request
    original_update = scheduler_class._update_after_schedule
    info = logger.info

    @functools.wraps(original_add)
    def add_request(self, request, *args, **kwargs):
        is_new = request.request_id not in self.requests
        result = original_add(self, request, *args, **kwargs)
        if is_new and _enabled(self) and request.request_id in self.requests:
            info(
                "Prefill received: request=%s prompt_tokens=%d cache_lookup=pending",
                request.request_id,
                request.num_prompt_tokens,
            )
        return result

    @functools.wraps(original_update)
    def update_after_schedule(self, scheduler_output, *args, **kwargs):
        rows = []
        if _enabled(self):
            for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
                request = self.requests.get(request_id)
                if request is not None and (row := _progress(request, scheduled)) is not None:
                    rows.append(row)
        result = original_update(self, scheduler_output, *args, **kwargs)
        for row in rows:
            info(
                "Prefill scheduled: request=%s prompt_tokens=%d cached_tokens=%s "
                "prompt_chunk=%d scheduled_through=%d remaining_to_schedule=%d",
                *row,
            )
        return result

    update_after_schedule.__ascend_prefill_progress__ = True
    scheduler_class.add_request = add_request
    scheduler_class._update_after_schedule = update_after_schedule
    return True


install_prefill_progress(Scheduler)
