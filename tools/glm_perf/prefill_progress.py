# SPDX-License-Identifier: Apache-2.0
"""Prompt progress from CPU SchedulerOutput metadata, including resident reloads."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    total: int
    reused: int | str


class PrefillProgress:
    def __init__(self):
        self.requests = {}

    def scheduled(self, output):
        for request_id in output.finished_req_ids:
            self.requests.pop(request_id, None)
        starts = {}
        for request in output.scheduled_new_reqs:
            ids = request.prompt_token_ids
            total = (
                len(ids)
                if ids is not None
                else request.prompt_embeds.shape[0]
                if request.prompt_embeds is not None
                else 0
            )
            before = max(0, request.num_computed_tokens)
            self.requests[request.req_id] = Prompt(total, min(total, before))
            starts[request.req_id] = before
        cached = output.scheduled_cached_reqs
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        for request_id in cached.resumed_req_ids:
            prompt = self.requests.get(request_id)
            if prompt is not None:
                self.requests[request_id] = Prompt(prompt.total, "unknown_after_preemption")
        rows = []
        for request_id, scheduled in output.num_scheduled_tokens.items():
            prompt = self.requests.get(request_id)
            before = starts.get(request_id)
            if prompt is None or before is None:
                continue
            before = min(prompt.total, max(0, before))
            chunk = min(max(0, scheduled), prompt.total - before)
            if chunk:
                through = before + chunk
                rows.append((request_id, prompt.total, prompt.reused, chunk, through, prompt.total - through))
        return rows
