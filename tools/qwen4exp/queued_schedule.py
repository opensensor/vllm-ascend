# SPDX-License-Identifier: Apache-2.0
"""Materialize an optimistic engine schedule from ordered worker results.

This bridge permits an engine batch queue without enabling the worker's
placeholder-based async history path. Worker execution remains ordered, and
its sampled-token history is authoritative when a queued RPC starts. This
removes the engine round trip; it does not remove worker CPU bookkeeping.
"""

from dataclasses import replace


def can_use_device_drafts(runner, schedule, transition, history):
    """Admit an unchanged MTP2 batch whose input/history packets are on device.

    Zero-filled CPU draft slots below are shape reservations, not model input.
    The consumer hooks must assert that both target IDs and PLE history came
    from this exact device packet before allowing model execution.
    """
    if transition is None or history is None:
        return False
    batch = runner.input_batch
    ids = tuple(batch.req_ids)
    pending = transition.next_pending
    return (
        1 <= len(ids) <= 4
        and transition.enabled
        and history.enabled
        and not transition.failed
        and not history.failed
        and not transition.verify
        and not history.verify
        and pending is not None
        and pending[0] == ids
        and history.pending == ids
        and not schedule.scheduled_new_reqs
        and not schedule.finished_req_ids
        and not schedule.scheduled_cached_reqs.resumed_req_ids
        and tuple(schedule.scheduled_cached_reqs.req_ids) == ids
        and set(schedule.num_scheduled_tokens) == set(ids)
        and schedule.total_num_scheduled_tokens == len(ids) * 3
        and all(schedule.num_scheduled_tokens[key] == 3 for key in ids)
        and all(len(schedule.scheduled_spec_decode_tokens.get(key, ())) == 2 for key in ids)
        and not schedule.has_structured_output_requests
        and not runner.num_prompt_logprobs
        and not batch.sampling_metadata.output_token_ids
        and not runner.use_dcp
        and not runner.uses_mrope
        and not runner.lora_config
        and not runner.enable_prompt_embeds
    )


def materialize_schedule(schedule, requests, draft_ids, *, max_model_len):
    """Resolve optimistic counts/drafts without changing allocation or geometry.

    The engine retains its original optimistic schedule for rejection rollback.
    Only the worker-local copy is adjusted. Prefix-recompute/resumed requests
    retain their engine-provided computed offset. No page ownership is inferred
    from token IDs, and a missing draft is an error rather than a -1 model input.
    """
    cached = schedule.scheduled_cached_reqs
    computed = list(cached.num_computed_tokens)
    outputs = list(cached.num_output_tokens)
    drafts = dict(schedule.scheduled_spec_decode_tokens)
    corrected = 0
    for row, req_id in enumerate(cached.req_ids):
        request = requests.get(req_id)
        if request is None:
            raise ValueError(f"queued schedule has no worker request: {req_id}")
        if req_id in cached.resumed_req_ids:
            if any(token < 0 for token in drafts.get(req_id, ())):
                raise ValueError("resumed requests cannot consume speculative placeholders")
            continue
        actual_outputs = len(request.output_token_ids)
        if not actual_outputs:
            if any(token < 0 for token in drafts.get(req_id, ())):
                raise ValueError("prefill cannot consume speculative placeholders")
            continue
        # The final sampled (bonus) token is the next target input.
        actual_computed = request.num_prompt_tokens + actual_outputs - 1
        scheduled = schedule.num_scheduled_tokens[req_id]
        width = len(drafts.get(req_id, ()))
        if computed[row] < actual_computed and width == 0:
            # Recompute can span several prefill chunks after the one carrying
            # resumed_req_ids. Existing generated tokens remain input history;
            # their presence alone does not make this a decode iteration.
            if computed[row] < 0 or scheduled <= 0 or computed[row] + scheduled > actual_computed + 1:
                raise ValueError("queued recompute chunk exceeds authoritative worker history")
            continue
        if scheduled != width + 1 or actual_computed > computed[row]:
            raise ValueError("queued decode is not an optimistic one-token-plus-drafts schedule")
        if actual_computed < 0 or actual_computed + scheduled > max_model_len:
            raise ValueError("queued decode exceeds the model context")
        if outputs[row] < actual_outputs:
            # Engine counts may trail the worker by one in-flight step.
            # Preserve worker history; upstream _update_states otherwise
            # truncates tokens the engine has not yet observed.
            outputs[row] = actual_outputs
        elif outputs[row] > actual_outputs:
            # Optimistic output counts include placeholders, never real tokens.
            outputs[row] = actual_outputs
        corrected += computed[row] != actual_computed
        computed[row] = actual_computed
        if width:
            available = draft_ids.get(req_id)
            if available is None or len(available) < width or any(token < 0 for token in available[:width]):
                raise ValueError("queued decode requires actual worker-owned draft tokens")
            drafts[req_id] = list(available[:width])
    resolved = replace(
        schedule,
        scheduled_cached_reqs=replace(cached, num_computed_tokens=computed, num_output_tokens=outputs),
        scheduled_spec_decode_tokens=drafts,
    )
    return resolved, corrected


class QueuedScheduleBridge:
    """Worker-owned controls and receipts; no mutation of scheduler config."""

    def __init__(self, runner, transition=None, history=None):
        self.runner = runner
        self.enabled = False
        self.steps = self.corrected_rows = 0
        self.transition, self.history = transition, history
        self.device_drafts = False
        self.device_draft_steps = 0
        self.expected_device_inputs = self.expected_device_history = None

    def resolve(self, schedule):
        if not self.enabled:
            return schedule
        if self.runner.use_async_scheduling:
            raise RuntimeError("queued bridge requires authoritative synchronous worker history")
        if self.expected_device_inputs is not None or self.expected_device_history is not None:
            raise RuntimeError("previous queued device packet was not consumed")
        draft_ids = {}
        use_device = self.device_drafts and can_use_device_drafts(self.runner, schedule, self.transition, self.history)
        if use_device:
            draft_ids = {key: [0, 0] for key in self.runner.input_batch.req_ids}
        elif schedule.scheduled_spec_decode_tokens:
            values, ids = self.runner._get_draft_token_ids_cpu()
            draft_ids = dict(zip(ids, values))
        resolved, corrected = materialize_schedule(
            schedule, self.runner.requests, draft_ids, max_model_len=self.runner.max_model_len
        )
        self.steps += 1
        self.corrected_rows += corrected
        if use_device:
            self.expected_device_inputs = tuple(self.runner.input_batch.req_ids)
            self.device_draft_steps += 1
        return resolved

    def status(self):
        return dict(
            enabled=self.enabled,
            steps=self.steps,
            corrected_rows=self.corrected_rows,
            device_drafts=self.device_drafts,
            device_draft_steps=self.device_draft_steps,
            pending_device_inputs=self.expected_device_inputs is not None,
            pending_device_history=self.expected_device_history is not None,
            worker_cpu_bookkeeping_retained=True,
        )
