# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit synchronous Qwen scheduler with bounded prefixes and prefill work.

Select with --scheduler-cls; configuration lives in additional_config under
qwen_prefill_pacing. This does not poll sensors or replace the 94/85 thermal
controller. No request or encoder tokens are discarded.
"""

import time
from copy import copy

from vllm_ascend.core.prefix_mamba_scheduler import PrefixMambaBoundedScheduler
from vllm_ascend.core.qwen_prefill_pacing import PrefillPacingConfig, PrefillPacingPolicy


class QwenPrefillPacedScheduler(PrefixMambaBoundedScheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        options = (vllm_config.additional_config or {}).get("qwen_prefill_pacing", {})
        if not isinstance(options, dict):
            raise ValueError("qwen_prefill_pacing must be a dictionary")
        self._prefill_pacing = PrefillPacingPolicy(PrefillPacingConfig(**options))
        if not vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Qwen prefill pacing requires chunked prefill")
        self._prefill_pending_step = None
        self._prefill_decode_steps_remaining = 0
        super().__init__(vllm_config, *args, **kwargs)

    def _encoder_prefill_budget(self, budget):
        """Widen only an overlapping atomic image span, never its entire prompt.

        Decoder-side chunking and encoder admission are separate. Upstream still
        admits the full encoder item using its unchanged encoder compute/cache
        budget. Only disable_chunked_mm_input requires the decoder token window
        to cover an entire item. Consumed and future items must not bypass pacing.
        """
        if not self.scheduler_config.disable_chunked_mm_input:
            return budget
        required = budget
        running_ids = {request.request_id for request in self.running}
        for request in (*self.running, *self.waiting):
            if not request.has_encoder_inputs or (request.request_id in running_ids and not request.is_prefill_chunk):
                continue
            start = request.num_computed_tokens
            # Use the original paced window: widening for one item must not
            # recursively admit every later image in a very long prompt.
            for feature in request.mm_features:
                position = feature.mm_position
                end = position.offset + position.length
                if position.offset < start + budget and end > start:
                    required = max(required, end - start)
        return required

    def schedule(self, throttle_prefills: bool = False):
        self._prefill_pending_step = None
        original_config = self.scheduler_config
        original_maximum = self.max_num_scheduled_tokens
        original_capacity_bound = self.prefill_capacity_bound
        original_order = {request.request_id: index for index, request in enumerate(self.running)}
        decoders = [request for request in self.running if not request.is_prefill_chunk]
        decode_only = bool(decoders and self._prefill_decode_steps_remaining)
        if not decoders:
            self._prefill_decode_steps_remaining = 0
        started = time.monotonic()
        prefill_ids = {request.request_id for request in self.running if request.is_prefill_chunk}
        prefill_ids.update(request.request_id for request in self.waiting)
        try:
            if decoders:
                self.running.sort(key=lambda request: request.is_prefill_chunk)
                decode_reserve = sum(
                    max(1, request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens)
                    for request in decoders
                )
                if decode_only:
                    self.max_num_scheduled_tokens = min(original_maximum, decode_reserve)
                    # DP's throughput override can otherwise schedule prefills even
                    # on a requested decode-only step when waiting requests exist.
                    self.prefill_capacity_bound = False
                else:
                    budget = self._prefill_pacing.budget(original_maximum)
                    threshold = original_config.long_prefill_token_threshold
                    if threshold > 0:
                        budget = min(threshold, budget)
                    budget = self._encoder_prefill_budget(budget)
                    budget = min(budget, max(0, original_maximum - decode_reserve))
                    self.max_num_scheduled_tokens = min(original_maximum, decode_reserve + budget)
                    self.scheduler_config = copy(original_config)
                    self.scheduler_config.long_prefill_token_threshold = budget
            output = super().schedule(throttle_prefills=throttle_prefills or decode_only)
        finally:
            self.scheduler_config = original_config
            self.max_num_scheduled_tokens = original_maximum
            if decode_only:
                self.prefill_capacity_bound = original_capacity_bound
            # Keep FCFS order across calls, preserving upstream admissions and
            # removals rather than restoring a stale copy of running requests.
            self.running.sort(key=lambda request: original_order.get(request.request_id, len(original_order)))
        prefill_tokens = sum(
            tokens for request_id, tokens in output.num_scheduled_tokens.items() if request_id in prefill_ids
        )
        self._prefill_pending_step = (output, started, prefill_tokens, decode_only)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        pending = self._prefill_pending_step
        self._prefill_pending_step = None
        result = super().update_from_output(scheduler_output, model_runner_output)
        if pending is not None and pending[0] is scheduler_output:
            self._prefill_pacing.observe(pending[2], (time.monotonic() - pending[1]) * 1000)
            if pending[2]:
                self._prefill_decode_steps_remaining = self._prefill_pacing.config.decode_only_steps
            elif pending[3] and scheduler_output.num_scheduled_tokens:
                self._prefill_decode_steps_remaining = max(0, self._prefill_decode_steps_remaining - 1)
        return result
