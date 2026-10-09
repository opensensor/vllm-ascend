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
        super().__init__(vllm_config, *args, **kwargs)

    def schedule(self, throttle_prefills: bool = False):
        self._prefill_pending_step = None
        original_config = self.scheduler_config
        original_maximum = self.max_num_scheduled_tokens
        original_order = {request.request_id: index for index, request in enumerate(self.running)}
        decoders = [request for request in self.running if not request.is_prefill_chunk]
        # An indivisible encoder span may exceed the paced budget. Let the
        # upstream multimodal scheduler allocate it with the original limits;
        # otherwise a fresh image could wait forever behind a small token cap.
        encoder_prefill = any(
            request.is_prefill_chunk and request.has_encoder_inputs for request in self.running
        ) or any(request.has_encoder_inputs for request in self.waiting)
        if decoders and not encoder_prefill:
            self.running.sort(key=lambda request: request.is_prefill_chunk)
            decode_reserve = sum(
                max(1, request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens)
                for request in decoders
            )
            budget = self._prefill_pacing.budget(original_maximum)
            self.max_num_scheduled_tokens = min(original_maximum, decode_reserve + budget)
            self.scheduler_config = copy(original_config)
            threshold = original_config.long_prefill_token_threshold
            self.scheduler_config.long_prefill_token_threshold = min(threshold, budget) if threshold > 0 else budget
        started = time.monotonic()
        prefill_ids = {request.request_id for request in self.running if request.is_prefill_chunk}
        prefill_ids.update(request.request_id for request in self.waiting)
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        finally:
            self.scheduler_config = original_config
            self.max_num_scheduled_tokens = original_maximum
            # Keep FCFS order across calls, preserving upstream admissions and
            # removals rather than restoring a stale copy of running requests.
            self.running.sort(key=lambda request: original_order.get(request.request_id, len(original_order)))
        prefill_tokens = sum(
            tokens for request_id, tokens in output.num_scheduled_tokens.items() if request_id in prefill_ids
        )
        self._prefill_pending_step = (output, started, prefill_tokens)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        pending = self._prefill_pending_step
        self._prefill_pending_step = None
        result = super().update_from_output(scheduler_output, model_runner_output)
        if pending is not None and pending[0] is scheduler_output:
            self._prefill_pacing.observe(pending[2], (time.monotonic() - pending[1]) * 1000)
        return result
