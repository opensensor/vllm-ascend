# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-request timings from vLLM's CPU-side finished-request stats."""

from vllm.logger import init_logger
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

from vllm_ascend import envs

logger = init_logger("vllm")


def _duration_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}" if seconds >= 0 else "n/a"


def _rate(tokens: int, seconds: float) -> str:
    return f"{tokens / seconds:.1f}" if tokens > 0 and seconds > 0 else "n/a"


def _ms_per_token(tokens: int, seconds: float) -> str:
    return f"{seconds * 1000 / tokens:.1f}" if tokens > 0 and seconds > 0 else "n/a"


class AscendRequestTimingLogger(StatLoggerBase):
    """Log completed-request prefill and decode rates without NPU synchronizations."""

    def __init__(self, vllm_config: object, engine_index: int = 0) -> None:
        del vllm_config
        self.engine_index = engine_index
        self.enabled = envs.VLLM_ASCEND_LOG_REQUEST_TIMINGS

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        del scheduler_stats, mm_cache_stats, engine_idx
        if not self.enabled or iteration_stats is None:
            return

        for request in iteration_stats.finished_requests:
            computed_tokens = max(0, request.num_prompt_tokens - request.num_cached_tokens)
            decode_steps = max(0, request.num_generation_tokens - 1)
            prefill_time = request.prefill_time if request.num_generation_tokens > 0 else -1.0
            decode_time = request.decode_time if decode_steps else -1.0
            logger.info(
                "Engine %03d request %s (%s): prompt %d computed + %d cached "
                "in %s ms (%s ms/tok, %s effective tok/s); "
                "decode %d tokens, %d gaps in %s ms (%s ms/tok, %s tok/s); "
                "queue %s ms; total %s ms",
                self.engine_index,
                request.request_id,
                request.finish_reason,
                computed_tokens,
                request.num_cached_tokens,
                _duration_ms(prefill_time),
                _ms_per_token(computed_tokens, prefill_time),
                _rate(computed_tokens, prefill_time),
                request.num_generation_tokens,
                decode_steps,
                _duration_ms(decode_time),
                _ms_per_token(decode_steps, decode_time),
                _rate(decode_steps, decode_time),
                _duration_ms(request.queued_time),
                _duration_ms(request.e2e_latency),
            )

    def log_engine_initialized(self) -> None:
        pass
