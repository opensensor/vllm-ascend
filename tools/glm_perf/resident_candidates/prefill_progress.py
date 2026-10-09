# SPDX-License-Identifier: Apache-2.0
"""Add rank-zero prompt progress to an existing resident candidate."""

import functools

from tools.glm_perf.prefill_progress import PrefillProgress

TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.execute_model"


def extend_replacements(changes):
    # Worker-only imports keep CPU tooling independent of NPU initialization.
    from vllm.logger import logger

    from vllm_ascend._310p.worker_310p import NPUWorker310

    result = dict(changes)
    original = result.get(TARGET, NPUWorker310.execute_model)
    original = getattr(original, "__prefill_progress_original__", original)
    progress = PrefillProgress()
    info = logger.info

    @functools.wraps(original)
    def execute(self, scheduler_output, *args, **kwargs):
        if self.rank == 0 and getattr(self.vllm_config.observability_config, "enable_logging_iteration_details", False):
            for row in progress.scheduled(scheduler_output):
                info(
                    "Prefill scheduled: request=%s prompt_tokens=%d reused_at_admission=%s "
                    "prompt_chunk=%d scheduled_through=%d remaining_to_schedule=%d",
                    *row,
                )
        return original(self, scheduler_output, *args, **kwargs)

    execute.__prefill_progress_original__ = original
    result[TARGET] = execute
    return result
