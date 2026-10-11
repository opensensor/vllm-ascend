# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in first-replay prefix checkpoints for synchronous Qwen serving."""

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler


def replay_checkpoint_split(start: int, scheduled: int, prompt_tokens: int, alignment: int) -> int:
    """Materialize the last checkpoint reusable by an identical prompt.

    Prefix lookup excludes the prompt's final token. A fully aligned prompt
    must therefore retain the preceding boundary as well as its prompt-end
    state. Otherwise the first repeat recomputes it before later repeats hit.
    """
    if start < 0 or scheduled < 0 or prompt_tokens < 1 or alignment < 1:
        raise ValueError("invalid recurrent replay geometry")
    replay_position = (prompt_tokens - 1) // alignment * alignment
    if start < replay_position < start + scheduled:
        return replay_position - start
    return scheduled


class QwenPrefixReplayScheduler(Scheduler):
    """Stop once at the identical-replay endpoint, preserving parent rules."""

    def __init__(self, vllm_config, *args, **kwargs):
        # Hardware lookup is deferred for CPU tests and EngineCore isolation.
        # Other models and schedulers retain their existing behavior.
        from vllm_ascend.utils import is_310p

        if not is_310p() or getattr(vllm_config.model_config.hf_text_config, "model_type", None) != "qwen4_exp_text":
            raise ValueError("Qwen prefix replay requires Qwen4Exp on Ascend 310P")
        if vllm_config.scheduler_config.async_scheduling or vllm_config.kv_transfer_config is not None:
            raise ValueError("Qwen prefix replay requires synchronous standalone serving")
        if not vllm_config.cache_config.enable_prefix_caching or vllm_config.cache_config.mamba_cache_mode != "align":
            raise ValueError("Qwen prefix replay requires align-mode prefix caching")
        super().__init__(vllm_config, *args, **kwargs)
        coordinator = self.kv_cache_manager.coordinator
        self._qwen_replay_alignment = getattr(coordinator, "_cache_hit_alignment_tokens", self.hash_block_size)
        if self._qwen_replay_alignment < 1:
            raise ValueError("Qwen prefix replay requires positive checkpoint alignment")
        logger.info("Qwen first-replay checkpoint alignment: %d tokens.", self._qwen_replay_alignment)

    def _mamba_block_aligned_split(
        self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0
    ):
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        num_new_tokens = replay_checkpoint_split(
            start, num_new_tokens, request.num_prompt_tokens, self._qwen_replay_alignment
        )
        return super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
