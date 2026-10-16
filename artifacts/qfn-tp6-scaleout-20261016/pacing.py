"""Upgrade a drained bounded scheduler using the existing paced policy."""


def prepare(engine):
    from vllm_ascend.core.prefix_mamba_scheduler import PrefixMambaBoundedScheduler
    from vllm_ascend.core.qwen_prefill_pacing import PrefillPacingConfig, PrefillPacingPolicy
    from vllm_ascend.core.qwen_prefill_scheduler import QwenPrefillPacedScheduler

    class Candidate:
        def validate(self, engine):
            assert engine._pause_complete() and not engine.batch_queue
            assert not engine.scheduler.get_num_unfinished_requests()
            assert type(engine.scheduler) is PrefixMambaBoundedScheduler
            assert not engine.async_scheduling and engine.batch_queue_size == 1
            assert engine.vllm_config.scheduler_config.enable_chunked_prefill
            return dict(passed=True, preserves_cache_manager=True)

        def apply(self, engine):
            self.validate(engine)
            scheduler = engine.scheduler
            cache = scheduler.kv_cache_manager
            scheduler._prefill_pacing = PrefillPacingPolicy(
                PrefillPacingConfig(
                    target_step_ms=200,
                    min_tokens=128,
                    initial_tokens=128,
                    max_tokens=640,
                    alignment=128,
                    smoothing=0.25,
                    decode_only_steps=8,
                )
            )
            scheduler._prefill_pending_step = None
            scheduler._prefill_decode_steps_remaining = 0
            scheduler.__class__ = QwenPrefillPacedScheduler
            assert scheduler.kv_cache_manager is cache
            return dict(passed=True, scheduler=type(scheduler).__name__, cache_manager_preserved=True)

    return Candidate()
