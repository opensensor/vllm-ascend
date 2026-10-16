# SPDX-License-Identifier: Apache-2.0
"""Drained engine switch to a two-batch optimistic scheduling pipeline.

Preserve the existing scheduler, cache manager, and BalanceScheduler behavior.
Only the upstream async placeholder lifecycle and engine queue are installed.
Workers must first install/enable QueuedScheduleBridge while the service is
paused. The bridge deliberately keeps worker async-history flags disabled.
"""

from collections import deque


class QueuedEngineControl:
    def __init__(self, engine, *, enabled, async_scheduler_class):
        self.enabled = enabled
        self.async_scheduler_class = async_scheduler_class
        self.original_class = type(engine.scheduler)
        self.original_step = engine.step_fn
        self.original_queue_size = engine.batch_queue_size
        self.original_async = engine.async_scheduling
        self.original_config_async = engine.vllm_config.scheduler_config.async_scheduling
        self.original_token_budget = getattr(engine.vllm_config.scheduler_config, "max_num_batched_tokens", None)
        self.original_scheduled_budget = getattr(engine.scheduler, "max_num_scheduled_tokens", None)
        self.prefix_mode = self.original_class.__name__ == "QwenPrefillPacedScheduler"
        self.step_budget = None
        self.queued_class = None

    def validate(self, engine):
        scheduler = engine.scheduler
        if engine.batch_queue or scheduler.get_num_unfinished_requests() or not engine._pause_complete():
            raise RuntimeError("drain and pause the engine before changing its scheduling pipeline")
        config = engine.vllm_config
        parallel = config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("queued bridge requires PP1/DP1/DCP1")
        if scheduler.connector is not None or scheduler.ec_connector is not None:
            raise ValueError("connector allocation ownership is outside the queued bridge contract")
        if getattr(scheduler, "_balance_enabled", False):
            raise ValueError("cross-DP balance is outside the queued bridge contract")
        if scheduler.num_spec_tokens != 2 or getattr(scheduler, "use_v2_model_runner", True):
            raise ValueError("queued bridge requires the v1 MTP2 scheduler")
        managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        if self.enabled and self.prefix_mode:
            leases = getattr(scheduler, "_prefix_mamba_step_leases", None)
            if getattr(scheduler, "_prefix_mamba_queue_depth", 1) != 2 or leases is None or not leases.drained:
                raise ValueError("queued prefixes require cold two-step state reservation and drained leases")
            if getattr(scheduler, "_prefill_queued_steps", None) != {}:
                raise ValueError("queued pacing must have no pending outputs")
            if type(self.original_token_budget) is not int or type(self.original_scheduled_budget) is not int:
                raise ValueError("queued prefixes require an explicit startup token reservation")
            # Two half-budget steps never exceed the original cold KV token
            # reservation. Worker buffers remain at their larger startup size.
            self.step_budget = min(self.original_token_budget // 2, self.original_scheduled_budget)
            if self.step_budget < config.scheduler_config.max_num_seqs * (1 + scheduler.num_spec_tokens):
                raise ValueError("startup token reservation cannot hold two complete decode batches")
        if self.enabled:
            for manager in managers:
                cap = getattr(manager, "_max_admission_blocks_per_request", None)
                if cap is None:
                    cap = getattr(manager, "max_admission_blocks_per_request", None)
                if cap is None:
                    continue
                spec = getattr(manager, "kv_cache_spec", None)
                bound = getattr(spec, "max_admission_blocks_per_request", None)
                if not self.prefix_mode or not callable(bound) or type(cap) is not int:
                    raise ValueError("recycling cache admission must be rebuilt for a deeper queue")
                required = bound(self.step_budget * 2, config.model_config.max_model_len)
                if required > cap:
                    raise ValueError("queued steps exceed the cold recycling cache admission bound")
        if self.enabled:
            if engine.async_scheduling or engine.batch_queue_size != 1:
                raise ValueError("expected the retained synchronous engine")
            if type(scheduler).__name__ != "BalanceScheduler" and not self.prefix_mode:
                raise ValueError("unreviewed scheduler class")
        elif not hasattr(engine, "_qfn_queued_engine_owner"):
            raise ValueError("no queued engine owner to restore")
        elif engine._qfn_queued_engine_owner.prefix_mode and (
            not scheduler._prefix_mamba_step_leases.drained or scheduler._prefill_queued_steps
        ):
            raise ValueError("retire all prefix and pacing outputs before restoring synchronous execution")
        return dict(
            passed=True,
            enabled=self.enabled,
            scheduler=type(scheduler).__name__,
            cache_managers=[type(manager).__name__ for manager in managers],
            step_token_budget=self.step_budget,
        )

    def apply(self, engine):
        self.validate(engine)
        scheduler = engine.scheduler
        if self.enabled:
            # AsyncScheduler's super() retains BalanceScheduler in the MRO.
            # No request/cache manager is reconstructed or discarded.
            self.queued_class = type(
                "Queued" + self.original_class.__name__, (self.async_scheduler_class, self.original_class), {}
            )
            engine._qfn_queued_engine_owner = self
            scheduler.__class__ = self.queued_class
            scheduler._spec_token_placeholders = [-1] * scheduler.num_spec_tokens
            scheduler.pp_size = 1
            if self.prefix_mode:
                scheduler._qfn_queued_prefix_enabled = True
                scheduler.max_num_scheduled_tokens = self.step_budget
                engine.vllm_config.scheduler_config.max_num_batched_tokens = self.step_budget
            engine.vllm_config.scheduler_config.async_scheduling = True
            engine.async_scheduling = True
            engine.batch_queue_size = 2
            engine.batch_queue = deque(maxlen=2)
            engine.step_fn = engine.step_with_batch_queue
        else:
            owner = engine._qfn_queued_engine_owner
            if owner.prefix_mode:
                scheduler._qfn_queued_prefix_enabled = False
                scheduler.max_num_scheduled_tokens = owner.original_scheduled_budget
                engine.vllm_config.scheduler_config.max_num_batched_tokens = owner.original_token_budget
            scheduler.__class__ = owner.original_class
            engine.vllm_config.scheduler_config.async_scheduling = owner.original_config_async
            engine.async_scheduling = owner.original_async
            engine.batch_queue_size = owner.original_queue_size
            engine.batch_queue = None
            engine.step_fn = owner.original_step
        return dict(
            passed=True,
            enabled=self.enabled,
            scheduler=type(scheduler).__name__,
            batch_queue_size=engine.batch_queue_size,
            worker_cpu_bookkeeping_retained=True,
        )
