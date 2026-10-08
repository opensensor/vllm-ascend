# SPDX-License-Identifier: Apache-2.0
"""Inspect batch admission with copied configs; do not allocate or change caches."""

import copy

from tools.glm_perf.resident_worker import ResidentWorkerExtension


def replacements():
    cached = {}

    def status(self):
        if not cached:
            try:
                import torch

                from vllm_ascend.models.glm5next.cache_config import get_glm5_next_max_memory_usage

                runner = self.model_runner
                config = copy.copy(runner.vllm_config)
                config.scheduler_config = copy.copy(config.scheduler_config)
                groups = runner.kv_cache_config.kv_cache_groups
                requirements = {}
                for batch in (640, 1280, 2560):
                    config.scheduler_config.max_num_batched_tokens = batch
                    requirements[str(batch)] = get_glm5_next_max_memory_usage(config, groups)
                free, total = torch.npu.mem_get_info()
                cached.update(
                    context=config.model_config.max_model_len,
                    required_cache_bytes=requirements,
                    allocated=torch.npu.memory_allocated(),
                    reserved=torch.npu.memory_reserved(),
                    peak_allocated=torch.npu.max_memory_allocated(),
                    peak_reserved=torch.npu.max_memory_reserved(),
                    free=free,
                    total=total,
                )
            except Exception as exc:
                cached["error"] = repr(exc)
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["capacity_audit"] = cached
        return receipt

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status}
