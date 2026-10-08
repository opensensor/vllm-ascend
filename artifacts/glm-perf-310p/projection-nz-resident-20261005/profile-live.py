# SPDX-License-Identifier: Apache-2.0
"""Temporary explicit profiler RPC; no device work during factory preparation."""


def replacements(native_resources=None):
    changes = serving_replacements(native_resources)  # noqa: F821 - combined with serving source

    def profile(self, is_start=True, profile_prefix=None):
        import torch
        import torch_npu

        path = (
            "/home/matteius/experiments/glm-projection-nz-resident-20261005/trace"
            + f"/worker_rank{torch.distributed.get_rank()}_decode"
        )
        current = getattr(self, "_glm_live_probe_profiler", None)
        if is_start:
            if current is not None:
                raise RuntimeError("diagnostic profiler already active")
            current = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
            )
            current.start()
            self._glm_live_probe_profiler = current
        elif current is not None:
            torch.npu.synchronize()
            current.stop()
            self._glm_live_probe_profiler = None
        return {"rank": torch.distributed.get_rank(), "active": bool(is_start), "path": path}

    changes["vllm_ascend._310p.worker_310p:NPUWorker310.profile"] = profile
    return changes
