# SPDX-License-Identifier: Apache-2.0
"""Append after renaming the resident factory to serving_replacements."""


def replacements(native_resources=None):
    # Imports are lazy because this factory executes inside resident workers.
    import importlib
    import time
    from pathlib import Path

    import torch
    import torch_npu

    changes = serving_replacements(native_resources)  # noqa: F821
    runner_cls = importlib.import_module("vllm_ascend._310p.model_runner_310p").NPUModelRunner310
    original_execute = runner_cls.execute_model
    state = {"profiler": None, "step": 0, "chunks": [], "rank": None, "nonce": None}
    status_target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original_status = changes[status_target]
    profile_root = "/home/matteius/experiments/glm-prompt-profile-20261005/trace"
    recorded_steps = {1, 2, 25, 26}

    def schedule(step):
        action = torch_npu.profiler.ProfilerAction
        if step in (0, 24):
            return action.WARMUP
        if step in (1, 25):
            return action.RECORD
        if step in (2, 26):
            return action.RECORD_AND_SAVE
        return action.NONE

    def profile(self, is_start=True, profile_prefix=None):
        import json

        rank = torch.distributed.get_rank()
        path = f"{profile_root}/worker_rank{rank}_prefill"
        if is_start:
            if state["profiler"] is not None:
                raise RuntimeError("prompt profiler already active")
            state.update(step=0, chunks=[], rank=rank, nonce=profile_prefix)
            profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=schedule,
                record_shapes=True,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
            )
            profiler.start()
            state["profiler"] = profiler
        else:
            profiler = state["profiler"]
            if profiler is not None:
                torch.npu.synchronize()
                profiler.stop()
                state["profiler"] = None
            Path(profile_root).mkdir(parents=True, exist_ok=True)
            Path(f"{profile_root}/rank{rank}-chunks.json").write_text(json.dumps(state["chunks"], indent=2))
        return {"rank": rank, "active": bool(is_start), "path": path, "steps": state["step"]}

    def status(self):
        receipt = original_status(self)
        receipt["prompt_profile"] = {
            "nonce": state["nonce"],
            "active": state["profiler"] is not None,
            "steps": state["step"],
        }
        return receipt

    def execute(self, scheduler_output, intermediate_tensors=None):
        profiler = state["profiler"]
        if profiler is None:
            return original_execute(self, scheduler_output, intermediate_tensors)
        step = state["step"]
        tokens = scheduler_output.total_num_scheduled_tokens
        started = time.time_ns()
        try:
            with torch.autograd.profiler.record_function(f"glm_prefill_step={step};tokens={tokens}"):
                return original_execute(self, scheduler_output, intermediate_tensors)
        finally:
            finished = time.time_ns()
            state["chunks"].append(
                {
                    "step": step,
                    "tokens": tokens,
                    "start_ns": started,
                    "end_ns": finished,
                    "host_elapsed_ms": (finished - started) / 1_000_000,
                    "recorded": step in recorded_steps,
                }
            )
            state["step"] += 1
            profiler.step()

    def scoped(original, label):
        def wrapped(*args, **kwargs):
            if state["profiler"] is None or state["step"] not in recorded_steps:
                return original(*args, **kwargs)
            with torch.autograd.profiler.record_function(label):
                return original(*args, **kwargs)

        return wrapped

    targets = {
        "vllm_ascend.models.glm5next_w2.moe:Glm5NextW2MoE.forward": "glm_moe",
        "vllm_ascend.models.glm5next_w2.kda_310:_run_prefill": "glm_kda_prefill",
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool.forward_oot": "glm_indexer",
        "vllm_ascend.models.glm5next.kpool_ops:score_and_select_kpool_tokens": "glm_pool_score_select",
        "vllm.model_executor.layers.mhc:MHCPreOp.forward_oot": "glm_mhc_pre",
        "vllm.model_executor.layers.mhc:MHCFusedPostPreOp.forward_oot": "glm_mhc_fused",
    }
    for target, label in targets.items():
        module_name, attributes = target.split(":")
        original = importlib.import_module(module_name)
        for attribute in attributes.split("."):
            original = getattr(original, attribute)
        changes[target] = scoped(changes.get(target, original), label)
    changes["vllm_ascend._310p.model_runner_310p:NPUModelRunner310.execute_model"] = execute
    changes["vllm_ascend._310p.worker_310p:NPUWorker310.profile"] = profile
    changes[status_target] = status
    return changes
