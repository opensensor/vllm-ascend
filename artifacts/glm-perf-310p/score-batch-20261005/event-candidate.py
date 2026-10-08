# SPDX-License-Identifier: Apache-2.0
"""Append after renaming the fused factory to qualified_replacements."""


def replacements(native_resources=None):
    import importlib
    import time

    import torch

    changes = qualified_replacements(native_resources)  # noqa: F821
    state = {"armed": False, "records": []}
    max_records = 1200
    status_target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original_status = changes[status_target]

    def scoped(original, label, token_count):
        def wrapped(*args, **kwargs):
            if not state["armed"] or len(state["records"]) >= max_records:
                return original(*args, **kwargs)
            tokens = token_count(args, kwargs)
            if tokens <= 8:
                return original(*args, **kwargs)
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            began = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                state["records"].append((label, tokens, start, end, (time.perf_counter_ns() - began) / 1e6))

        return wrapped

    targets = {
        "vllm_ascend.models.glm5next_w2.moe:Glm5NextW2MoE.forward": ("moe_total", lambda a, k: a[1].shape[0]),
        "vllm_ascend.models.glm5next_w2.moe:_all_reduce_routed": ("moe_all_reduce", lambda a, k: a[0].shape[0]),
        "vllm_ascend._310p.quantization.methods.w2_dynamic:AscendW2DynamicFusedMoEMethod310._apply_device_grouped": (
            "moe_grouped_total",
            lambda a, k: a[3].shape[0],
        ),
        "vllm_ascend.models.glm5next_w2.kda_310:_run_prefill": ("kda_prefill", lambda a, k: a[1].shape[1]),
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool.forward_oot": (
            "indexer",
            lambda a, k: a[1].shape[0],
        ),
    }
    for target, (label, tokens) in targets.items():
        module, attributes = target.split(":")
        original = importlib.import_module(module)
        for attribute in attributes.split("."):
            original = getattr(original, attribute)
        changes[target] = scoped(changes.get(target, original), label, tokens)

    module = importlib.import_module("vllm_ascend._310p.quantization.methods.w2_dynamic")
    original_getter = module._w2_grouped_mm_op
    operation = original_getter()
    if operation is None:
        raise RuntimeError("qualified grouped projection unavailable")
    grouped = scoped(operation, "expert_projection", lambda a, k: a[0].shape[0])

    def getter():
        return grouped

    def profile(self, is_start=True, profile_prefix=None):
        # No profiler object, allocator hooks, device waits, or trace service.
        if is_start:
            if state["armed"]:
                raise RuntimeError("event sampling already armed")
            state["records"].clear()
        state["armed"] = bool(is_start)
        return {
            "rank": torch.distributed.get_rank(),
            "event_sampling": state["armed"],
            "records": len(state["records"]),
        }

    def status(self):
        receipt = original_status(self)
        rows = []
        for label, tokens, start, end, host_ms in state["records"]:
            ready = end.query()
            rows.append(
                {
                    "label": label,
                    "tokens": tokens,
                    "ready": ready,
                    "host_ms": host_ms,
                    "stream_ms": start.elapsed_time(end) if ready else None,
                }
            )
        receipt["prefill_events"] = {"armed": state["armed"], "records": rows, "limit": max_records}
        return receipt

    changes["vllm_ascend._310p.quantization.methods.w2_dynamic:_w2_grouped_mm_op"] = getter
    changes["vllm_ascend._310p.worker_310p:NPUWorker310.profile"] = profile
    changes[status_target] = status
    return changes
