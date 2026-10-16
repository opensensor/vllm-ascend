# SPDX-License-Identifier: Apache-2.0
"""Temporary host spans for TP6 mixed batches; no synchronization is added.

Install/remove only after a cache-preserving engine drain. Records contain
shapes and schedules, never token contents. Durations include existing waits;
nested spans overlap and do not establish isolated device execution time.
"""

import inspect

from tools.qwen4exp.mixed_prefill_profile import Recorder

MAX_RECORDS = 65536


def schedule_metadata(schedule, *args, **kwargs):
    return dict(
        scheduled=sorted(schedule.num_scheduled_tokens.values()),
        drafts=sorted(len(value) for value in schedule.scheduled_spec_decode_tokens.values()),
    )


def ple_metadata(layer, hidden_states, input_ids, *args, **kwargs):
    # _forward_eager's first tensor is hidden states, not input IDs.
    return dict(tokens=input_ids.shape[0])


def collective_metadata(collective, tensor, fallback):
    dtype = str(tensor.dtype).removeprefix("torch.")
    elements = tensor.numel()
    native = tensor.is_contiguous() and (dtype, elements) in collective.owners
    return dict(elements=elements, dtype=dtype, native=native)


def install_worker(worker):
    from vllm.forward_context import get_forward_context

    from tools.sidecard.qfn_collective_worker import _communicator
    from vllm_ascend.models.qwen4_exp.model import _PLEInjection
    from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer

    if getattr(worker, "_qfn_tp6_mixed_profile", None) is not None:
        raise RuntimeError("mixed profiler already installed")
    runner = worker.model_runner
    if getattr(runner, "execute_model_state", None) is not None:
        raise RuntimeError("drain worker before installing instrumentation")
    forward = _PLEInjection._forward_eager
    manager = inspect.getclosurevars(forward).nonlocals["manager"]
    lookup_type = type(manager).lookup.__globals__["Lookup"]
    if manager.failed or manager.busy:
        raise RuntimeError("native PLE owner is failed or busy")
    collective = _communicator().custom_sum
    recorder = worker._qfn_tp6_mixed_profile = Recorder(limit=MAX_RECORDS)

    def forward_metadata(num_tokens_padded, *args, **kwargs):
        return dict(tokens=num_tokens_padded, mode=str(get_forward_context().cudagraph_runtime_mode))

    def gather_metadata(layer, ngram_ids, *args, **kwargs):
        return dict(tokens=ngram_ids.shape[0], native=getattr(layer, "_native_host_ple_active", None) is not None)

    def hash_metadata(layer, input_ids, *args, **kwargs):
        return dict(tokens=input_ids.shape[0], device=str(input_ids.device))

    def fetch_metadata(owner, tokens, *args, **kwargs):
        return dict(tokens=tokens.shape[0])

    try:
        recorder.wrap(runner, "execute_model", "execute_model", schedule_metadata)
        recorder.wrap(runner, "_model_forward", "model_forward", forward_metadata)
        recorder.wrap(_PLEInjection, "_forward_eager", "ple", ple_metadata)
        recorder.wrap(_PLEInjection, "_real_ngram_ids", "ple_hash", hash_metadata)
        recorder.wrap(AscendQwen4ExpPLELayer, "gather_embeddings", "ple_gather", gather_metadata)
        recorder.wrap(type(collective), "all_reduce", "collective", collective_metadata)
        recorder.wrap(lookup_type, "fetch", "ple_fetch", fetch_metadata)
        recorder.wrap(manager.lib, "ple_host_produce_fetch_inputs", "ple_native")
        # Python Stream wrapper: retain the exact original descriptor. These
        # spans observe barriers already present in serving; they add none.
        import torch

        recorder.wrap(type(torch.npu.current_stream()), "synchronize", "stream_wait")
    except Exception:
        recorder.restore()
        worker._qfn_tp6_mixed_profile = None
        raise
    return dict(rank=worker.rank, passed=True, hooks=len(recorder.hooks))


def stop_worker(worker):
    recorder = worker._qfn_tp6_mixed_profile
    recorder.restore()
    result = dict(rank=worker.rank, passed=True, **recorder.snapshot())
    worker._qfn_tp6_mixed_profile = None
    return result
