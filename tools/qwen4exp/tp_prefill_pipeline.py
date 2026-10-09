# SPDX-License-Identifier: Apache-2.0
"""Explicit eager prefill experiment: overlap chunk reduction with next compute.

Collective count increases; no cross-layer or decode-graph overlap is attempted.
Every rank must use the same chunk size and token shape. The combined local
routed + TP-sharded shared expert is reduced once per chunk, preserving the
shared-expert ownership policy and FP32 reduction boundary.
"""

import torch
import torch.nn.functional as F

from vllm_ascend._310p.transfer_audit import TransferLedger

MIN_CHUNK_TOKENS = 128
MAX_CHUNK_TOKENS = 2560
MAX_IN_FLIGHT = 2


def pipelined_prefill(module, inputs, route, submit_reduce, wait, *, chunk_tokens=1024):
    if type(chunk_tokens) is not int or not MIN_CHUNK_TOKENS <= chunk_tokens <= MAX_CHUNK_TOKENS:
        raise ValueError("chunk_tokens must be an integer in [128, 2560]")
    if module.shared_expert_replicated or module.expert_tp_size <= 1 or not module.grouped_routing:
        raise ValueError("pipeline requires grouped routing with TP-sharded shared experts")
    weights, ids = route(
        F.linear(inputs, module.gate),
        module.top_k,
        renormalize=module.renormalize,
        routed_scaling_factor=module.routed_scaling_factor,
    )
    outputs, pending = [], []
    for start in range(0, inputs.shape[0], chunk_tokens):
        if len(pending) >= MAX_IN_FLIGHT:
            wait(pending.pop(0))
        stop = min(start + chunk_tokens, inputs.shape[0])
        chunk = inputs[start:stop]
        # Match baseline grouped destination dtype before shared addition.
        local = module._forward_grouped_chunk(chunk, weights[start:stop], ids[start:stop]).to(module.compute_dtype)
        if module.has_shared_expert:
            local += module._forward_shared(chunk)
        reduced, completion = submit_reduce(local)
        outputs.append(reduced)
        pending.append(completion)
    for completion in pending:
        wait(completion)
    return torch.cat(outputs, dim=0).to(module.params_dtype)


def npu_pipelined_prefill(module, inputs, route, *, chunk_tokens=1024):
    from vllm_ascend.models.qwen4_exp.w4_moe import DeferredReduceStream
    from vllm_ascend.utils import current_stream, npu_stream_switch

    if module._tp_reduce is None:
        raise ValueError("TP reduction is not configured")
    if not hasattr(module, "_tp_prefill_stream"):
        module._tp_prefill_stream = DeferredReduceStream()
    if not hasattr(module, "_transfer_ledger"):
        module._transfer_ledger = TransferLedger(event_limit=256)
    main, comm = current_stream(), module._tp_prefill_stream.get()

    def submit(local):
        ready = main.record_event()
        local.record_stream(comm)
        with npu_stream_switch(comm):
            comm.wait_event(ready)
            reduced = module._tp_reduce(local)
            completed = comm.record_event()
        reduced.record_stream(main)
        module._transfer_ledger.record(
            "collective", "prefill_chunk_all_reduce", nbytes=local.numel() * local.element_size()
        )
        return reduced, completed

    def wait(completed):
        main.wait_event(completed)
        module._transfer_ledger.record("device_wait", "prefill_chunk_completion")

    return pipelined_prefill(module, inputs, route, submit, wait, chunk_tokens=chunk_tokens)
