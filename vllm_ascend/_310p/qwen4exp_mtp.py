# SPDX-License-Identifier: Apache-2.0
"""Host-side 310P Qwen4Exp MTP runner decisions and PLE input staging."""

from __future__ import annotations

from copy import copy

import numpy as np
import torch

MAX_EXACT_FLOAT32_TOKEN_ID = 1 << 24


def greedy_tokens_from_local_logits(logits, vocab_start, num_padding, all_gather):
    """Reduce shard maxima with exact FP32 token IDs and first-index ties."""
    if (
        logits.ndim != 2
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or logits.shape[1] < 1
        or not 0 <= num_padding <= logits.shape[1]
        or vocab_start < 0
        or vocab_start + logits.shape[1] - num_padding > MAX_EXACT_FLOAT32_TOKEN_ID
    ):
        raise ValueError("local draft logits require a nonempty vocabulary with exactly representable token IDs")
    if num_padding:
        logits[..., -num_padding:] = -float("inf")
    values, local_ids = logits.max(dim=-1)
    # 310P's INT64 -> FP32 conversion takes the AI CPU path. The bounded
    # vocabulary indices fit exactly through two AI Core casts instead.
    global_ids = local_ids.to(torch.int32) + vocab_start
    pairs = torch.stack((values.float(), global_ids.float()), dim=-1)
    gathered = all_gather(pairs, dim=-1).view(logits.shape[0], -1, 2)
    rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(-1, rank).squeeze(-1).to(torch.int32).to(torch.int64)


def try_qwen4exp_draft_argmax(proposer, hidden_states, *, lmhead_tp_enabled, all_gather=None):
    """Avoid full-vocabulary gather for Qwen's ordinary greedy proposals.

    Probabilistic proposals, remapped vocabularies, added embeddings and the
    separate Ascend LM-head parallel mode keep their established logits path.
    Target sampling is unaffected, including nonzero user temperatures.
    """
    config = getattr(getattr(proposer, "draft_model_config", None), "hf_config", None)
    architectures = getattr(config, "architectures", ()) or ()
    if (
        proposer.method != "mtp"
        or "Qwen4ExpMTP" not in architectures
        or lmhead_tp_enabled
        or getattr(proposer, "_enable_probabilistic_draft_probs", True)
        or getattr(proposer, "use_heterogeneous_vocab", False)
    ):
        return None
    model = proposer.model
    unwrap = getattr(model, "unwrap", None)
    if callable(unwrap):
        model = unwrap()
    head = model.lm_head
    processor = model.logits_processor
    if (
        not 0 < processor.org_vocab_size <= MAX_EXACT_FLOAT32_TOKEN_ID
        or getattr(processor, "head_dtype", None) not in (None, torch.float16, torch.bfloat16, torch.float32)
        or getattr(head, "num_added_embeddings", 0)
        or getattr(model, "draft_id_to_target_id", None) is not None
    ):
        return None
    local_logits = processor(head, hidden_states, skip_gather=True)
    shards = head.shard_indices
    if all_gather is None:
        # Distributed state exists only in a worker. Keep the host decision
        # helpers importable for offline validation without initializing vLLM.
        from vllm.distributed import tensor_model_parallel_all_gather

        all_gather = tensor_model_parallel_all_gather
    return greedy_tokens_from_local_logits(
        local_logits, shards.org_vocab_start_index, shards.num_org_vocab_padding, all_gather
    )


def qwen4exp_draft_supports_stream_ordering(vllm_config, is_draft_model: bool) -> bool:
    """Whether draft metadata is prepared synchronously on the replay stream.

    Qwen's fixed QSA gathers consume the staged device tables directly. They
    do not use Ascend's asynchronous, host-side attention task-group updates.
    Keep other families, target graphs, async scheduling and DCP on the
    established host completion boundary.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    draft_config = getattr(spec, "draft_model_config", None)
    architectures = getattr(getattr(draft_config, "hf_config", None), "architectures", ()) or ()
    scheduler = getattr(vllm_config, "scheduler_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    return (
        is_draft_model
        and getattr(spec, "method", None) == "mtp"
        and "Qwen4ExpMTP" in architectures
        and getattr(scheduler, "async_scheduling", True) is False
        and getattr(parallel, "decode_context_parallel_size", None) == 1
    )


def qwen4exp_target_has_no_task_updates(vllm_config, is_draft_model, graph_params):
    """Recognize a synchronous Qwen target with no mutable attention tasks.

    Ordinary attention backends can mutate captured host task parameters.
    Their completion boundaries remain necessary. Qwen's fixed QSA/GDN
    kernels instead consume persistent tensors staged on the replay stream.
    Require an initialized, entirely empty task registry before skipping the
    otherwise empty update-stream round trip.
    """
    if (
        is_draft_model
        or not is_qwen4exp_mtp_config(
            getattr(vllm_config, "model_config", None), getattr(vllm_config, "speculative_config", None)
        )
        or getattr(getattr(vllm_config, "scheduler_config", None), "async_scheduling", True) is not False
        or getattr(getattr(vllm_config, "parallel_config", None), "decode_context_parallel_size", None) != 1
    ):
        return False
    handles = getattr(graph_params, "handles", None)
    params = getattr(graph_params, "attn_params", None)
    return (
        isinstance(handles, dict)
        and isinstance(params, dict)
        and bool(handles)
        and handles.keys() == params.keys()
        and all(isinstance(value, (list, tuple)) and not value for value in handles.values())
        and all(isinstance(value, (list, tuple)) and not value for value in params.values())
    )


def stage_qwen_later_draft_block_table(proposer, metadata, draft_index):
    """Bind later Qwen draft gathers to an owned, refreshed table address.

    Capture metadata is built before entering the graph memory pool. A table
    produced there by clone/padding must not become a captured kernel operand:
    that allocation can disappear after capture and is not updated on replay.
    Keep independent storage for each later step, covering the largest capture
    bucket, and refresh its contents during ordinary metadata preparation.
    """
    config = getattr(proposer.draft_model_config, "hf_config", None)
    architectures = getattr(config, "architectures", ()) or ()
    if proposer.method != "mtp" or draft_index == 0 or "Qwen4ExpMTP" not in architectures:
        return metadata
    source = metadata.block_table_tensor
    rows = metadata.num_reqs
    if source.ndim != 2 or source.dtype != torch.int32 or rows < 1:
        raise ValueError("Qwen draft block tables require positive rows and an int32 matrix")
    capture_sizes = proposer.vllm_config.compilation_config.cudagraph_capture_sizes or ()
    capacity = max(proposer.runner.max_num_reqs, max(capture_sizes, default=0))
    if rows > capacity:
        raise ValueError("Qwen draft block table exceeds its configured capture capacity")
    buffers = getattr(proposer, "_qwen_draft_block_table_buffers", None)
    if buffers is None:
        buffers = {}
        proposer._qwen_draft_block_table_buffers = buffers
    buffer = buffers.get(draft_index)
    if buffer is None:
        buffer = torch.zeros((capacity, source.shape[1]), dtype=source.dtype, device=source.device)
        buffers[draft_index] = buffer
    if buffer.shape != (capacity, source.shape[1]) or buffer.device != source.device:
        raise ValueError("captured Qwen draft block table storage cannot change geometry or device")
    copied_rows = min(rows, source.shape[0])
    if buffer.data_ptr() != source.data_ptr():
        buffer[:copied_rows].copy_(source[:copied_rows])
    buffer[copied_rows:].zero_()
    result = copy(metadata)
    result.block_table_tensor = buffer[:rows]
    return result


def is_qwen4exp_mtp_config(model_config: object, speculative_config: object) -> bool:
    if getattr(speculative_config, "method", None) != "mtp":
        return False
    architectures = getattr(model_config, "architectures", ()) or ()
    return any(arch in {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"} for arch in architectures)


def validate_mtp_ple_scheduling(is_qwen4exp_mtp: bool, async_scheduling: bool) -> None:
    """Do not feed async sampled-token placeholders into the PLE n-gram hash.

    The v1 async runner keeps sampled IDs on the device while its CPU token
    table contains placeholders. PLE history currently comes from that table,
    so async MTP would silently change the target model's distribution.
    """
    if is_qwen4exp_mtp and async_scheduling:
        raise ValueError("Qwen4Exp MTP requires --no-async-scheduling until PLE history is device-resident")


def qwen4exp_mtp_hidden_width(draft_model_config: object, method: str) -> int | None:
    """Return the target's complete hyperconnection width for this draft."""
    hf_config = draft_model_config.hf_config
    if method != "mtp" or "Qwen4ExpMTP" not in (getattr(hf_config, "architectures", ()) or ()):
        return None
    text_config = getattr(hf_config, "text_config", None)
    hc_count = int(getattr(hf_config, "hc_count", getattr(text_config, "hc_count", 1)))
    if hc_count < 1:
        raise ValueError("Qwen4Exp MTP requires hc_count >= 1")
    return draft_model_config.get_hidden_size() * hc_count


def stage_ple_history(
    context: torch.Tensor,
    boundaries: torch.Tensor,
    token_ids: np.ndarray,
    computed: np.ndarray,
    query_start_loc: np.ndarray,
    num_reqs: int,
    num_tokens_padded: int,
    eos_token_id: int,
) -> None:
    """Build per-request n-gram history from the authoritative token table.

    Rebuilding it each step also handles speculative rejection and rollback.
    Unused boundary rows describe zero-length requests, keeping graph inputs
    at fixed addresses. Padding tokens, when present, form one dummy request.
    """
    if num_reqs > context.shape[0] - 1 or boundaries.numel() < context.shape[0] + 1:
        raise ValueError("Qwen4Exp PLE batch exceeds max_num_reqs")
    history_len = context.shape[1]
    context.fill_(eos_token_id)
    for req_idx in range(num_reqs):
        end = int(computed[req_idx])
        if end < 0 or end > token_ids.shape[1]:
            raise ValueError("Qwen4Exp PLE history exceeds the token buffer")
        start = max(0, end - history_len)
        history = token_ids[req_idx, start:end]
        if len(history):
            if np.any(history < 0):
                raise ValueError("Qwen4Exp PLE history contains placeholder token IDs")
            context[req_idx, -len(history) :] = torch.as_tensor(history, dtype=context.dtype)

    if num_reqs:
        boundaries[: num_reqs + 1] = torch.as_tensor(query_start_loc[: num_reqs + 1])
        last = int(boundaries[num_reqs])
    else:
        boundaries[0] = 0
        last = 0
    if last > num_tokens_padded:
        raise ValueError("Qwen4Exp PLE query length exceeds padded tokens")
    num_segments = num_reqs
    if last < num_tokens_padded:
        boundaries[num_reqs + 1] = num_tokens_padded
        num_segments += 1
    boundaries[num_segments + 1 :] = num_tokens_padded
