# SPDX-License-Identifier: Apache-2.0
"""Host-side 310P Qwen4Exp MTP runner decisions and PLE input staging."""

from __future__ import annotations

import numpy as np
import torch


def is_qwen4exp_mtp_config(model_config: object, speculative_config: object) -> bool:
    if getattr(speculative_config, "method", None) != "mtp":
        return False
    architectures = getattr(model_config, "architectures", ()) or ()
    return any(arch in {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"} for arch in architectures)


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
