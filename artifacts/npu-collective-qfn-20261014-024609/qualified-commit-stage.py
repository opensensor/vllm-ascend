def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    """Stage all per-request inputs the fused mamba postprocess kernel reads.

    Walks ``req_ids[:num_reqs]`` once, writing each request's mamba block
    index and scheduled/computed/draft token counts into the matching pinned
    numpy views, then issues four non-blocking H→D copies. The fused kernel
    indexes the resulting GPU tensors by ``req_idx``. Buffers live on ``ctx``
    and only exist when the postprocess kernel is enabled.

    Invariant: ``preprocess_mamba`` must have run first for the same batch so
    that every ``req_ids[i]`` has an entry in ``mamba_state_idx``.
    """
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_scheduled = scheduler_output.num_scheduled_tokens
    state_idx_np = ctx.mamba_state_idx_buf.np
    scheduled_np = ctx.num_scheduled_tokens_buf.np
    computed_np = ctx.num_computed_tokens_buf.np
    draft_np = ctx.num_draft_tokens_buf.np

    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before stage_postprocess_inputs_to_gpu"
        )
        state_idx_np[i] = state_idx
        scheduled_np[i] = num_scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, []))

    ctx.mamba_state_idx_buf.copy_to_gpu(num_reqs)
    ctx.num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_computed_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_draft_tokens_buf.copy_to_gpu(num_reqs)
