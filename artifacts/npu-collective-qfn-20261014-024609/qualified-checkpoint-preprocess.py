def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: MambaStateCopyFuncsByType,
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    # assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    _reset_layerwise_copy_meta(copy_bufs)
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if num_scheduled_tokens == 0:
            # Async KV connectors can surface a request in a load-only step
            # before any model tokens are scheduled.  Persisting the derived
            # ``-1`` state index here makes the next real forward skip the
            # copy from the remotely loaded h(N-1) state into its running
            # block.  Re-resolve the index from the updated computed-token
            # count when the request is actually scheduled instead.
            mamba_state_idx.pop(req_id, None)
            continue
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            mamba_utils.collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    if _can_launch_triton_batch_memcpy():
        # Layerwise mode: stage per-layer pointer tables here (same protected
        # input-preparation scope), then each layer's state copy is executed by
        # do_mamba_copy_block_for_layer() right after that layer's KV load
        # finishes. Non-layerwise callers still get the bulk staging below.
        if getattr(copy_bufs, "_layer_copy_metadata", None):
            prepare_mamba_copy_by_layer(copy_bufs)
        else:
            _stage_mamba_copy_metadata(copy_bufs)
