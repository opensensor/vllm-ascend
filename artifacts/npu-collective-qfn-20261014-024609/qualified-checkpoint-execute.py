@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | IntermediateTensors | None:
    self._cpp_execution_time_ms = None
    profiling_chunk_config = self.ascend_config.scheduler_config.profiling_chunk_config
    execution_start_time = _start_profiling_chunk_timing(
        profiling_chunk_config,
        scheduler_output,
    )
    if self.execute_model_state is not None:
        raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

    # If ngram_gpu is used, we need to copy the scheduler_output to avoid
    # the modification has influence on the scheduler_output in engine core process.
    # The replace is much faster than deepcopy.
    if (
        self.speculative_config is not None
        and self.speculative_config.use_ngram_gpu()
    ):
        num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
        spec_decode_tokens_copy = (
            scheduler_output.scheduled_spec_decode_tokens.copy()
        )
        scheduler_output = replace(
            scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens_copy,
            scheduled_spec_decode_tokens=spec_decode_tokens_copy,
        )

    # NOTE: The async-scheduling deepcopy was removed in
    # vllm-project/vllm#29821. Draft token ids are now propagated via
    # _copy_draft_token_ids_to_cpu()/take_draft_token_ids(), and the
    # scheduler output is updated by EngineCore in its own process
    # (Scheduler.update_draft_token_ids_in_output). Nothing below mutates
    # scheduler_output in the worker, so no copy is needed.
    pp_group = get_pp_group()
    if pp_group.world_size > 1 and not pp_group.is_last_rank:
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_token_ids = cached_reqs.new_token_ids
        if new_token_ids and all(not token_ids for token_ids in new_token_ids):
            scheduler_output = replace(
                scheduler_output,
                scheduled_cached_reqs=replace(
                    cached_reqs,
                    new_token_ids=[],
                ),
            )

    if has_kv_transfer_group():
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        # Preemption stores must run before _update_states() zeroes newly
        # allocated blocks that may reuse the same physical KV cache IDs.
        get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    with record_function_or_nullcontext("prepare input"):
        with self.synchronize_input_prep():
            # Fix up prev_req_id_to_index for requests that were discarded
            # in the previous sample_tokens step. If a request has
            # prev_num_draft_len > 0 but is missing from
            # prev_req_id_to_index, the parent _update_states would
            # hit a KeyError. Reset prev_num_draft_len to 0 for such
            # requests so they fall through safely.
            if (
                self.use_async_scheduling
                and self.num_spec_tokens
                and self.input_batch.prev_req_id_to_index is not None
            ):
                for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                    if (
                        req_id not in self.input_batch.prev_req_id_to_index
                        and (req_state := self.requests.get(req_id)) is not None
                        and req_state.prev_num_draft_len
                    ):
                        req_state.prev_num_draft_len = 0

            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(
                scheduler_output
            )

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    self._finalize_dump_data()
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1, skip_gdn_state_update=True)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            if (scheduler_output.total_num_scheduled_tokens <= 0
                    or not tokens or sum(tokens) == 0):
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                force_eager=self.model_config.enforce_eager,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            if self.dynamic_eplb:
                self.update_eplb_heat_collection_status(num_tokens_padded, num_tokens_across_dp)

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            # postprocess_mamba runs later in _update_states_after_model_execute.
            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                preprocess_bufs = mamba_bufs.preprocess
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self._get_mamba_state_copy_funcs(),
                    preprocess_bufs,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            if self.use_compress:
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                num_reqs = self.input_batch.num_reqs
                req_indices = np.repeat(
                    self.arange_np[:num_reqs], num_scheduled_tokens_np
                )
                dsa_positions_np = self._dsa_positions_np_buf[
                    :total_num_scheduled_tokens
                ]
                np.add(
                    self.input_batch.num_computed_tokens_cpu[req_indices],
                    self.query_pos.np[:total_num_scheduled_tokens],
                    out=dsa_positions_np,
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            if (
                cudagraph_mode == CUDAGraphMode.FULL
                or (enable_sp() and not self.model_config.use_mla)
                and self.dcp_size == 1
            ):
                # Currently, Graph Mode and SP will both pad num_tokens,
                # Another possible condition is num_tokens_padded != num_tokens_unpadded
                # but this scope is way too big and the consequences are unpredictable
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    self.query_start_loc,
                    num_tokens_padded,
                    num_reqs_padded,
                    num_reqs,
                    cudagraph_mode,
                    batch_desc.num_reqs,
                )

            (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
            )

            self._sanitize_placeholder_input_ids_for_forward(
                scheduler_output,
                num_tokens_padded,
            )

        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = self._preprocess(
            scheduler_output,
            num_tokens_padded,
            intermediate_tensors,
        )

        # update global cos, sin
        update_cos_sin(positions)

    if self.kvpp.scheduler is not None:
        self.kvpp.prepare_forward(bool(np.any(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)))

    if self.dynamic_eplb:
        self.eplb_updator.forward_before()

    # Encoder-decoder models and raw-token multimodal models can only
    # compile pure decode steps where no encoder inputs are present. The
    # DeepSeek-V4 vision router needs raw sentinel ids during image
    # prefill, so keep that pass eager.
    num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
    has_encoder_input = num_encoder_reqs > 0 and (
        self.model_config.is_encoder_decoder
        or self.model_config.requires_raw_input_tokens
    )
    # Run forward pass
    defer_kv_connector_finalize = self.speculative_config is not None and (
        get_pp_group().is_last_rank or self.broadcast_pp_output
    )
    active_device_metadata_executor = self._prepare_device_metadata_for_forward(cudagraph_mode)
    with (
        record_function_or_nullcontext("forward"),
        set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
            model_instance=self.model,
            device_metadata_executor=active_device_metadata_executor,
            skip_compiled=has_encoder_input,
            has_sinks=self._has_sinks,
            eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
        ),
        self.maybe_get_kv_connector_output(
            scheduler_output,
            **(
                {"defer_finalize": defer_kv_connector_finalize}
            ),
        ) as kv_connector_output,
    ):
        # Mamba state copy must run AFTER the KV transfer load finishes,
        # otherwise the copy would race with in-flight layerwise loads and
        # read half-loaded state. With a layerwise-capable connector we
        # defer the copy: prepare_mamba_state_copy stages per-layer copy
        # metadata here, and each layer's copy is executed right after its
        # own KV load completes (wait_for_layer_load), overlapping the copy
        # with the remaining layers' loads. Non-layerwise connectors keep
        # the batched copy after all loads finish (do_mamba_copy_block).
        mamba_copy_connector = None
        if self.cache_config.mamba_cache_mode == "align":
            if has_kv_transfer_group():
                connector = get_kv_transfer_group()
                prepare_mamba_state_copy = getattr(
                    connector,
                    "prepare_mamba_state_copy",
                    None,
                )
                if callable(prepare_mamba_state_copy) and prepare_mamba_state_copy(preprocess_bufs):
                    mamba_copy_connector = connector
            if mamba_copy_connector is None:
                mamba_utils.do_mamba_copy_block(preprocess_bufs)
        hidden_states = self._model_forward(
            num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
        )
        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_chunk_config,
            execution_start_time,
        )
        # Verify every scheduled layer executed its deferred copy.
        if self.cache_config.mamba_cache_mode == "align" and mamba_copy_connector is not None:
            mamba_copy_connector.finish_mamba_state_copy()
    if active_device_metadata_executor is not None and active_device_metadata_executor.submission_in_flight:
        active_device_metadata_executor.release()
    self.kvpp.complete_forward()

    with record_function_or_nullcontext("post process"):
        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states
        if not self.broadcast_pp_output:
            # Common case.
            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                if self.dynamic_eplb:
                    self.eplb_updator.forward_end(self.eplb_heat_collection_status)
                return hidden_states
            if self.is_pooling_model:
                # Return the pooling output.
                output = self._pool(
                    hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                )
                output.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                return output

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            # Rare case.
            assert not self.is_pooling_model

            if not get_pp_group().is_last_rank:
                sample_hidden_states = hidden_states[logits_indices]
                get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)

            model_output_broadcast_data: dict[str, Any] = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            broadcasted = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert broadcasted is not None
            logits = broadcasted["logits"]

        # Apply structured output bitmasks if present
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
        )
        self.kv_connector_output = kv_connector_output

    # Now the batch has been launched we can wait for corrections from the
    # previous model forward without breaking async scheduling.
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
    return None
