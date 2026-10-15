def _prepare_inputs(  # type: ignore[override]
    self,
    scheduler_output: SchedulerOutput,
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
    """
    310P cannot use the Triton slot-mapping kernel or the generic NPU Add
    kernels used by the base runner for decode metadata. Keep those pieces
    on CPU and upload the prepared tensors.
    """
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    self.input_batch.block_table.commit_block_table(num_reqs)
    self._remap_compact_mamba_block_tables(num_reqs, num_scheduled_tokens)

    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    if not scheduler_output.scheduled_spec_decode_tokens:
        num_valid_tokens = num_scheduled_tokens
    else:
        num_valid_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[i]
                - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for i in self.input_batch.req_ids
            ],
            dtype=np.int32,
        )
    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

    with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
    self.with_prefill = with_prefill

    cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)

    if self.num_accepted_tokens_event is not None:
        self.num_accepted_tokens_event.synchronize()
        # Keep accepted-token ownership by request when async scheduling
        # condenses/reorders the 310P input batch.
        self._sync_num_accepted_tokens(num_reqs, has_prev_mapping=bool(prev_req_id_to_index))
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()
    else:
        if is_rc_device():
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

    need_async_num_computed_update = (
        self.use_async_spec_decode and self.valid_sampled_token_count_gpu is not None and prev_req_id_to_index
    )

    if need_async_num_computed_update:
        self.prev_positions.copy_to_gpu(num_reqs)
        self.prev_num_draft_tokens.copy_to_gpu()
        cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
            device=self.device, non_blocking=True
        )
        update_num_computed_tokens_for_batch_change(
            self.num_computed_tokens,
            self.num_accepted_tokens.gpu[:num_reqs],
            self.prev_positions.gpu[:num_reqs],
            self.valid_sampled_token_count_gpu,
            self.prev_num_draft_tokens.gpu,
            cpu_values,
        )
        # Make sure D2H is synchronized.
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].copy_(
            self.num_computed_tokens[:num_reqs], non_blocking=False
        )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
    # Make sure when you update the positions and slot mapping, num_computed_tokens_cpu
    # has been corrected.
    positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        self.query_pos.np[: cu_num_tokens[-1]],
        out=positions_np,
    )
    block_table = cast(MultiGroupBlockTable310, self.input_batch.block_table)
    block_table.compute_slot_mapping(
        req_indices,
        positions_np[:total_num_scheduled_tokens],
    )

    if self.use_dcp:
        self.dcp_manager.init_batch_info(
            num_scheduled_tokens,
            self.input_batch.num_reqs,
            self.input_batch.num_computed_tokens_cpu,
            self.input_batch.num_prompt_tokens,
        )

    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)
    prev_positions_gpu = None
    if self.use_async_scheduling and self.input_batch.prev_sampled_token_ids is not None and prev_req_id_to_index:
        self.prev_positions.copy_to_gpu(num_reqs)
        prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

    if self.speculative_config and self.use_dcp:
        self.dcp_manager.generate_dcp_mtp_input(
            total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            self.input_batch,
            self.arange_np,
            req_indices,
            positions_np,
            cu_num_tokens,
            self._draft_token_ids,  # type: ignore[has-type]
            scheduler_output,
            self.num_spec_tokens,
            prev_positions=prev_positions_gpu,
        )

    self.query_lens = torch.from_numpy(num_scheduled_tokens)

    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        torch.index_select(
            is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
        )

    if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue

            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos

            if actual_num_sched > 0:
                self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                    req_embeds[start_pos:actual_end]
                )

            output_idx += num_sched

    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    if is_rc_device():
        self.query_start_loc.np[num_reqs + 1 :].fill(-1)
    self.query_start_loc.copy_to_gpu()

    if self._has_gdn:
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.gdn_query_start_loc.copy_to_gpu()

    torch.add(
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        torch.from_numpy(num_scheduled_tokens),
        out=self.optimistic_seq_lens_cpu[:num_reqs],
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

    if not is_rc_device():
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

    self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)
        self.mrope_positions.gpu.copy_(
            self.mrope_positions.cpu,
            non_blocking=True,
        )
    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)
    base_num_reqs = self.input_batch.num_reqs
    num_reqs = base_num_reqs
    discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

    discard_request_indices = np.nonzero(discard_requests_mask)[0]
    self.num_discarded_requests = len(discard_request_indices)
    self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
    self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)

    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
    self.positions[:total_num_scheduled_tokens].copy_(
        self._positions_cpu_buf[:total_num_scheduled_tokens],
        non_blocking=True,
    )
    if need_async_num_computed_update:
        self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        if is_rc_device():
            tail_len = self.seq_lens.shape[0] - num_reqs
            if tail_len > 0:
                self.seq_lens[num_reqs:].copy_(
                    self.optimistic_seq_lens_cpu[num_reqs:].to(self.device, non_blocking=True),
                )
    else:
        if is_rc_device():
            self.seq_lens.copy_(
                self.optimistic_seq_lens_cpu[: self.seq_lens.shape[0]],
                non_blocking=True,
            )
        else:
            self.seq_lens[:num_reqs].copy_(
                self.optimistic_seq_lens_cpu[:num_reqs],
                non_blocking=True,
            )
    if not is_rc_device():
        self.seq_lens[num_reqs:].fill_(0)

    if (
        self._needs_seq_lens_cpu_sync
        and self.use_async_spec_decode
        and self.valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        # Correct optimistic_seq_lens_cpu on CPU using the asynchronously
        # copied valid-sampled-token counts; avoids an extra NPU->CPU copy
        # of seq_lens and the event.synchronize() in attention metadata.
        # The shared helper synchronizes on the D2H copy event before the
        # host read to avoid consuming stale counts (see its docstring).
        self._correct_optimistic_seq_lens_cpu(num_reqs)

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        spec_decode_metadata = None
        num_draft_tokens = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
    else:
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if (self.is_kv_consumer and req_id in new_schedule_reqs) or (
                self.input_batch.num_computed_tokens_cpu[req_idx] >= self.input_batch.num_prompt_tokens[req_idx]
            ):
                num_decode_draft_tokens[req_idx] = draft_len
            else:
                num_decode_draft_tokens[req_idx] = -1

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens,
            cu_num_tokens,
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1

        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()

    self.logits_indices = logits_indices

    if self.lora_config:
        assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
        self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
    if lmhead_tp_enable():
        max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
        logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

    if self.supports_compact_mamba_state:
        self._stage_prefix_mamba_request_ids()

    return (
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    )
