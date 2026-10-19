# SPDX-License-Identifier: Apache-2.0
"""Admission for the explicitly requested, bounded QFN DSpark hardware probe."""

from vllm_ascend._310p.qwen4exp_mtp import is_qwen4exp_speculative_target

QFN_LEGACY_SPEC_ERROR = (
    "Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers"
)


def validate_qfn_dspark_probe(config):
    spec = config.speculative_config
    model, parallel, scheduler, cache = (
        config.model_config,
        config.parallel_config,
        config.scheduler_config,
        config.cache_config,
    )
    if not is_qwen4exp_speculative_target(model, spec) or spec.method != "dspark":
        raise ValueError("QFN DSpark probe requires the Qwen4Exp target and DSpark method")
    eager_probe = config.additional_config.get("qfn_dspark_eager_probe") is True
    graph_probe = config.additional_config.get("qfn_dspark_graph_probe") is True
    if eager_probe == graph_probe:
        raise ValueError("QFN DSpark is experimental; select exactly one eager or graph probe explicitly")
    if not spec.enforce_eager or scheduler.async_scheduling:
        raise ValueError("QFN DSpark probe requires an eager draft and synchronous scheduling")
    if eager_probe and not model.enforce_eager:
        raise ValueError("QFN DSpark eager probe requires target eager mode")
    if graph_probe:
        compilation = config.compilation_config
        mode = getattr(compilation.cudagraph_mode, "name", compilation.cudagraph_mode)
        # Each explicitly bounded probe uses K+1 target rows per request.
        # K7 is an experimental geometry requiring separate collective, HC,
        # PLE, QSA and whole-model rollback qualification on the hardware.
        captures = {2: [3, 6, 12], 7: [8, 16, 32]}.get(spec.num_speculative_tokens)
        if (
            model.enforce_eager
            or mode != "FULL_DECODE_ONLY"
            or captures is None
            or list(compilation.cudagraph_capture_sizes or ()) != captures
            or scheduler.max_num_seqs not in (1, 4)
        ):
            raise ValueError("QFN DSpark graph probe requires C1/C4 and K2 [3, 6, 12] or K7 [8, 16, 32]")
        if (
            not cache.enable_prefix_caching
            or cache.mamba_cache_mode != "align"
            or scheduler.scheduler_cls != "vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler"
        ):
            raise ValueError("QFN DSpark graph probe requires the paced scheduler and align-mode prefix caching")
    if (
        parallel.tensor_parallel_size != 6
        or spec.draft_tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise ValueError("QFN DSpark probe requires target TP6, draft TP1, PP1, and no context parallelism")
    # Upstream normalizes align -> none when prefix caching is disabled.
    # Both use live speculative slots; prefix checkpoint restoration is excluded.
    if eager_probe and (cache.enable_prefix_caching or cache.mamba_cache_mode not in {"align", "none"}):
        raise ValueError("QFN DSpark probe requires live state caching without prefix caching")
    if (
        (eager_probe and scheduler.max_num_seqs != 1)
        or model.max_model_len > (8192 if graph_probe else 4096)
        or spec.num_speculative_tokens not in (1, 2, 3, 7)
    ):
        raise ValueError("QFN DSpark exceeds its bounded request, context, or draft-length probe limits")
    mm = model.multimodal_config
    if mm is not None and not mm.language_model_only:
        raise ValueError("QFN DSpark probe requires language-model-only mode")
    draft = spec.draft_model_config.hf_config
    ids = getattr(draft, "dflash_config", {}).get("target_layer_ids")
    if list(ids or ()) != [1, 12, 23, 34, 45]:
        raise ValueError("QFN DSpark requires the checkpoint's five pre-layer feature taps")
    target = model.hf_text_config
    if (
        target.hc_count != 4
        or target.hidden_size != 2560
        or target.vocab_size != 248320
        or draft.hidden_size != target.hidden_size
        or draft.vocab_size != target.vocab_size
        or draft.num_attention_heads != 32
        or draft.num_key_value_heads != 8
    ):
        raise ValueError("QFN DSpark probe checkpoint geometry does not match the qualified draft")


def verify_qfn_config(original, config, *, is_310p):
    """Preserve upstream validation, except its terminal method allowlist.

    The retained vLLM pin ends Qwen4Exp verification with this exact error,
    after HC, RoPE, PLE, PP, microbatch and inherited hybrid-cache validation.
    Every other error propagates. No temporary method/config mutation is used.
    Remove this wrapper once upstream admits backend-qualified DSpark.
    """
    spec = config.speculative_config
    enabled = is_310p and getattr(spec, "method", None) == "dspark"
    if enabled:
        validate_qfn_dspark_probe(config)
    try:
        original(config)
    except NotImplementedError as exc:
        if not enabled or str(exc) != QFN_LEGACY_SPEC_ERROR:
            raise
    if enabled:
        validate_qfn_dspark_probe(config)
