# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse one sampled-token observation for synchronous QFN Mamba commit.

This is an intermediate removal of a duplicate D2H dependency, not device-side
sampling or an autonomous iteration loop. No captured arithmetic is changed.
Counts precede vocabulary/discard filtering and live for exactly one step.
"""

import hashlib
import inspect
from dataclasses import replace

import torch

RUNNER = "vllm_ascend.worker.model_runner_v1:NPUModelRunner"
MAMBA = "vllm.v1.worker.mamba_utils:postprocess_mamba_align_gpu"
QUALIFIED_FALLBACK_SHA256 = "c7fdbf0591c4d36e6853728f91d5df98e2f967223b30d52593f18f4eafdf35ce"


def eligible(runner):
    return (
        getattr(runner, "_qfn_acceptance_reuse", False)
        and getattr(runner, "_qwen4exp_mtp_ple", False)
        and not runner.use_async_scheduling
        and runner.need_accepted_tokens
        and runner.cache_config.mamba_cache_mode == "align"
    )


def make_bookkeeping(original):
    original = original_base(original)

    def bookkeeping(runner, scheduler_output, sampler_output, *args, **kwargs):
        runner._qfn_acceptance_snapshot = None
        if eligible(runner):
            sampled = sampler_output.sampled_token_ids
            host_sampled = sampled.cpu()
            # The state commit consumes raw sampler acceptance, even for rows
            # whose tokens are subsequently discarded or outside the vocabulary.
            counts = (host_sampled != -1).sum(dim=1)
            runner._qfn_acceptance_snapshot = (sampled, tuple(runner.input_batch.req_ids), counts)
            sampler_output = replace(sampler_output, sampled_token_ids=host_sampled)
        try:
            return original(runner, scheduler_output, sampler_output, *args, **kwargs)
        except BaseException:
            runner._qfn_acceptance_snapshot = None
            raise

    return bookkeeping


def make_state_update(original):
    original = original_base(original)

    def update(runner, output_token_ids, scheduler_output):
        snapshot = getattr(runner, "_qfn_acceptance_snapshot", None)
        runner._qfn_acceptance_snapshot = None
        batch = runner.input_batch
        if hasattr(batch, "_qfn_commit_counts"):
            raise RuntimeError("previous acceptance-count scope was not retired")
        if snapshot is not None and eligible(runner):
            sampled, req_ids, counts = snapshot
            if sampled is output_token_ids and req_ids == tuple(batch.req_ids):
                batch._qfn_commit_counts = counts
                batch._qfn_commit_verify = getattr(runner, "_qfn_acceptance_verify", False)
        try:
            return original(runner, output_token_ids, scheduler_output)
        finally:
            if hasattr(batch, "_qfn_commit_counts"):
                del batch._qfn_commit_counts
            if hasattr(batch, "_qfn_commit_verify"):
                del batch._qfn_commit_verify

    return update


def make_postprocess(original):
    original = original_base(original)

    def postprocess(**kwargs):
        counts = getattr(kwargs["input_batch"], "_qfn_commit_counts", None)
        if counts is not None:
            if counts.device.type != "cpu" or counts.dtype != torch.int64 or counts.shape != (kwargs["num_reqs"],):
                raise ValueError("invalid one-step raw acceptance-count snapshot")
            batch = kwargs["input_batch"]
            if getattr(batch, "_qfn_commit_verify", False):
                # Diagnostic mode only: preserve the old readback to qualify
                # actual sampler counts, rather than comparing generated text.
                expected = kwargs["num_accepted_tokens_gpu"][: kwargs["num_reqs"]].cpu()
                torch.testing.assert_close(counts, expected.to(torch.int64), rtol=0, atol=0)
                batch._qfn_acceptance_checks = getattr(batch, "_qfn_acceptance_checks", 0) + 1
            # The qualified CPU fallback only uses this argument to initialize
            # its host counts. Keep all subsequent block-copy decisions intact.
            kwargs["num_accepted_tokens_gpu"] = counts
        return original(**kwargs)

    return postprocess


def switch(worker, enabled, verify=False):
    """Cold A/B control; controller drains inference before this RPC."""
    try:
        runner = worker.model_runner
        if type(enabled) is not bool or type(verify) is not bool or (verify and not enabled):
            raise ValueError("enabled/verify must be booleans; verification requires reuse")
        if runner.execute_model_state is not None or getattr(runner, "_qfn_acceptance_snapshot", None) is not None:
            raise RuntimeError("cannot change acceptance reuse with a pending step")
        if hasattr(runner.input_batch, "_qfn_commit_counts"):
            raise RuntimeError("cannot change acceptance reuse during commit")
        if (
            runner.use_async_scheduling
            or not runner._qwen4exp_mtp_ple
            or runner.cache_config.mamba_cache_mode != "align"
        ):
            raise ValueError("requires synchronous QFN MTP with aligned Mamba cache")
        runner._qfn_acceptance_reuse = enabled
        runner._qfn_acceptance_verify = verify
        return dict(
            rank=worker.rank,
            passed=True,
            enabled=enabled,
            verify=verify,
            verified_steps=getattr(runner.input_batch, "_qfn_acceptance_checks", 0),
        )
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=str(error))


def original_base(function):
    """Unwrap only this candidate when preparing a successive resident revision."""
    while True:
        if hasattr(function, "_qfn_transition_base"):
            function = function._qfn_transition_base
        elif (function.__module__ == __name__ and function.__name__ in ("bookkeeping", "update", "postprocess")) or (
            function.__module__ == "tools.qwen4exp.resident_candidates.qfn_mamba_commit"
            and function.__name__ == "postprocess"
        ):
            function = inspect.getclosurevars(function).nonlocals["original"]
        else:
            break
    return function


def replacements():
    # Worker-only imports keep CPU contract tests independent of vLLM/NPU init.
    from vllm.v1.worker import mamba_utils

    from vllm_ascend.patch.worker.patch_mamba_utils import _postprocess_mamba_align_gpu_cpu_fallback
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if original_base(mamba_utils.postprocess_mamba_align_gpu) is not _postprocess_mamba_align_gpu_cpu_fallback:
        raise RuntimeError("acceptance snapshot requires the original 310P CPU Mamba fallback")
    source = inspect.getsource(_postprocess_mamba_align_gpu_cpu_fallback)
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_FALLBACK_SHA256:
        raise RuntimeError("acceptance snapshot requires the qualified CPU Mamba fallback source")
    return {
        RUNNER + "._bookkeeping_sync": make_bookkeeping(NPUModelRunner._bookkeeping_sync),
        RUNNER + "._update_states_after_model_execute": make_state_update(
            NPUModelRunner._update_states_after_model_execute
        ),
        MAMBA: make_postprocess(mamba_utils.postprocess_mamba_align_gpu),
        "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop": switch,
    }
