# SPDX-License-Identifier: Apache-2.0
"""Overlap qualified output observation with the device MTP draft.

This owner removes host output parsing from the path *before* draft submission.
It retains a join before returning the model result, so it is not autonomous
device scheduling. Mutable worker history is published exactly once, while
request admission/row movement remains excluded until the join completes.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import torch

from tools.qwen4exp.resident_candidates.qfn_early_tp_draft import eligible


class OutputObserver:
    def __init__(self, transition, bookkeeping):
        self.transition, self.runner = transition, transition.runner
        self.bookkeeping = bookkeeping
        self.enabled = self.failed = False
        self.future = None
        self.observed = self.completed_before_join = 0
        self.host_samples = {
            dtype: torch.empty((4, 3), dtype=dtype, pin_memory=True) for dtype in (torch.int32, torch.int64)
        }
        self.copy_stream = torch.npu.Stream(device=self.runner.device)
        self.copy_done = torch.npu.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qfn-output-observer")

    def start(self, runner, sampler, scheduler, logits, hidden, spec, batch, padded):
        if self.failed or self.future is not None:
            raise RuntimeError("output observer has an unretired or failed observation")
        if (
            not self.enabled
            or runner is not self.runner
            or not padded
            or not self.transition.enabled
            or self.transition.verify
            or sampler.logprobs_tensors is not None
            or sampler.sampled_token_ids.dtype not in self.host_samples
            or not eligible(runner, spec, batch, scheduler)
        ):
            return False
        # The original bookkeeping may read logits for NaN diagnostics. That
        # mode must retain the original ordering and target storage lifetime.
        vllm_envs = self.bookkeeping.__globals__.get("envs")
        if getattr(vllm_envs, "VLLM_COMPUTE_NANS_IN_LOGITS", False):
            return False
        if not self.transition.begin(sampler.sampled_token_ids, scheduler):
            return False
        self.failed = True
        n = sampler.sampled_token_ids.shape[0]
        host_samples = self.host_samples[sampler.sampled_token_ids.dtype][:n]
        producer = torch.npu.current_stream(self.runner.device)
        with torch.npu.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer)
            host_samples.copy_(sampler.sampled_token_ids, non_blocking=True)
            self.copy_done.record()
        host_sampler = replace(sampler, sampled_token_ids=host_samples)
        self.future = self.executor.submit(self._observe, host_sampler, scheduler, logits, hidden, spec)
        self.failed = False
        return True

    def _observe(self, sampler, scheduler, logits, hidden, spec):
        # Driver context is thread-local. The sampler copy/event belong to this
        # worker's rank; never let a helper thread initialize device zero.
        torch.npu.set_device(self.runner.device)
        self.copy_done.synchronize()
        with torch.inference_mode():
            return self.bookkeeping(
                self.runner, scheduler, sampler, logits, hidden, scheduler.total_num_scheduled_tokens, spec
            )

    def finish(self):
        if self.future is None:
            raise RuntimeError("no output observation to retire")
        self.failed = True
        self.completed_before_join += self.future.done()
        result = self.future.result()
        self.future = None
        self.observed += 1
        self.failed = False
        return result

    def status(self):
        return dict(
            enabled=self.enabled,
            failed=self.failed,
            pending=self.future is not None,
            observed=self.observed,
            completed_before_join=self.completed_before_join,
            worker_cpu_bookkeeping_retained=True,
        )
