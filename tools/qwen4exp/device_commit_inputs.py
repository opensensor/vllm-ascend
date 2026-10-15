# SPDX-License-Identifier: Apache-2.0
"""Use current device positions/tables for commit without pinned staging.

CPU page admission remains authoritative. Only checkpoint-qualified unchanged
decode rows use these graphs; the original transition handles all other rows.
"""

import hashlib
import inspect

import torch

from tools.qwen4exp.iteration_transition import accept_step
from tools.qwen4exp.iteration_transition_runtime import admit_all_acceptances
from tools.qwen4exp.mamba_commit import MambaCommitGroup
from tools.qwen4exp.mamba_commit_runtime import admit_commit

QUALIFIED_BEGIN_SHA256 = "7c519614cba9a66d9ef3161dcf94c1ac04e8f9ce91ec3094a3072e85635845d0"
METADATA_FIELDS = ("mamba_state_idx_buf", "num_computed_tokens_buf", "num_scheduled_tokens_buf", "num_draft_tokens_buf")


def publish_commit_inputs(inputs, computed, *, block_size):
    """All tensors are device-owned; the three-slot query determines state index."""
    n = computed.numel()
    inputs[0, :n].copy_(torch.div(computed + 2, block_size, rounding_mode="floor"))
    inputs[1, :n].copy_(computed)
    inputs[2, :n].fill_(3)
    inputs[3, :n].fill_(2)


def eligible(owner, sampled, schedule):
    runner, transition, checkpoint = owner.runner, owner.transition, owner.checkpoint
    ids = tuple(runner.input_batch.req_ids)
    n = len(ids)
    published = checkpoint.published
    return (
        owner.enabled
        and transition.enabled
        and checkpoint.enabled
        and not checkpoint.failed
        and checkpoint.backend == "fused"
        and published is not None
        and published[0] is schedule
        and published[1] == ids
        and n in owner.records
        and not runner.use_async_scheduling
        and sampled.shape == (n, 3)
        and sampled.dtype in (torch.int32, torch.int64)
        and not runner.num_discarded_requests
        and not runner.with_prefill
        and not runner.uses_mrope
        and all(schedule.num_scheduled_tokens.get(key) == 3 for key in ids)
        and all(len(schedule.scheduled_spec_decode_tokens.get(key, ())) == 2 for key in ids)
    )


class DeviceCommitInputs:
    def __init__(self, commit):
        self.commit, self.runner = commit, commit.runner
        self.transition = commit.device_transition_v1
        self.checkpoint = commit.checkpoint_preparation_v1
        self.original = self.transition.begin
        if hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest() != QUALIFIED_BEGIN_SHA256:
            raise RuntimeError("device commit inputs require the qualified transition source")
        self.enabled = self.verify = False
        self.failed = True
        self.calls = self.checked = self.checked_blocks = self.fallbacks = 0
        self.avoided_uploads = self.avoided_staging_waits = 0
        self.records = {}
        transition = self.transition
        stream = torch.npu.Stream(device=self.runner.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            saved_inputs, saved_samples = transition.inputs.clone(), transition.samples.clone()
            for n, original in transition.records.items():
                group = MambaCommitGroup(
                    commit.records[n][0].kernel,
                    commit.records[n][0].launch,
                    commit.banks,
                    [self.checkpoint.tables[key][:n] for key in commit.bank_groups],
                    commit.convolution,
                    block_size=commit.block_size,
                )
                record = dict(original, group=group, acceptance_graph=torch.npu.NPUGraph())
                self.records[n] = record
                computed = original["next"].computed
                saved = tuple(value.clone() for value in (computed, *original["accepted"]))
                computed.fill_(10)
                transition.samples.fill_(-1)

                def step(n=n, computed=computed, group=group, original=original):
                    publish_commit_inputs(transition.inputs, computed, block_size=commit.block_size)
                    accepted = accept_step(transition.samples[:n], transition.inputs, block_size=commit.block_size)
                    group(transition.inputs)
                    # Keep addresses consumed by the existing next-input and
                    # PLE graphs; changing tensor objects would leave stale reads.
                    for destination, source in zip(original["accepted"], accepted):
                        destination.copy_(source)
                    return accepted

                for _ in range(3):
                    step()
                torch.npu.synchronize()
                with torch.npu.graph(record["acceptance_graph"], stream=stream):
                    record["work"] = step()
                torch.npu.synchronize()
                if group.receipt.cpu().count_nonzero():
                    raise RuntimeError("device commit no-copy warmup differs")
                for value, previous in zip((computed, *original["accepted"]), saved):
                    value.copy_(previous)
            transition.inputs.copy_(saved_inputs)
            transition.samples.copy_(saved_samples)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.failed = False

    def begin(self, sampled, schedule):
        transition, runner = self.transition, self.runner
        if self.failed or transition.failed or transition.pending is not None:
            raise RuntimeError("device commit has failed or unretired work; retain owners")
        if not eligible(self, sampled, schedule):
            self.fallbacks += 1
            return self.original(sampled, schedule)
        n = runner.input_batch.num_reqs
        ctx = runner._mamba_bufs.postprocess_align
        mapped = getattr(runner.input_batch, "_prefix_mamba_postprocess_tables", None)
        if mapped is None or any(
            key not in mapped or mapped[key].shape != (n, self.commit.capacity[key]) for key in self.commit.groups
        ):
            self.fallbacks += 1
            return self.original(sampled, schedule)
        columns = [getattr(ctx, field).np[:n] for field in METADATA_FIELDS]
        try:
            packed = admit_all_acceptances(columns, mapped, block_size=self.commit.block_size, slots=64)
        except ValueError:
            self.fallbacks += 1
            return self.original(sampled, schedule)
        record = self.records[n]
        expected = ()
        if self.verify or transition.verify:
            host_inputs = torch.from_numpy(packed.copy())
            accepted = accept_step(sampled.cpu(), host_inputs, block_size=self.commit.block_size)
            if accepted.error.any():
                raise RuntimeError("sampler violates admitted acceptance")
            _, _, copies = admit_commit(
                columns, accepted.counts.numpy(), mapped, block_size=self.commit.block_size, slots=64
            )
            expected = self.commit.expected_blocks(mapped, copies)
        self.failed = transition.failed = True
        transition.next_pending = None
        transition.samples[:n].copy_(sampled)
        record["acceptance_graph"].replay()
        transition.accept_done.record()
        if self.verify or transition.verify:
            torch.testing.assert_close(transition.inputs[:, :n].cpu(), host_inputs[:, :n], rtol=0, atol=0)
            for actual, reference in zip(record["accepted"], accepted):
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            if record["group"].receipt[:, 1].cpu().any():
                raise RuntimeError("device commit rejected admitted tables")
            for bank, block, value in expected:
                torch.testing.assert_close(bank[block].cpu().view(torch.int32), value.view(torch.int32), rtol=0, atol=0)
            self.checked += 1
            self.checked_blocks += len(expected)
        transition.pending = (sampled, tuple(runner.input_batch.req_ids), record)
        self.calls += 1
        self.avoided_uploads += 1 + len(self.commit.groups)
        self.avoided_staging_waits += int(transition.staging_pending)
        self.failed = transition.failed = False
        return True

    def install(self):
        self.transition.begin = self.begin

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            calls=self.calls,
            checked=self.checked,
            checked_blocks=self.checked_blocks,
            fallbacks=self.fallbacks,
            avoided_uploads=self.avoided_uploads,
            avoided_staging_waits=self.avoided_staging_waits,
            host_page_admission_retained=True,
        )
