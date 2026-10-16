# SPDX-License-Identifier: Apache-2.0
"""Resident device acceptance/copy and next-token graph ownership.

This bridge removes the CPU acceptance decision from state commit and keeps
next token IDs on device. The existing synchronous engine still schedules each
step. It must not be described as an autonomous model loop.
"""

import numpy as np
import torch

from tools.qwen4exp.iteration_transition import accept_step, prepare_next_step
from tools.qwen4exp.mamba_commit import MambaCommitGroup
from tools.qwen4exp.mamba_commit_runtime import admit_commit


def admit_all_acceptances(columns, tables, *, block_size, slots):
    """Validate every possible MTP2 copy before observing sampler decisions."""
    if len(columns) != 4:
        raise ValueError("four metadata columns required")
    n = len(columns[0])
    if any(np.asarray(column).shape != (n,) for column in columns):
        raise ValueError("one metadata row per admitted request required")
    if not np.all(columns[2] == 3) or not np.all(columns[3] == 2):
        raise ValueError("device transition requires uniform MTP2 decode")
    # Zero acceptance cannot cross a checkpoint boundary. Validate/pack the
    # common geometry once, then check the union of all possible copy accesses.
    # Values are INT64 here so admission itself cannot overflow INT32 inputs.
    packed, _, _ = admit_commit(columns, np.zeros(n, dtype=np.int32), tables, block_size=block_size, slots=slots)
    source = packed[0, :n].astype(np.int64)[:, None]
    running = packed[1, :n].astype(np.int64)[:, None] + 1
    counts = np.arange(1, 4, dtype=np.int64)[None, :]
    aligned = (running + counts - 1) // block_size * block_size
    destination = aligned // block_size - 1
    bias = aligned - running
    copy = (aligned >= running) & ((source != destination) | (bias != 0))
    rows, choices = np.nonzero(copy)
    if not len(rows):
        return packed
    selected_bias = bias[rows, choices]
    if np.any((selected_bias < 0) | (selected_bias > 2)):
        raise ValueError("copy bias outside speculative window")
    source = source[rows, 0]
    destination = destination[rows, choices]
    indices = np.stack((source, source + selected_bias, destination), axis=1)
    for table in tables.values():
        if table.ndim != 2 or table.shape[0] < n or table.dtype != np.int32:
            raise ValueError("compact table must be an INT32 request matrix")
        if np.any((indices < 0) | (indices >= table.shape[1])):
            raise ValueError("copy index outside compact table")
        ids = table[rows[:, None], indices]
        if np.any((ids < 0) | (ids >= slots)):
            raise ValueError("global or invalid block ID cannot address compact state")
    return packed


class ResidentDeviceTransition:
    """Own two ordered graphs per batch shape and retain all live-bank owners."""

    def __init__(self, commit):
        self.commit, self.runner = commit, commit.runner
        self.enabled = self.verify = self.failed = False
        self.pending = self.next_pending = None
        self.checked_steps = self.checked_inputs = self.checked_banks = 0
        self.host_inputs = torch.zeros_like(commit.host_inputs, pin_memory=True)
        self.inputs = torch.zeros_like(commit.inputs)
        self.samples = torch.full((commit.max_requests, 3), -1, dtype=torch.int64, device=self.inputs.device)
        self.drafts = torch.zeros((commit.max_requests, 2), dtype=torch.int64, device=self.inputs.device)
        self.host_tables = {key: torch.zeros_like(value, pin_memory=True) for key, value in commit.host_tables.items()}
        self.tables = {key: torch.zeros_like(value) for key, value in commit.tables.items()}
        self.records = {}
        self.accept_done, self.next_done, self.staging_done = (torch.npu.Event() for _ in range(3))
        self.staging_pending = False
        self.inputs[1].fill_(10)
        self.inputs[2].fill_(3)
        self.inputs[3].fill_(2)
        stream = torch.npu.Stream(device=self.inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for n in sorted(commit.records):
                group = MambaCommitGroup(
                    commit.records[n][0].kernel,
                    commit.records[n][0].launch,
                    commit.banks,
                    [self.tables[key][:n] for key in commit.bank_groups],
                    commit.convolution,
                    block_size=commit.block_size,
                )
                acceptance_graph, next_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
                record = dict(group=group, acceptance_graph=acceptance_graph, next_graph=next_graph)
                self.records[n] = record

                def accept_and_commit(n=n, group=group):
                    accepted = accept_step(self.samples[:n], self.inputs, block_size=commit.block_size)
                    group(self.inputs)
                    return accepted

                for _ in range(3):
                    accept_and_commit()
                torch.npu.synchronize()
                with torch.npu.graph(acceptance_graph, stream=stream):
                    record["accepted"] = accept_and_commit()
                torch.npu.synchronize()

                def make_next(n=n, record=record):
                    return prepare_next_step(
                        self.samples[:n],
                        self.drafts[:n],
                        self.inputs,
                        record["accepted"],
                        max_model_len=self.runner.max_model_len,
                    )

                for _ in range(3):
                    make_next()
                torch.npu.synchronize()
                with torch.npu.graph(next_graph, stream=stream):
                    record["next"] = make_next()
                torch.npu.synchronize()
                # Zero acceptance warmup must never modify a model bank.
                if group.receipt.cpu().count_nonzero():
                    raise RuntimeError("zero-acceptance warmup receipt differs")
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            pending=self.pending is not None,
            next_pending=self.next_pending is not None,
            checked_steps=self.checked_steps,
            checked_inputs=self.checked_inputs,
            checked_banks=self.checked_banks,
            capture_requests=sorted(self.records),
        )

    def begin(self, sampled, scheduler):
        if self.failed:
            raise RuntimeError("device transition submission failed; retain owners")
        if self.pending is not None:
            raise RuntimeError("previous device acceptance was not retired")
        self.next_pending = None
        runner, batch = self.runner, self.runner.input_batch
        n = batch.num_reqs
        if (
            not self.enabled
            or runner.use_async_scheduling
            or n not in self.records
            or sampled.shape != (n, 3)
            or sampled.dtype not in (torch.int32, torch.int64)
            or runner.num_discarded_requests
            or runner.with_prefill
            or runner.uses_mrope
            or any(scheduler.num_scheduled_tokens.get(key) != 3 for key in batch.req_ids)
            or any(len(scheduler.scheduled_spec_decode_tokens.get(key, ())) != 2 for key in batch.req_ids)
        ):
            return False
        ctx = runner._mamba_bufs.postprocess_align
        mapped = getattr(batch, "_prefix_mamba_postprocess_tables", None)
        if mapped is None or any(
            key not in mapped or mapped[key].shape != (n, self.commit.capacity[key]) for key in self.commit.groups
        ):
            return False
        columns = [
            getattr(ctx, field).np[:n]
            for field in (
                "mamba_state_idx_buf",
                "num_computed_tokens_buf",
                "num_scheduled_tokens_buf",
                "num_draft_tokens_buf",
            )
        ]
        try:
            packed = admit_all_acceptances(columns, mapped, block_size=self.commit.block_size, slots=self.commit.slots)
        except ValueError:
            return False
        record = self.records[n]
        expected = ()
        if self.verify:
            host_sampled = sampled.cpu()
            host_inputs = torch.from_numpy(packed.copy())
            accepted = accept_step(host_sampled, host_inputs, block_size=self.commit.block_size)
            if accepted.error.any():
                raise RuntimeError("sampler violates contiguous MTP2 acceptance")
            _, _, copies = admit_commit(
                columns, accepted.counts.numpy(), mapped, block_size=self.commit.block_size, slots=self.commit.slots
            )
            expected = self.commit.expected_blocks(mapped, copies)
        # Once submission starts, no fallback may replay writes on uncertainty.
        self.failed = True
        if self.staging_pending:
            self.staging_done.synchronize()
        self.host_inputs.copy_(torch.from_numpy(packed))
        self.inputs.copy_(self.host_inputs, non_blocking=True)
        for key in self.commit.groups:
            self.host_tables[key][:n].copy_(torch.from_numpy(mapped[key]))
            self.tables[key][:n].copy_(self.host_tables[key][:n], non_blocking=True)
        self.staging_done.record()
        self.staging_pending = True
        self.samples[:n].copy_(sampled)
        record["acceptance_graph"].replay()
        self.accept_done.record()
        if self.verify:
            for actual, reference in zip(record["accepted"], accepted):
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            receipt = record["group"].receipt.cpu()
            if receipt[:, 1].count_nonzero():
                raise RuntimeError("device transition commit rejected admitted metadata")
            for bank, block, value in expected:
                torch.testing.assert_close(bank[block].cpu().view(torch.int32), value.view(torch.int32), rtol=0, atol=0)
            self.checked_steps += 1
            self.checked_banks += len(expected)
        self.pending = (sampled, tuple(batch.req_ids), record)
        self.failed = False
        return True

    def make_next(self, drafts):
        if self.pending is None:
            return
        _, requests, record = self.pending
        n = len(requests)
        if not isinstance(drafts, torch.Tensor) or drafts.shape != (n, 2):
            return  # no device next-input fast path; commit still retires normally
        self.failed = True
        torch.npu.current_stream().wait_event(self.accept_done)
        self.drafts[:n].copy_(drafts)
        record["next_graph"].replay()
        self.next_done.record()
        self.next_pending = (requests, record)
        self.failed = False

    def retire_commit(self, sampled):
        if self.pending is None:
            return False
        original, requests, record = self.pending
        if original is not sampled or requests != tuple(self.runner.input_batch.req_ids):
            raise RuntimeError("device commit lifetime/request order changed")
        n = len(requests)
        torch.npu.current_stream().wait_event(self.accept_done)
        self.runner.num_accepted_tokens.gpu[:n].copy_(record["accepted"].counts)
        # Legacy synchronous bookkeeping observes a completed result. Device
        # commit itself did not wait for this readback. Scheduler removal is
        # still required before claiming host-independent iteration progress.
        self.runner.input_batch.num_accepted_tokens_cpu_tensor[:n].copy_(
            record["accepted"].next_counts, non_blocking=True
        )
        if self.runner.num_accepted_tokens_event is None:
            self.runner.num_accepted_tokens_event = torch.npu.Event()
        self.runner.num_accepted_tokens_event.record()
        self.pending = None
        return True

    def take_inputs(self, scheduler, n, total):
        pending, self.next_pending = self.next_pending, None
        if not self.enabled or pending is None:
            return False
        requests, record = pending
        if (
            requests != tuple(self.runner.input_batch.req_ids)
            or n != len(requests)
            or total != n * 3
            or any(scheduler.num_scheduled_tokens.get(key) != 3 for key in requests)
            or any(len(scheduler.scheduled_spec_decode_tokens.get(key, ())) != 2 for key in requests)
            or np.any(self.runner.input_batch.num_computed_tokens_cpu[:n] + 3 > self.runner.max_model_len)
        ):
            return False
        torch.npu.current_stream().wait_event(self.next_done)
        if self.verify:
            expected = self.runner.input_ids.cpu[:total]
            actual = record["next"].input_ids.reshape(-1).cpu().to(expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                record["next"].computed.cpu(),
                self.runner.input_batch.num_computed_tokens_cpu_tensor[:n].to(torch.int32),
                rtol=0,
                atol=0,
            )
            if not record["next"].active.cpu().all() or record["next"].error.cpu().any():
                raise RuntimeError("invalid continuation cannot consume prepared inputs")
            self.checked_inputs += 1
        self.runner.input_ids.gpu[:total].copy_(record["next"].input_ids.reshape(-1))
        return True
