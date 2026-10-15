# SPDX-License-Identifier: Apache-2.0
"""Resident integration of grouped commits; CPU sampling/bookkeeping retained.

Cold construction captures no-copy inputs against existing banks. Admission
uses the compact CPU mapping already owned by the pinned synchronous runtime.
The optional verification path observes real bank contents; normal operation
does not read back kernel receipts or add request-time captures.
"""

import numpy as np
import torch

from tools.qwen4exp.mamba_commit import PADDED_ROWS, MambaCommitGroup

MAX_REQUESTS = 4
MAX_TOKEN_VALUE = (1 << 31) - 1


def admit_commit(columns, counts, tables, *, block_size, slots):
    """Pure host admission before any write; return packed input and copy rows."""
    n = len(counts)
    if not 1 <= n <= MAX_REQUESTS or len(columns) != 4 or not 0 < block_size <= MAX_TOKEN_VALUE:
        raise ValueError("unsupported commit shape")
    data = np.zeros((5, PADDED_ROWS), dtype=np.int32)
    values = [np.asarray(column)[:n] for column in (*columns, counts)]
    if any(value.shape != (n,) or value.dtype.kind not in "iu" for value in values):
        raise ValueError("commit metadata requires integer request vectors")
    if any(np.any(value < 0) or np.any(value > MAX_TOKEN_VALUE) for value in values):
        raise ValueError("commit metadata exceeds nonnegative INT32 contract")
    for index, value in enumerate(values):
        data[index, :n] = value
    next_counts = np.asarray(counts, dtype=np.int64).copy()
    copies = []
    for row in range(n):
        source, computed, scheduled, drafts, accepted = map(int, data[:, row])
        if scheduled <= drafts or accepted > drafts + 1:
            raise ValueError("invalid scheduled/draft/acceptance relationship")
        running = computed + scheduled - drafts
        aligned = (running + accepted - 1) // block_size * block_size
        if aligned < running:
            continue
        destination, bias = aligned // block_size - 1, aligned - running
        if source == destination:
            next_counts[row] = 1
            if bias == 0:
                continue
        if not 0 <= bias <= drafts:
            raise ValueError("copy bias outside speculative window")
        for table in tables.values():
            if table.ndim != 2 or table.shape[0] < n or table.dtype != np.int32:
                raise ValueError("compact table must be an INT32 request matrix")
            if min(source, destination) < 0 or max(source + bias, destination) >= table.shape[1]:
                raise ValueError("copy index outside compact table")
            ids = table[row, [source, source + bias, destination]]
            if np.any(ids < 0) or np.any(ids >= slots):
                raise ValueError("global or invalid block ID cannot address compact state")
        copies.append((row, source, destination, bias))
    return data, next_counts, copies


class ResidentMambaCommit:
    """Worker-owned code, banks, pinned staging buffers and private graphs."""

    def __init__(self, runner, kernel, launch):
        # Worker-only dependency: this module's host admission tests need no vLLM.
        from vllm.model_executor.layers.mamba.mamba_utils import (
            get_conv_copy_spec,
            get_temporal_copy_spec,
            is_conv_state_dim_first,
        )
        from vllm.v1.worker import mamba_utils

        if runner.use_async_scheduling or is_conv_state_dim_first():
            raise ValueError("resident commit requires synchronous SD-layout Mamba")
        self.runner = runner
        self.enabled, self.verify, self.all_steps = False, False, False
        self.failed = False
        self.checked_calls = self.checked_blocks = self.native_calls = 0
        self.fallbacks = {}
        self.bank_groups, self.banks, self.convolution = [], [], []
        context = runner._mamba_bufs.postprocess_align
        self.groups, self.block_size = tuple(context.mamba_group_ids), context.block_size
        mapped = runner.input_batch._prefix_mamba_postprocess_tables
        self.capacity = {group: mapped[group].shape[1] for group in self.groups}
        functions = runner._get_mamba_state_copy_funcs()
        for group in self.groups:
            cache_group = runner.kv_cache_config.kv_cache_groups[group]
            for name in cache_group.layer_names:
                spec = mamba_utils._get_mamba_spec_for_layer(cache_group, name)
                if functions[spec.mamba_type] != (get_conv_copy_spec, get_temporal_copy_spec):
                    raise ValueError("unsupported Mamba copy functions")
                states = runner.compilation_config.static_forward_context[name].kv_cache
                if len(states) != 2:
                    raise ValueError("expected one convolution/recurrent pair")
                for bank, conv in zip(states, (True, False)):
                    if (conv and tuple(bank.shape) != (64, 5, 2560)) or (
                        not conv and tuple(bank.shape) != (64, 12, 128, 128)
                    ):
                        raise ValueError("live state differs from qualified inner geometry")
                    self.banks.append(bank)
                    self.convolution.append(conv)
                    self.bank_groups.append(group)
        if len(self.banks) != 72 or self.block_size != 128:
            raise ValueError("requires the qualified 72-bank QFN configuration")
        device = self.banks[0].device
        self.host_inputs = torch.zeros((5, PADDED_ROWS), dtype=torch.int32, pin_memory=True)
        self.host_inputs[1].fill_(10)
        self.host_inputs[2].fill_(3)
        self.host_inputs[3].fill_(2)
        self.host_inputs[4].fill_(1)
        self.inputs = self.host_inputs.to(device)
        self.host_tables = {
            group: torch.zeros((MAX_REQUESTS, size), dtype=torch.int32, pin_memory=True)
            for group, size in self.capacity.items()
        }
        self.tables = {group: value.to(device) for group, value in self.host_tables.items()}
        self.records = {}
        self.staging_done = torch.npu.Event()
        self.staging_pending = False
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream())
        # Invalid or boundary-crossing warmup inputs could mutate live state.
        # These valid inputs align below running=11, so every bank is skipped.
        with torch.npu.stream(stream):
            for n in range(1, MAX_REQUESTS + 1):
                resource = MambaCommitGroup(
                    kernel,
                    launch,
                    self.banks,
                    [self.tables[group][:n] for group in self.bank_groups],
                    self.convolution,
                    block_size=self.block_size,
                )
                graph = torch.npu.NPUGraph()
                self.records[n] = (resource, graph)  # retain ownership before capture
                for _ in range(3):
                    resource(self.inputs)
                torch.npu.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    resource(self.inputs)
                torch.npu.synchronize()
                receipt = resource.receipt.cpu()
                if receipt[:, 1].count_nonzero() or not torch.all(receipt[:, 0, :n] == 1):
                    raise RuntimeError("no-copy graph warmup receipt differs")
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            all_steps=self.all_steps,
            failed=self.failed,
            capture_requests=sorted(self.records),
            banks=len(self.banks),
            checked_calls=self.checked_calls,
            checked_blocks=self.checked_blocks,
            diagnostic_native_calls=self.native_calls,
            diagnostic_fallbacks=dict(self.fallbacks),
        )

    def expected_blocks(self, tables, copies):
        expected = []
        for bank, group, conv in zip(self.banks, self.bank_groups, self.convolution):
            table = tables[group]
            snapshots = {}
            destinations = set()
            for row, source, destination, bias in copies:
                src, dst = int(table[row, source if conv else source + bias]), int(table[row, destination])
                for block in (src, dst):
                    if block not in snapshots:
                        snapshots[block] = bank[block].cpu()
                selected = snapshots[src][bias:] if conv else snapshots[src]
                snapshots[dst].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1).clone())
                destinations.add(dst)
            expected.extend((bank, block, snapshots[block]) for block in destinations)
        return expected

    def execute(self, kwargs):
        """False means admission failed before writes; caller retains fallback."""
        if self.failed:
            raise RuntimeError("native commit failed; retain owners for inspection")
        batch, n = kwargs["input_batch"], kwargs["num_reqs"]
        counts = getattr(batch, "_qfn_commit_counts", None)
        if counts is None or counts.device.type != "cpu" or counts.shape != (n,) or counts.dtype != torch.int64:
            return False
        ctx = kwargs["bufs"].postprocess_align
        mapped = getattr(batch, "_prefix_mamba_postprocess_tables", None)
        if (
            batch is not self.runner.input_batch
            or ctx.block_size != self.block_size
            or mapped is None
            or tuple(ctx.mamba_group_ids) != self.groups
            or n not in self.records
        ):
            return False
        if any(group not in mapped or mapped[group].shape != (n, self.capacity[group]) for group in self.groups):
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
            packed, next_counts, copies = admit_commit(
                columns, counts.numpy(), mapped, block_size=self.block_size, slots=64
            )
        except ValueError as error:
            if self.verify:
                self.fallbacks[str(error)] = self.fallbacks.get(str(error), 0) + 1
            return False
        if self.verify:
            actual_counts = kwargs["num_accepted_tokens_gpu"][:n].cpu().to(torch.int64)
            torch.testing.assert_close(actual_counts, counts, rtol=0, atol=0)
        if copies or self.all_steps:
            # Scope includes staging: after submission uncertainty, never rerun
            # a CPU fallback against banks that may already have been modified.
            self.failed = True
            expected = self.expected_blocks(mapped, copies) if self.verify else ()
            if self.staging_pending:
                self.staging_done.synchronize()
            self.host_inputs.copy_(torch.from_numpy(packed))
            self.inputs.copy_(self.host_inputs, non_blocking=True)
            for group in self.groups:
                self.host_tables[group][:n].copy_(torch.from_numpy(mapped[group]))
                self.tables[group][:n].copy_(self.host_tables[group][:n], non_blocking=True)
            self.staging_done.record()
            self.staging_pending = True
            resource, graph = self.records[n]
            graph.replay()
            if self.verify:
                receipt = resource.receipt.cpu()
                torch.testing.assert_close(
                    receipt[:, 0, :n], torch.from_numpy(next_counts).int().expand(72, n), rtol=0, atol=0
                )
                if receipt[:, 1].count_nonzero():
                    raise RuntimeError("native commit rejected an admitted row")
                for bank, block, value in expected:
                    torch.testing.assert_close(
                        bank[block].cpu().view(torch.int32), value.view(torch.int32), rtol=0, atol=0
                    )
                self.checked_blocks += len(expected)
                self.native_calls += 1
            self.failed = False
        kwargs["num_accepted_tokens_cpu_tensor"][:n].copy_(torch.from_numpy(next_counts))
        if self.verify:
            self.checked_calls += 1
        return True
