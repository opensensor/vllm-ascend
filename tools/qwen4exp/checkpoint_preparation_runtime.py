# SPDX-License-Identifier: Apache-2.0
"""Resident device pre-copy and normalized-count ownership for exact MTP2 rows.

Host admission, page mapping and lifecycle remain authoritative. Only unchanged
device-continuation rows bypass accepted-count synchronization and re-upload.
Deferred counts are materialized before layout changes or fallback consumers.
"""

import hashlib
import inspect
import itertools
import textwrap
from types import MethodType

import numpy as np
import torch

from tools.qwen4exp.checkpoint_preparation import MAX_REQUESTS, MTP_WIDTH, prepare_checkpoint
from tools.qwen4exp.decode_preparation_graph import admits_packet, compile_preparation
from tools.qwen4exp.mamba_commit import PADDED_ROWS, MambaCommitGroup

PREPROCESS_SHA256 = "2ca8840d7f6ccc513c02797e05ee384914a351a20f9f1a64e6040e8c6c6c3834"
EXECUTE_SHA256 = "07ab8d64a4795936cba5b3ed8a279440551b4c918243a2a28066ee154761ce25"


def compile_execution(original, owner, *, source=None):
    """Replace only the redundant host count publication after pre-copy."""
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != EXECUTE_SHA256:
        raise RuntimeError("checkpoint execution requires exact qualified source")
    old = """                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)"""
    if source.count(old) != 1:
        raise RuntimeError("checkpoint count publication seam differs")
    source = source.replace(old, "                _owner.publish_counts(self, scheduler_output, num_reqs)")
    factory = "def factory(_owner):\n" + textwrap.indent(source, "    ") + "    return execute_model\n"
    namespace = dict(inspect.unwrap(original).__globals__)
    exec(compile(factory, "<qfn-checkpoint-execution>", "exec"), namespace)
    return namespace["factory"](owner)


def admit_precopy(previous, computed, tables, *, block_size, slots):
    """Check every possible accepted bias without observing device counts."""
    previous, computed = np.asarray(previous), np.asarray(computed)
    n = previous.size
    if (
        not 1 <= n <= MAX_REQUESTS
        or previous.shape != (n,)
        or computed.shape != (n,)
        or previous.dtype.kind not in "iu"
        or computed.dtype.kind not in "iu"
        or np.any(previous < -1)
        or np.any(computed < 0)
        or np.any(computed > (1 << 31) - block_size - MTP_WIDTH - 1)
        or not tables
    ):
        raise ValueError("invalid admitted checkpoint rows")
    current = (computed.astype(np.int64) + MTP_WIDTH - 1) // block_size
    copied = (previous >= 0) & (previous != current)
    rows = np.flatnonzero(copied)
    for table in tables.values():
        if table.ndim != 2 or table.shape[0] != n or table.dtype != np.int32:
            raise ValueError("invalid compact table")
        if rows.size:
            indices = np.column_stack((previous[rows], previous[rows] + 1, previous[rows] + 2, current[rows]))
            if np.any(indices < 0) or np.any(indices >= table.shape[1]):
                raise ValueError("checkpoint columns outside compact table")
            ids = table[rows[:, None], indices]
            if np.any(ids < 0) or np.any(ids >= slots):
                raise ValueError("checkpoint physical slot not admitted")
    return current, copied


class CheckpointPreparationGraphs:
    def __init__(self, commit, original_preprocess, runner_class):
        source = textwrap.dedent(inspect.getsource(original_preprocess))
        if hashlib.sha256(source.encode()).hexdigest() != PREPROCESS_SHA256:
            raise RuntimeError("checkpoint pre-copy requires exact qualified source")
        self.commit, self.runner = commit, commit.runner
        self.preparation = commit.decode_preparation_v1
        self.transition = commit.device_transition_v1
        self.original_preprocess = original_preprocess
        self.runner_class = runner_class
        self.original_execute = runner_class.execute_model
        base = self.original_execute._qfn_queued_base
        self.execute = compile_execution(base, self)
        self.original_prepare = self.runner._prepare_inputs
        self.original_fast = self.preparation.fast
        self.enabled = self.verify = False
        self.failed = True
        self.armed = self.published = None
        self.replays = self.checked = self.checked_blocks = self.fallbacks = 0
        self.avoided_syncs = self.avoided_uploads = 0
        self.lazy_readback = False
        self.deferred_counts = None
        self.avoided_readbacks = self.materialized_readbacks = 0
        self.original_retire = self.transition.retire_commit
        self.records = {}
        self.inputs = torch.zeros((5, PADDED_ROWS), dtype=torch.int32, device=self.runner.device)
        self.tables = {
            key: self.runner.input_batch.block_table.block_tables[key].block_table.gpu for key in commit.groups
        }
        if any(table.shape[1] != commit.capacity[key] for key, table in self.tables.items()):
            raise RuntimeError("checkpoint table storage differs from admitted commit geometry")
        self.normalized = self.runner.num_accepted_tokens.gpu
        stream = torch.npu.Stream(device=self.runner.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for n in range(1, MAX_REQUESTS + 1):
                transition = self.transition.records[n]
                values = (
                    self.transition.inputs[0, :n],
                    transition["next"].computed,
                    transition["accepted"].next_counts,
                )
                saved = tuple(value.clone() for value in values)
                group = MambaCommitGroup(
                    commit.records[n][0].kernel,
                    commit.records[n][0].launch,
                    commit.banks,
                    [self.tables[key][:n] for key in commit.bank_groups],
                    commit.convolution,
                    block_size=commit.block_size,
                )
                record = dict(graph=torch.npu.NPUGraph(), group=group, values=values)
                self.records[n] = record

                def prepare(values=values, group=group, n=n):
                    plan = prepare_checkpoint(*values, self.inputs, block_size=commit.block_size)
                    group(self.inputs)
                    self.normalized.fill_(1)
                    self.normalized[:n].copy_(plan.normalized_counts)
                    return plan

                # Cold capture must never copy live banks. Preserve every
                # transition input; no request owns these buffers while drained.
                values[0].zero_()
                values[1].fill_(10)
                values[2].fill_(1)
                for _ in range(3):
                    prepare()
                torch.npu.synchronize()
                with torch.npu.graph(record["graph"], stream=stream):
                    record["plan"] = prepare()
                torch.npu.synchronize()
                if group.receipt.cpu().count_nonzero():
                    raise RuntimeError("checkpoint no-copy warmup touched bank metadata")
                for value, old in zip(values, saved):
                    value.copy_(old)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.failed = False

    def prepare_inputs(self, schedule, counts):
        if self.failed:
            raise RuntimeError("checkpoint submission failed; retain all owners")
        self.armed = self.published = None
        if self.enabled and not schedule.preempted_req_ids and admits_packet(self.preparation, schedule, counts):
            self.armed = (schedule, tuple(self.runner.input_batch.req_ids))
        else:
            self.materialize_counts()
        return self.original_prepare(schedule, counts)

    def before_execute(self, schedule):
        """Publish before any host request insertion, compaction, or reordering."""
        ids = tuple(self.runner.input_batch.req_ids)
        if not (
            self.enabled
            and not self.verify
            and self.preparation.enabled
            and not self.preparation.verify
            and not schedule.preempted_req_ids
            and self.commit.queued_schedule_v1.expected_device_inputs == ids
            and self.transition.next_pending is not None
            and self.transition.next_pending[0] == ids
        ):
            self.materialize_counts()

    def materialize_counts(self):
        pending = self.deferred_counts
        if pending is None:
            return
        ids, counts = pending
        if ids != tuple(self.runner.input_batch.req_ids):
            raise RuntimeError("deferred counts must be published before request layout changes")
        stream = torch.npu.current_stream()
        stream.wait_event(self.transition.accept_done)
        # A previous legacy fallback may still own a D2H into the same CPU
        # array. Order both writers before publishing deferred counts there.
        legacy_event = getattr(self.runner, "num_accepted_tokens_event", None)
        if legacy_event is not None:
            stream.wait_event(legacy_event)
        self.runner.input_batch.num_accepted_tokens_cpu_tensor[: len(ids)].copy_(counts, non_blocking=False)
        self.deferred_counts = None
        self.materialized_readbacks += 1

    def retire_commit(self, sampled):
        transition = self.transition
        if transition.pending is None:
            # The legacy postprocess below publishes this iteration's counts.
            self.deferred_counts = None
            return self.original_retire(sampled)
        if not self.enabled or self.verify or not self.lazy_readback:
            self.deferred_counts = None
            return self.original_retire(sampled)
        original, ids, record = transition.pending
        if original is not sampled or ids != tuple(self.runner.input_batch.req_ids):
            raise RuntimeError("device count retirement lifetime/request order changed")
        torch.npu.current_stream().wait_event(transition.accept_done)
        self.runner.num_accepted_tokens.gpu[: len(ids)].copy_(record["accepted"].counts)
        self.deferred_counts = (ids, record["accepted"].next_counts)
        self.avoided_readbacks += 1
        transition.pending = None
        return True

    def prepare_counts(self, n):
        """True skips the legacy accepted-count event and H2D in hot preparation."""
        if self.enabled and not self.verify and self.armed is not None and len(self.armed[1]) == n:
            self.avoided_syncs += int(self.runner.num_accepted_tokens_event is not None)
            self.avoided_uploads += 1
            return True
        return False

    def restore_counts(self, n):
        runner = self.runner
        self.materialize_counts()
        if runner.num_accepted_tokens_event is not None:
            runner.num_accepted_tokens_event.synchronize()
        runner._sync_num_accepted_tokens(n, has_prev_mapping=bool(runner.input_batch.prev_req_id_to_index))
        runner.num_accepted_tokens.np[n:].fill(1)
        runner.num_accepted_tokens.copy_to_gpu()

    def preprocess(self, schedule, config, cache, state_idx, batch, requests, context, functions, bufs):
        args = (schedule, config, cache, state_idx, batch, requests, context, functions, bufs)
        if batch is not self.runner.input_batch:
            return self.original_preprocess(*args)
        n = batch.num_reqs
        if self.armed != (schedule, tuple(batch.req_ids)):
            self.fallbacks += 1
            return self.original_preprocess(*args)
        previous = np.array(
            [
                state_idx.get(key, (requests[key].num_computed_tokens - 1) // self.commit.block_size)
                for key in batch.req_ids
            ],
            dtype=np.int64,
        )
        computed = np.array([requests[key].num_computed_tokens for key in batch.req_ids], dtype=np.int64)
        mapped = getattr(batch, "_prefix_mamba_postprocess_tables", None)
        try:
            if mapped is None or any(
                key not in mapped or mapped[key].shape != (n, self.commit.capacity[key]) for key in self.commit.groups
            ):
                raise ValueError("missing compact checkpoint tables")
            current, copied = admit_precopy(previous, computed, mapped, block_size=self.commit.block_size, slots=64)
        except ValueError:
            self.restore_counts(n)
            self.armed = None
            self.fallbacks += 1
            return self.original_preprocess(*args)
        record = self.records[n]
        expected = ()
        if self.verify:
            counts = batch.num_accepted_tokens_cpu[:n].copy()
            for actual, reference in zip(record["values"], (previous, computed, counts)):
                torch.testing.assert_close(
                    actual.cpu().to(torch.int64), torch.from_numpy(reference).to(torch.int64), rtol=0, atol=0
                )
            copies = [(i, int(previous[i]), int(current[i]), int(counts[i]) - 1) for i in np.flatnonzero(copied)]
            expected = self.commit.expected_blocks(mapped, copies)
        self.failed = True
        record["graph"].replay()
        if self.verify:
            if record["plan"].error.cpu().any() or record["group"].receipt[:, 1].cpu().any():
                raise RuntimeError("device checkpoint rejected admitted rows")
            normalized = np.where(copied, 1, counts)
            torch.testing.assert_close(
                self.normalized[:n].cpu().to(torch.int64), torch.from_numpy(normalized).to(torch.int64), rtol=0, atol=0
            )
            for bank, block, value in expected:
                torch.testing.assert_close(bank[block].cpu().view(torch.int32), value.view(torch.int32), rtol=0, atol=0)
            self.checked += 1
            self.checked_blocks += len(expected)
        # Keep CPU state-index ownership for the current host scheduler/commit
        # admission. No acceptance value is needed for this lifecycle mirror.
        for key in itertools.chain(
            schedule.finished_req_ids, schedule.preempted_req_ids or (), schedule.scheduled_cached_reqs.resumed_req_ids
        ):
            state_idx.pop(key, None)
        state_idx.update(zip(batch.req_ids, map(int, current)))
        bufs.offset = 0
        self.original_preprocess.__globals__["_reset_layerwise_copy_meta"](bufs)
        self.published = self.armed
        self.replays += 1
        self.failed = False

    def publish_counts(self, runner, schedule, n):
        if runner is self.runner and self.published == (schedule, tuple(runner.input_batch.req_ids)):
            self.avoided_uploads += 1
            return
        runner.num_accepted_tokens.np[:n] = runner.input_batch.num_accepted_tokens_cpu[:n]
        runner.num_accepted_tokens.copy_to_gpu(n)

    def install(self, mamba_utils):
        owner = self

        def prepare(runner, schedule, counts):
            if runner is not owner.runner:
                raise RuntimeError("checkpoint owner belongs to another runner")
            return owner.prepare_inputs(schedule, counts)

        def execute(runner, schedule, *args, **kwargs):
            if runner is not owner.runner:
                return owner.original_execute(runner, schedule, *args, **kwargs)
            resolved = owner.commit.queued_schedule_v1.resolve(schedule)
            owner.before_execute(resolved)
            return owner.execute(runner, resolved, *args, **kwargs)

        self.preparation.checkpoint_preparation_v1 = self
        self.preparation.fast = compile_preparation(
            self.preparation.original.__func__, self.preparation, checkpoint=True
        )
        self.runner._prepare_inputs = MethodType(prepare, self.runner)
        self.runner_class.execute_model = execute
        mamba_utils.preprocess_mamba = self.preprocess
        self.transition.retire_commit = self.retire_commit

    def status(self):
        return dict(
            backend=getattr(self, "backend", "tensor"),
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            replays=self.replays,
            checked=self.checked,
            checked_blocks=self.checked_blocks,
            fallbacks=self.fallbacks,
            avoided_syncs=self.avoided_syncs,
            avoided_uploads=self.avoided_uploads,
            capture_requests=sorted(self.records),
            lazy_readback=self.lazy_readback,
            avoided_readbacks=self.avoided_readbacks,
            materialized_readbacks=self.materialized_readbacks,
            deferred_count_rows=0 if self.deferred_counts is None else len(self.deferred_counts[0]),
            host_lifecycle_retained=True,
        )
