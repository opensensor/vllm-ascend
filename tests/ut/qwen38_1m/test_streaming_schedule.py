# SPDX-License-Identifier: Apache-2.0
"""Deterministic all-rank arithmetic, buffer ownership and faulted scheduling."""

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from tools.qwen4exp.streaming_schedule import (
    PoisonedSchedule,
    ScheduledMoE,
    SchedulePlan,
    SchedulePolicy,
    validate_rank_plan,
)


def plan(policy=None):
    return SchedulePlan(policy or SchedulePolicy(), 7, "1" * 64, "2" * 64, "3" * 64, max_input_tokens=8192)


def receipts(value):
    return [
        {
            "rank": rank,
            "pid": 100 + rank,
            "execution_namespace": "qwen_stream_test",
            "generation": value.generation,
            "plan_sha256": value.sha256,
        }
        for rank in range(4)
    ]


def baseline(inputs, placement):
    routed = [inputs * np.float32(rank + 1) for rank in range(4)]
    shared = [inputs * np.float32((rank + 1) / 4) for rank in range(4)]
    if placement == "tp_sharded":
        routed = [r + s for r, s in zip(routed, shared)]
    result = routed[0].copy()
    for rank in range(1, 4):
        result += routed[rank]
    if placement == "replicated":
        result += inputs * np.float32(0.375)
    return result


class Harness:
    def __init__(self, inputs, agreed, rank=0):
        self.inputs, self.plan, self.rank = inputs, agreed, rank
        self.log, self.slots, self.pending = [], [], {}
        self.max_pending = 0
        self.route_calls = 0
        self.submit_calls = 0
        self.fail_submit = None
        self.fail_wait = None
        self.fail_local = None
        self.fail_store = False
        self.fail_shared_after_reduce = False
        self.output = None

    def route(self, inputs):
        self.route_calls += 1
        return np.ones((len(inputs), 2), dtype=np.float32), np.zeros((len(inputs), 2), dtype=np.int32)

    def local(self, chunk, weights, ids):
        self.log.append(("local", len(chunk)))
        if self.fail_local == len([entry for entry in self.log if entry[0] == "local"]):
            raise RuntimeError("local fault")
        assert weights.shape == ids.shape == (len(chunk), 2)
        return chunk * np.float32(self.rank + 1)

    def shared(self, chunk):
        self.log.append(("shared", len(chunk)))
        if self.fail_shared_after_reduce:
            raise RuntimeError("shared fault")
        scale = 0.375 if self.plan.policy.shared_policy == "replicated" else (self.rank + 1) / 4
        return chunk * np.float32(scale)

    def allocate_output(self, inputs):
        self.output = np.full(inputs.shape, np.nan, dtype=np.float32)
        return self.output

    def allocate_slot(self, inputs, tokens):
        slot = np.empty((tokens, inputs.shape[1]), dtype=np.float32)
        self.slots.append(slot)
        return slot

    def store(self, destination, source):
        if np.shares_memory(destination, self.output):
            self.log.append(("store_output", len(source)))
            if self.fail_store:
                raise RuntimeError("store fault")
        else:
            slot = next(index for index, buffer in enumerate(self.slots) if np.shares_memory(buffer, destination))
            assert not any(entry["slot"] == slot for entry in self.pending.values()), "slot reused before completion"
            self.log.append(("store_slot", slot))
        destination[:] = source

    def submit(self, local, slot):
        self.submit_calls += 1
        index = self.submit_calls - 1
        self.log.append(("submit", index, slot))
        if self.fail_submit == self.submit_calls:
            raise RuntimeError("collective enqueue unresolved")
        start, stop = self.plan.chunks(len(self.inputs))[index]
        contributions = []
        for rank in range(4):
            expected = self.inputs[start:stop] * np.float32(rank + 1)
            if self.plan.policy.shared_policy == "tp_sharded":
                expected += self.inputs[start:stop] * np.float32((rank + 1) / 4)
            if rank == self.rank:
                np.testing.assert_array_equal(local, expected)
                expected = local
            contributions.append(expected)
        reduced = contributions[0].copy()
        for rank in range(1, 4):
            reduced += contributions[rank]
        self.pending[index] = {"slot": slot, "local": local, "snapshot": local.copy()}
        self.max_pending = max(self.max_pending, len(self.pending))
        return reduced, index

    def wait(self, handle):
        self.log.append(("wait", handle))
        if self.fail_wait == handle:
            raise RuntimeError("completion unresolved")
        entry = self.pending.pop(handle)
        np.testing.assert_array_equal(entry["local"], entry["snapshot"])

    def scheduler(self, rank_receipts=None):
        return ScheduledMoE(
            self.plan,
            receipts(self.plan) if rank_receipts is None else rank_receipts,
            route=self.route,
            local=self.local,
            shared=self.shared,
            submit_reduce=self.submit,
            wait=self.wait,
            store=self.store,
            allocate_output=self.allocate_output,
            allocate_slot=self.allocate_slot,
        )


@pytest.mark.parametrize("tokens", [0, 129, 2560, 2561])
@pytest.mark.parametrize("chunk", [128, 2560])
@pytest.mark.parametrize("placement", ["none", "tp_sharded", "replicated"])
def test_all_rank_arithmetic_full_tail_and_bounded_two_slots(tokens, chunk, placement):
    inputs = np.random.default_rng(17).normal(size=(tokens, 3)).astype(np.float32)
    agreed = plan(SchedulePolicy(chunk_tokens=chunk, shared_policy=placement))
    sequences = []
    for rank in range(4):
        harness = Harness(inputs, agreed, rank)
        result = harness.scheduler().run(inputs)
        np.testing.assert_array_equal(result, baseline(inputs, placement))
        assert harness.route_calls == int(tokens > 0)
        assert not harness.pending and harness.max_pending <= 2
        assert len(harness.slots) <= 2
        sequences.append([(e[1], e[2]) for e in harness.log if e[0] == "submit"])
        for index, event in enumerate(harness.log):
            if event[0] == "store_output":
                assert any(entry[0] == "wait" for entry in harness.log[:index])
    assert all(sequence == sequences[0] for sequence in sequences)


@pytest.mark.parametrize(
    "change", ["none", "missing", "duplicate", "generation", "shape", "policy", "namespace", "pid"]
)
def test_rank_mismatch_rejected_before_any_alloc_route_or_collective(change):
    agreed = plan()
    rank_receipts = receipts(agreed)
    if change == "none":
        rank_receipts = []
    elif change == "missing":
        rank_receipts.pop()
    elif change == "duplicate":
        rank_receipts[-1] = rank_receipts[0]
    elif change == "generation":
        rank_receipts[0]["generation"] += 1
    elif change == "shape":
        rank_receipts[0]["plan_sha256"] = replace(agreed, max_input_tokens=4096).sha256
    elif change == "policy":
        rank_receipts[0]["plan_sha256"] = replace(agreed, policy=SchedulePolicy(chunk_tokens=1024)).sha256
    elif change == "namespace":
        rank_receipts[0]["execution_namespace"] = "different"
    else:
        rank_receipts[0]["pid"] = 0
    harness = Harness(np.ones((129, 3), np.float32), agreed)
    with pytest.raises(ValueError):
        harness.scheduler(rank_receipts)
    assert harness.route_calls == harness.submit_calls == 0
    assert harness.output is None and not harness.slots


def test_default_geometry_and_immutable_plan():
    value = plan()
    assert value.policy.chunk_tokens == 2560
    assert value.chunks(2561) == ((0, 2560), (2560, 2561))
    with pytest.raises(FrozenInstanceError):
        value.policy.chunk_tokens = 1024
    with pytest.raises(ValueError):
        value.chunks(8193)
    assert validate_rank_plan(value, receipts(value))


@pytest.mark.parametrize("limit", [1, 2])
def test_one_or_two_pending_and_output_read_only_after_wait(limit):
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128, max_inflight=limit)))
    result = harness.scheduler().run(inputs)
    assert harness.max_pending == limit
    assert len(harness.slots) == limit
    np.testing.assert_array_equal(result, baseline(inputs, "tp_sharded"))


def test_unresolved_completion_stops_all_followup_submits_and_waits():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128)))
    harness.fail_wait = 0
    owner = harness.scheduler()
    with pytest.raises(PoisonedSchedule) as failure:
        owner.run(inputs)
    assert failure.value.unresolved_handles == 2
    assert harness.submit_calls == 2
    assert [entry for entry in harness.log if entry[0] == "wait"] == [("wait", 0)]
    assert np.isnan(harness.output).all()
    with pytest.raises(PoisonedSchedule):
        owner.run(inputs)
    assert harness.submit_calls == 2


def test_uncertain_submit_does_not_add_collective_or_attempt_cleanup_wait():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128)))
    harness.fail_submit = 2
    with pytest.raises(PoisonedSchedule) as failure:
        harness.scheduler().run(inputs)
    assert failure.value.submission_uncertain is True
    assert failure.value.unresolved_handles == 1
    assert not any(entry[0] == "wait" for entry in harness.log)
    assert harness.submit_calls == 2


def test_compute_failure_drains_only_known_completions_once():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128)))
    harness.fail_local = 2
    with pytest.raises(PoisonedSchedule):
        harness.scheduler().run(inputs)
    assert harness.submit_calls == 1 and not harness.pending
    assert [entry for entry in harness.log if entry[0] == "wait"] == [("wait", 0)]
    assert np.isnan(harness.output).all()


def test_compute_failure_stops_draining_after_failed_completion():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128)))
    harness.fail_local = 2
    harness.fail_wait = 0
    with pytest.raises(PoisonedSchedule) as failure:
        harness.scheduler().run(inputs)
    assert failure.value.unresolved_handles == 1
    assert harness.submit_calls == 1
    assert [entry for entry in harness.log if entry[0] == "wait"] == [("wait", 0)]


def test_replicated_shared_failure_waits_only_other_known_handle():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128, shared_policy="replicated")))
    harness.fail_shared_after_reduce = True
    with pytest.raises(PoisonedSchedule):
        harness.scheduler().run(inputs)
    assert harness.submit_calls == 2
    assert [entry for entry in harness.log if entry[0] == "wait"] == [("wait", 0), ("wait", 1)]
    assert not harness.pending


def test_output_copy_failure_does_not_wait_completed_handle_twice():
    inputs = np.ones((513, 3), np.float32)
    harness = Harness(inputs, plan(SchedulePolicy(chunk_tokens=128)))
    harness.fail_store = True
    with pytest.raises(PoisonedSchedule):
        harness.scheduler().run(inputs)
    assert harness.submit_calls == 2 and not harness.pending
    assert [entry for entry in harness.log if entry[0] == "wait"] == [("wait", 0), ("wait", 1)]


def test_no_device_readback_or_global_tensor_slots():
    path = Path(__file__).resolve().parents[3] / "tools/qwen4exp/streaming_schedule.py"
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(n, ast.Attribute) and n.attr in ("item", "cpu", "tolist", "synchronize") for n in ast.walk(tree)
    )
    assert not any(
        isinstance(n, ast.Import) and any(a.name in ("torch", "torch_npu") for a in n.names) for n in tree.body
    )
    owner = Harness(np.ones((129, 3), np.float32), plan()).scheduler()
    owner.run(np.ones((129, 3), np.float32))
    assert not any(isinstance(value, np.ndarray) for value in vars(owner).values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_inflight": 0},
        {"max_inflight": 3},
        {"chunk_tokens": 127},
        {"chunk_tokens": 2561},
        {"shared_policy": "automatic"},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        SchedulePolicy(**kwargs)
