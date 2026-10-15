# SPDX-License-Identifier: Apache-2.0
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp import checkpoint_preparation_runtime as module


def test_admission_checks_all_acceptance_biases():
    table = np.array([[4, 8, 9, 10]], dtype=np.int32)
    current, copied = module.admit_precopy([0], [126], {2: table}, block_size=128, slots=64)
    np.testing.assert_array_equal(current, [1])
    np.testing.assert_array_equal(copied, [True])
    table[0, 2] = 64
    with pytest.raises(ValueError, match="physical slot"):
        module.admit_precopy([0], [126], {2: table}, block_size=128, slots=64)


@pytest.mark.parametrize(
    "previous,computed", [([-2], [10]), ([0], [-1]), ([0], [1 << 31]), ([1.5], [10]), ([0], [10, 11])]
)
def test_invalid_host_admission(previous, computed):
    with pytest.raises(ValueError):
        module.admit_precopy(previous, computed, {2: np.zeros((1, 4), dtype=np.int32)}, block_size=128, slots=64)


def test_source_drift_is_fail_closed():
    with pytest.raises(RuntimeError, match="exact qualified"):
        module.compile_execution(lambda: None, None, source="def execute_model(): pass\n")


def test_publication_transform_keeps_other_execution_work(monkeypatch):
    source = """def execute_model(self, scheduler_output, num_reqs):
    if True:
        if True:
            if True:
                self.before()
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)
                self.after()
"""
    namespace = {}
    exec(source, namespace)
    monkeypatch.setattr(module, "EXECUTE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    calls = []
    owner = SimpleNamespace(publish_counts=lambda *args: calls.append("publish"))
    compiled = module.compile_execution(namespace["execute_model"], owner, source=source)
    compiled(SimpleNamespace(before=lambda: calls.append("before"), after=lambda: calls.append("after")), object(), 2)
    assert calls == ["before", "publish", "after"]


def test_skip_sync_requires_enabled_normal_armed_path():
    owner = module.CheckpointPreparationGraphs.__new__(module.CheckpointPreparationGraphs)
    owner.runner = SimpleNamespace(num_accepted_tokens_event=object())
    owner.enabled = True
    owner.verify = False
    owner.armed = None
    owner.avoided_syncs = owner.avoided_uploads = 0
    assert not owner.prepare_counts(1)
    owner.armed = (object(), ("a",))
    owner.verify = True
    assert not owner.prepare_counts(1)
    owner.verify = False
    assert not owner.prepare_counts(2)
    assert owner.prepare_counts(1)
    assert owner.avoided_syncs == owner.avoided_uploads == 1
    owner.enabled = False
    assert not owner.prepare_counts(1)


def test_fallback_publishes_host_counts():
    owner = module.CheckpointPreparationGraphs.__new__(module.CheckpointPreparationGraphs)
    calls = []
    owner.runner = SimpleNamespace(
        num_accepted_tokens=SimpleNamespace(np=np.zeros(4, dtype=np.int32), copy_to_gpu=calls.append),
        input_batch=SimpleNamespace(req_ids=["a"], num_accepted_tokens_cpu=np.array([3], dtype=np.int32)),
    )
    owner.published = None
    owner.avoided_uploads = 0
    schedule = object()
    owner.publish_counts(owner.runner, schedule, 1)
    assert owner.runner.num_accepted_tokens.np[0] == 3 and calls == [1]
    owner.published = (schedule, ("a",))
    owner.publish_counts(owner.runner, schedule, 1)
    assert calls == [1] and owner.avoided_uploads == 1


@pytest.fixture
def deferred_owner(monkeypatch):
    owner = module.CheckpointPreparationGraphs.__new__(module.CheckpointPreparationGraphs)
    owner.enabled = owner.lazy_readback = True
    owner.verify = False
    owner.deferred_counts = None
    owner.avoided_readbacks = owner.materialized_readbacks = 0
    batch = SimpleNamespace(req_ids=["a", "b"], num_accepted_tokens_cpu_tensor=torch.full((4,), 99, dtype=torch.int32))
    owner.runner = SimpleNamespace(
        input_batch=batch, num_accepted_tokens=SimpleNamespace(gpu=torch.ones(4, dtype=torch.int32))
    )
    owner.preparation = SimpleNamespace(enabled=True, verify=False)
    owner.commit = SimpleNamespace(queued_schedule_v1=SimpleNamespace(expected_device_inputs=("a", "b")))
    owner.transition = SimpleNamespace(accept_done=object(), next_pending=(("a", "b"), {}), pending=None)
    events = []
    monkeypatch.setattr(
        module.torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=events.append)),
        raising=False,
    )
    owner.original_retire = lambda sampled: False
    return owner


def test_deferred_counts_have_no_host_write_until_needed(deferred_owner):
    owner = deferred_owner
    sampled = object()
    accepted = SimpleNamespace(
        counts=torch.tensor([3, 2], dtype=torch.int32), next_counts=torch.tensor([1, 2], dtype=torch.int32)
    )
    owner.transition.pending = (sampled, ("a", "b"), {"accepted": accepted})
    assert owner.retire_commit(sampled)
    assert owner.transition.pending is None
    assert owner.runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [99] * 4
    assert owner.deferred_counts[1] is accepted.next_counts
    assert owner.avoided_readbacks == 1
    owner.before_execute(SimpleNamespace(preempted_req_ids=set()))
    assert owner.materialized_readbacks == 0
    # Flush before changing request layout; downstream insertion/compaction can
    # now preserve counts using the same CPU arrays as the original runtime.
    owner.commit.queued_schedule_v1.expected_device_inputs = None
    owner.before_execute(SimpleNamespace(preempted_req_ids=set()))
    assert owner.runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [1, 2, 99, 99]
    assert owner.deferred_counts is None and owner.materialized_readbacks == 1


def test_deferred_counts_cannot_be_published_into_reordered_rows(deferred_owner):
    owner = deferred_owner
    owner.deferred_counts = (("a", "b"), torch.tensor([1, 3], dtype=torch.int32))
    owner.runner.input_batch.req_ids.reverse()
    with pytest.raises(RuntimeError, match="before request layout"):
        owner.materialize_counts()
    assert owner.deferred_counts is not None
    assert owner.runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [99] * 4


@pytest.mark.parametrize("reason", ["disabled", "verify", "preparation_disabled", "preempted", "packet_changed"])
def test_fallback_flushes_counts_before_host_lifecycle(deferred_owner, reason):
    owner = deferred_owner
    owner.deferred_counts = (("a", "b"), torch.tensor([2, 3], dtype=torch.int32))
    schedule = SimpleNamespace(preempted_req_ids=set())
    if reason == "disabled":
        owner.enabled = False
    elif reason == "verify":
        owner.verify = True
    elif reason == "preparation_disabled":
        owner.preparation.enabled = False
    elif reason == "preempted":
        schedule.preempted_req_ids = {"a"}
    else:
        owner.transition.next_pending = None
    owner.before_execute(schedule)
    assert owner.materialized_readbacks == 1 and owner.deferred_counts is None


def test_legacy_commit_clears_superseded_device_counts(deferred_owner):
    owner = deferred_owner
    owner.deferred_counts = (("a", "b"), torch.tensor([2, 3], dtype=torch.int32))
    assert not owner.retire_commit(object())
    assert owner.deferred_counts is None


def test_retirement_identity_failure_preserves_owner(deferred_owner):
    owner = deferred_owner
    owner.transition.pending = (object(), ("a", "b"), {})
    with pytest.raises(RuntimeError, match="lifetime/request order"):
        owner.retire_commit(object())
    assert owner.transition.pending is not None


def test_materialization_orders_legacy_and_device_writers(deferred_owner, monkeypatch):
    owner = deferred_owner
    events = []
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: SimpleNamespace(wait_event=events.append))
    owner.runner.num_accepted_tokens_event = object()
    owner.deferred_counts = (("a", "b"), torch.tensor([1, 3], dtype=torch.int32))
    owner.materialize_counts()
    assert events == [owner.transition.accept_done, owner.runner.num_accepted_tokens_event]
    assert owner.runner.input_batch.num_accepted_tokens_cpu_tensor.tolist() == [1, 3, 99, 99]
