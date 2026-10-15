# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.device_commit_inputs import DeviceCommitInputs, eligible, publish_commit_inputs
from tools.qwen4exp.iteration_transition import accept_step


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("position", [10, 124, 125, 126, 127, 128, 254, 255, 256, 8191, 262140])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_publication_and_acceptance_match_host_admission(n, position, count):
    inputs = torch.full((5, 8), -99, dtype=torch.int32)
    computed = torch.arange(n, dtype=torch.int32) + position
    publish_commit_inputs(inputs, computed, block_size=128)
    expected = inputs.clone()
    for row, value in enumerate(computed.tolist()):
        expected[:, row] = torch.tensor([(value + 3 + 127) // 128 - 1, value, 3, 2, count])
    samples = torch.full((n, 3), -1, dtype=torch.int64)
    samples[:, :count] = torch.arange(count) + 123
    result = accept_step(samples, inputs, block_size=128)
    assert not result.error.any()
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    assert torch.all(inputs[:, n:] == -99)


def fixture():
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 3}, scheduled_spec_decode_tokens={"a": [1, 2]})
    checkpoint = SimpleNamespace(enabled=True, failed=False, backend="fused", published=(schedule, ("a",)))
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a"], num_reqs=1),
        use_async_scheduling=False,
        num_discarded_requests=0,
        with_prefill=False,
        uses_mrope=False,
    )
    owner = SimpleNamespace(
        enabled=True, transition=SimpleNamespace(enabled=True), checkpoint=checkpoint, runner=runner, records={1: {}}
    )
    return owner, schedule, torch.ones((1, 3), dtype=torch.int64)


@pytest.mark.parametrize(
    "condition", ["off", "tensor", "other_schedule", "reorder", "prefill", "discard", "drafts", "shape"]
)
def test_only_exact_published_packet_is_admitted(condition):
    owner, schedule, samples = fixture()
    assert eligible(owner, samples, schedule)
    if condition == "off":
        owner.enabled = False
    elif condition == "tensor":
        owner.checkpoint.backend = "tensor"
    elif condition == "other_schedule":
        owner.checkpoint.published = (object(), ("a",))
    elif condition == "reorder":
        owner.runner.input_batch.req_ids = ["b"]
    elif condition == "prefill":
        owner.runner.with_prefill = True
    elif condition == "discard":
        owner.runner.num_discarded_requests = 1
    elif condition == "drafts":
        schedule.scheduled_spec_decode_tokens["a"] = []
    else:
        samples = samples[:, :2]
    assert not eligible(owner, samples, schedule)


def test_unretired_or_failed_submission_never_falls_back():
    owner = DeviceCommitInputs.__new__(DeviceCommitInputs)
    owner.runner = object()
    for failed, pending in [(True, None), (False, object())]:
        owner.failed = failed
        owner.transition = SimpleNamespace(failed=False, pending=pending)
        with pytest.raises(RuntimeError, match="retain owners"):
            owner.begin(None, None)


def test_ineligible_input_uses_original_before_writes():
    state, schedule, samples = fixture()
    owner = DeviceCommitInputs.__new__(DeviceCommitInputs)
    owner.__dict__.update(state.__dict__)
    owner.enabled = owner.failed = owner.transition.failed = False
    owner.transition.pending = None
    owner.fallbacks = 0
    calls = []
    owner.original = lambda *args: calls.append(args) or False
    assert not owner.begin(samples, schedule)
    assert calls == [(samples, schedule)] and owner.fallbacks == 1
