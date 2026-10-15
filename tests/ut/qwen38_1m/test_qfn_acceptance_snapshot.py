# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance reuse must not change raw counts, request ownership or commit."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qfn_acceptance_snapshot as candidate


@dataclass
class Sampler:
    sampled_token_ids: object
    logprobs_tensors: object = None


class DeviceSamples:
    def __init__(self, values):
        self.host = torch.tensor(values)
        self.reads = 0

    def cpu(self):
        self.reads += 1
        return self.host.clone()


def runner():
    return SimpleNamespace(
        _qfn_acceptance_reuse=True,
        _qwen4exp_mtp_ple=True,
        use_async_scheduling=False,
        need_accepted_tokens=True,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"]),
        execute_model_state=None,
    )


def test_raw_acceptance_reused_once_before_vocab_and_discard_filtering():
    owner = runner()
    samples = DeviceSamples([[7, 8, -1], [1000000, -1, -1], [9, 10, 11]])
    logprobs = object()
    observed = []

    def bookkeeping(r, scheduler, sampler):
        assert sampler.logprobs_tensors is logprobs
        assert isinstance(sampler.sampled_token_ids, torch.Tensor)
        # Simulate vocabulary filtering and a discarded prefill row. Neither
        # changes the raw acceptance consumed by Mamba alignment.
        return [[7, 8], [], []]

    def fallback(**kwargs):
        host = kwargs["num_accepted_tokens_cpu_tensor"]
        host.copy_(kwargs["num_accepted_tokens_gpu"])
        observed.append(host.clone())
        # The real align fallback can reset counts after a state copy. This
        # must not mutate the reusable snapshot itself.
        host[0] = 1

    commit = candidate.make_postprocess(fallback)

    def update(r, ids, scheduler):
        assert ids is samples
        commit(
            input_batch=r.input_batch,
            num_reqs=3,
            num_accepted_tokens_gpu=object(),
            num_accepted_tokens_cpu_tensor=torch.empty(3, dtype=torch.int64),
        )

    output = candidate.make_bookkeeping(bookkeeping)(owner, None, Sampler(samples, logprobs))
    assert output == [[7, 8], [], []]
    counts = owner._qfn_acceptance_snapshot[2]
    candidate.make_state_update(update)(owner, samples, None)
    assert observed[0].tolist() == [2, 1, 3]
    assert counts.tolist() == [2, 1, 3]
    assert samples.reads == 1
    assert owner._qfn_acceptance_snapshot is None
    assert not hasattr(owner.input_batch, "_qfn_commit_counts")


@pytest.mark.parametrize("reason", ["disabled", "async", "other_model", "no_acceptance", "all_cache"])
def test_unadmitted_path_is_unchanged(reason):
    owner = runner()
    if reason == "disabled":
        owner._qfn_acceptance_reuse = False
    elif reason == "async":
        owner.use_async_scheduling = True
    elif reason == "other_model":
        owner._qwen4exp_mtp_ple = False
    elif reason == "no_acceptance":
        owner.need_accepted_tokens = False
    else:
        owner.cache_config.mamba_cache_mode = "all"
    samples = DeviceSamples([[1, -1, -1]])
    sampler = Sampler(samples)
    result = candidate.make_bookkeeping(lambda r, s, original: original)(owner, None, sampler)
    assert result is sampler and samples.reads == 0


@pytest.mark.parametrize("change", ["reorder", "replacement", "different_tensor", "disabled"])
def test_mismatched_step_never_reuses_counts(change):
    owner = runner()
    samples = DeviceSamples([[7, -1, -1]] * 3)
    candidate.make_bookkeeping(lambda *args: None)(owner, None, Sampler(samples))
    ids = samples
    if change == "reorder":
        owner.input_batch.req_ids.reverse()
    elif change == "replacement":
        owner.input_batch.req_ids[-1] = "new-request"
    elif change == "different_tensor":
        ids = object()
    else:
        owner._qfn_acceptance_reuse = False

    def update(r, *args):
        assert not hasattr(r.input_batch, "_qfn_commit_counts")

    candidate.make_state_update(update)(owner, ids, None)
    assert owner._qfn_acceptance_snapshot is None


@pytest.mark.parametrize("phase", ["bookkeeping", "commit"])
def test_error_discards_snapshot(phase):
    owner = runner()
    samples = DeviceSamples([[1, -1, -1]] * 3)

    def fail(*args):
        raise RuntimeError("injected")

    if phase == "bookkeeping":
        with pytest.raises(RuntimeError, match="injected"):
            candidate.make_bookkeeping(fail)(owner, None, Sampler(samples))
    else:
        candidate.make_bookkeeping(lambda *args: None)(owner, None, Sampler(samples))
        with pytest.raises(RuntimeError, match="injected"):
            candidate.make_state_update(fail)(owner, samples, None)
    assert owner._qfn_acceptance_snapshot is None
    assert not hasattr(owner.input_batch, "_qfn_commit_counts")


def test_postprocess_fallback_preserves_device_argument():
    batch = SimpleNamespace()
    device_counts = object()
    result = candidate.make_postprocess(lambda **kwargs: kwargs)(
        input_batch=batch, num_reqs=3, num_accepted_tokens_gpu=device_counts
    )
    assert result["num_accepted_tokens_gpu"] is device_counts


@pytest.mark.parametrize("counts", [torch.zeros(2, dtype=torch.int64), torch.zeros(3, dtype=torch.int32)])
def test_invalid_snapshot_rejected(counts):
    batch = SimpleNamespace(_qfn_commit_counts=counts)
    with pytest.raises(ValueError, match="snapshot"):
        candidate.make_postprocess(lambda **kwargs: pytest.fail("must not commit"))(input_batch=batch, num_reqs=3)


@pytest.mark.parametrize("pending", ["execution", "snapshot", "commit"])
def test_switch_rejects_inflight_ownership(pending):
    owner = runner()
    if pending == "execution":
        owner.execute_model_state = object()
    elif pending == "snapshot":
        owner._qfn_acceptance_snapshot = object()
    else:
        owner.input_batch._qfn_commit_counts = object()
    result = candidate.switch(SimpleNamespace(rank=2, model_runner=owner), False)
    assert result["passed"] is False and owner._qfn_acceptance_reuse is True


def test_cold_switch_is_typed_and_reversible():
    owner = runner()
    worker = SimpleNamespace(rank=0, model_runner=owner)
    assert not candidate.switch(worker, 1)["passed"]
    for value in (False, True):
        assert candidate.switch(worker, value) == dict(
            rank=0, passed=True, enabled=value, verify=False, verified_steps=0
        )
        assert owner._qfn_acceptance_reuse is value


def test_hardware_count_verification_is_explicit_and_detects_mismatch():
    batch = SimpleNamespace(_qfn_commit_counts=torch.tensor([1, 3]), _qfn_commit_verify=True)
    postprocess = candidate.make_postprocess(lambda **kwargs: kwargs)
    args = dict(input_batch=batch, num_reqs=2, num_accepted_tokens_gpu=torch.tensor([1, 3], dtype=torch.int32))
    assert postprocess(**args)["num_accepted_tokens_gpu"] is batch._qfn_commit_counts
    assert batch._qfn_acceptance_checks == 1
    args["num_accepted_tokens_gpu"] = torch.tensor([1, 2])
    with pytest.raises(AssertionError):
        postprocess(**args)
    assert batch._qfn_acceptance_checks == 1


def test_repeated_factory_does_not_stack_wrappers():
    def base(*args):
        return args

    first = candidate.make_bookkeeping(base)
    second = candidate.make_bookkeeping(first)
    assert candidate.original_base(second) is base
