# SPDX-License-Identifier: Apache-2.0
"""Admission, storage lifetime and reference-preserving WY replay checks."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.wy_graph import WYGraph, matches


def make_owner():
    runtime = SimpleNamespace(capturing=False, stream=7)
    runtime.is_current_stream_capturing = lambda: runtime.capturing
    runtime.current_stream = lambda device: runtime.stream
    values = tuple(torch.arange(8).reshape(2, 4).half() + n for n in range(5))
    original = lambda *args: tuple(value * (n + 1) for n, value in enumerate(args[:5]))
    record = SimpleNamespace(
        inputs=tuple(value.clone() for value in values),
        outputs=original(*values, 64),
        original=original,
        replay_stream=7,
        failed=False,
        receipt=dict(passed=True),
    )

    def replay():
        for target, output in zip(record.outputs, original(*record.inputs, 64)):
            target.copy_(output)

    record.graph = SimpleNamespace(replay=replay)
    return WYGraph(record, runtime=runtime), values, runtime


def test_changed_inputs_noncontiguous_views_and_output_lifetime():
    owner, values, runtime = make_owner()
    owner.verify = False
    first = owner(*values, 64)
    saved = tuple(value.clone() for value in first)
    changed = tuple((torch.arange(16).half() + n).reshape(2, 8)[:, ::2] for n in range(5))
    assert not changed[0].is_contiguous()
    assert matches(changed, owner.record, 64)
    second = owner(*changed, 64)
    for actual, expected in zip(second, owner.original(*changed, 64)):
        assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(first, saved))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first, owner.record.outputs))
    assert owner.replays == 2


@pytest.mark.parametrize("reason", ["disabled", "failed", "shape", "dtype", "chunk", "capture", "stream"])
def test_unsupported_call_retains_original_without_replay(reason):
    owner, values, runtime = make_owner()
    chunk = 64
    if reason == "disabled":
        owner.enabled = False
    if reason == "failed":
        owner.failed = True
    if reason == "shape":
        values = tuple(value[:1] for value in values)
    if reason == "dtype":
        values = tuple(value.float() for value in values)
    if reason == "chunk":
        chunk = 32
    if reason == "capture":
        runtime.capturing = True
    if reason == "stream":
        runtime.stream = 8
    assert all(torch.equal(a, b) for a, b in zip(owner(*values, chunk), owner.original(*values, chunk)))
    assert owner.replays == 0 and owner.fallbacks == 1


def test_verification_failure_returns_reference_then_falls_back():
    owner, values, runtime = make_owner()
    owner.record.graph.replay = lambda: owner.record.outputs[2].fill_(123)
    actual = owner(*values, 64)
    assert all(torch.equal(a, b) for a, b in zip(actual, owner.original(*values, 64)))
    assert owner.failed and owner.mismatched_outputs == ["w"]
    assert owner.checked == owner.replays == 1
    owner.verify = False
    owner(*values, 64)
    assert owner.fallbacks == 1 and owner.replays == 1


def test_shadow_success_returns_independent_reference_outputs():
    owner, values, runtime = make_owner()
    output = owner(*values, 64)
    assert owner.checked == 1 and not owner.failed
    assert all(torch.equal(a, b) for a, b in zip(output, owner.original(*values, 64)))


def test_unqualified_graph_cannot_be_bound():
    owner, _, runtime = make_owner()
    owner.record.failed = True
    with pytest.raises(ValueError, match="qualified"):
        WYGraph(owner.record, runtime=runtime)
