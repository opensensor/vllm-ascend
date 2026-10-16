# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.tp6_hc_graph import HCMixGraph, replay


def fixture():
    staged = torch.zeros(3, 4)
    mixed, normalized = staged.clone(), staged.clone()
    graph = SimpleNamespace(replay=lambda: (mixed.copy_(staged * 2), normalized.copy_(staged * 3)))
    record = dict(input=staged, output=(mixed, (staged, normalized)), graph=graph, stream=1, qualified=True)
    runtime = SimpleNamespace(current_stream=lambda device: 1, is_current_stream_capturing=lambda: False)
    return record, runtime


def test_outputs_survive_next_replay_and_residual_aliases_input():
    record, _ = fixture()
    value = torch.ones(3, 4)
    first = replay(value, record)
    replay(torch.full_like(value, 7), record)
    assert torch.equal(first[0], value * 2)
    assert torch.equal(first[1][1], value * 3)
    assert first[1][0] is value


@pytest.mark.parametrize("reason", ["grad", "capture", "stream", "shape", "dtype", "failed", "strides"])
def test_unsupported_calls_keep_original(reason):
    record, runtime = fixture()
    value = torch.ones(3, 4)
    if reason == "capture":
        runtime.is_current_stream_capturing = lambda: True
    elif reason == "stream":
        runtime.current_stream = lambda device: 2
    elif reason == "shape":
        value = value[:1]
    elif reason == "dtype":
        value = value.double()
    elif reason == "strides":
        value = torch.ones(4, 3).T
    graph = HCMixGraph(lambda v: "reference", record, runtime)
    if reason == "failed":
        record["failed"] = True
    with torch.set_grad_enabled(reason == "grad"):
        assert graph(value) == "reference"
    assert graph.calls == 0 and graph.fallbacks == 1


def test_requires_qualified_record_and_counts_replays():
    record, runtime = fixture()
    record["qualified"] = False
    with pytest.raises(ValueError):
        HCMixGraph(None, record, runtime)
    record["qualified"] = True
    graph = HCMixGraph(None, record, runtime)
    with torch.inference_mode():
        result = graph(torch.ones(3, 4))
    assert graph.calls == 1 and graph.fallbacks == 0
    assert torch.equal(result[0], torch.full((3, 4), 2))


def test_retirement_keeps_uncertain_graphs_and_resets_completed_private_ones():
    from tools.qwen4exp.tp6_hc_graph_gate import retire

    calls = []
    runtime = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    record = dict(graph=SimpleNamespace(reset=lambda: calls.append("reset")), input=object())
    owner = dict(records=[record])
    with pytest.raises(RuntimeError):
        retire(owner, runtime)
    assert not calls and record
    owner["completed"] = True
    assert retire(owner, runtime)["passed"]
    assert calls == ["sync", "reset"] and not record and not owner["records"]
    with pytest.raises(RuntimeError):
        retire(owner, runtime)
