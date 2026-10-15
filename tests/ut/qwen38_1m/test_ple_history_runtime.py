# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp.ple_history_runtime import ResidentPLEHistory, uniform_history_rows
from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER
from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC
from tools.qwen4exp.resident_candidates.qfn_ple_history import make_replacements


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_admission_requires_exact_mtp2_segments(n):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=n),
        with_prefill=False,
        query_start_loc=SimpleNamespace(np=np.arange(n + 1) * 3),
    )
    assert uniform_history_rows(runner, n * 3)
    assert not uniform_history_rows(runner, n * 3 + 3)
    runner.with_prefill = True
    assert not uniform_history_rows(runner, n * 3)
    runner.with_prefill = False
    runner.query_start_loc.np[1] -= 1
    assert not uniform_history_rows(runner, n * 3)


def test_rejected_input_fast_path_discards_history_continuation():
    owner = ResidentPLEHistory.__new__(ResidentPLEHistory)
    owner.pending = ("old",)
    owner.input_result(False)
    assert owner.pending is None and owner.continuation is None
    owner.pending = ("same",)
    owner.input_result(True)
    assert owner.continuation == ("same",) and owner.pending is None


def test_changed_request_order_cannot_advance_history():
    owner = ResidentPLEHistory.__new__(ResidentPLEHistory)
    owner.enabled = True
    owner.staged = ("a", "b")
    owner.pending = ("stale",)
    owner.transition = SimpleNamespace(next_pending=(("b", "a"), {}))
    owner.advance()
    assert owner.pending is None


def test_uncertain_submission_never_falls_back_to_cpu_history():
    owner = ResidentPLEHistory.__new__(ResidentPLEHistory)
    owner.failed = True
    with pytest.raises(RuntimeError, match="retain graph owners"):
        owner.stage(3, lambda *args: pytest.fail("must not silently continue"))


def staged_owner(monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=lambda event: None)),
        raising=False,
    )
    owner = ResidentPLEHistory.__new__(ResidentPLEHistory)
    owner.enabled = True
    owner.verify = owner.failed = False
    owner.checked_histories = 0
    owner.early_readback = False
    owner.early_packets = owner.ready_packets = 0
    owner.packet_requests = None
    owner.staged = owner.pending = None
    owner.continuation = ("a", "b")
    owner.transition = SimpleNamespace(enabled=True)
    owner.done = object()
    owner.history = torch.tensor([[4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0]], dtype=torch.int32)
    owner.history_width = 3
    owner.packet = torch.tensor([[4, 5, 6, 10, 11, 12], [7, 8, 9, 13, 14, 15]], dtype=torch.int64)
    owner.host_packet = torch.empty_like(owner.packet)
    owner.first_input = torch.zeros(4, dtype=torch.int64)
    owner.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["a", "b"]),
        with_prefill=False,
        query_start_loc=SimpleNamespace(np=np.array([0, 3, 6])),
        input_ids=SimpleNamespace(gpu=torch.tensor([10, 11, 12, 13, 14, 15]), cpu=torch.full((6,), -1)),
        _ple_eos_token_id=99,
        _ple_context_cpu=torch.full((5, 3), -1, dtype=torch.int32),
        _ple_context_gpu=torch.empty((5, 3), dtype=torch.int32),
        _ple_query_start_loc_cpu=torch.empty(6, dtype=torch.int32),
        _ple_query_start_loc_gpu=torch.empty(6, dtype=torch.int32),
    )
    return owner


def test_continuation_never_uses_cpu_token_placeholders(monkeypatch):
    owner = staged_owner(monkeypatch)
    starts, context = owner.stage(6, lambda *args: pytest.fail("must not rebuild history from CPU tokens"))
    assert context.tolist() == [[4, 5, 6], [7, 8, 9], [99, 99, 99], [99, 99, 99], [99, 99, 99]]
    assert starts.tolist() == [0, 3, 6, 6, 6, 6]
    assert owner.first_input[:2].tolist() == [10, 13]
    assert owner.runner.input_ids.cpu.tolist() == [10, 11, 12, 13, 14, 15]
    assert owner.staged == ("a", "b") and owner.continuation is None and not owner.failed


def test_early_packet_waits_before_host_lookup_and_never_recopies(monkeypatch):
    owner = staged_owner(monkeypatch)
    owner.early_readback = True
    owner.packet_requests = ("a", "b")
    packet = owner.packet.clone()
    owner.packet.fill_(-999)  # The consumer must use the queued host snapshot.
    events = []

    def complete():
        events.append("complete")
        owner.host_packet.copy_(packet)

    owner.packet_done = SimpleNamespace(query=lambda: False, synchronize=complete)
    owner.stage(6, lambda *args: pytest.fail("unexpected CPU fallback"))
    assert events == ["complete"]
    assert owner.runner.input_ids.cpu.tolist() == [10, 11, 12, 13, 14, 15]
    assert owner.packet_requests is None and owner.ready_packets == 0


def test_early_packet_rejects_stale_request_layout(monkeypatch):
    owner = staged_owner(monkeypatch)
    owner.early_readback = True
    owner.packet_requests = ("b", "a")
    with pytest.raises(RuntimeError, match="different iteration layout"):
        owner.stage(6, lambda *args: pytest.fail("unexpected fallback"))
    assert owner.failed


def test_early_packet_copy_follows_history_graph_on_producer_stream(monkeypatch):
    owner = staged_owner(monkeypatch)
    events = []
    owner.early_readback = True
    owner.staged = ("a", "b")
    owner.transition.next_pending = (("a", "b"), {})
    owner.transition.next_done = object()
    owner.graphs = {2: SimpleNamespace(replay=lambda: events.append("graph"))}
    owner.done = SimpleNamespace(record=lambda: events.append("history_done"))
    owner.packet_done = SimpleNamespace(record=lambda: events.append("packet_done"))

    class HostPacket:
        def __getitem__(self, rows):
            assert rows == slice(None, 2)
            return self

        def copy_(self, source, *, non_blocking):
            assert non_blocking and torch.equal(source, owner.packet)
            events.append("copy")

    owner.host_packet = HostPacket()
    owner.advance()
    assert events == ["graph", "history_done", "copy", "packet_done"]
    assert owner.pending == owner.packet_requests == ("a", "b")
    assert owner.early_packets == 1 and not owner.failed


def test_verification_checks_history_before_consuming_it(monkeypatch):
    owner = staged_owner(monkeypatch)
    owner.verify = True
    with pytest.raises(AssertionError):
        owner.stage(6, lambda *args: None)  # deliberately wrong host reference
    assert owner.failed and not owner.checked_histories


def test_verification_rejects_wrong_lookup_token_packet(monkeypatch):
    owner = staged_owner(monkeypatch)
    owner.verify = True

    def original(runner, padded):
        runner._ple_context_cpu[:2].copy_(owner.history[:2])
        # Keep deliberately invalid CPU token IDs as the comparison oracle.

    with pytest.raises(AssertionError):
        owner.stage(6, original)
    assert owner.failed and not owner.checked_histories


def test_new_layout_bootstraps_authoritative_history_once(monkeypatch):
    owner = staged_owner(monkeypatch)
    owner.continuation = ("different", "rows")
    calls = []

    def original(runner, padded):
        calls.append(padded)
        runner._ple_context_gpu.fill_(17)
        return runner._ple_query_start_loc_gpu, runner._ple_context_gpu

    owner.stage(6, original)
    assert calls == [6] and owner.history[:2].tolist() == [[17, 17, 17], [17, 17, 17]]


def test_candidate_keeps_transition_order_and_rejects_undrained_control():
    calls = []
    transition = SimpleNamespace(take_inputs=lambda *args: True, enabled=True, failed=False, pending=None)
    history = SimpleNamespace(
        transition=transition,
        failed=False,
        input_result=lambda used: calls.append(("input", used)),
        advance=lambda: calls.append("history"),
        status=lambda: dict(host_scheduler_retained=True),
    )

    def draft(*args):
        calls.append("draft_and_next_ids")
        return "draft_ids"

    base = {RUNNER + ".propose_draft_token_ids": draft, RPC: lambda *args: "base_control"}
    runner = SimpleNamespace(_stage_qwen4exp_ple_inputs=lambda *args: None, _prepare_input_ids=lambda *args: None)
    wrappers = make_replacements(base, history, runner)
    assert wrappers[RUNNER + ".propose_draft_token_ids"](None) == "draft_ids"
    wrappers[INPUT_TARGET](None, None, 1, 3, None)
    assert calls == ["draft_and_next_ids", "history", ("input", True)]
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(execute_model_state=None))
    request = json.dumps(dict(ple_history=True, enabled=True, verify=True))
    assert wrappers[RPC](worker, request)["passed"]
    assert history.enabled and history.verify and history.pending is None
    transition.pending = object()
    assert not wrappers[RPC](worker, request)["passed"]
    assert wrappers[RPC](worker, True) == "base_control"
