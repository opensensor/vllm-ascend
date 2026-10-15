# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.qwen4exp.iteration_transition_runtime import ResidentDeviceTransition, admit_all_acceptances
from tools.qwen4exp.mamba_commit_runtime import admit_commit
from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER, original_base
from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET, make_replacements
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import MAMBA, RPC
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import make_replacements as commit_replacements


def test_admission_checks_unobserved_acceptances_before_any_submission():
    columns = [np.array([value], dtype=np.int32) for value in (0, 126, 3, 2)]
    tables = {1: np.array([[0, -1, 2, 3]], dtype=np.int32)}
    # Count one does not copy; count three needs the temporal source at column
    # one. It must be rejected without waiting to discover which is sampled.
    with pytest.raises(ValueError, match="invalid block ID"):
        admit_all_acceptances(columns, tables, block_size=128, slots=4)
    tables[1][0, 1] = 1
    packed = admit_all_acceptances(columns, tables, block_size=128, slots=4)
    assert np.all(packed[4] == 0)


@pytest.mark.parametrize("scheduled,drafts", [(2, 1), (1, 0), (4, 2)])
def test_nonuniform_decode_rejected(scheduled, drafts):
    columns = [np.array([value], dtype=np.int32) for value in (0, 10, scheduled, drafts)]
    with pytest.raises(ValueError, match="uniform MTP2"):
        admit_all_acceptances(columns, {1: np.zeros((1, 4), dtype=np.int32)}, block_size=128, slots=4)


def reference_admission(columns, tables, *, block_size, slots):
    for count in range(4):
        packed, _, _ = admit_commit(
            columns,
            np.full(len(columns[0]), count, dtype=np.int32),
            tables,
            block_size=block_size,
            slots=slots,
        )
    packed[4].fill(0)
    return packed


@pytest.mark.parametrize("block_size", [1, 2, 3, 128, (1 << 31) - 1])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_union_admission_matches_all_four_reference_checks(block_size, n):
    rng = np.random.default_rng(739 + n)
    for trial in range(120):
        computed = rng.integers(0, 1024, n, dtype=np.int64)
        if trial < 10:
            computed[:] = max(0, min((1 << 31) - 1, block_size + trial - 5))
        if trial == 10:
            computed[:] = (1 << 31) - 1
        source = np.maximum((computed - 1) // block_size, 0)
        columns = [source, computed, np.full(n, 3), np.full(n, 2)]
        table = rng.integers(0, 64, (n, 1030), dtype=np.int32)
        if trial % 3 == 0:
            # Corrupt an accessed or potentially unaccessed speculative slot.
            table[np.arange(n), np.minimum(source + trial % 4, 1029)] = -1 if trial % 2 else 64
        tables = {1: table, 2: table.copy()}
        if trial % 11 == 0:
            tables[2] = table.astype(np.int64)
        elif trial % 13 == 0:
            tables[2] = table[:, :1]
        try:
            expected = reference_admission(columns, tables, block_size=block_size, slots=64)
        except ValueError:
            with pytest.raises(ValueError):
                admit_all_acceptances(columns, tables, block_size=block_size, slots=64)
        else:
            actual = admit_all_acceptances(columns, tables, block_size=block_size, slots=64)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [-1, 1 << 31, 1.5])
def test_union_admission_rejects_invalid_common_metadata(value):
    columns = [np.array([0]), np.array([value]), np.array([3]), np.array([2])]
    with pytest.raises(ValueError):
        admit_all_acceptances(columns, {}, block_size=128, slots=64)


def wrapper_fixture():
    calls = []

    def bookkeeping(*args, **kwargs):
        calls.append("bookkeeping")
        return "output"

    def legacy_bookkeeping(*args, **kwargs):
        calls.append("legacy")
        return bookkeeping(*args, **kwargs)

    legacy_bookkeeping._qfn_transition_base = bookkeeping
    base = {
        RUNNER + "._bookkeeping_sync": legacy_bookkeeping,
        RUNNER + "._update_states_after_model_execute": lambda *args: calls.append("legacy_commit"),
        RPC: lambda *args: "old_control",
    }
    runner = SimpleNamespace(
        propose_draft_token_ids=lambda *args, **kwargs: "drafts",
        _prepare_input_ids=lambda *args: calls.append("host_inputs"),
    )
    resource = SimpleNamespace(
        begin=lambda *args: True,
        retire_commit=lambda *args: True,
        make_next=lambda value: calls.append(value),
        take_inputs=lambda *args: True,
        pending=None,
        next_pending=None,
        failed=False,
        enabled=False,
        verify=False,
        status=lambda: {},
    )
    return calls, make_replacements(base, resource, runner), resource


def test_device_path_does_not_repeat_legacy_commit_or_host_count_snapshot():
    calls, wrappers, _ = wrapper_fixture()
    sampled = SimpleNamespace(sampled_token_ids=object())
    assert wrappers[RUNNER + "._bookkeeping_sync"](None, None, sampled) == "output"
    wrappers[RUNNER + "._update_states_after_model_execute"](None, None, None)
    assert wrappers[RUNNER + ".propose_draft_token_ids"](None) == "drafts"
    wrappers[INPUT_TARGET](None, None, 1, 3, None)
    assert calls == ["bookkeeping", "drafts"]


def test_unsupported_rows_keep_original_fallback():
    calls, wrappers, resource = wrapper_fixture()
    resource.begin = resource.retire_commit = resource.take_inputs = lambda *args: False
    wrappers[RUNNER + "._bookkeeping_sync"](None, None, SimpleNamespace(sampled_token_ids=None))
    wrappers[RUNNER + "._update_states_after_model_execute"](None, None, None)
    wrappers[INPUT_TARGET](None, None, 1, 3, None)
    assert calls == ["legacy", "bookkeeping", "legacy_commit", "host_inputs"]


def test_failure_after_submission_does_not_fallback():
    calls, wrappers, resource = wrapper_fixture()

    def uncertain(*args):
        raise RuntimeError("uncertain native submission")

    resource.begin = uncertain
    with pytest.raises(RuntimeError, match="uncertain"):
        wrappers[RUNNER + "._bookkeeping_sync"](None, None, SimpleNamespace(sampled_token_ids=None))
    assert not calls


def test_control_refuses_pending_and_keeps_legacy_controls():
    _, wrappers, resource = wrapper_fixture()
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(execute_model_state=None))
    request = json.dumps(dict(transition=True, enabled=True, verify=True))
    result = wrappers[RPC](worker, request)
    assert result["passed"] and result["host_scheduler_retained"] and resource.enabled
    resource.pending = object()
    assert not wrappers[RPC](worker, request)["passed"]
    assert wrappers[RPC](worker, True) == "old_control"


def test_reordered_or_finished_requests_cannot_consume_previous_tokens():
    owner = ResidentDeviceTransition.__new__(ResidentDeviceTransition)
    owner.enabled = True
    owner.next_pending = (("old",), {})
    owner.runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["new"]))
    assert not owner.take_inputs(None, 1, 3)
    assert owner.next_pending is None


def test_commit_snapshot_unwrapper_handles_existing_native_wrapper():
    def original(**kwargs):
        return None

    functions = commit_replacements({MAMBA: original, RPC: lambda *args: None}, SimpleNamespace(enabled=False))
    assert original_base(functions[MAMBA]) is original
