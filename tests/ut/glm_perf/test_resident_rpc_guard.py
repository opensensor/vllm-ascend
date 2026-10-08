# SPDX-License-Identifier: Apache-2.0
"""Failed diagnostics must return every rank's receipt without new broadcasts."""

from types import SimpleNamespace

import pytest

from tools.glm_perf.resident_rpc_guard import WORKER_PREFIX, check_profile_label, guard_replacements, guard_rpc


def test_faulting_status_returns_errors_on_all_ranks():
    def status(self):
        raise AttributeError("bad candidate status counter")

    function = guard_rpc(status)
    receipts = []
    for rank in range(4):
        worker = SimpleNamespace(_resident_error=lambda error, rank=rank: {"rank": rank, "error": str(error)})
        receipts.append(function(worker))
    assert [r["rank"] for r in receipts] == list(range(4))
    assert all(r["error"] == "bad candidate status counter" for r in receipts)


def test_valid_rpc_preserves_arguments_return_value_and_metadata():
    def profile(self, start, label=None):
        return (self, start, label)

    profile.audit = object()
    guarded = guard_rpc(profile)
    worker = object()
    assert guarded(worker, False, label="decode_c1") == (worker, False, "decode_c1")
    assert guarded.__name__ == profile.__name__ and guarded.audit is profile.audit


def test_guard_replacements_only_wraps_diagnostic_callbacks():
    original = lambda self: {}
    mapping = {WORKER_PREFIX + name: original for name in ("profile", "resident_status", "resident_apply")}
    guarded = guard_replacements(mapping)
    assert mapping[WORKER_PREFIX + "profile"] is original
    assert guarded[WORKER_PREFIX + "profile"] is not original
    assert guarded[WORKER_PREFIX + "resident_status"] is not original
    assert guarded[WORKER_PREFIX + "resident_apply"] is original


@pytest.mark.parametrize("label", [None, "cleanup", "prefill_1280", 1])
def test_profile_workload_is_checked_before_any_rpc(label):
    with pytest.raises(ValueError, match="unsupported"):
        check_profile_label(label, ("decode_c1",))


def test_explicitly_allowed_prefill_workload_passes_local_check():
    assert check_profile_label("prefill_1280", ("decode_c1", "prefill_1280")) == "prefill_1280"
