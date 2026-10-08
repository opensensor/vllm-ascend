"""Real-weight gate: switch all modes, recapture a Python patch, restore baseline."""

import uuid
from pathlib import Path

import pytest

from tools.glm_perf.resident_control import MODES, Control
from tools.glm_perf.suite import Case, run_request

ROOT = Path(__file__).resolve().parents[6]


def test_glm_weights_stay_resident_through_modes_and_python_recapture(resident_client):
    client = resident_client
    initial = client.rpc("resident_status")
    if any(worker["candidate"] != "baseline" for worker in initial):
        raise AssertionError("qualification requires baseline Python on entry")
    identities = {(worker["rank"], worker["pid"], worker["weight_storage_digest"]) for worker in initial}
    case = Case("resident_smoke", "fault", "What is 17 + 28? Answer only the number.", None, 32)
    for mode in MODES:
        workers = client.switch(Control(uuid.uuid4().hex, mode))
        assert {(worker["rank"], worker["pid"], worker["weight_storage_digest"]) for worker in workers} == identities
        row = run_request(case, client.base_url, "glm53-flash-selective-w3", 42, mode)
        assert row["valid"], row.get("error")

    source = (ROOT / "tools/glm_perf/resident_candidates/mtp_norm_reference.py").read_text()
    client.switch(Control(uuid.uuid4().hex, "graph", "mtp_norm_reference", source))
    row = run_request(case, client.base_url, "glm53-flash-selective-w3", 42, "patched")
    assert row["valid"], row.get("error")
    workers = client.switch(Control(uuid.uuid4().hex))
    assert {(worker["rank"], worker["pid"], worker["weight_storage_digest"]) for worker in workers} == identities
    row = run_request(case, client.base_url, "glm53-flash-selective-w3", 42, "restored")
    assert row["valid"], row.get("error")


def test_capture_failure_keeps_pause_and_can_recover_without_reloading(resident_client):
    client = resident_client
    initial = client.rpc("resident_status")
    assert all(worker["candidate"] == "baseline" for worker in initial)
    identities = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in initial}
    source = """def fail_capture(self):
    raise RuntimeError("resident gate injected capture failure")
def replacements():
    return {"vllm_ascend.worker.model_runner_v1:NPUModelRunner.capture_model": fail_capture}
"""
    with pytest.raises(RuntimeError, match="remains paused"):
        client.switch(Control(uuid.uuid4().hex, "graph", "capture_failure", source))
    assert client.request("/is_paused", method="GET")["is_paused"]
    # An uncaught executor error leaves stale rank replies behind. A status
    # round-trip must succeed and report all four dirty workers after failure.
    workers = client.rpc("resident_status")
    assert all(r["graphs_dirty"] for r in workers)
    assert {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in workers} == identities
    workers = client.switch(Control(uuid.uuid4().hex))
    assert all(not r["graphs_dirty"] for r in workers)
    assert {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in workers} == identities
    assert client.request("/is_paused", method="GET")["is_paused"]
    client.resume()
    case = Case("resident_recovered", "fault", "What is 17 + 28? Answer only the number.", None, 32)
    row = run_request(case, client.base_url, "glm53-flash-selective-w3", 42, "recovered")
    assert row["valid"], row.get("error")
