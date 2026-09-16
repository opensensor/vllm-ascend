# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Qwen4Exp 310P run-log / first-fatal observability (plan TOBS, R13).

Exercised entirely on host with synthetic sub-reports; no NPU is required.
"""

import json

import pytest

from vllm_ascend.observability.qwen38_runlog import (
    LINK_CROSS_CARD,
    LINK_UNKNOWN,
    LINK_WITHIN_CARD,
    REQUIRED_TRANSFER_METRIC_FIELDS,
    RankIdentity,
    RunLog,
    classify_link,
)

_GIB = 1024**3


# A minimal T0.3-shaped topology: chips 0/1 on card 0, chips 2/3 on card 1.
_TOPOLOGY = {
    "schema_version": 1,
    "timestamp": 0.0,
    "chips": [
        {"chip_id": 0, "card_id": 0},
        {"chip_id": 1, "card_id": 0},
        {"chip_id": 2, "card_id": 1},
        {"chip_id": 3, "card_id": 1},
    ],
    "links": [],
}

_REVISIONS = {"schema_version": 1, "components": {"vllm": {"version": "x"}}}
_MEMORY = {"world_size": 4, "ranks": [{"rank": 0, "device_bytes": 31 * _GIB}]}
_PLE_METRICS = {
    "host_bytes": 95 * _GIB,
    "page_faults": 12,
    "transfer_bytes": 4096,
    "hit_rate": 0.97,
    "lookup_latency": 1.5,
}
_QSA_METRICS = {
    "host_bytes": 0,
    "page_faults": 3,
    "transfer_bytes": 2048,
    "hit_rate": 0.88,
    "lookup_latency": 0.9,
}


def _clock():
    """Deterministic monotonic clock for timestamp fields."""
    state = {"t": 0.0}

    def now():
        state["t"] += 1.0
        return state["t"]

    return now


def _populated_runlog() -> RunLog:
    runlog = RunLog(run_id="run-abc", world_size=4, rank=2, now=_clock())
    runlog.set_rank_identity(RankIdentity(rank=2, world_size=4, local_rank=2, chip_id=2, card_id=1))
    runlog.set_revisions(_REVISIONS)
    runlog.set_topology(_TOPOLOGY)
    runlog.set_memory(_MEMORY)
    runlog.set_token_limits(max_model_len=1_048_576, max_num_batched_tokens=8192, prefill_chunk_size=4096)
    runlog.set_ple_metrics(_PLE_METRICS)
    runlog.set_qsa_metrics(_QSA_METRICS)
    runlog.record_host_memory(2, ple_bytes=95 * _GIB, pinned_bytes=_GIB, rss_bytes=2 * _GIB, swap_bytes=0, numa_node=1)
    runlog.record_lifecycle("gdn_state_alloc", phase="startup", rank=2, detail={"bytes": _GIB})
    return runlog


# --------------------------------------------------------------------------- #
# First-fatal-rank capture (acceptance #1)
# --------------------------------------------------------------------------- #


def test_first_fatal_names_rank_and_preserves_traceback():
    runlog = _populated_runlog()

    def failing_worker():
        raise RuntimeError("expert weight load OOM on shard 3")

    try:
        failing_worker()
    except RuntimeError as exc:
        runlog.record_fatal(rank=3, exc=exc)

    payload = runlog.to_dict()
    fatal = payload["first_fatal"]
    assert fatal is not None
    assert fatal["rank"] == 3
    assert fatal["exception_type"] == "RuntimeError"
    assert "expert weight load OOM on shard 3" in fatal["message"]
    # Traceback preserved with the real frame -- never a bare "worker died".
    assert "failing_worker" in fatal["traceback"]
    assert "Traceback" in fatal["traceback"]
    assert "worker died" not in fatal["traceback"].lower()


def test_first_fatal_wins_over_later_fatals():
    runlog = _populated_runlog()
    try:
        raise ValueError("root cause on rank 1")
    except ValueError as exc:
        runlog.record_fatal(rank=1, exc=exc)
    try:
        raise RuntimeError("secondary cascade on rank 0")
    except RuntimeError as exc:
        runlog.record_fatal(rank=0, exc=exc)

    fatal = runlog.first_fatal
    assert fatal.rank == 1
    assert fatal.exception_type == "ValueError"
    assert "root cause" in fatal.message
    assert runlog.fatal_count == 2


def test_human_summary_includes_fatal_rank_and_traceback():
    runlog = _populated_runlog()
    try:
        raise RuntimeError("kv cache alloc failed")
    except RuntimeError as exc:
        runlog.record_fatal(rank=2, exc=exc)
    text = runlog.human_summary()
    assert "FIRST FATAL on rank 2" in text
    assert "kv cache alloc failed" in text
    assert "traceback (preserved)" in text


# --------------------------------------------------------------------------- #
# All metric sections present (acceptance #2)
# --------------------------------------------------------------------------- #


def test_all_sections_present():
    runlog = _populated_runlog()
    payload = runlog.to_dict()
    for section in (
        "revisions",
        "topology",
        "token_limits",
        "memory",
        "ple_metrics",
        "qsa_metrics",
        "rank_identity",
        "collective_trace",
        "lifecycle_events",
        "host_memory",
        "chunk_progress",
        "heartbeats",
    ):
        assert section in payload, f"missing section {section}"
    assert payload["revisions"] == _REVISIONS
    assert payload["token_limits"]["max_model_len"] == 1_048_576
    assert payload["lifecycle_events"][0]["event"] == "gdn_state_alloc"
    assert runlog.missing_sections() == []


def test_ple_qsa_metrics_carry_required_transfer_fields():
    runlog = _populated_runlog()
    payload = runlog.to_dict()
    for section in ("ple_metrics", "qsa_metrics"):
        for field_name in REQUIRED_TRANSFER_METRIC_FIELDS:
            assert field_name in payload[section], f"{section} missing {field_name}"


def test_missing_sections_flags_unpopulated():
    runlog = RunLog(run_id="bare", world_size=4, now=_clock())
    missing = runlog.missing_sections()
    for expected in ("revisions", "topology", "token_limits", "memory", "ple_metrics", "lifecycle_events"):
        assert expected in missing


def test_json_round_trips():
    runlog = _populated_runlog()
    payload = json.loads(runlog.to_json())
    assert payload["run_id"] == "run-abc"
    assert payload["world_size"] == 4


# --------------------------------------------------------------------------- #
# Collective link-class tracing (acceptance #3, R3)
# --------------------------------------------------------------------------- #


def test_classify_link_from_topology():
    assert classify_link(_TOPOLOGY, 0, 1) == LINK_WITHIN_CARD
    assert classify_link(_TOPOLOGY, 0, 2) == LINK_CROSS_CARD
    assert classify_link(_TOPOLOGY, 0, 99) == LINK_UNKNOWN
    assert classify_link(None, 0, 1) == LINK_UNKNOWN


def test_collective_entries_carry_link_class_from_topology():
    runlog = _populated_runlog()
    within = runlog.record_collective("all_reduce", src_chip=0, dst_chip=1, num_bytes=1024)
    cross = runlog.record_collective("all_reduce", src_chip=0, dst_chip=2, num_bytes=2048)
    assert within.link_class == LINK_WITHIN_CARD
    assert cross.link_class == LINK_CROSS_CARD

    entries = runlog.to_dict()["collective_trace"]
    assert all("link_class" in e for e in entries)
    assert {e["link_class"] for e in entries} == {LINK_WITHIN_CARD, LINK_CROSS_CARD}


def test_collective_summary_aggregates_by_link_class():
    runlog = _populated_runlog()
    runlog.record_collective("all_reduce", src_chip=0, dst_chip=1, num_bytes=1000)
    runlog.record_collective("all_reduce", src_chip=1, dst_chip=0, num_bytes=500)
    runlog.record_collective("all_gather", src_chip=0, dst_chip=3, num_bytes=4000)

    summary = runlog.collective_summary()
    assert summary[LINK_WITHIN_CARD]["count"] == 2
    assert summary[LINK_WITHIN_CARD]["bytes"] == 1500
    assert summary[LINK_CROSS_CARD]["count"] == 1
    assert summary[LINK_CROSS_CARD]["bytes"] == 4000


def test_explicit_link_class_overrides_topology():
    runlog = _populated_runlog()
    entry = runlog.record_collective("all_reduce", src_chip=0, dst_chip=1, num_bytes=1024, link_class=LINK_CROSS_CARD)
    assert entry.link_class == LINK_CROSS_CARD


# --------------------------------------------------------------------------- #
# Progress / heartbeat / throughput + artifact write
# --------------------------------------------------------------------------- #


def test_chunk_progress_and_heartbeat_and_throughput():
    runlog = _populated_runlog()
    progress = runlog.record_chunk_progress(chunk_index=0, tokens_done=4096, elapsed_s=2.0, num_chunks=256)
    assert progress.tokens_per_s == pytest.approx(2048.0)
    ping = runlog.heartbeat(tokens_done=500_000, tokens_total=1_048_576, elapsed_s=100.0, note="long prefill")
    assert ping.tokens_per_s == pytest.approx(5000.0)
    assert ping.fraction == pytest.approx(500_000 / 1_048_576)

    payload = runlog.to_dict()
    assert payload["chunk_progress"][0]["tokens_per_s"] == pytest.approx(2048.0)
    assert payload["heartbeats"][0]["note"] == "long prefill"


def test_write_emits_json_and_human_log(tmp_path):
    runlog = _populated_runlog()
    try:
        raise RuntimeError("boom on rank 2")
    except RuntimeError as exc:
        runlog.record_fatal(rank=2, exc=exc)
    json_path = tmp_path / "runlog.json"
    log_path = tmp_path / "runlog.log"
    runlog.write(json_path=str(json_path), log_path=str(log_path))

    payload = json.loads(json_path.read_text())
    assert payload["first_fatal"]["rank"] == 2
    text = log_path.read_text()
    assert "FIRST FATAL on rank 2" in text


def test_ple_metric_schema_matches_required_prefix():
    from vllm_ascend.observability.qwen38_runlog import ple_metric_schema

    schema = ple_metric_schema()
    # The run log's required transfer fields are the leading entries of T4.4's schema.
    assert schema[: len(REQUIRED_TRANSFER_METRIC_FIELDS)] == REQUIRED_TRANSFER_METRIC_FIELDS
