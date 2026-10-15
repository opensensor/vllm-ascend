# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from tools.sidecard.qfn_collective_compare import compare, prefix_length


@pytest.fixture
def receipts():
    trial = dict(
        label="c1-a",
        concurrency=1,
        generated_tokens=2,
        elapsed_seconds=1,
        outputs=[dict(prompt_tokens=8, num_cached_tokens=0, completions=[dict(token_ids=[1, 2])])],
    )
    baseline = dict(mode="hccl", eager=True, ownership_released=True, measurements=[trial])
    candidate = copy.deepcopy(baseline)
    candidate["mode"] = "custom"
    candidate["measurements"][0].update(
        elapsed_seconds=0.5,
        collective_coverage=[dict(geometries=[dict(calls=10)], fallback_schemas=[]) for _ in range(4)],
    )
    return (
        baseline,
        candidate,
        {"enforce_eager": True},
        {"enforce_eager": True, "additional_config": {"npu_collective": {}}},
    )


def test_comparison_uses_completed_time_and_does_not_certify_quality(receipts):
    baseline, candidate, bo, co = receipts
    candidate["measurements"][0]["outputs"][0]["completions"][0]["token_ids"] = [1, 3]
    result = compare(baseline, candidate, bo, co)
    assert result["summaries"][0]["median_rate_ratio"] == 2
    assert result["trials"][0]["requests"][0]["shared_prefix_tokens"] == 1
    assert result["model_numerical_qualification"] is False
    assert result["confidence_interval_computed"] is False


@pytest.mark.parametrize("mutation", ["options", "ownership", "cache", "budget", "coverage", "duration"])
def test_rejects_unmatched_or_unproved_result(receipts, mutation):
    b, c, bo, co = receipts
    trial = c["measurements"][0]
    if mutation == "options":
        co["enforce_eager"] = False
    elif mutation == "ownership":
        c["ownership_released"] = False
    elif mutation == "cache":
        trial["outputs"][0]["num_cached_tokens"] = 4
    elif mutation == "budget":
        trial["generated_tokens"] = 1
    elif mutation == "coverage":
        trial["collective_coverage"][2]["geometries"][0]["calls"] = 0
    else:
        trial["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        compare(b, c, bo, co)


def test_prefix_handles_different_lengths():
    assert prefix_length([1, 2], [1]) == 1
    assert prefix_length([1, 2], [1, 3]) == 1


def graph_receipts(receipts, profile="c1-c2"):
    baseline, candidate, bo, co = receipts
    compilation = dict(
        mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[3, 6] if profile == "c1-c2" else [9, 12]
    )
    for result in (baseline, candidate):
        result.update(eager=False, graph_profile=profile)
        result["measurements"][0]["expected_graph_replay"] = True
    for options in (bo, co):
        options.update(enforce_eager=False, compilation_config=copy.deepcopy(compilation))
    co["additional_config"]["npu_collective"]["graph_replay_tracking"] = True
    candidate["measurements"][0]["graph_replay_coverage"] = [graph_evidence() for _ in range(4)]
    candidate["closed_workers"] = [
        dict(
            state="closed",
            graph_observer=dict(
                registered_graphs=2,
                captured_graphs=2,
                failed_graphs=0,
                replay_submissions=9,
                captured_calls=8,
                replayed_calls=72,
                reset_graphs=2,
                retired=True,
                active_capture=False,
                lifecycle_only=True,
                tracking_enabled=True,
                count_semantics="capture-and-replay-submissions",
                native_leases_acquired=4,
                native_leases_released=4,
            ),
        )
        for _ in range(4)
    ]
    return baseline, candidate, bo, co


def graph_evidence(tracking=True, replays=9):
    return dict(
        replay_submissions=replays if tracking else None,
        replayed_calls=replays * 8 if tracking else None,
        tracking_enabled=tracking,
        count_semantics="capture-and-replay-submissions" if tracking else "capture-only",
        registered_graphs=2,
        captured_graphs=2,
        captured_calls=8,
        native_leases_acquired=4,
        native_leases_released=0,
    )


@pytest.mark.parametrize("profile", ("c1-c2", "c3-c4"))
def test_matching_graph_profiles_require_replay_and_retirement_evidence(receipts, profile):
    values = graph_receipts(receipts, profile)
    result = compare(*values)
    assert result["eager"] is False and result["graph_profile"] == profile
    assert result["trials"][0]["expected_graph_replay"] is True
    assert result["trials"][0]["graph_replay_coverage"][0] == graph_evidence()
    assert result["model_numerical_qualification"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "profile",
        "sizes",
        "eager",
        "expectation",
        "missing-replays",
        "zero-replays",
        "coerced-replays",
        "rank-replays",
        "missing-retirement",
        "not-retired",
        "not-reset",
        "not-closed",
    ),
)
def test_graph_comparison_rejects_different_profiles_unproved_replay_and_live_graphs(receipts, mutation):
    baseline, candidate, bo, co = graph_receipts(receipts)
    trial = candidate["measurements"][0]
    if mutation == "profile":
        candidate["graph_profile"] = "c3-c4"
    elif mutation == "sizes":
        bo["compilation_config"]["cudagraph_capture_sizes"] = [3, 6, 12]
        co["compilation_config"] = copy.deepcopy(bo["compilation_config"])
    elif mutation == "eager":
        candidate["eager"] = True
    elif mutation == "expectation":
        trial["expected_graph_replay"] = False
    elif mutation == "missing-replays":
        del trial["graph_replay_coverage"]
    elif mutation == "zero-replays":
        trial["graph_replay_coverage"][1]["replay_submissions"] = 0
    elif mutation == "coerced-replays":
        trial["graph_replay_coverage"][1]["replay_submissions"] = True
    elif mutation == "rank-replays":
        trial["graph_replay_coverage"][1]["replayed_calls"] += 1
    elif mutation == "missing-retirement":
        del candidate["closed_workers"]
    elif mutation == "not-retired":
        candidate["closed_workers"][1]["graph_observer"]["retired"] = False
    elif mutation == "not-reset":
        candidate["closed_workers"][1]["graph_observer"]["reset_graphs"] = 1
    else:
        candidate["closed_workers"][1]["state"] = "active"
    with pytest.raises(ValueError):
        compare(baseline, candidate, bo, co)


def test_c4_default_graph_profile_does_not_claim_required_primary_replays(receipts):
    baseline, candidate, bo, co = graph_receipts(receipts)
    for result in (baseline, candidate):
        trial = result["measurements"][0]
        trial.update(concurrency=4, generated_tokens=8, expected_graph_replay=False)
        trial["outputs"] *= 4
    candidate["measurements"][0]["graph_replay_coverage"] = [graph_evidence(replays=0) for _ in range(4)]
    result = compare(baseline, candidate, bo, co)
    assert result["trials"][0]["expected_graph_replay"] is False


def test_untracked_production_graph_comparison_accepts_zero_python_enqueue_deltas_without_inventing_replays(receipts):
    baseline, candidate, bo, co = graph_receipts(receipts)
    co["additional_config"]["npu_collective"]["graph_replay_tracking"] = False
    candidate["measurements"][0]["graph_replay_coverage"] = [graph_evidence(tracking=False) for _ in range(4)]
    for rank in candidate["measurements"][0]["collective_coverage"]:
        rank["geometries"] = []
    for worker in candidate["closed_workers"]:
        worker["graph_observer"].update(
            tracking_enabled=False, count_semantics="capture-only", replay_submissions=0, replayed_calls=0
        )
    result = compare(baseline, candidate, bo, co)
    trial = result["trials"][0]
    assert trial["replay_counter_observed"] is False
    assert trial["custom_calls_per_rank"] == [0] * 4
    assert all(
        rank["replay_submissions"] is None and rank["replayed_calls"] is None for rank in trial["graph_replay_coverage"]
    )
    assert "unobserved" in result["graph_evidence_scope"]


@pytest.mark.parametrize("mutation", ("missing-capture", "missing-lease", "unreleased-lease", "invented-replay"))
def test_untracked_production_requires_cold_capture_and_native_lease_release_evidence(receipts, mutation):
    baseline, candidate, bo, co = graph_receipts(receipts)
    co["additional_config"]["npu_collective"]["graph_replay_tracking"] = False
    candidate["measurements"][0]["graph_replay_coverage"] = [graph_evidence(tracking=False) for _ in range(4)]
    if mutation == "missing-capture":
        candidate["measurements"][0]["graph_replay_coverage"][1]["captured_calls"] = 0
    elif mutation == "missing-lease":
        candidate["measurements"][0]["graph_replay_coverage"][1]["native_leases_acquired"] = 0
    elif mutation == "unreleased-lease":
        candidate["closed_workers"][1]["graph_observer"]["native_leases_released"] = 0
    else:
        candidate["measurements"][0]["graph_replay_coverage"][1]["replay_submissions"] = 0
    with pytest.raises(ValueError):
        compare(baseline, candidate, bo, co)
