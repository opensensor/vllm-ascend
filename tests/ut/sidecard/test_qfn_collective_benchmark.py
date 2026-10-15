# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from tools.sidecard import qfn_collective_benchmark as benchmark


def graph_snapshot(replays=0, retired=False, tracking=True):
    return dict(
        registered_graphs=2,
        captured_graphs=2,
        failed_graphs=0,
        replay_submissions=replays,
        captured_calls=8,
        replayed_calls=replays * 8,
        reset_graphs=2 if retired else 0,
        active_capture=False,
        retired=retired,
        lifecycle_only=True,
        tracking_enabled=tracking,
        count_semantics="capture-and-replay-submissions" if tracking else "capture-only",
        native_leases_acquired=4,
        native_leases_released=4 if retired else 0,
    )


def status(custom, closed=False, calls=1, replays=None, tracking=True):
    values = [
        dict(enabled=custom, state="closed" if closed else "active", geometries=[dict(calls=calls)]) for _ in range(4)
    ]
    if custom and replays is not None:
        for value in values:
            value["graph_observer"] = graph_snapshot(replays, retired=closed, tracking=tracking)
    return values


def test_matched_engine_configuration():
    baseline = benchmark.engine_options()
    candidate = benchmark.engine_options({"pinned": "configuration"})
    assert candidate.pop("additional_config") == {"npu_collective": {"pinned": "configuration"}}
    assert candidate == baseline
    assert baseline["enforce_eager"] is True
    assert baseline["compilation_config"]["cudagraph_mode"] == "NONE"
    assert baseline["kv_cache_memory_bytes"] == 88673894400
    assert baseline["speculative_config"]["num_speculative_tokens"] == 2
    assert baseline["max_model_len"] == 262144


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("profile", ("eager", "c1-c2", "c3-c4"))
@pytest.mark.parametrize("tracking", (False, True))
def test_controlled_run_releases_before_engine_shutdown(tmp_path, custom, profile, tracking):
    events = []

    class FakeLLM:
        def __init__(self, **options):
            self.custom = "additional_config" in options
            self.calls = 0 if profile == "eager" else 100
            self.replays = 0 if options["enforce_eager"] is False else None
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown")))

        def collective_rpc(self, name):
            events.append(name)
            return status(
                self.custom,
                closed=name == "npu_collective_close",
                calls=self.calls,
                replays=self.replays,
                tracking=tracking,
            )

        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")

        def generate(self, prompts, params, **kw):
            if profile == "eager" or tracking:
                self.calls += 100
            if self.replays is not None and tracking:
                self.replays += 10
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    metrics=None,
                    outputs=[SimpleNamespace(text="output", token_ids=[2] * params.max_tokens, finish_reason="length")],
                )
                for _ in prompts
            ]

    result = benchmark.run(
        tmp_path / "run",
        {"config": True, "graph_replay_tracking": tracking} if custom else None,
        llm_factory=FakeLLM,
        sampling_factory=lambda **kw: SimpleNamespace(**kw),
        graph_profile=profile,
    )
    assert events[-2:] == ["npu_collective_close", "shutdown"]
    assert result["ownership_released"]
    assert len(result["measurements"]) == 5
    assert result["graph_profile"] == profile and result["eager"] is (profile == "eager")
    assert result["measurements"][1]["generated_tokens"] == (256 if profile == "eager" else 512)
    assert result["measurements"][2]["generated_tokens"] == 4 * 128
    if profile != "eager":
        assert result["measurements"][1]["expected_graph_replay"] is True
        assert result["measurements"][2]["expected_graph_replay"] is (profile == "c3-c4")
        if custom:
            evidence = result["measurements"][1]["graph_replay_coverage"]
            assert all(rank["replay_submissions"] == (10 if tracking else None) for rank in evidence)
            assert all(rank["replayed_calls"] == (80 if tracking else None) for rank in evidence)
            assert result["measurements"][1]["replay_counter_observed"] is tracking
            if not tracking:
                assert all(not rank["geometries"] for rank in result["measurements"][1]["collective_coverage"])
    assert json.loads((tmp_path / "run/result.json").read_text()) == result


@pytest.mark.parametrize("profile,sizes", (("c1-c2", [3, 6]), ("c3-c4", [9, 12])))
def test_qualified_graph_options_are_matched_and_keep_mtp_context_and_cache(profile, sizes):
    baseline = benchmark.engine_options(graph_profile=profile)
    candidate = benchmark.engine_options({"graph_mode": "fixed"}, graph_profile=profile)
    assert candidate.pop("additional_config") == {"npu_collective": {"graph_mode": "fixed"}}
    assert candidate == baseline
    assert baseline["enforce_eager"] is False
    assert baseline["compilation_config"] == dict(
        mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=sizes
    )
    eager = benchmark.engine_options()
    for field in (
        "speculative_config",
        "max_model_len",
        "kv_cache_memory_bytes",
        "max_num_seqs",
        "max_num_batched_tokens",
    ):
        assert baseline[field] == eager[field]


@pytest.mark.parametrize("profile", (None, True, [], "graph", "3,6"))
def test_unsupported_graph_profile_rejected_before_output_or_engine_creation(tmp_path, profile):
    with pytest.raises(ValueError, match="graph_profile"):
        benchmark.run(
            tmp_path / "run",
            graph_profile=profile,
            llm_factory=lambda **kwargs: pytest.fail("invalid profile reached engine"),
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "mutation", ("missing", "zero", "negative", "coerced", "failed", "capture", "retired", "ranks")
)
def test_expected_replay_evidence_rejects_missing_stale_or_inconsistent_rank_counters(mutation):
    before, after = status(True, replays=10), status(True, replays=15)
    packet = after[2]["graph_observer"]
    if mutation == "missing":
        del after[2]["graph_observer"]
    elif mutation == "zero":
        packet["replay_submissions"] = 10
    elif mutation == "negative":
        packet["replayed_calls"] = 0
    elif mutation == "coerced":
        packet["replay_submissions"] = True
    elif mutation == "failed":
        packet["failed_graphs"] = 1
    elif mutation == "capture":
        packet["active_capture"] = True
    elif mutation == "retired":
        packet["retired"] = True
    else:
        packet["replay_submissions"] += 1
    with pytest.raises(RuntimeError):
        benchmark.replay_deltas(before, after, required=True)


def test_c4_eager_primary_dispatch_can_report_zero_or_positive_graph_tail_replays():
    assert benchmark.expects_graph_replay("c1-c2", 4) is False
    zero = benchmark.replay_deltas(status(True, replays=10), status(True, replays=10), required=False)
    assert all(rank["replay_submissions"] == rank["replayed_calls"] == 0 for rank in zero)
    positive = benchmark.replay_deltas(status(True, replays=10), status(True, replays=12), required=False)[0]
    assert positive["replay_submissions"] == 2 and positive["replayed_calls"] == 16


@pytest.mark.parametrize(
    "workers,custom,closed,require_calls",
    [
        ([], True, False, False),
        (status(False), True, False, False),
        (status(True), False, False, False),
        (status(True), True, True, False),
        (status(True, calls=0), True, False, True),
    ],
)
def test_worker_proof_rejected(workers, custom, closed, require_calls):
    with pytest.raises(RuntimeError):
        benchmark.verify_workers(workers, custom=custom, closed=closed, require_calls=require_calls)


def test_measurement_requires_new_custom_calls():
    with pytest.raises(RuntimeError, match="measurement did not execute"):
        benchmark.coverage_deltas(status(True, calls=10), status(True, calls=10), custom=True)


def test_measurement_keeps_call_deltas():
    assert (
        benchmark.coverage_deltas(status(True, calls=10), status(True, calls=15), custom=True)[0]["geometries"][0][
            "calls"
        ]
        == 5
    )


@pytest.mark.parametrize(
    "outputs,count,tokens",
    [
        ([], 1, 2),
        ([{"completions": []}], 1, 2),
        ([{"completions": [{"token_ids": [1], "finish_reason": "length"}]}], 1, 2),
        ([{"completions": [{"token_ids": [1, 2], "finish_reason": "stop"}]}], 1, 2),
    ],
)
def test_completion_budget_rejects_short_or_missing_outputs(outputs, count, tokens):
    with pytest.raises(RuntimeError):
        benchmark.verify_completion_budget(outputs, count, tokens)
