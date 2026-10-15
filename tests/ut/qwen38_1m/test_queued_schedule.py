# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from tools.qwen4exp.queued_engine import QueuedEngineControl
from tools.qwen4exp.queued_schedule import QueuedScheduleBridge, can_use_device_drafts, materialize_schedule
from tools.qwen4exp.resident_candidates import qfn_queued_schedule as candidate
from tools.qwen4exp.resident_candidates.qfn_device_transition import INPUT_TARGET
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC
from tools.qwen4exp.resident_candidates.qfn_ple_history import STAGE_TARGET


@dataclass
class Cached:
    req_ids: list
    num_computed_tokens: list
    num_output_tokens: list
    resumed_req_ids: set = field(default_factory=set)


@dataclass
class Schedule:
    scheduled_cached_reqs: Cached
    scheduled_spec_decode_tokens: dict
    num_scheduled_tokens: dict
    total_num_scheduled_tokens: int = 3
    new_block_ids_to_zero: list = field(default_factory=lambda: [42])
    scheduled_new_reqs: list = field(default_factory=list)
    finished_req_ids: set = field(default_factory=set)
    has_structured_output_requests: bool = False


@pytest.mark.parametrize("accepted", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_rejection_corrects_only_worker_copy_and_preserves_allocations(accepted, n):
    ids = [str(i) for i in range(n)]
    schedule = Schedule(Cached(ids, [131] * n, [4] * n), {i: [-1, -1] for i in ids}, {i: 3 for i in ids})
    requests = {i: SimpleNamespace(num_prompt_tokens=128, output_token_ids=[7] * (1 + accepted)) for i in ids}
    result, count = materialize_schedule(schedule, requests, {i: [9, 10] for i in ids}, max_model_len=512)
    assert result.scheduled_cached_reqs.num_computed_tokens == [128 + accepted] * n
    assert result.scheduled_cached_reqs.num_output_tokens == [1 + accepted] * n
    assert count == (n if accepted < 3 else 0)
    assert result.scheduled_spec_decode_tokens == {i: [9, 10] for i in ids}
    assert schedule.scheduled_spec_decode_tokens == {i: [-1, -1] for i in ids}
    assert schedule.scheduled_cached_reqs.num_computed_tokens == [131] * n
    assert result.num_scheduled_tokens is schedule.num_scheduled_tokens
    assert result.new_block_ids_to_zero is schedule.new_block_ids_to_zero


def test_worker_tokens_are_not_truncated_to_lagging_engine_output_count():
    schedule = Schedule(Cached(["a"], [131], [1]), {"a": [-1, -1]}, {"a": 3})
    result, _ = materialize_schedule(
        schedule,
        {"a": SimpleNamespace(num_prompt_tokens=128, output_token_ids=[7, 8])},
        {"a": [9, 10]},
        max_model_len=512,
    )
    assert result.scheduled_cached_reqs.num_output_tokens == [2]


@pytest.mark.parametrize("case", ["missing_draft", "negative_draft", "short_draft", "behind", "context", "geometry"])
def test_invalid_queue_contract_fails_before_model_execution(case):
    schedule = Schedule(Cached(["a"], [131], [1]), {"a": [-1, -1]}, {"a": 3})
    requests = {"a": SimpleNamespace(num_prompt_tokens=128, output_token_ids=[7, 8])}
    drafts, limit = {"a": [9, 10]}, 512
    if case == "missing_draft":
        drafts = {}
    elif case == "negative_draft":
        drafts["a"][0] = -1
    elif case == "short_draft":
        drafts["a"] = [9]
    elif case == "behind":
        schedule.scheduled_cached_reqs.num_computed_tokens = [128]
    elif case == "context":
        limit = 130
    else:
        schedule.num_scheduled_tokens["a"] = 2
    with pytest.raises(ValueError):
        materialize_schedule(schedule, requests, drafts, max_model_len=limit)


def test_prefill_and_resumed_offsets_are_not_replaced_with_output_length():
    schedule = Schedule(Cached(["a", "b"], [32, 64], [0, 5], {"b"}), {}, {"a": 32, "b": 32})
    requests = {"a": SimpleNamespace(output_token_ids=[]), "b": SimpleNamespace(output_token_ids=[7] * 5)}
    result, count = materialize_schedule(schedule, requests, {}, max_model_len=512)
    assert result.scheduled_cached_reqs.num_computed_tokens == [32, 64]
    assert count == 0
    schedule.scheduled_spec_decode_tokens = {"b": [-1, -1]}
    with pytest.raises(ValueError, match="resumed"):
        materialize_schedule(schedule, requests, {}, max_model_len=512)


def test_later_recompute_chunks_with_generated_history_are_still_prefill():
    schedule = Schedule(Cached(["a"], [128], [7]), {}, {"a": 128})
    request = SimpleNamespace(num_prompt_tokens=512, output_token_ids=[7] * 7)
    result, count = materialize_schedule(schedule, {"a": request}, {}, max_model_len=1024)
    assert result.scheduled_cached_reqs.num_computed_tokens == [128]
    assert result.scheduled_cached_reqs.num_output_tokens == [7] and count == 0
    schedule.num_scheduled_tokens["a"] = 1024
    with pytest.raises(ValueError, match="recompute chunk"):
        materialize_schedule(schedule, {"a": request}, {}, max_model_len=1024)


def test_disabled_worker_bridge_does_not_observe_or_touch_drafts():
    bridge = QueuedScheduleBridge(SimpleNamespace())
    schedule = object()
    assert bridge.resolve(schedule) is schedule


class Scheduler:
    def _update_after_schedule(self, output):
        output.append("scheduler")


class BalanceScheduler(Scheduler):
    def schedule(self):
        return "balanced"


class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, output):
        super()._update_after_schedule(output)
        output.append("placeholders")


def engine_fixture():
    scheduler = BalanceScheduler()
    scheduler.get_num_unfinished_requests = lambda: 0
    scheduler.connector = scheduler.ec_connector = None
    scheduler.num_spec_tokens = 2
    scheduler.use_v2_model_runner = False
    manager = SimpleNamespace()
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[manager]))
    return SimpleNamespace(
        scheduler=scheduler,
        batch_queue=None,
        batch_queue_size=1,
        async_scheduling=False,
        _pause_complete=lambda: True,
        step_fn=object(),
        step_with_batch_queue=object(),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1
            ),
        ),
    )


def test_engine_queue_keeps_scheduler_and_cache_ownership_and_restores_drained():
    engine = engine_fixture()
    scheduler, cache, original_step = engine.scheduler, engine.scheduler.kv_cache_manager, engine.step_fn
    owner = QueuedEngineControl(engine, enabled=True, async_scheduler_class=AsyncScheduler)
    assert owner.validate(engine)["passed"]
    assert type(engine.scheduler) is BalanceScheduler  # preparation does not mutate
    assert owner.apply(engine)["batch_queue_size"] == 2
    assert engine.scheduler is scheduler and scheduler.kv_cache_manager is cache
    assert scheduler.schedule() == "balanced"
    output = []
    scheduler._update_after_schedule(output)
    assert output == ["scheduler", "placeholders"]
    restore = QueuedEngineControl(engine, enabled=False, async_scheduler_class=AsyncScheduler)
    restore.apply(engine)
    assert type(scheduler) is BalanceScheduler and engine.step_fn is original_step
    assert engine.batch_queue is None and engine.batch_queue_size == 1 and not engine.async_scheduling


@pytest.mark.parametrize("case", ["queued", "unfinished", "unpaused", "recycling", "connector", "dp"])
def test_engine_switch_rejects_unaudited_lifetimes(case):
    engine = engine_fixture()
    if case == "queued":
        engine.batch_queue = [object()]
    elif case == "unfinished":
        engine.scheduler.get_num_unfinished_requests = lambda: 1
    elif case == "unpaused":
        engine._pause_complete = lambda: False
    elif case == "recycling":
        engine.scheduler.kv_cache_manager.coordinator.single_type_managers[0].max_admission_blocks_per_request = 32
    elif case == "connector":
        engine.scheduler.connector = object()
    else:
        engine.vllm_config.parallel_config.data_parallel_size = 2
    control = QueuedEngineControl(engine, enabled=True, async_scheduler_class=AsyncScheduler)
    with pytest.raises((ValueError, RuntimeError)):
        control.apply(engine)
    assert type(engine.scheduler) is BalanceScheduler


def device_fixture():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a"], sampling_metadata=SimpleNamespace(output_token_ids=[])),
        num_prompt_logprobs={},
        use_dcp=False,
        uses_mrope=False,
        lora_config=None,
        enable_prompt_embeds=False,
        use_async_scheduling=False,
        max_model_len=512,
        requests={"a": SimpleNamespace(num_prompt_tokens=128, output_token_ids=[7, 8])},
    )
    transition = SimpleNamespace(enabled=True, failed=False, verify=False, next_pending=(("a",), {}))
    history = SimpleNamespace(enabled=True, failed=False, verify=False, pending=("a",), continuation=("a",))
    schedule = Schedule(Cached(["a"], [131], [4]), {"a": [-1, -1]}, {"a": 3})
    return runner, transition, history, schedule


def test_device_reservation_never_waits_for_host_draft_ids():
    runner, transition, history, schedule = device_fixture()
    runner._get_draft_token_ids_cpu = lambda: pytest.fail("device continuation cannot observe host draft IDs")
    bridge = QueuedScheduleBridge(runner, transition, history)
    bridge.enabled = bridge.device_drafts = True
    result = bridge.resolve(schedule)
    assert result.scheduled_spec_decode_tokens == {"a": [0, 0]}
    assert bridge.expected_device_inputs == ("a",)
    assert bridge.device_draft_steps == 1
    with pytest.raises(RuntimeError, match="not consumed"):
        bridge.resolve(schedule)


@pytest.mark.parametrize(
    "case", ["changed", "prefill", "finished", "resumed", "history", "verify", "penalty", "grammar", "shape"]
)
def test_device_drafts_exclude_every_unowned_layout_or_host_token_consumer(case):
    runner, transition, history, schedule = device_fixture()
    assert can_use_device_drafts(runner, schedule, transition, history)
    if case == "changed":
        runner.input_batch.req_ids = ["b"]
    elif case == "prefill":
        schedule.scheduled_new_reqs = [object()]
    elif case == "finished":
        schedule.finished_req_ids = {"old"}
    elif case == "resumed":
        schedule.scheduled_cached_reqs.resumed_req_ids = {"a"}
    elif case == "history":
        history.pending = None
    elif case == "verify":
        transition.verify = True
    elif case == "penalty":
        runner.input_batch.sampling_metadata.output_token_ids = [[7]]
    elif case == "grammar":
        schedule.has_structured_output_requests = True
    else:
        schedule.num_scheduled_tokens["a"] = 2
    assert not can_use_device_drafts(runner, schedule, transition, history)


def test_missing_device_consumer_cannot_silently_hash_reservations(monkeypatch):
    runner, transition, history, _ = device_fixture()
    bridge = QueuedScheduleBridge(runner, transition, history)
    bridge.expected_device_inputs = ("a",)
    functions = candidate.make_replacements(
        {RPC: lambda *args: None, INPUT_TARGET: lambda *args: None, STAGE_TARGET: lambda *args: "stage"},
        bridge,
        SimpleNamespace(execute_model=lambda *args: None),
    )
    history.continuation = None
    with pytest.raises(RuntimeError, match="missing device input packet"):
        functions[INPUT_TARGET](runner)
    history.continuation = ("a",)
    functions[INPUT_TARGET](runner)
    assert bridge.expected_device_inputs is None and bridge.expected_device_history == ("a",)
    monkeypatch.setattr(candidate, "uniform_history_rows", lambda *args: False)
    with pytest.raises(RuntimeError, match="fallback PLE hash"):
        functions[STAGE_TARGET](runner, 6)
    monkeypatch.setattr(candidate, "uniform_history_rows", lambda *args: True)
    assert functions[STAGE_TARGET](runner, 3) == "stage"
    assert bridge.expected_device_history is None
