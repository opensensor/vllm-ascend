# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual patch using CPU scheduler metadata and import stubs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def installed(monkeypatch):
    messages = []

    class Scheduler:
        def __init__(self):
            self.requests = {}
            self.log_stats = True
            self.observability_config = SimpleNamespace(enable_logging_iteration_details=True)

        def add_request(self, request):
            self.requests[request.request_id] = request
            return "added"

        def _update_after_schedule(self, output):
            if getattr(output, "fail", False):
                raise RuntimeError("original update failed")
            for key, count in output.num_scheduled_tokens.items():
                self.requests[key].num_computed_tokens += count
            return "updated"

    logger_module = ModuleType("vllm.logger")
    logger_module.logger = SimpleNamespace(info=lambda text, *args: messages.append(text % args))
    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    path = ROOT / "vllm_ascend/patch/platform/patch_prefill_progress.py"
    spec = importlib.util.spec_from_file_location("prefill_progress_offline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Scheduler(), messages


def request(name="req-1", total=6400, before=0, cached=0):
    return SimpleNamespace(
        request_id=name,
        num_prompt_tokens=total,
        num_computed_tokens=before,
        prefill_stats=SimpleNamespace(num_cached_tokens=cached),
    )


def test_cold_prompt_total_and_every_chunk_then_decode(installed):
    _, scheduler, messages = installed
    assert scheduler.add_request(request()) == "added"
    assert messages == ["Prefill received: request=req-1 prompt_tokens=6400 cache_lookup=pending"]
    for end in range(1280, 6401, 1280):
        assert scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 1280})) == "updated"
        assert f"prompt_tokens=6400 cached_tokens=0 prompt_chunk=1280 scheduled_through={end}" in messages[-1]
        assert messages[-1].endswith(f"remaining_to_schedule={6400 - end}")
    previous = len(messages)
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 2}))
    assert len(messages) == previous


@pytest.mark.parametrize(
    "cached,before,total,budget,chunk,remaining",
    [
        (5760, 5760, 6400, 1280, 640, 0),
        (6400, 6400, 6529, 129, 129, 0),
        (6400, 6400, 6529, 131, 129, 0),  # MTP verifier padding is not prompt work.
        (0, 75520, 100000, 1280, 1280, 23200),
        (640, 640, 6400, 1280, 1280, 4480),  # Preempted/re-admitted position.
    ],
)
def test_cache_prefix_tail_spec_padding_and_recomputed_cursor(
    installed, cached, before, total, budget, chunk, remaining
):
    _, scheduler, messages = installed
    scheduler.add_request(request(total=total, before=before, cached=cached))
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": budget}))
    assert f"cached_tokens={cached} prompt_chunk={chunk} scheduled_through={before + chunk}" in messages[-1]
    assert messages[-1].endswith(f"remaining_to_schedule={remaining}")


@pytest.mark.parametrize("log_stats,iteration_logs", [(False, True), (True, False), (False, False)])
def test_existing_flags_disable_all_logs_without_changing_scheduling(installed, log_stats, iteration_logs):
    _, scheduler, messages = installed
    scheduler.log_stats = log_stats
    scheduler.observability_config.enable_logging_iteration_details = iteration_logs
    scheduler.add_request(request())
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 1280}))
    assert scheduler.requests["req-1"].num_computed_tokens == 1280
    assert messages == []


def test_multiple_requests_and_idempotent_install(installed):
    module, scheduler, messages = installed
    assert module.install_prefill_progress(type(scheduler)) is False
    scheduler.add_request(request("a", total=2000))
    scheduler.add_request(request("b", total=3000, before=1000, cached=1000))
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"a": 640, "b": 640}))
    assert len(messages) == 4
    assert "request=a prompt_tokens=2000" in messages[2]
    assert "remaining_to_schedule=1360" in messages[2]
    assert "request=b prompt_tokens=3000 cached_tokens=1000" in messages[3]
    assert "remaining_to_schedule=1360" in messages[3]


@pytest.mark.parametrize("legacy_cached", [-1, None, 320])
def test_legacy_reuse_field_and_unknown_values(installed, legacy_cached):
    _, scheduler, messages = installed
    value = request(before=320)
    del value.prefill_stats
    value.num_cached_tokens = legacy_cached
    scheduler.add_request(value)
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 640}))
    expected = "unknown" if legacy_cached is None or legacy_cached < 0 else str(legacy_cached)
    assert f"cached_tokens={expected}" in messages[-1]


def test_failed_schedule_update_emits_no_progress(installed):
    _, scheduler, messages = installed
    scheduler.add_request(request())
    with pytest.raises(RuntimeError, match="original update failed"):
        scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 1280}, fail=True))
    assert len(messages) == 1


def test_zero_prompt_and_duplicate_admission(installed):
    _, scheduler, messages = installed
    scheduler.add_request(request(total=0))
    scheduler.add_request(request(total=0))
    scheduler._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"req-1": 0}))
    assert len(messages) == 1
