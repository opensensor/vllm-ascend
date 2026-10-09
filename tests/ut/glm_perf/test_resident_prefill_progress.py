# SPDX-License-Identifier: Apache-2.0
"""CPU scheduler metadata, never device values, drives resident prompt logs."""

from types import SimpleNamespace

import pytest

from tools.glm_perf.prefill_progress import PrefillProgress


def step(new=(), cached=(), finished=(), resumed=(), scheduled=None):
    ids = [key for key, _ in cached]
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        finished_req_ids=set(finished),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=ids, num_computed_tokens=[count for _, count in cached], resumed_req_ids=set(resumed)
        ),
        num_scheduled_tokens={} if scheduled is None else scheduled,
    )


def new_request(name="r", length=6400, before=0):
    return SimpleNamespace(req_id=name, prompt_token_ids=[0] * length, prompt_embeds=None, num_computed_tokens=before)


def test_full_prompt_progress_and_cleanup():
    progress = PrefillProgress()
    assert progress.scheduled(step(new=[new_request()], scheduled={"r": 1280})) == [("r", 6400, 0, 1280, 1280, 5120)]
    for before in range(1280, 6400, 1280):
        assert progress.scheduled(step(cached=[("r", before)], scheduled={"r": 1280})) == [
            ("r", 6400, 0, 1280, before + 1280, 5120 - before)
        ]
    assert progress.scheduled(step(cached=[("r", 6400)], scheduled={"r": 2})) == []
    assert progress.scheduled(step(finished=["r"])) == []
    assert not progress.requests


@pytest.mark.parametrize("before,expected", [(5760, 640), (6399, 1), (6400, 0)])
def test_cache_reuse_and_final_verifier_padding(before, expected):
    progress = PrefillProgress()
    rows = progress.scheduled(step(new=[new_request(before=before)], scheduled={"r": 1280}))
    assert rows == ([("r", 6400, before, expected, 6400, 0)] if expected else [])


def test_multiple_requests_async_snapshots_preemption_and_same_id_reuse():
    progress = PrefillProgress()
    progress.scheduled(step(new=[new_request("a"), new_request("b", before=320)], scheduled={"a": 640, "b": 640}))
    # Each output carries its own before position; later queued scheduling does
    # not make this an assertion that the earlier chunk completed on device.
    assert progress.scheduled(step(cached=[("a", 640), ("b", 960)], scheduled={"a": 640, "b": 640})) == [
        ("a", 6400, 0, 640, 1280, 5120),
        ("b", 6400, 320, 640, 1600, 4800),
    ]
    assert progress.scheduled(step(cached=[("a", 320)], resumed=["a"], scheduled={"a": 640})) == [
        ("a", 6400, "unknown_after_preemption", 640, 960, 5440)
    ]
    assert progress.scheduled(step(finished=["a"], new=[new_request("a", length=100)], scheduled={"a": 128})) == [
        ("a", 100, 0, 100, 100, 0)
    ]


def test_prompt_embeddings_use_shape_without_reading_values():
    progress = PrefillProgress()
    request = SimpleNamespace(
        req_id="embed", prompt_token_ids=None, prompt_embeds=SimpleNamespace(shape=(2000, 4096)), num_computed_tokens=0
    )
    assert progress.scheduled(step(new=[request], scheduled={"embed": 1280})) == [("embed", 2000, 0, 1280, 1280, 720)]


def test_unknown_existing_request_is_not_guessed_after_hot_reload():
    progress = PrefillProgress()
    assert progress.scheduled(step(cached=[("already_running", 3200)], scheduled={"already_running": 1280})) == []


def test_resident_wrapper_logs_through_serving_logger_and_unwraps(monkeypatch):
    import sys
    from types import ModuleType

    from tools.glm_perf.resident_candidates.prefill_progress import TARGET, extend_replacements

    messages, calls = [], []
    logger_module = ModuleType("vllm.logger")
    logger_module.logger = SimpleNamespace(info=lambda text, *args: messages.append(text % args))
    worker_module = ModuleType("vllm_ascend._310p.worker_310p")

    class Worker:
        rank = 0
        vllm_config = SimpleNamespace(observability_config=SimpleNamespace(enable_logging_iteration_details=True))

        def execute_model(self, output):
            calls.append(output)
            return "executed"

    worker_module.NPUWorker310 = Worker
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    original = Worker.execute_model
    first = extend_replacements({})[TARGET]
    # Replacing the candidate keeps one logger wrapper, not two.
    Worker.execute_model = first
    second = extend_replacements({})[TARGET]
    assert second.__prefill_progress_original__ is original
    worker = Worker()
    output = step(new=[new_request()], scheduled={"r": 1280})
    assert second(worker, output) == "executed"
    assert len(calls) == len(messages) == 1
    assert "prompt_tokens=6400" in messages[0]
    assert "remaining_to_schedule=5120" in messages[0]
    worker.rank = 1
    second(worker, output)
    worker.rank = 0
    worker.vllm_config = SimpleNamespace(observability_config=SimpleNamespace(enable_logging_iteration_details=False))
    second(worker, output)
    assert len(calls) == 3 and len(messages) == 1
