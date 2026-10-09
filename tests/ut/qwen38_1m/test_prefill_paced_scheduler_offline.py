# SPDX-License-Identifier: Apache-2.0
"""Exercise the real subclass with a synchronous scheduling boundary stub."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.core.qwen_prefill_pacing import PrefillPacingConfig, PrefillPacingPolicy

ROOT = Path(__file__).resolve().parents[3]


class BaseScheduler:
    def schedule(self, throttle_prefills=False):
        self.seen = (
            self.scheduler_config,
            self.max_num_scheduled_tokens,
            [request.request_id for request in self.running],
            throttle_prefills,
        )
        if self.fail:
            self.running = [r for r in self.running if r.request_id != "prefill"] + [request("new", True)]
            raise RuntimeError("parent failed")
        budget = self.max_num_scheduled_tokens
        scheduled = {}
        for req in self.running:
            needed = req.num_tokens_with_spec - req.num_computed_tokens
            threshold = self.scheduler_config.long_prefill_token_threshold
            needed = min(needed, threshold) if threshold > 0 else needed
            tokens = min(budget, needed)
            if tokens:
                scheduled[req.request_id] = tokens
                budget -= tokens
        return SimpleNamespace(num_scheduled_tokens=scheduled, ascend_prefix_mamba_block_ids={1: [1, 4]})

    def update_from_output(self, scheduler_output, model_runner_output):
        return "parent_update"


def request(name, prefill, image=False):
    return SimpleNamespace(
        request_id=name,
        is_prefill_chunk=prefill,
        has_encoder_inputs=image,
        num_tokens_with_spec=10000 if prefill else 103,
        num_computed_tokens=0 if prefill else 100,
        num_output_placeholders=0,
    )


@pytest.fixture
def paced():
    path = ROOT / "vllm_ascend/core/qwen_prefill_scheduler.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    clock = SimpleNamespace(monotonic=Mock(side_effect=[1, 3, 4, 5]))
    scope = {
        "PrefixMambaBoundedScheduler": BaseScheduler,
        "copy": copy,
        "time": clock,
        "PrefillPacingConfig": PrefillPacingConfig,
        "PrefillPacingPolicy": PrefillPacingPolicy,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    instance = object.__new__(scope[cls.name])
    instance.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    instance.max_num_scheduled_tokens = 2560
    instance._prefill_pacing = PrefillPacingPolicy(PrefillPacingConfig())
    instance._prefill_pending_step = None
    instance.running = [request("prefill", True), request("decode", False)]
    instance.waiting = []
    instance.fail = False
    return instance


def test_decoder_first_aggregate_budget_and_fcfs_order_restored(paced):
    original = paced.scheduler_config
    output = paced.schedule(throttle_prefills=True)
    assert output.num_scheduled_tokens == {"decode": 3, "prefill": 256}
    assert paced.seen[1:] == (259, ["decode", "prefill"], True)
    assert paced.seen[0] is not original
    assert paced.scheduler_config is original and original.long_prefill_token_threshold == 0
    assert paced.max_num_scheduled_tokens == 2560
    assert [r.request_id for r in paced.running] == ["prefill", "decode"]
    assert output.ascend_prefix_mamba_block_ids == {1: [1, 4]}
    assert paced.update_from_output(output, None) == "parent_update"
    assert paced._prefill_pacing.budget(2560) == 128
    assert paced.schedule().num_scheduled_tokens == {"decode": 3, "prefill": 128}


def test_multiple_prefills_cannot_exceed_aggregate_budget(paced):
    paced.running.append(request("prefill2", True))
    result = paced.schedule()
    assert sum(result.num_scheduled_tokens.values()) <= 259
    assert result.num_scheduled_tokens["decode"] == 3


def test_existing_smaller_threshold_is_retained(paced):
    paced.scheduler_config.long_prefill_token_threshold = 128
    output = paced.schedule()
    assert output.num_scheduled_tokens["prefill"] == 128
    assert paced.scheduler_config.long_prefill_token_threshold == 128


@pytest.mark.parametrize("image_location", ["running", "waiting"])
def test_uncached_image_span_keeps_original_encoder_admission_limits(paced, image_location):
    getattr(paced, image_location).append(request("image", True, image=True))
    original = paced.scheduler_config
    paced.schedule()
    assert paced.seen[0] is original and paced.seen[1] == 2560
    assert paced.seen[2][:2] == ["prefill", "decode"]


def test_prefill_only_retains_high_throughput_budget(paced):
    paced.running = paced.running[:1]
    assert paced.schedule().num_scheduled_tokens == {"prefill": 2560}


def test_exception_restores_config_and_surviving_order_without_resurrecting_requests(paced):
    paced.fail = True
    original = paced.scheduler_config
    with pytest.raises(RuntimeError, match="parent failed"):
        paced.schedule()
    assert paced.scheduler_config is original and paced.max_num_scheduled_tokens == 2560
    assert [r.request_id for r in paced.running] == ["decode", "new"]


def test_mismatched_and_repeated_output_cannot_reuse_timing_snapshot(paced):
    output = paced.schedule()
    paced.update_from_output(SimpleNamespace(), None)
    paced.update_from_output(output, None)
    assert paced._prefill_pacing.ms_per_token is None
