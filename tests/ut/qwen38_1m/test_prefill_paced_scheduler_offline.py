# SPDX-License-Identifier: Apache-2.0
"""Exercise the real subclass with a synchronous scheduling boundary stub."""

import ast
import dataclasses
from copy import copy
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import field_validator

from vllm_ascend.config_utils import config
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
            if throttle_prefills and not self.prefill_capacity_bound and req.is_prefill_chunk:
                continue
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


def request(name, prefill, image=False, image_start=0, image_length=1024):
    return SimpleNamespace(
        request_id=name,
        is_prefill_chunk=prefill,
        has_encoder_inputs=image,
        num_tokens_with_spec=10000 if prefill else 103,
        num_computed_tokens=0 if prefill else 100,
        num_output_placeholders=0,
        mm_features=[SimpleNamespace(mm_position=SimpleNamespace(offset=image_start, length=image_length))]
        if image
        else [],
    )


@pytest.fixture
def paced():
    path = ROOT / "vllm_ascend/core/qwen_prefill_scheduler.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    clock = SimpleNamespace(monotonic=Mock(side_effect=count(1)))
    scope = {
        "PrefixMambaBoundedScheduler": BaseScheduler,
        "copy": copy,
        "time": clock,
        "PrefillPacingConfig": PrefillPacingConfig,
        "PrefillPacingPolicy": PrefillPacingPolicy,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    instance = object.__new__(scope[cls.name])
    instance.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0, disable_chunked_mm_input=False)
    instance.max_num_scheduled_tokens = 2560
    instance._prefill_pacing = PrefillPacingPolicy(PrefillPacingConfig())
    instance._prefill_pending_step = None
    instance._prefill_decode_steps_remaining = 0
    instance.prefill_capacity_bound = False
    instance.running = [request("prefill", True), request("decode", False)]
    instance.waiting = []
    instance.fail = False
    return instance


def test_decoder_first_aggregate_budget_and_fcfs_order_restored(paced):
    original = paced.scheduler_config
    output = paced.schedule()
    assert output.num_scheduled_tokens == {"decode": 3, "prefill": 256}
    assert paced.seen[1:] == (259, ["decode", "prefill"], False)
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
def test_image_requests_keep_pacing_when_decoder_image_chunking_is_allowed(paced, image_location):
    getattr(paced, image_location).append(request("image", True, image=True))
    original = paced.scheduler_config
    paced.schedule()
    assert paced.seen[0] is not original and paced.seen[1] == 259
    assert paced.seen[2][:2] == ["decode", "prefill"]


@pytest.mark.parametrize("image_location", ["running", "waiting"])
def test_atomic_image_expands_only_the_overlapping_span(paced, image_location):
    paced.scheduler_config.disable_chunked_mm_input = True
    paced.scheduler_config.long_prefill_token_threshold = 128
    getattr(paced, image_location).append(request("image", True, image=True, image_start=64, image_length=512))
    paced.schedule()
    assert paced.seen[1] == 3 + 64 + 512
    assert paced.seen[0].long_prefill_token_threshold == 64 + 512
    assert paced.scheduler_config.long_prefill_token_threshold == 128


@pytest.mark.parametrize("image_start,computed", [(2048, 0), (0, 1024)])
def test_future_or_consumed_image_cannot_disable_pacing(paced, image_start, computed):
    paced.scheduler_config.disable_chunked_mm_input = True
    image = request("image", True, image=True, image_start=image_start, image_length=512)
    image.num_computed_tokens = computed
    paced.running.append(image)
    paced.schedule()
    assert paced.seen[1] == 259


def test_widening_one_image_does_not_recursively_admit_later_images(paced):
    paced.scheduler_config.disable_chunked_mm_input = True
    image = request("image", True, image=True, image_length=512)
    image.mm_features.append(SimpleNamespace(mm_position=SimpleNamespace(offset=512, length=1024)))
    paced.running.append(image)
    paced.schedule()
    assert paced.seen[1] == 515


def test_decode_only_steps_preserve_graph_geometry_and_prefill_eventual_progress(paced):
    paced._prefill_pacing = PrefillPacingPolicy(PrefillPacingConfig(decode_only_steps=3))
    output = paced.schedule()
    assert output.num_scheduled_tokens == {"decode": 3, "prefill": 256}
    paced.update_from_output(output, None)
    for remaining in [2, 1, 0]:
        # Upstream's DP throughput override must not defeat latency pacing.
        paced.prefill_capacity_bound = True
        output = paced.schedule()
        assert output.num_scheduled_tokens == {"decode": 3}
        assert paced.seen[1:] == (3, ["decode", "prefill"], True)
        assert paced.prefill_capacity_bound is True
        assert output.ascend_prefix_mamba_block_ids == {1: [1, 4]}
        paced.update_from_output(output, None)
        assert paced._prefill_decode_steps_remaining == remaining
    output = paced.schedule()
    assert output.num_scheduled_tokens == {"decode": 3, "prefill": 128}


def test_encoder_admission_does_not_bypass_decode_only_cadence(paced):
    paced._prefill_decode_steps_remaining = 2
    paced.scheduler_config.disable_chunked_mm_input = True
    paced.running.append(request("image", True, image=True))
    assert paced.schedule().num_scheduled_tokens == {"decode": 3}


def test_multiple_decoders_reserve_all_speculative_tokens(paced):
    paced.running.append(request("decode2", False))
    paced._prefill_decode_steps_remaining = 1
    assert paced.schedule().num_scheduled_tokens == {"decode": 3, "decode2": 3}


def test_no_decoders_restore_full_prefill_budget_and_clear_cadence(paced):
    paced.running = paced.running[:1]
    paced._prefill_decode_steps_remaining = 4
    assert paced.schedule().num_scheduled_tokens == {"prefill": 2560}
    assert paced._prefill_decode_steps_remaining == 0


def test_failed_preparation_restores_running_order_and_configuration(paced):
    paced.scheduler_config.disable_chunked_mm_input = True
    image = request("image", True, image=True)
    image.mm_features = [SimpleNamespace()]
    paced.waiting.append(image)
    original = paced.scheduler_config
    with pytest.raises(AttributeError):
        paced.schedule()
    assert paced.scheduler_config is original and paced.max_num_scheduled_tokens == 2560
    assert [r.request_id for r in paced.running] == ["prefill", "decode"]


@pytest.mark.parametrize("value", [-1, True, 65, 1.5])
def test_invalid_decode_only_cadence_is_rejected(value):
    with pytest.raises(ValueError, match="decode_only_steps"):
        PrefillPacingConfig(decode_only_steps=value)


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


def test_failed_decode_only_step_restores_capacity_and_does_not_consume_cadence(paced):
    paced.prefill_capacity_bound = True
    paced._prefill_decode_steps_remaining = 2
    paced.fail = True
    original = paced.scheduler_config
    with pytest.raises(RuntimeError, match="parent failed"):
        paced.schedule()
    assert paced.prefill_capacity_bound is True
    assert paced.scheduler_config is original and paced.max_num_scheduled_tokens == 2560
    assert paced._prefill_decode_steps_remaining == 2
    assert paced._prefill_pending_step is None
    assert [r.request_id for r in paced.running] == ["decode", "new"]


def test_only_the_matching_completed_step_can_consume_decode_cadence(paced):
    paced._prefill_decode_steps_remaining = 2
    output = paced.schedule()
    paced.update_from_output(SimpleNamespace(num_scheduled_tokens={"decode": 3}), None)
    paced.update_from_output(output, None)
    assert paced._prefill_decode_steps_remaining == 2
    output = paced.schedule()
    paced.update_from_output(output, None)
    paced.update_from_output(output, None)
    assert paced._prefill_decode_steps_remaining == 1


def test_empty_step_does_not_consume_decode_cadence(paced):
    paced._prefill_decode_steps_remaining = 2
    paced.running[-1].num_tokens_with_spec = paced.running[-1].num_computed_tokens
    output = paced.schedule()
    assert not output.num_scheduled_tokens
    paced.update_from_output(output, None)
    assert paced._prefill_decode_steps_remaining == 2


@pytest.fixture
def pacing_schema():
    """Use actual registration nodes with the real strict config decorator.

    Other Ascend fields import hardware, unrelated model types, and upstream
    vLLM. Startup below is also gated against the full schema on the serving
    host; this focused CPU check catches omitted registration and validation.
    """
    path = ROOT / "vllm_ascend/ascend_config.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendConfig"
    )
    cls.body = [
        node
        for node in cls.body
        if (isinstance(node, ast.AnnAssign) and node.target.id == "qwen_prefill_pacing")
        or (isinstance(node, ast.FunctionDef) and node.name == "_validate_qwen_prefill_pacing")
    ]
    assert len(cls.body) == 2, "Pacing requires a declared field and validator in AscendConfig"
    scope = {
        "config": config,
        "field_validator": field_validator,
        "dataclasses": dataclasses,
        "Any": Any,
        "PrefillPacingConfig": PrefillPacingConfig,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["AscendConfig"]


def test_pacing_options_are_registered_in_the_strict_ascend_schema(pacing_schema):
    value = {"target_step_ms": 200, "initial_tokens": 128, "max_tokens": 640, "decode_only_steps": 8}
    assert pacing_schema(qwen_prefill_pacing=value).qwen_prefill_pacing == value
    assert pacing_schema().qwen_prefill_pacing == {}
    with pytest.raises(ValueError):
        pacing_schema(qwen_prefill_pacing_typo={})


@pytest.mark.parametrize(
    "value",
    [None, [], {"decode_only_steps": True}, {"decode_only_steps": 65}, {"unknown_option": 1}, {"initial_tokens": 129}],
)
def test_strict_ascend_pacing_schema_rejects_invalid_options(pacing_schema, value):
    with pytest.raises(ValueError):
        pacing_schema(qwen_prefill_pacing=value)
