# SPDX-License-Identifier: Apache-2.0
"""Bounded graph dispatch never suppresses explicit eager or mixed-batch guards."""

import __future__

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.glm_perf.resident_candidates.prefill_graphs import (
    PrefillGraphPolicy,
    eligible_prefill,
    original_prefill_function,
    release_native_scratch,
)


def test_idle_native_scratch_is_released_without_changing_weights_or_resources():
    weights = object()
    fused = SimpleNamespace(scratch={"640": object(), "8": object()}, weights=weights)
    resources = {"old": {"fused_int4a4": fused}, "other": object()}
    assert release_native_scratch(resources, [SimpleNamespace(entries={})]) == 2
    assert fused.scratch == {}
    assert fused.weights is weights
    assert resources["old"]["fused_int4a4"] is fused
    assert release_native_scratch(resources, [SimpleNamespace(entries={})]) == 0


def test_native_scratch_is_kept_while_any_target_or_draft_graph_exists():
    scratch = {"640": object()}
    resources = {"active": {"fused_int4a4": SimpleNamespace(scratch=scratch)}}
    with pytest.raises(RuntimeError, match="release old graphs"):
        release_native_scratch(resources, [SimpleNamespace(entries={}), SimpleNamespace(entries={8: object()})])
    assert len(scratch) == 1


def test_legacy_draft_wrappers_do_not_stack_across_resident_candidates():
    def original_propose():
        return "original"

    def wrap(original_propose):
        def draft_propose():
            return original_propose()

        return draft_propose

    wrapped = wrap(wrap(original_propose))
    assert original_prefill_function(wrapped, "original_propose") is original_propose


def test_policy_executes_with_postponed_annotations_in_unregistered_resident_module():
    from tools.glm_perf.resident_candidates import prefill_graphs

    source = Path(prefill_graphs.__file__).read_text()
    module = ModuleType("test_unregistered_resident_prefill_candidate")
    exec(compile(source, "<prefill_candidate>", "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    policy = module.PrefillGraphPolicy()
    assert policy.saved == {} and policy.prefill_replays == 0


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"num_tokens": 639}, False),
        ({"num_reqs": 2}, False),
        ({"max_num_scheduled_tokens": 2}, False),
        ({"force_eager": True}, False),
        ({"force_uniform_decode": True}, False),
        ({"force_has_lora": True}, False),
        ({"force_num_active_loras": 1}, False),
        ({"num_encoder_reqs": 1}, False),
    ],
)
def test_prefill_graph_eligibility(changes, expected):
    arguments = {"num_tokens": 640, "num_reqs": 1, "max_num_scheduled_tokens": 640, **changes}
    assert eligible_prefill(arguments) is expected


def test_dummy_capture_allows_bounded_multi_request_descriptor_without_enabling_mixed_runtime():
    arguments = {"num_tokens": 640, "num_reqs": 4, "max_num_scheduled_tokens": 640}
    assert eligible_prefill(arguments, capturing=True)
    assert not eligible_prefill(arguments, capturing=False)


def test_failed_or_removed_prefill_capture_restores_dispatcher_and_padding():
    class Mode(Enum):
        FULL = 1
        PIECEWISE = 2
        FULL_AND_PIECEWISE = 3

    @dataclass(frozen=True)
    class Key:
        num_tokens: int

    config = SimpleNamespace(cudagraph_capture_sizes=[2, 8], max_cudagraph_capture_size=8)

    class Dispatcher:
        cudagraph_mode = Mode.FULL
        cudagraph_keys = {Mode.FULL: {Key(2), Key(8)}, Mode.PIECEWISE: set()}
        _bs_to_padded_graph_size = list(range(9))
        keys_initialized = True

        def _compute_bs_to_padded_graph_size(self):
            self._bs_to_padded_graph_size = list(range(config.max_cudagraph_capture_size + 1))

        def initialize_cudagraph_keys(self, mode, query_len):
            assert query_len == 2
            self.cudagraph_mode = mode
            self.cudagraph_keys = {
                Mode.FULL: {Key(2), Key(8)},
                Mode.PIECEWISE: {Key(value) for value in config.cudagraph_capture_sizes},
            }

    runner = SimpleNamespace(compilation_config=config, cudagraph_dispatcher=Dispatcher(), uniform_decode_query_len=2)
    policy = PrefillGraphPolicy()
    policy.configure(runner, Mode)
    policy.configure(runner, Mode)
    assert config.cudagraph_capture_sizes == [2, 8, 640]
    assert runner.cudagraph_dispatcher.cudagraph_keys[Mode.PIECEWISE] == {Key(640)}
    draft = policy.draft_dispatcher
    assert draft is not runner.cudagraph_dispatcher
    assert draft.compilation_config.max_cudagraph_capture_size == 8
    assert draft.compilation_config.cudagraph_capture_sizes == [2, 8]
    assert draft.cudagraph_mode == Mode.FULL
    assert draft.cudagraph_keys == {Mode.FULL: {Key(2), Key(8)}, Mode.PIECEWISE: set()}
    assert draft._bs_to_padded_graph_size == list(range(9))
    policy.restore(runner)
    assert config.cudagraph_capture_sizes == [2, 8] and config.max_cudagraph_capture_size == 8
    assert runner.cudagraph_dispatcher.cudagraph_mode == Mode.FULL
    assert runner.cudagraph_dispatcher.cudagraph_keys == {Mode.FULL: {Key(2), Key(8)}, Mode.PIECEWISE: set()}
    assert runner.cudagraph_dispatcher._bs_to_padded_graph_size == list(range(9))
    prefill = policy.prefill_dispatcher
    assert prefill is not runner.cudagraph_dispatcher
    assert prefill.compilation_config is not config
    assert prefill.compilation_config.cudagraph_capture_sizes == [2, 8, 640]
    assert prefill.compilation_config.max_cudagraph_capture_size == 640
    assert prefill.cudagraph_keys[Mode.PIECEWISE] == {Key(640)}
    assert len(prefill._bs_to_padded_graph_size) == 641
    assert not policy.saved
