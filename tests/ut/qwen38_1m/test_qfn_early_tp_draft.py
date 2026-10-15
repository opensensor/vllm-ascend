# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the exact pinned runner function with CPU-only dependency fixtures."""

import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.qwen4exp.resident_candidates import qfn_early_tp_draft as candidate

SOURCE = Path(__file__).with_name("fixtures").joinpath("qfn_pinned_sample_tokens.txt").read_text()


def setup_runner(*, pipeline=1):
    events = []
    sampled_ids = object()
    hidden = {"intact": True}
    scheduler = SimpleNamespace(num_spec_tokens_to_schedule=2, total_num_scheduled_tokens=3)
    config = SimpleNamespace(
        method="mtp",
        disable_padded_drafter_batch=False,
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram_gpu=lambda: False,
    )
    model_type = type(candidate.QUALIFIED_MODEL_CLASS, (), {"__module__": candidate.QUALIFIED_MODEL_MODULE})
    model = model_type()
    model.model = SimpleNamespace(_mtp_hidden_buffer=object())
    batch = SimpleNamespace(uniform=True)
    spec_metadata = object()
    input_batch = SimpleNamespace(sampling_metadata=object(), req_ids=["request"], num_tokens_no_spec=[10])
    runner = SimpleNamespace(
        speculative_config=config,
        num_spec_tokens=2,
        use_async_scheduling=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=pipeline, data_parallel_size=1),
        use_dcp=False,
        num_discarded_requests=0,
        num_prompt_logprobs={},
        drafter=SimpleNamespace(_enable_probabilistic_draft_probs=False),
        get_model=lambda: model,
        kv_connector_output=None,
        need_accepted_tokens=False,
        dynamic_eplb=False,
        supports_mm_inputs=False,
        input_batch=input_batch,
        _input_fits_in_drafter=lambda metadata: True,
        _sample=lambda logits, metadata: SimpleNamespace(sampled_token_ids=sampled_ids),
        _copy_draft_token_ids_to_cpu=lambda output: events.append("copy_draft"),
        finalize_kv_connector=lambda: events.append("finalize"),
        _finalize_dump_data=lambda: None,
    )
    runner.execute_model_state = (
        scheduler,
        object(),
        spec_metadata,
        object(),
        hidden,
        object(),
        None,
        {},
        None,
        None,
        None,
        batch,
    )

    def propose(*args):
        assert args[0] is sampled_ids
        events.append("propose")
        hidden["intact"] = False
        return [[7, 8]]

    def bookkeeping(*args):
        if runner.num_prompt_logprobs:
            assert hidden["intact"], "draft replay invalidated target prompt-logprob input"
        events.append("bookkeeping")
        input_batch.num_tokens_no_spec[0] += 1
        return ({}, None, None, [[6]], {}, ["request"], {"request": 0}, [])

    runner.propose_draft_token_ids = propose
    runner._bookkeeping_sync = bookkeeping
    namespace = dict(
        torch=SimpleNamespace(inference_mode=lambda: lambda fn: fn, Tensor=type("Tensor", (), {})),
        ModelRunnerOutput=type("Output", (SimpleNamespace,), {}),
        AsyncModelRunnerOutput=object,
        IntermediateTensors=object,
        get_pp_group=lambda: SimpleNamespace(world_size=pipeline),
        record_function_or_nullcontext=lambda name: nullcontext(),
    )
    # Postponed annotations avoid importing the real model runner in CPU tests.
    exec(compile("from __future__ import annotations\n" + SOURCE, "<pinned-fixture>", "exec"), namespace)
    original = namespace["sample_tokens"]
    patched = candidate.build_forward(original, source=SOURCE)
    return SimpleNamespace(**locals())


def test_exact_source_fingerprint_and_single_early_proposer():
    assert hashlib.sha256(SOURCE.encode()).hexdigest() == candidate.QUALIFIED_SOURCE_SHA256
    gate = setup_runner()
    result = gate.patched(gate.runner, None)
    assert gate.events == ["propose", "copy_draft", "bookkeeping", "finalize"]
    assert gate.runner.execute_model_state is None
    assert result.sampled_token_ids == [[6]]
    assert gate.input_batch.num_tokens_no_spec == [11]
    assert result.spec_token_ids is None
    assert gate.patched.__globals__ is gate.original.__globals__


def test_original_qualified_function_has_late_tp_proposer():
    gate = setup_runner()
    gate.original(gate.runner, None)
    assert gate.events == ["bookkeeping", "propose", "copy_draft", "finalize"]


def test_qualified_conditional_wrapper_uses_audited_language_model():
    gate = setup_runner()
    wrapper_type = type(candidate.QUALIFIED_CONDITIONAL_CLASS, (), {"__module__": candidate.QUALIFIED_MODEL_MODULE})
    wrapper = wrapper_type()
    wrapper.language_model = gate.model
    gate.runner.get_model = lambda: wrapper
    gate.patched(gate.runner, None)
    assert gate.events == ["propose", "copy_draft", "bookkeeping", "finalize"]


def test_existing_pp_early_behavior_is_unchanged():
    gate = setup_runner(pipeline=2)
    result = gate.patched(gate.runner, None)
    assert gate.events == ["propose", "copy_draft", "bookkeeping", "finalize"]
    assert result.spec_token_ids == [[7, 8]]


def test_prompt_logprobs_consume_target_hidden_before_draft_overwrites_it():
    gate = setup_runner()
    gate.runner.num_prompt_logprobs = {"request": 2}
    gate.patched(gate.runner, None)
    assert gate.events == ["bookkeeping", "propose", "copy_draft", "finalize"]


@pytest.mark.parametrize(
    "case", ["discarded", "probabilistic", "dcp", "dp", "tp", "prefill", "wrong_model", "no_hidden", "different_k"]
)
def test_unaudited_cases_preserve_original_order(case):
    gate = setup_runner()
    if case == "discarded":
        gate.runner.num_discarded_requests = 1
    elif case == "probabilistic":
        gate.runner.drafter._enable_probabilistic_draft_probs = True
    elif case == "dcp":
        gate.runner.use_dcp = True
    elif case == "dp":
        gate.runner.parallel_config.data_parallel_size = 2
    elif case == "tp":
        gate.runner.parallel_config.tensor_parallel_size = 2
    elif case == "prefill":
        gate.batch.uniform = False
    elif case == "wrong_model":
        gate.runner.get_model = lambda: SimpleNamespace()
    elif case == "no_hidden":
        gate.model.model._mtp_hidden_buffer = None
    else:
        gate.scheduler.num_spec_tokens_to_schedule = 1
    gate.patched(gate.runner, None)
    assert gate.events == ["bookkeeping", "propose", "copy_draft", "finalize"]


def test_missing_spec_metadata_and_async_are_not_admitted():
    gate = setup_runner()
    assert not candidate.eligible(gate.runner, None, gate.batch, gate.scheduler)
    gate.runner.use_async_scheduling = True
    assert not candidate.eligible(gate.runner, gate.spec_metadata, gate.batch, gate.scheduler)


def test_source_drift_fails_closed():
    gate = setup_runner()
    with pytest.raises(RuntimeError, match="exact qualified"):
        candidate.build_forward(gate.original, source=SOURCE + "\n")


def test_unrelated_live_globals_remain_visible():
    gate = setup_runner()
    gate.original.__globals__["ModelRunnerOutput"] = lambda **kwargs: {"live_factory": True, **kwargs}
    assert gate.patched(gate.runner, None)["live_factory"]
