# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU unit tests for the Qwen4Exp 1M context-window budgeting (T7.2).

Self-contained: only imports the 310P host-side context-budget module (which
pulls the T7.1 ceiling constant). No server or NPU. Run with
``pytest --noconftest`` because the shared ut conftest fails to import.
"""

import threading

import pytest

from vllm_ascend._310p.worker.v2.rope import MAX_EXTENDED_POSITION_EMBEDDINGS
from vllm_ascend.models.qwen4_exp.context_budget import (
    QWEN4EXP_1M_CONTEXT_CEILING,
    QWEN4EXP_1M_MAX_CONCURRENCY,
    QWEN4EXP_1M_MAX_DRAFT_TOKENS,
    QWEN4EXP_1M_MAX_INPUT_TOKENS,
    QWEN4EXP_1M_MAX_OUTPUT_TOKENS,
    ConcurrencyLimitExceededError,
    ContextBudgetExceededError,
    ContextBudgetPolicy,
    Qwen4Exp1MAdmissionController,
    check_context_budget,
)


# ---------------------------------------------------------------------------
# Constants / budget split
# ---------------------------------------------------------------------------
def test_ceiling_matches_t71():
    assert QWEN4EXP_1M_CONTEXT_CEILING == 1_048_576
    assert QWEN4EXP_1M_CONTEXT_CEILING == MAX_EXTENDED_POSITION_EMBEDDINGS


def test_budget_split_sums_to_ceiling_exactly():
    assert QWEN4EXP_1M_MAX_INPUT_TOKENS == 1_048_060
    assert QWEN4EXP_1M_MAX_OUTPUT_TOKENS == 512
    assert QWEN4EXP_1M_MAX_DRAFT_TOKENS == 4
    assert (
        QWEN4EXP_1M_MAX_INPUT_TOKENS + QWEN4EXP_1M_MAX_OUTPUT_TOKENS + QWEN4EXP_1M_MAX_DRAFT_TOKENS
        == QWEN4EXP_1M_CONTEXT_CEILING
    )


def test_default_policy_valid():
    policy = ContextBudgetPolicy.default()
    assert policy.max_concurrency == QWEN4EXP_1M_MAX_CONCURRENCY == 1
    assert policy.context_ceiling == QWEN4EXP_1M_CONTEXT_CEILING


def test_policy_rejects_component_over_ceiling():
    with pytest.raises(ValueError):
        ContextBudgetPolicy(
            max_input_tokens=QWEN4EXP_1M_CONTEXT_CEILING + 1,
            max_output_tokens=512,
            max_draft_tokens=4,
            context_ceiling=QWEN4EXP_1M_CONTEXT_CEILING,
            max_concurrency=1,
        )


# ---------------------------------------------------------------------------
# Boundary accept / reject (no truncation)
# ---------------------------------------------------------------------------
def test_exactly_at_limit_accepts():
    decision = check_context_budget(
        QWEN4EXP_1M_MAX_INPUT_TOKENS,
        output_tokens=QWEN4EXP_1M_MAX_OUTPUT_TOKENS,
        draft_tokens=QWEN4EXP_1M_MAX_DRAFT_TOKENS,
    )
    assert decision.accepted is True
    assert decision.total_tokens == QWEN4EXP_1M_CONTEXT_CEILING
    assert decision.accepted_prompt_tokens == QWEN4EXP_1M_MAX_INPUT_TOKENS


def test_input_one_over_limit_rejects_no_truncation():
    with pytest.raises(ContextBudgetExceededError) as excinfo:
        check_context_budget(
            QWEN4EXP_1M_MAX_INPUT_TOKENS + 1,
            output_tokens=QWEN4EXP_1M_MAX_OUTPUT_TOKENS,
            draft_tokens=QWEN4EXP_1M_MAX_DRAFT_TOKENS,
        )
    # Explicit rejection, not a silent trim.
    assert "truncat" in str(excinfo.value).lower()


def test_total_over_ceiling_rejects():
    # Input at cap but output + draft push the total past the ceiling.
    policy = ContextBudgetPolicy(
        max_input_tokens=QWEN4EXP_1M_CONTEXT_CEILING,
        max_output_tokens=QWEN4EXP_1M_CONTEXT_CEILING,
        max_draft_tokens=QWEN4EXP_1M_CONTEXT_CEILING,
        context_ceiling=QWEN4EXP_1M_CONTEXT_CEILING,
        max_concurrency=1,
    )
    with pytest.raises(ContextBudgetExceededError) as excinfo:
        check_context_budget(
            QWEN4EXP_1M_CONTEXT_CEILING - 1,
            output_tokens=1,
            draft_tokens=1,
            policy=policy,
        )
    assert "ceiling" in str(excinfo.value).lower()


def test_output_over_budget_rejects():
    with pytest.raises(ContextBudgetExceededError):
        check_context_budget(100, output_tokens=QWEN4EXP_1M_MAX_OUTPUT_TOKENS + 1)


def test_draft_over_budget_rejects():
    with pytest.raises(ContextBudgetExceededError):
        check_context_budget(100, draft_tokens=QWEN4EXP_1M_MAX_DRAFT_TOKENS + 1)


def test_negative_input_rejects():
    with pytest.raises(ContextBudgetExceededError):
        check_context_budget(-1)


def test_small_prompt_accepts_and_defaults_reserve_full_output():
    decision = check_context_budget(1000)
    assert decision.accepted_prompt_tokens == 1000
    assert decision.output_tokens == QWEN4EXP_1M_MAX_OUTPUT_TOKENS
    assert decision.draft_tokens == QWEN4EXP_1M_MAX_DRAFT_TOKENS


# ---------------------------------------------------------------------------
# Accepted-token-count reporting (response metadata)
# ---------------------------------------------------------------------------
def test_metadata_reports_accepted_token_count():
    decision = check_context_budget(500_000, output_tokens=10, draft_tokens=2)
    meta = decision.to_metadata()
    assert meta["accepted_prompt_tokens"] == 500_000
    assert meta["total_tokens"] == 500_012
    assert meta["truncated"] is False
    assert meta["context_ceiling"] == QWEN4EXP_1M_CONTEXT_CEILING


# ---------------------------------------------------------------------------
# Concurrency = 1 (admission + execution)
# ---------------------------------------------------------------------------
def test_second_concurrent_request_rejected():
    controller = Qwen4Exp1MAdmissionController()
    decision = controller.admit("req-1", 1000)
    assert decision.accepted_prompt_tokens == 1000
    assert controller.in_flight == 1

    with pytest.raises(ConcurrencyLimitExceededError):
        controller.admit("req-2", 2000)
    assert controller.in_flight == 1


def test_slot_freed_allows_next_request():
    controller = Qwen4Exp1MAdmissionController()
    controller.admit("req-1", 1000)
    controller.release("req-1")
    assert controller.in_flight == 0
    # Now a second request can be admitted.
    controller.admit("req-2", 2000)
    assert controller.in_flight == 1


def test_over_budget_admission_consumes_no_slot():
    controller = Qwen4Exp1MAdmissionController()
    with pytest.raises(ContextBudgetExceededError):
        controller.admit("req-big", QWEN4EXP_1M_MAX_INPUT_TOKENS + 1)
    assert controller.in_flight == 0
    # A well-formed request still admits afterwards.
    controller.admit("req-ok", 10)
    assert controller.in_flight == 1


def test_double_admit_same_id_rejected():
    controller = Qwen4Exp1MAdmissionController()
    controller.admit("req-1", 1000)
    with pytest.raises(ConcurrencyLimitExceededError):
        controller.admit("req-1", 1000)


def test_admission_slot_context_manager_releases():
    controller = Qwen4Exp1MAdmissionController()
    with controller.admission_slot("req-1", 4242) as decision:
        assert decision.accepted_prompt_tokens == 4242
        assert controller.in_flight == 1
        # A concurrent request is rejected while the slot is held.
        with pytest.raises(ConcurrencyLimitExceededError):
            controller.admit("req-2", 10)
    assert controller.in_flight == 0


def test_release_is_idempotent():
    controller = Qwen4Exp1MAdmissionController()
    controller.release("never-admitted")  # no raise
    assert controller.in_flight == 0


def test_concurrency_enforced_under_threads():
    controller = Qwen4Exp1MAdmissionController()
    admitted: list[str] = []
    rejected: list[str] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            controller.admit(f"req-{idx}", 100)
            with lock:
                admitted.append(f"req-{idx}")
        except ConcurrencyLimitExceededError:
            with lock:
                rejected.append(f"req-{idx}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one request wins the single 1M slot.
    assert len(admitted) == 1
    assert len(rejected) == 7
    assert controller.in_flight == 1
