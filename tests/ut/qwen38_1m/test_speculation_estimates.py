# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression gates for fixed-width draft accounting and unknown widths."""

import pytest

from tools.qwen38_decode_study import speculation, summarize


def row(**changes):
    return {
        "drafted_delta": 384,
        "accepted_delta": 320,
        "client_decode_tok_s": 30.910185017223437,
        **changes,
    }


def test_mtp2_uses_two_drafts_in_round_estimate():
    estimate = speculation.round_estimate(row(), 2)
    assert estimate["estimated_rounds"] == 192
    assert estimate["estimated_emitted_tokens_per_round"] == pytest.approx(8 / 3)
    assert estimate["mean_accepted_drafts_per_round"] == pytest.approx(5 / 3)
    assert estimate["estimated_round_ms"] == pytest.approx(86.2714560000)
    budget = speculation.added_draft_break_even(estimate, 0.6)
    assert budget["maximum_round_time_increase_fraction"] == pytest.approx(0.225)
    assert budget["maximum_round_time_increase_ms"] == pytest.approx(19.4110776000)


def test_mtp_width_and_server_timing_take_precedence():
    estimate = speculation.round_estimate(row(drafted_delta=100, accepted_delta=80, server_decode_tok_s=20), 1)
    assert estimate["estimated_emitted_tokens_per_round"] == 1.8
    assert estimate["estimated_round_ms"] == 90


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_invalid_width(width):
    with pytest.raises(ValueError, match="positive integer"):
        speculation.round_estimate(row(), width)


@pytest.mark.parametrize(
    "changes",
    [
        {"drafted_delta": 0},
        {"accepted_delta": -1},
        {"accepted_delta": 385},
        {"accepted_delta": None},
        {"client_decode_tok_s": 0},
        {"client_decode_tok_s": float("nan")},
        {"drafted_delta": float("inf")},
    ],
)
def test_invalid_evidence(changes):
    with pytest.raises(ValueError):
        speculation.round_estimate(row(**changes), 2)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_invalid_marginal_acceptance(probability):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        speculation.added_draft_break_even(speculation.round_estimate(row(), 2), probability)


def test_summary_requires_width_and_uses_counters_for_known_width():
    record = row(
        request={"messages": ["same"]},
        server_decode_tok_s=20,
        text_sha256="same",
        acceptance=320 / 384,
        usage={"prompt_tokens": 32},
    )
    control = {("short", 512, 0): record}
    result = summarize.compare(control, control)
    assert result["pairs"][0]["baseline_estimated_step_ms"] is None
    record["num_speculative_tokens"] = 2
    result = summarize.compare(control, control)
    assert result["pairs"][0]["baseline_estimated_step_ms"] == pytest.approx(1000 * (8 / 3) / 20)


def test_new_position_cannot_exceed_unchanged_prefix_acceptance():
    estimate = speculation.round_estimate(row(), 2)
    assert estimate["maximum_new_position_acceptance"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="unchanged-prefix"):
        speculation.added_draft_break_even(estimate, 0.9)
