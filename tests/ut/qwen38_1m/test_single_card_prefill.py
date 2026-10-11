# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.qwen4exp.single_card_prefill import pair_summary, prompt_accounting


def result(cached, correct=True, tokens=8192):
    return {
        "usage": {"prompt_tokens": tokens, "prompt_tokens_details": {"cached_tokens": cached}},
        "ttft_s": 32,
        "correct": correct,
    }


def test_uncached_rate_excludes_reused_tokens():
    counts = prompt_accounting(result(7168))
    assert counts["uncached_tokens"] == 1024
    assert counts["uncached_prompt_tok_s_including_first_token"] == 32


def test_missing_cache_details_are_unknown_not_zero():
    record = result(None)
    record["usage"].pop("prompt_tokens_details")
    counts = prompt_accounting(record)
    assert counts["cached_tokens"] is None
    assert counts["uncached_prompt_tok_s_including_first_token"] is None
    summary = pair_summary(record, record)
    assert not summary["cold_verified"] and not summary["prefix_reuse_verified"]


def test_pair_checks_cache_recall_and_exact_lengths_independently():
    assert all(
        pair_summary(result(0), result(7168))[key]
        for key in ("cold_verified", "prefix_reuse_verified", "recall_passed", "same_prompt_length")
    )
    assert not pair_summary(result(10), result(7168))["cold_verified"]
    assert not pair_summary(result(0), result(0))["prefix_reuse_verified"]
    assert not pair_summary(result(0), result(7168, correct=False))["recall_passed"]
    assert not pair_summary(result(0), result(4096, tokens=4096))["same_prompt_length"]


@pytest.mark.parametrize("cached", [-1, 8193, True, "10"])
def test_invalid_cache_counts_are_rejected(cached):
    with pytest.raises(ValueError, match="cached token"):
        prompt_accounting(result(cached))
