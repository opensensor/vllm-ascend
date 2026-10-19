# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.dflash_research.qfn_dspark_reference import Reference, prefix_length, summarize


def test_markov_chain_uses_anchor_then_previous_proposal():
    reference = Reference.__new__(Reference)
    reference.weights = {
        "markov_head.markov_w1.weight": torch.eye(3).half(),
        "markov_head.markov_w2.weight": torch.tensor([[0, 0, 9], [9, 0, 0], [0, 9, 0]]).half(),
    }
    result = reference.propose(torch.zeros(3, 2).half(), 0, torch.zeros(3, 2).half())
    assert result["tokens"] == [1, 2, 0]
    assert result["margins"] == [9, 9, 9]


def test_rope_matches_complex_rotation():
    reference = Reference.__new__(Reference)
    reference.config = {"rope_theta": 10000}
    reference.head_dim = 4
    reference.device = "cpu"
    value = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]).half()
    positions = torch.tensor([0, 17])
    complex_value = torch.complex(value[..., :2].float(), value[..., 2:].float())
    angles = positions[:, None, None] * torch.tensor([1, 0.01])[None, None, :]
    rotated = complex_value * torch.polar(torch.ones_like(angles), angles)
    expected = torch.cat((rotated.real, rotated.imag), dim=-1).half()
    torch.testing.assert_close(reference.rope(value, positions), expected, rtol=0, atol=0)


def fixture_trace():
    # First round accepts token 11, rejects 99, then samples token 12.
    return [
        {
            "query_inputs": {"positions": torch.tensor([1, 2])},
            "target_positions": torch.tensor([0]),
            "target_token_ids": torch.tensor([9]),
            "next_token_ids": torch.tensor([10]),
        },
        {
            "query_inputs": {"positions": torch.tensor([3, 4])},
            "target_positions": torch.tensor([1, 2, 3]),
            "target_token_ids": torch.tensor([10, 11, 99]),
            "next_token_ids": torch.tensor([12]),
        },
    ]


def fixture_records():
    return [
        {
            "step": 0,
            "final": {"relative_rms": 0.001, "cosine": 0.999},
            "proposals": {
                "ascend": [11, 99],
                "cuda_head_on_ascend_hidden": {"tokens": [11, 99]},
                "cuda_reference": {"tokens": [11, 12]},
            },
        },
        {
            "step": 1,
            "final": {"relative_rms": 0.002, "cosine": 0.998},
            "proposals": {
                "ascend": [13, 14],
                "cuda_head_on_ascend_hidden": {"tokens": [13, 14]},
                "cuda_reference": {"tokens": [13, 14]},
            },
        },
    ]


def test_summary_excludes_unverified_final_round_and_rejected_target_rows():
    summary = summarize(fixture_trace(), fixture_records())
    result = summary["proposal_comparison"]["cuda_reference"]
    assert result["matching_tokens"] == 3
    assert result["total_tokens"] == 4
    assert result["verified_rounds"] == 1
    assert result["observed_accepted"] == 1
    assert result["reference_prefix_matches_on_recorded_continuation"] == 2


def test_summary_rejects_inconsistent_target_trajectory():
    trace = fixture_trace()
    trace[1]["target_token_ids"][1] = 100
    with pytest.raises(ValueError, match="committed target trajectory"):
        summarize(trace, fixture_records())


def test_prefix_stops_at_first_rejection():
    assert prefix_length([1, 99, 3], [1, 2, 3]) == 1
    assert prefix_length([1, 2], []) == 0
