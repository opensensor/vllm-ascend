# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline speculative-round estimates from one isolated request's counters.

The server must draft the specified fixed width every round, without other
traffic in the counter window. Endpoint timing and periodic metric publication
make these estimates, not measured verification/draft kernel timings.
"""

import argparse
import json
import math
from pathlib import Path


def round_estimate(row: dict, draft_length: int) -> dict:
    if isinstance(draft_length, bool) or not isinstance(draft_length, int) or draft_length <= 0:
        raise ValueError("draft length must be a positive integer")
    drafted = row.get("drafted_delta")
    accepted = row.get("accepted_delta")
    rate = row.get("server_decode_tok_s", row.get("client_decode_tok_s"))
    if any(value is None or not math.isfinite(value) for value in (drafted, accepted, rate)):
        raise ValueError("finite draft/accepted counters and decode rate are required")
    if drafted <= 0 or accepted < 0 or accepted > drafted or rate <= 0:
        raise ValueError("invalid speculative counters or decode rate")
    rounds = drafted / draft_length
    tokens = 1 + draft_length * accepted / drafted
    return {
        "draft_length": draft_length,
        "drafted_delta": drafted,
        "accepted_delta": accepted,
        "estimated_rounds": rounds,
        "mean_accepted_drafts_per_round": tokens - 1,
        "estimated_emitted_tokens_per_round": tokens,
        "maximum_new_position_acceptance": (tokens - 1) / draft_length,
        "estimated_round_ms": 1000 * tokens / rate,
        "observed_decode_tok_s": rate,
        "assumption": "isolated_request_full_fixed_width_drafts_one_target_token_per_round",
        "timing_warning": "endpoint_and_counter_window_bias_not_direct_round_or_kernel_timing",
    }


def added_draft_break_even(estimate: dict, marginal_acceptance: float) -> dict:
    """Cost budget if one additional draft retains earlier-position acceptance.

    Marginal acceptance is the unconditional probability of accepting the new
    position (including rejection at earlier positions), not a conditional rate
    after earlier tokens have already been accepted.
    """
    if not math.isfinite(marginal_acceptance) or not 0 <= marginal_acceptance <= 1:
        raise ValueError("marginal acceptance must be in [0, 1]")
    # Accepted drafts form a prefix. The last existing position cannot exceed
    # the average acceptance of those positions; a new position cannot exceed
    # it either if earlier acceptance is unchanged.
    if marginal_acceptance > estimate["maximum_new_position_acceptance"] + 1e-12:
        raise ValueError("marginal acceptance exceeds the unchanged-prefix upper bound")
    tokens = estimate["estimated_emitted_tokens_per_round"]
    cycle = estimate["estimated_round_ms"]
    headroom = marginal_acceptance / tokens
    return {
        "unconditional_new_position_acceptance": marginal_acceptance,
        "maximum_round_time_increase_fraction": headroom,
        "maximum_round_time_increase_ms": cycle * headroom,
        "break_even_round_ms": cycle * (1 + headroom),
        "assumption": "earlier_acceptance_unchanged_including_target_graph_shape_effects",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--draft-length", type=int, required=True)
    args = parser.parse_args()
    results = []
    for line in args.records.read_text().splitlines():
        row = json.loads(line)
        if row.get("warmup"):
            continue
        estimate = round_estimate(row, args.draft_length)
        results.append(
            {
                "request_id": row.get("request_id"),
                "round_estimate": estimate,
                "one_extra_draft_break_even": [
                    added_draft_break_even(estimate, probability)
                    for probability in sorted(
                        {estimate["maximum_new_position_acceptance"]}
                        | {
                            value
                            for value in (0, 0.2, 0.4, 0.6, 0.8)
                            if value <= estimate["maximum_new_position_acceptance"]
                        }
                    )
                ],
            }
        )
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
