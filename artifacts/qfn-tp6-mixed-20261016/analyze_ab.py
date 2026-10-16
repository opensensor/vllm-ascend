# SPDX-License-Identifier: Apache-2.0
"""Summarize the controlled ABBAAB sequence without claiming token-level gaps."""

import json
import statistics
import sys
from pathlib import Path


def summarize(rows):
    result = {}
    for enabled in (False, True):
        state = result["shared" if enabled else "checkpoint"] = {}
        for kind in ("prefill", "mixed_prefill", "mixed_decode"):
            samples = [row for row in rows if row["enabled"] == enabled and row["kind"] == kind]
            assert samples
            assert all(row["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 for row in samples)
            state[kind] = dict(
                trials=len(samples),
                median_ttft_s=statistics.median(row["ttft_s"] for row in samples),
                ttft_s=[row["ttft_s"] for row in samples],
                prompt_tokens=[row["usage"]["prompt_tokens"] for row in samples],
                median_max_content_gap_s=statistics.median(row["max_content_gap_s"] for row in samples),
                max_content_gaps_s=[row["max_content_gap_s"] for row in samples],
                median_prompt_tokens_per_ttft_s=statistics.median(
                    row["usage"]["prompt_tokens"] / row["ttft_s"] for row in samples
                ),
            )
    result["prefill_ttft_reduction_percent"] = 100 * (
        1 - result["shared"]["prefill"]["median_ttft_s"] / result["checkpoint"]["prefill"]["median_ttft_s"]
    )
    return result


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    destination.write_text(json.dumps(summarize(json.loads(source.read_text())), indent=2) + "\n")
