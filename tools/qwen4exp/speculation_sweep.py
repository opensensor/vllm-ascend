# SPDX-License-Identifier: Apache-2.0
"""Prepare MTP0/1/2 arms and compare matched, sustained records offline.

This tool never starts, pauses, changes or contacts a server. Each arm requires
a separately planned cache and graph capture; these are not live hot swaps.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

HARD_THERMAL_LIMIT_C = 96


def profiles() -> dict:
    arms = []
    for depth in (0, 1, 2):
        query_len = depth + 1
        arms.append(
            {
                "name": f"mtp{depth}",
                "speculative_config": {"method": "mtp", "num_speculative_tokens": depth} if depth else None,
                "omit_speculative_config": depth == 0,
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [query_len, 3 * query_len],
                },
                "requires_cache_replan_and_graph_recapture": True,
                "requires_image_gate": True,
                "concurrencies_to_measure": [1, 2, 3],
            }
        )
    return {
        "autostart": False,
        "hardware_validated": False,
        "preserve_image_enabled": True,
        "thermal_hold_c": 94,
        "thermal_resume_c": 85,
        "arms": arms,
    }


def request_fingerprint(request: dict) -> str:
    # Serving aliases differ by arm. Every other request field, including
    # image payload, thinking policy, seed and output cap, must match.
    value = {key: item for key, item in request.items() if key != "model"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare(records: list[dict], min_sustained_seconds: float = 600) -> dict:
    if not math.isfinite(min_sustained_seconds) or min_sustained_seconds <= 0:
        raise ValueError("min_sustained_seconds must be finite and positive")
    matched = defaultdict(dict)
    for row in records:
        if row.get("arm_failed"):
            raise ValueError("sweep includes a failed arm; no recommendation")
        if row.get("warmup"):
            continue
        depth = row["draft_length"]
        if type(depth) is not int or depth not in (0, 1, 2):
            raise ValueError("draft_length must be 0, 1 or 2")
        concurrency = row["concurrency"]
        if type(concurrency) is not int or concurrency not in (1, 2, 3):
            raise ValueError("concurrency must be 1, 2 or 3")
        key = (request_fingerprint(row["request"]), concurrency, row["repeat"])
        if depth in matched[key]:
            raise ValueError("duplicate arm for one matched workload/repeat")
        for field in ("completion_tokens", "decode_seconds", "ttft_seconds", "sustained_seconds", "max_core_c"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        rounds = row.get("decode_rounds", 1)
        if type(rounds) is not int or rounds <= 0:
            raise ValueError("decode_rounds must be a positive integer")
        if row["completion_tokens"] <= concurrency * rounds or row["decode_seconds"] <= 0:
            raise ValueError("positive decode duration and more than one token per request required")
        if row["ttft_seconds"] < 0 or row["sustained_seconds"] < 0 or row["max_core_c"] < 0:
            raise ValueError("timings and temperature cannot be negative")
        energy = row.get("energy_joules")
        if energy is not None and (not math.isfinite(energy) or energy < 0):
            raise ValueError("energy must be finite and nonnegative")
        # Same policy as N-1 token gap timing for each of C concurrent requests.
        tokens = row["completion_tokens"] - concurrency * rounds
        wall = row.get("wall_seconds")
        if wall is not None and (not math.isfinite(wall) or wall <= 0):
            raise ValueError("wall_seconds must be finite and positive")
        matched[key][depth] = {
            "decode_tokens_per_second": tokens / row["decode_seconds"],
            "ttft_seconds": row["ttft_seconds"],
            "wall_tokens_per_second": row["completion_tokens"] / wall if wall is not None else None,
            "joules_per_completion_token": energy / row["completion_tokens"] if energy is not None else None,
            "max_core_c": row["max_core_c"],
            "qualified": (
                row.get("quality_pass") is True
                and row.get("image_pass") is True
                and row.get("thermal_shutdown") is False
                and row.get("thermal_policy_pass") is True
                and row["max_core_c"] < HARD_THERMAL_LIMIT_C
                and row["sustained_seconds"] >= min_sustained_seconds
            ),
        }
    if not matched:
        raise ValueError("no non-warmup measurements")
    incomplete = [key for key, arms in matched.items() if set(arms) != {0, 1, 2}]
    if incomplete:
        raise ValueError(f"{len(incomplete)} workload/repeats lack a matched MTP0/1/2 arm")
    workloads = defaultdict(lambda: defaultdict(list))
    for (fingerprint, concurrency, _), arms in matched.items():
        if len({row["wall_tokens_per_second"] is not None for row in arms.values()}) != 1:
            raise ValueError("all matched arms must use the same timing basis")
        for depth, row in arms.items():
            workloads[fingerprint, concurrency][depth].append(row)
    comparisons = []
    for (fingerprint, concurrency), arms in sorted(workloads.items()):
        summaries = []
        for depth, rows in sorted(arms.items()):
            energies = [row["joules_per_completion_token"] for row in rows]
            walls = [row["wall_tokens_per_second"] for row in rows]
            summaries.append(
                {
                    "draft_length": depth,
                    "repeats": len(rows),
                    "median_decode_tokens_per_second": median(row["decode_tokens_per_second"] for row in rows),
                    "median_ttft_seconds": median(row["ttft_seconds"] for row in rows),
                    "median_wall_tokens_per_second": median(walls) if all(w is not None for w in walls) else None,
                    "median_joules_per_completion_token": median(energies)
                    if all(e is not None for e in energies)
                    else None,
                    "qualified": len(rows) >= 3 and all(row["qualified"] for row in rows),
                }
            )
        eligible = [arm for arm in summaries if arm["qualified"]]
        comparisons.append(
            {
                "request_sha256": fingerprint,
                "concurrency": concurrency,
                "arms": summaries,
                "recommended_draft_length": (
                    max(
                        eligible,
                        key=lambda arm: arm["median_wall_tokens_per_second"] or arm["median_decode_tokens_per_second"],
                    )["draft_length"]
                    if eligible
                    else None
                ),
            }
        )
    return {"comparisons": comparisons, "min_sustained_seconds": min_sustained_seconds, "server_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--min-sustained-seconds", type=float, default=600)
    args = parser.parse_args()
    value = (
        compare(
            [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()],
            args.min_sustained_seconds,
        )
        if args.records
        else profiles()
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
